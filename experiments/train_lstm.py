#!/usr/bin/env python3
"""Обучение per-SKU LSTM-бейзлайна для прогноза продаж.

Скрипт повторяет логику остальных экспериментов: готовит выборки из features,
обучает модель на заданном окне, сохраняет веса/скейлеры и метрики.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype, is_bool_dtype, is_object_dtype
from tqdm import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - torch ставится опционально
    raise SystemExit(
        "Torch не установлен. Установите PyTorch: pip install torch>=2.0.0"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from make_features import make_features  # noqa: E402
from train_forecast import (  # noqa: E402
    _sanitize_family_name,
    _safe_metrics,
    mape,
    pick_top_sku,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-SKU LSTM baseline")
    parser.add_argument("--train", default=None)
    parser.add_argument("--transactions", default=None)
    parser.add_argument("--oil", default=None)
    parser.add_argument("--holidays", default=None)
    parser.add_argument("--stores", default=None)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--warehouse_dir", default="data_dw")
    parser.add_argument("--top_n_sku", type=int, default=20)
    parser.add_argument("--top_recent_days", type=int, default=90)
    parser.add_argument("--valid_days", type=int, default=28)
    parser.add_argument("--window", type=int, default=30, help="Размер временного окна")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Патience для early stopping")
    parser.add_argument("--store", type=int, default=None, help="Опционально обучить конкретную пару")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (по умолчанию auto)")
    return parser.parse_args()


def fill_paths_from_env(args: argparse.Namespace) -> argparse.Namespace:
    raw_dir = os.getenv("RAW_DIR", "data_raw")

    def _need(x: Optional[str]) -> bool:
        return x is None or str(x).strip() == ""

    if _need(args.train):
        args.train = str(Path(raw_dir) / "train.csv")
    if _need(args.transactions):
        args.transactions = str(Path(raw_dir) / "transactions.csv")
    if _need(args.oil):
        args.oil = str(Path(raw_dir) / "oil.csv")
    if _need(args.holidays):
        args.holidays = str(Path(raw_dir) / "holidays_events.csv")
    if _need(args.stores):
        args.stores = str(Path(raw_dir) / "stores.csv")
    return args


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.x = torch.from_numpy(sequences.astype(np.float32))
        self.y = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.squeeze(-1)


@dataclass
class PairData:
    store: int
    family: str
    feature_cols: List[str]
    train_sequences: np.ndarray
    train_targets: np.ndarray
    val_sequences: np.ndarray
    val_targets: np.ndarray
    val_dates: Sequence[pd.Timestamp]
    scaler: StandardScaler


def ensure_categorical(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            try:
                out[col] = out[col].astype("category")
            except Exception:
                pass
    return out


def to_numeric_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col]
        if is_bool_dtype(series):
            out[col] = series.astype("int8")
        elif isinstance(series.dtype, CategoricalDtype):
            out[col] = series.cat.codes.astype("int16")
        elif is_object_dtype(series):
            out[col] = series.astype("category").cat.codes.astype("int16")
    return out.fillna(0.0)


def prepare_pair_data(
    df_pair: pd.DataFrame,
    window: int,
    valid_days: int,
    cat_cols: Iterable[str],
) -> Optional[PairData]:
    df_pair = df_pair.sort_values("date").reset_index(drop=True)
    if len(df_pair) < max(window + valid_days, 60):
        return None

    max_date = df_pair["date"].max()
    min_valid = max_date - pd.Timedelta(days=valid_days - 1)
    val_mask = df_pair["date"] >= min_valid
    if val_mask.sum() < max(7, valid_days // 2):
        return None
    val_start_idx = int(np.where(val_mask)[0][0])

    df_pair = ensure_categorical(df_pair, cat_cols)
    datetime_cols = df_pair.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    exclude_cols = {"id", "sales", "date", *datetime_cols}
    feature_cols = [c for c in df_pair.columns if c not in exclude_cols]

    feature_df = to_numeric_features(df_pair[feature_cols], feature_cols)

    scaler = StandardScaler()
    train_slice = feature_df.loc[: val_start_idx - 1]
    scaler.fit(train_slice)
    features_scaled = scaler.transform(feature_df)
    targets = df_pair["sales"].values.astype(np.float32)

    def build_sequences(end_idx: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        seqs: List[np.ndarray] = []
        tgts: List[float] = []
        dts: List[pd.Timestamp] = []
        for idx in range(window, end_idx):
            seqs.append(features_scaled[idx - window : idx])
            tgts.append(float(targets[idx]))
            dts.append(df_pair.loc[idx, "date"])
        if not seqs:
            return np.empty((0, window, features_scaled.shape[1]), dtype=np.float32), np.empty(
                (0,), dtype=np.float32
            ), []
        return (
            np.stack(seqs).astype(np.float32),
            np.array(tgts, dtype=np.float32),
            dts,
        )

    train_sequences, train_targets, _ = build_sequences(val_start_idx)
    val_sequences, val_targets, val_dates = build_sequences(len(df_pair))
    if len(train_targets) == 0 or len(val_targets) == 0:
        return None

    return PairData(
        store=int(df_pair.iloc[0]["store_nbr"]),
        family=str(df_pair.iloc[0]["family"]),
        feature_cols=feature_cols,
        train_sequences=train_sequences,
        train_targets=train_targets,
        val_sequences=val_sequences,
        val_targets=val_targets,
        val_dates=val_dates,
        scaler=scaler,
    )


def get_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lstm_for_pair(
    pair: PairData,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    model = LSTMRegressor(
        input_dim=pair.train_sequences.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        SequenceDataset(pair.train_sequences, pair.train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(pair.val_sequences, pair.val_targets),
        batch_size=args.batch_size,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.detach().cpu()))
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if mean_val_loss + 1e-6 < best_loss:
            best_loss = mean_val_loss
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy()
            preds.append(p)
        pred_array = np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)

    metrics = {
        "predictions": pred_array,
        "best_val_loss": best_loss,
    }
    return model, metrics


def main() -> None:
    args = parse_args()
    args = fill_paths_from_env(args)

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, parse_dates=["date"])
    transactions = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    holidays = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    Xfull, _ = make_features(train, holidays, transactions, oil, stores, dropna_target=True)

    cat_cols = ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]
    Xfull = ensure_categorical(Xfull, cat_cols)

    if args.store is not None and args.family is not None:
        top_pairs = {(int(args.store), str(args.family))}
    else:
        top_pairs_df = pick_top_sku(train, args.top_n_sku, args.top_recent_days)
        top_pairs = set(map(tuple, top_pairs_df[["store_nbr", "family"]].values.tolist()))

    device = get_device(args.device)
    print(f"Использую устройство: {device}")

    groupby_pairs = Xfull.groupby(["store_nbr", "family"], sort=False, observed=True)

    metrics_rows: List[dict] = []
    progress = tqdm(groupby_pairs, desc="Training LSTM per-SKU", unit="pair")
    for (store, family), df_pair in progress:
        store_int = int(store)
        family_str = str(family)
        if (store_int, family_str) not in top_pairs:
            continue

        pair_data = prepare_pair_data(df_pair, args.window, args.valid_days, cat_cols)
        if pair_data is None:
            continue

        model, info = train_lstm_for_pair(pair_data, args, device)

        preds = info["predictions"]
        if preds.size == 0:
            continue

        mae = float(np.mean(np.abs(pair_data.val_targets - preds)))
        mmape = float(mape(pair_data.val_targets, preds))

        # Базовые прогнозы для сравнения
        y_series = df_pair.sort_values("date").reset_index(drop=True)["sales"]
        val_mask = df_pair["date"].isin(pair_data.val_dates)
        naive_lag7 = y_series.shift(7).values[val_mask.values]
        mask_lag = ~np.isnan(naive_lag7)
        lag7_mae, lag7_mape = _safe_metrics(pair_data.val_targets[mask_lag], naive_lag7[mask_lag])

        naive_ma7_full = y_series.shift(1).rolling(7, min_periods=1).mean().values
        naive_ma7 = naive_ma7_full[val_mask.values]
        ma7_mae, ma7_mape = _safe_metrics(pair_data.val_targets, naive_ma7)

        safe_family = _sanitize_family_name(family_str)
        model_path = Path(args.models_dir) / f"{store_int}__{safe_family}__lstm.pt"
        scaler_path = model_path.with_suffix(".scaler.pkl")
        meta_path = model_path.with_suffix(".lstm.json")

        torch.save(model.state_dict(), model_path)
        joblib.dump(pair_data.scaler, scaler_path)

        meta = {
            "store_nbr": store_int,
            "family": family_str,
            "feature_columns": pair_data.feature_cols,
            "window": args.window,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "best_val_loss": info["best_val_loss"],
            "device": str(device),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics_rows.append(
            {
                "store_nbr": store_int,
                "family": family_str,
                "MAE": mae,
                "MAPE_%": mmape,
                "NAIVE_LAG7_MAE": lag7_mae,
                "NAIVE_LAG7_MAPE_%": lag7_mape,
                "NAIVE_MA7_MAE": ma7_mae,
                "NAIVE_MA7_MAPE_%": ma7_mape,
                "MAE_GAIN_vs_LAG7": (lag7_mae - mae) if lag7_mae == lag7_mae else None,
                "MAE_GAIN_vs_MA7": (ma7_mae - mae) if ma7_mae == ma7_mae else None,
                "VAL_SAMPLES": len(pair_data.val_targets),
            }
        )

    if not metrics_rows:
        print("Не удалось собрать метрики LSTM.")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = Path(args.warehouse_dir) / "metrics_lstm.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "Моделей обучено": int(len(metrics_rows)),
        "Средний MAE": float(metrics_df["MAE"].mean()),
        "Средний MAPE": float(metrics_df["MAPE_%"].mean()),
    }
    summary_path = Path(args.warehouse_dir) / "summary_metrics_lstm.txt"
    summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items())
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
