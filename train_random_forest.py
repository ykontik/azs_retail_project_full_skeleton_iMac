#!/usr/bin/env python3
"""Обучение per-SKU моделей RandomForestRegressor.

Скрипт повторяет пайплайн XGBoost (train_forecast_xgb.py):
- формирует признаки через make_features;
- отбирает top-N пар (store_nbr, family) по сумме продаж;
- бьёт хвост на holdout-валидацию по последним valid_days;
- обучает RandomForestRegressor и сохраняет артефакты в `models/`;
- считает MAE/MAPE + сравнение с простыми baseline (lag-7, MA(7));
- сводит метрики в `data_dw/metrics_random_forest.csv` и summary в `data_dw/summary_metrics_random_forest.txt`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from make_features import make_features
from train_forecast import mape, pick_top_sku, _safe_metrics  # pylint: disable=protected-access


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-SKU RandomForest models")
    parser.add_argument("--train", default=None, help="Путь к train.csv (по умолчанию RAW_DIR/train.csv)")
    parser.add_argument("--transactions", default=None)
    parser.add_argument("--oil", default=None)
    parser.add_argument("--holidays", default=None)
    parser.add_argument("--stores", default=None)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--warehouse_dir", default="data_dw")
    parser.add_argument("--top_n_sku", type=int, default=50, help="Сколько пар обучать")
    parser.add_argument("--top_recent_days", type=int, default=90, help="Окно дней для отбора top-N")
    parser.add_argument("--valid_days", type=int, default=28, help="Размер holdout окна")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--min_samples_split", type=int, default=6)
    parser.add_argument("--max_features", default="sqrt")
    return parser.parse_args()


def fill_paths_from_env(args: argparse.Namespace) -> argparse.Namespace:
    raw_dir = os.getenv("RAW_DIR", "data_raw")

    def _need(x: Optional[str]) -> bool:
        return (x is None) or (str(x).strip() == "")

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


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует категориальные/булевы/объектные колонки в числовые коды."""
    out = df.copy()
    # Булевы → int8
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype("int8")
    # Объектные → категории → коды
    obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in obj_cols:
        out[col] = out[col].astype("category").cat.codes.astype("int16")
    # Категориальные → коды
    cat_cols = out.select_dtypes(include=["category"]).columns.tolist()
    for col in cat_cols:
        out[col] = out[col].cat.codes.astype("int16")
    return out.fillna(0.0)


def main() -> None:
    args = parse_args()
    args = fill_paths_from_env(args)

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    # Загружаем сырые данные
    train = pd.read_csv(args.train, parse_dates=["date"])
    transactions = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    holidays = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    # Формируем признаки. dropna_target=True — выкидываем строки без продаж
    Xfull, _ = make_features(train, holidays, transactions, oil, stores, dropna_target=True)

    # Отберём топовые пары для обучения
    top_pairs_df = pick_top_sku(train, args.top_n_sku, args.top_recent_days)
    top_pairs = set(map(tuple, top_pairs_df[["store_nbr", "family"]].values.tolist()))

    # Нормализуем типы, чтобы избежать object в признаках
    for col in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]:
        if col in Xfull.columns:
            try:
                Xfull[col] = Xfull[col].astype("category")
            except Exception:
                pass

    datetime_cols = Xfull.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    exclude_cols = {"id", "sales", "date", *datetime_cols}
    feature_cols = [c for c in Xfull.columns if c not in exclude_cols]

    metrics: list[dict[str, float | int | None]] = []
    trained = 0

    groups = list(Xfull.groupby(["store_nbr", "family"], sort=False, observed=True))
    progress = tqdm(groups, desc="Training RandomForest per-SKU", unit="pair")
    for (store, family), df_pair in progress:
        try:
            store_int = int(store)
            family_str = str(family)
        except Exception:
            continue
        if (store_int, family_str) not in top_pairs:
            continue
        df_pair = df_pair.sort_values("date").reset_index(drop=True)
        if df_pair.empty:
            continue

        y = df_pair["sales"].values
        X_pair = ensure_numeric(df_pair[feature_cols])

        # Holdout по последним valid_days
        max_date = df_pair["date"].max()
        min_valid = max_date - pd.Timedelta(days=args.valid_days - 1)
        mask_val = df_pair["date"] >= min_valid
        if mask_val.sum() < max(7, min(args.valid_days, len(df_pair)) // 2 or 1):
            # пропускаем пары, где валидационное окно совсем маленькое
            continue
        X_train, y_train = X_pair[~mask_val], y[~mask_val]
        X_val, y_val = X_pair[mask_val], y[mask_val]
        if len(y_train) < 20 or len(y_val) < 7:
            continue

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = float(np.mean(np.abs(y_val - y_pred)))
        mmape = float(mape(y_val, y_pred))

        # Baseline: lag-7 и MA(7)
        y_series = df_pair["sales"].reset_index(drop=True)
        naive_lag7 = y_series.shift(7).values[mask_val.values]
        mask_lag = ~np.isnan(naive_lag7)
        lag7_mae, lag7_mape = _safe_metrics(y_val[mask_lag], naive_lag7[mask_lag])

        naive_ma7_full = y_series.shift(1).rolling(7, min_periods=1).mean().values
        naive_ma7 = naive_ma7_full[mask_val.values]
        ma7_mae, ma7_mape = _safe_metrics(y_val, naive_ma7)

        metrics.append(
            {
                "store_nbr": store_int,
                "family": family_str,
                "MAE": mae,
                "MAPE_%": mmape,
                "NAIVE_LAG7_MAE": lag7_mae,
                "NAIVE_LAG7_MAPE_%": lag7_mape,
                "NAIVE_MA7_MAE": ma7_mae,
                "NAIVE_MA7_MAPE_%": ma7_mape,
                "MAE_GAIN_vs_LAG7": (lag7_mae - mae) if (lag7_mae == lag7_mae) else None,
                "MAE_GAIN_vs_MA7": (ma7_mae - mae) if (ma7_mae == ma7_mae) else None,
            }
        )

        model_stem = f"{store_int}__{family_str.replace(' ', '_')}"
        model_path = Path(args.models_dir) / f"{model_stem}__rf.joblib"
        dump(model, model_path)
        # сохраняем список признаков рядом, чтобы UI мог использовать
        feat_path = Path(args.models_dir) / f"{model_stem}__rf.features.json"
        try:
            feat_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
            feat_path.write_text(json.dumps(feat_names, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        trained += 1

    if metrics:
        df_metrics = pd.DataFrame(metrics)
        metrics_path = Path(args.warehouse_dir) / "metrics_random_forest.csv"
        df_metrics.to_csv(metrics_path, index=False)
        summary_path = Path(args.warehouse_dir) / "summary_metrics_random_forest.txt"
        summary_path.write_text(
            "\n".join(
                [
                    f"Моделей обучено: {trained}",
                    f"Средний MAE: {df_metrics['MAE'].mean():.3f}",
                    f"Средний MAPE: {df_metrics['MAPE_%'].mean():.2f}%",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"OK: RandomForest per-SKU complete. Saved {trained} models.")
        print(f"Метрики → {metrics_path}")
    else:
        print("WARN: не удалось обучить ни одной пары. Проверьте параметры top_n_sku/top_recent_days/valid_days.")


if __name__ == "__main__":
    main()
