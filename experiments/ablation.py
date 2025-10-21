#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation-эксперименты: оцениваем вклад блоков фич, обучая модели без отдельных групп признаков.

Пример запуска:
  python experiments/ablation.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --warehouse_dir data_dw \
    --top_n_sku 10 --top_recent_days 90

Результат: data_dw/ablation_results.csv
"""
import argparse
import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from make_features import make_features


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def parse_args():
    p = argparse.ArgumentParser("Ablation per-SKU")
    p.add_argument("--train", required=True)
    p.add_argument("--transactions", required=True)
    p.add_argument("--oil", required=True)
    p.add_argument("--holidays", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--top_n_sku", type=int, default=10)
    p.add_argument("--top_recent_days", type=int, default=90)
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def pick_top_sku(df: pd.DataFrame, top_n: int, recent_days: int) -> set[tuple[int, str]]:
    df = df.copy()
    if recent_days and recent_days > 0 and "date" in df.columns:
        max_date = pd.to_datetime(df["date"]).max()
        min_date = max_date - pd.Timedelta(days=recent_days - 1)
        df = df[pd.to_datetime(df["date"]) >= min_date]
    top = (
        df.groupby(["store_nbr", "family"])["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )
    return set((int(r.store_nbr), str(r.family)) for _, r in top.iterrows())


def feature_groups():
    return {
        "calendar": [
            "year",
            "month",
            "week",
            "day",
            "dayofweek",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "trend",
        ],
        "special_days": ["is_christmas", "is_newyear", "is_black_friday"],
        "holidays": [
            "is_holiday",
            "is_holiday_national",
            "is_holiday_regional",
            "is_holiday_local",
        ],
        "transactions": ["transactions"],
        "oil": ["oil_price"],
        "lags": ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28"],
        "rolls": ["sales_rollmean_7", "sales_rollstd_7", "sales_rollmean_30", "sales_rollstd_30"],
        "store_meta": ["type", "city", "state", "cluster", "store_nbr", "family"],
    }


def train_eval(
    X: pd.DataFrame, y: np.ndarray, valid_days: int, cat_cols: list[str], random_state: int
) -> tuple[float, float, int]:
    df = X.copy()
    df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else pd.NaT
    df = df.sort_values("date").reset_index(drop=True)
    y_series = y
    max_date = df["date"].max()
    min_valid_date = max_date - pd.Timedelta(days=valid_days - 1)
    mask_val = df["date"] >= min_valid_date
    if mask_val.sum() < 7:
        return float("nan"), float("nan"), 0
    X_tr, y_tr = df.loc[~mask_val], y_series[~mask_val]
    X_va, y_va = df.loc[mask_val], y_series[mask_val]

    feat_cols = [c for c in df.columns if c not in {"id", "sales", "date"}]
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X_tr[feat_cols],
        y_tr,
        eval_set=[(X_va[feat_cols], y_va)],
        eval_metric="l1",
        categorical_feature=[c for c in cat_cols if c in feat_cols],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)],
    )
    pred = model.predict(X_va[feat_cols])
    return float(mean_absolute_error(y_va, pred)), float(mape(y_va, pred)), int(mask_val.sum())


def main():
    args = parse_args()
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    # Категориальные колонки, аналогично train_forecast
    cat_cols = [
        c
        for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]
        if c in Xfull.columns
    ]
    for c in cat_cols:
        Xfull[c] = Xfull[c].astype("category")

    dt_cols = Xfull.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")

    top_pairs = pick_top_sku(train, args.top_n_sku, args.top_recent_days)
    groups = feature_groups()

    rows = []
    groups_iter = list(Xfull.groupby(["store_nbr", "family"], sort=False, observed=True))
    total_pairs = len(groups_iter)
    t0 = time.perf_counter()
    bar = tqdm(
        groups_iter,
        desc="Ablation per-SKU",
        unit="pair",
        total=total_pairs,
        smoothing=0.3,
        mininterval=0.5,
    )
    for idx, ((store, fam), df_grp) in enumerate(bar, start=1):
        if (int(store), str(fam)) not in top_pairs:
            # обновим ETA/скорость, чтобы прогресс был стабильным
            elapsed = max(time.perf_counter() - t0, 1e-6)
            speed = idx / elapsed
            eta = max(total_pairs - idx, 0) / speed if speed > 0 else float("inf")
            bar.set_postfix_str(f"{speed:.2f} pair/s, ETA {eta:.1f}s")
            continue
        df_grp = df_grp.sort_values("date").reset_index(drop=True)
        y = df_grp["sales"].values
        # Базовая модель с полным набором фич
        base_mae, base_mape, nval = train_eval(
            df_grp, y, args.valid_days, cat_cols, args.random_state
        )
        rows.append(
            {
                "store_nbr": int(store),
                "family": str(fam),
                "setting": "full",
                "MAE": base_mae,
                "MAPE_%": base_mape,
                "n_val": nval,
            }
        )

        # Ablation: исключаем каждый блок фич по очереди
        inner = list(groups.items())
        inner_iter = (
            inner
            if len(inner) <= 1
            else tqdm(inner, desc=f"groups {int(store)}/{str(fam)}", unit="grp", leave=False)
        )
        for gname, cols in inner_iter:
            df_abl = df_grp.copy()
            drop_cols = [c for c in cols if c in df_abl.columns]
            if not drop_cols:
                continue
            df_abl = df_abl.drop(columns=drop_cols)
            mae, mape_v, nval = train_eval(df_abl, y, args.valid_days, cat_cols, args.random_state)
            rows.append(
                {
                    "store_nbr": int(store),
                    "family": str(fam),
                    "setting": f"w/o {gname}",
                    "MAE": mae,
                    "MAPE_%": mape_v,
                    "n_val": nval,
                }
            )
        # обновляем postfix скорости/ETA
        elapsed = max(time.perf_counter() - t0, 1e-6)
        speed = idx / elapsed
        eta = max(total_pairs - idx, 0) / speed if speed > 0 else float("inf")
        bar.set_postfix_str(f"{speed:.2f} pair/s, ETA {eta:.1f}s")

    out = pd.DataFrame(rows)
    out_path = Path(args.warehouse_dir) / "ablation_results.csv"
    out.to_csv(out_path, index=False)
    print("ablation →", out_path)


if __name__ == "__main__":
    main()
