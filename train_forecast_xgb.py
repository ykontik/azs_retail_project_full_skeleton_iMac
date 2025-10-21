#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm
from xgboost import XGBRegressor

from make_features import make_features


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def parse_args():
    p = argparse.ArgumentParser(description="Train per-SKU XGBoost models")
    p.add_argument("--train", required=False)
    p.add_argument("--transactions", required=False)
    p.add_argument("--oil", required=False)
    p.add_argument("--holidays", required=False)
    p.add_argument("--stores", required=False)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--top_n_sku", type=int, default=50)
    p.add_argument("--top_recent_days", type=int, default=90)
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def fill_paths_from_env(args):
    raw_dir = os.getenv("RAW_DIR", "data_raw")

    def _need(path_value):
        return (path_value is None) or (str(path_value).strip() == "")

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


def pick_top_sku(df: pd.DataFrame, top_n: int, recent_days: int) -> pd.DataFrame:
    df = df.copy()
    if recent_days and recent_days > 0 and "date" in df.columns:
        max_date = pd.to_datetime(df["date"]).max()
        min_date = max_date - pd.Timedelta(days=recent_days - 1)
        df = df[pd.to_datetime(df["date"]) >= min_date]
    return (
        df.groupby(["store_nbr", "family"])["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )


def main():
    args = parse_args()
    args = fill_paths_from_env(args)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]:
        if c in Xfull.columns:
            Xfull[c] = Xfull[c].astype("category")
    dt_cols = Xfull.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")
    exclude_cols = {"id", "sales", "date", *dt_cols}
    feat_cols = [c for c in Xfull.columns if c not in exclude_cols]

    top_pairs = set(
        map(
            tuple,
            pick_top_sku(train, args.top_n_sku, args.top_recent_days)[
                ["store_nbr", "family"]
            ].values.tolist(),
        )
    )

    groups = list(Xfull.groupby(["store_nbr", "family"], sort=False, observed=True))
    bar = tqdm(groups, desc="Training XGB per-SKU", unit="pair")
    for (store, fam), df_grp in bar:
        if (int(store), str(fam)) not in top_pairs:
            continue
        df_grp = df_grp.sort_values("date").reset_index(drop=True)
        if df_grp.empty:
            continue
        y = df_grp["sales"].values
        X = df_grp[feat_cols]
        max_date = df_grp["date"].max()
        min_valid = max_date - pd.Timedelta(days=args.valid_days - 1)
        m_val = df_grp["date"] >= min_valid
        if m_val.sum() < 7:
            continue
        X_tr, y_tr = X[~m_val], y[~m_val]
        X_va, y_va = X[m_val], y[m_val]

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=args.random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
            enable_categorical=True,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        base = f"{int(store)}__{str(fam).replace(' ', '_')}"
        model_path = Path(args.models_dir) / f"{base}__xgb.joblib"
        dump(model, model_path)
        # Сохраним список признаков в порядке обучения рядом с моделью
        try:
            import json

            feat_path = Path(args.models_dir) / f"{base}.features.json"
            feat_path.write_text(
                json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    print("OK: XGB per-SKU complete")


if __name__ == "__main__":
    main()
