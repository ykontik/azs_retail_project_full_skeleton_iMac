#!/usr/bin/env python3
"""
Глобальная модель CatBoost для прогнозирования спроса по всем парам (store_nbr, family).

Комментарии — на русском, код — на английском (Python стиль).

Запуск (пути берутся из RAW_DIR, либо задаются явно):
  python train_global_catboost.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --valid_days 28

Модель сохраняется в models/global_catboost.cbm, метрики — в data_dw/metrics_global_catboost.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm

from make_features import make_features


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def parse_args():
    p = argparse.ArgumentParser(description="Train global CatBoost model")
    p.add_argument("--train", default=None)
    p.add_argument("--transactions", default=None)
    p.add_argument("--oil", default=None)
    p.add_argument("--holidays", default=None)
    p.add_argument("--stores", default=None)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2_leaf_reg", type=float, default=4.0)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def fill_paths_from_env(args):
    raw_dir = os.getenv("RAW_DIR", "data_raw")

    def _need(path_value: Optional[str]) -> bool:
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


def main():
    args = parse_args()
    args = fill_paths_from_env(args)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    missing = [
        ("train", args.train),
        ("transactions", args.transactions),
        ("oil", args.oil),
        ("holidays", args.holidays),
        ("stores", args.stores),
    ]
    missing = [k for k, p in missing if (p is None) or (not Path(p).exists())]
    if missing:
        raise SystemExit("Отсутствуют входные файлы: " + ", ".join(missing))

    # Прогресс по шагам обучения глобальной модели
    pbar = tqdm(total=6, desc="Global CatBoost", unit="step")

    # Подготовка фич как при LGBM
    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)
    pbar.update(1)

    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    Xfull = Xfull.sort_values("date").reset_index(drop=True)
    pbar.update(1)

    # Разделение на train/val по последним valid_days
    max_date = Xfull["date"].max()
    min_valid_date = max_date - pd.Timedelta(days=args.valid_days - 1)
    m_val = Xfull["date"] >= min_valid_date
    if m_val.sum() < 7:
        raise SystemExit("Слишком короткий хвост для валидации (<7 точек). Уменьшите --valid_days.")
    pbar.update(1)

    # Категориальные фичи
    cat_cols: List[str] = [
        c
        for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]
        if c in Xfull.columns
    ]
    # Исключаем служебные
    feat_cols = [c for c in Xfull.columns if c not in {"id", "sales", "date"}]

    X_tr = Xfull.loc[~m_val, feat_cols]
    y_tr = Xfull.loc[~m_val, "sales"].values
    X_va = Xfull.loc[m_val, feat_cols]
    y_va = Xfull.loc[m_val, "sales"].values

    # Пулы CatBoost: имена фич + индексы категориальных
    cat_idx = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_idx, feature_names=feat_cols)
    pool_va = Pool(X_va, y_va, cat_features=cat_idx, feature_names=feat_cols)
    pbar.update(1)

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(pool_tr, eval_set=pool_va, verbose=False)
    pbar.update(1)

    y_pred = model.predict(pool_va)
    pbar.update(1)
    mae = float(np.mean(np.abs(y_va - y_pred)))
    mmape = mape(y_va, y_pred)

    out_path = Path(args.models_dir) / "global_catboost.cbm"
    model.save_model(str(out_path))

    metrics = {
        "MAE": mae,
        "MAPE_%": mmape,
        "valid_days": args.valid_days,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_state": args.random_state,
        "features": feat_cols,
        "categoricals": cat_cols,
    }
    with open(
        Path(args.warehouse_dir) / "metrics_global_catboost.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pbar.close()
    print(f"OK: global CatBoost saved → {out_path}\nMAE={mae:.4f}  MAPE={mmape:.2f}%")


if __name__ == "__main__":
    main()
