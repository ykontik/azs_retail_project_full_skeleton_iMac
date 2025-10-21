#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna-тюнинг гиперпараметров LGBM для одной пары (store_nbr, family) или автоматически выбранной топ‑пары.

Пример:
  python experiments/tune_optuna.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --store 1 --family AUTOMOTIVE \
    --valid_days 28 --n_trials 50

Результаты:
  - Лучшие параметры → data_dw/best_params_{store}__{family}.json
  - Модель с лучшими параметрами → models/{store}__{family}__optuna.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from joblib import dump

try:
    import optuna
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Optuna не установлена. Добавьте optuna в requirements.txt или установите локально."
    )

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from make_features import make_features


def parse_args():
    p = argparse.ArgumentParser("Optuna LGBM per-SKU")
    p.add_argument("--train", required=True)
    p.add_argument("--transactions", required=True)
    p.add_argument("--oil", required=True)
    p.add_argument("--holidays", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--store", type=int, default=None)
    p.add_argument("--family", type=str, default=None)
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def select_top_pair(train: pd.DataFrame) -> tuple[int, str]:
    top = (
        train.groupby(["store_nbr", "family"])["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(1)
    )
    r = top.iloc[0]
    return int(r.store_nbr), str(r.family)


def objective_factory(
    df_pair: pd.DataFrame, cat_cols: list[str], valid_days: int, random_state: int
):
    df_pair = df_pair.sort_values("date").reset_index(drop=True)
    y = df_pair["sales"].values
    feat_cols = [c for c in df_pair.columns if c not in {"id", "sales", "date"}]

    max_date = df_pair["date"].max()
    min_valid_date = max_date - pd.Timedelta(days=valid_days - 1)
    mask_val = df_pair["date"] >= min_valid_date
    X_tr, y_tr = df_pair.loc[~mask_val, feat_cols], y[~mask_val]
    X_va, y_va = df_pair.loc[mask_val, feat_cols], y[mask_val]

    def _objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
            "objective": "l1",
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            categorical_feature=[c for c in cat_cols if c in feat_cols],
            callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=0)],
        )
        pred = model.predict(X_va)
        mae = float(mean_absolute_error(y_va, pred))
        trial.set_user_attr("best_iteration_", getattr(model, "best_iteration_", None))
        return mae

    return _objective, (X_tr, y_tr, X_va, y_va, feat_cols)


def main():
    args = parse_args()
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    X, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    cat_cols = [
        c
        for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]
        if c in X.columns
    ]
    for c in cat_cols:
        X[c] = X[c].astype("category")

    if args.store is None or args.family is None:
        s, f = select_top_pair(train)
    else:
        s, f = int(args.store), str(args.family)

    df_pair = X[(X["store_nbr"] == s) & (X["family"] == f)].copy()
    if df_pair.empty:
        raise SystemExit("Не найдены данные для выбранной пары.")

    objective, ctx = objective_factory(df_pair, cat_cols, args.valid_days, args.random_state)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iteration_")
    if best_iter:
        best["n_estimators"] = best_iter

    out_json = Path(args.warehouse_dir) / f"best_params_{s}__{str(f).replace(' ', '_')}.json"
    out_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    # Обучим финальную модель на всех train данных пары, валидация прежняя
    X_tr, y_tr, X_va, y_va, feat_cols = ctx
    best.update(
        {"objective": "l1", "random_state": args.random_state, "n_jobs": -1, "verbosity": -1}
    )
    model = lgb.LGBMRegressor(**best)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=[c for c in cat_cols if c in feat_cols],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    name = f"{s}__{str(f).replace(' ', '_')}__optuna.joblib"
    dump(model, str(Path(args.models_dir) / name))
    print("Saved:", name)


if __name__ == "__main__":
    main()
