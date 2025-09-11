#!/usr/bin/env python3
"""
Глобальная модель XGBoost для прогнозирования спроса по всем парам (store_nbr, family).

Комментарии — на русском, код — на английском (Python стиль).

Запуск:
  python train_global_xgboost.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --valid_days 28

Модель сохраняется в models/global_xgboost.joblib, метрики — в data_dw/metrics_global_xgboost.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBRegressor

from make_features import make_features


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def parse_args():
    p = argparse.ArgumentParser(description="Train global XGBoost model")
    p.add_argument("--train", default=None)
    p.add_argument("--transactions", default=None)
    p.add_argument("--oil", default=None)
    p.add_argument("--holidays", default=None)
    p.add_argument("--stores", default=None)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--log1p_target", action="store_true", help="Учить на log1p(sales) и предсказывать с обратным expm1")
    p.add_argument("--poisson", action="store_true", help="Использовать objective=count:poisson для счётчиков")
    p.add_argument("--mape_weighting", action="store_true", help="Веса 1/max(y,1) для MAPE‑ориентированного обучения")
    p.add_argument("--n_estimators", type=int, default=1200)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    return p.parse_args()


def fill_paths_from_env(args):
    raw_dir = os.getenv("RAW_DIR", "data_raw")
    def _need(x: Optional[str]) -> bool: return (x is None) or (str(x).strip() == "")
    if _need(args.train): args.train = str(Path(raw_dir) / "train.csv")
    if _need(args.transactions): args.transactions = str(Path(raw_dir) / "transactions.csv")
    if _need(args.oil): args.oil = str(Path(raw_dir) / "oil.csv")
    if _need(args.holidays): args.holidays = str(Path(raw_dir) / "holidays_events.csv")
    if _need(args.stores): args.stores = str(Path(raw_dir) / "stores.csv")
    return args


def main():
    args = parse_args()
    args = fill_paths_from_env(args)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    missing = [
        ("train", args.train), ("transactions", args.transactions), ("oil", args.oil),
        ("holidays", args.holidays), ("stores", args.stores)
    ]
    missing = [k for k, p in missing if (p is None) or (not Path(p).exists())]
    if missing:
        raise SystemExit("Отсутствуют входные файлы: " + ", ".join(missing))

    # Подготовка фич как при LGBM
    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    Xfull = Xfull.sort_values("date").reset_index(drop=True)

    # Разделение на train/val по последним valid_days
    max_date = Xfull["date"].max()
    min_valid_date = max_date - pd.Timedelta(days=args.valid_days - 1)
    m_val = Xfull["date"] >= min_valid_date
    if m_val.sum() < 7:
        raise SystemExit("Слишком короткий хвост для валидации (<7 точек). Уменьшите --valid_days.")

    # Исключим служебные/дата-колонки
    feat_cols = [c for c in Xfull.columns if c not in {"id","sales","date"}]
    # Переведём категориальные столбцы в pandas.Categorical для XGBoost
    cat_cols = [c for c in ["store_nbr","family","type","city","state","cluster","is_holiday"] if c in Xfull.columns]
    for c in cat_cols:
        if not pd.api.types.is_numeric_dtype(Xfull[c]):
            Xfull[c] = Xfull[c].astype("category")
    # Гарантируем наличие тонких флагов праздников
    for c in ["is_holiday_national","is_holiday_regional","is_holiday_local"]:
        if c not in Xfull.columns:
            Xfull[c] = 0
    # Детерминированный порядок признаков, чтобы train/val совпадали
    feat_cols = sorted(set(feat_cols))
    X_tr = Xfull.loc[~m_val, feat_cols].reindex(columns=feat_cols)
    y_tr = Xfull.loc[~m_val, "sales"].values
    X_va = Xfull.loc[m_val, feat_cols].reindex(columns=feat_cols)
    y_va = Xfull.loc[m_val, "sales"].values

    # Опциональная лог-трансформация цели
    if args.log1p_target:
        y_tr_fit = np.log1p(y_tr)
        y_va_fit = np.log1p(y_va)
    else:
        y_tr_fit = y_tr
        y_va_fit = y_va

    objective = "count:poisson" if args.poisson else "reg:squarederror"
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=1.0,
        objective=objective,
        random_state=args.random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mae",
    )
    # Подготовим аргументы fit, учитывая MAPE‑веса
    fit_kwargs = {
        "X": X_tr,
        "y": y_tr_fit,
        "eval_set": [(X_va, y_va_fit)],
        "verbose": False,
    }
    if args.mape_weighting:
        w_tr = 1.0 / np.clip(y_tr, 1.0, np.inf)
        w_va = 1.0 / np.clip(y_va, 1.0, np.inf)
        fit_kwargs["sample_weight"] = w_tr
        fit_kwargs_eval = {**fit_kwargs, "sample_weight_eval_set": [w_va]}
    else:
        fit_kwargs_eval = fit_kwargs

    # Обучение с поддержкой старых/новых API XGBoost: пробуем разные способы ранней остановки
    try:
        model.fit(**{**fit_kwargs_eval, "early_stopping_rounds": 200})
    except TypeError:
        try:
            import xgboost as xgb  # type: ignore
            if hasattr(xgb, "callback") and hasattr(xgb.callback, "EarlyStopping"):
                model.fit(**{**fit_kwargs, "callbacks": [xgb.callback.EarlyStopping(rounds=200, save_best=True)]})
            else:
                model.fit(**fit_kwargs)
        except Exception:
            model.fit(**fit_kwargs)

    # Предсказания с использованием лучшей итерации (если есть)
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        try:
            y_pred_fit = model.predict(X_va, iteration_range=(0, best_it + 1))
        except TypeError:
            best_ntree = getattr(model, "best_ntree_limit", None)
            if best_ntree is not None:
                y_pred_fit = model.predict(X_va, ntree_limit=best_ntree)
            else:
                y_pred_fit = model.predict(X_va)
    else:
        y_pred_fit = model.predict(X_va)

    # Обратная трансформация предсказаний, если обучались в лог-пространстве
    if args.log1p_target:
        y_pred = np.expm1(y_pred_fit)
    else:
        y_pred = y_pred_fit

    mae = float(np.mean(np.abs(y_va - y_pred)))
    mmape = mape(y_va, y_pred)

    # Бэйслайны на валидации: Naive lag-7 и MA(7)
    base = Xfull.loc[m_val].copy()
    naive_mae = None
    naive_mape = None
    ma7_mae = None
    ma7_mape = None
    if "sales_lag_7" in base.columns:
        mask_ok = base["sales_lag_7"].notna().values
        if mask_ok.any():
            yp = base.loc[mask_ok, "sales_lag_7"].values
            yt = y_va[mask_ok]
            naive_mae = float(np.mean(np.abs(yt - yp)))
            naive_mape = mape(yt, yp)
    if "sales_rollmean_7" in base.columns:
        mask_ok = base["sales_rollmean_7"].notna().values
        if mask_ok.any():
            yp = base.loc[mask_ok, "sales_rollmean_7"].values
            yt = y_va[mask_ok]
            ma7_mae = float(np.mean(np.abs(yt - yp)))
            ma7_mape = mape(yt, yp)

    out_path = Path(args.models_dir) / "global_xgboost.joblib"
    dump(model, out_path)

    metrics = {
        "MAE": mae,
        "MAPE_%": mmape,
        "valid_days": args.valid_days,
        "random_state": args.random_state,
        "features": feat_cols,
        "best_iteration": int(best_it) if best_it is not None else None,
        "log1p_target": bool(args.log1p_target),
        "poisson": bool(args.poisson),
        "mape_weighting": bool(args.mape_weighting),
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
    }
    if naive_mae is not None:
        metrics.update({
            "NAIVE7_MAE": naive_mae,
            "NAIVE7_MAPE_%": naive_mape,
            "MAE_GAIN_vs_NAIVE7": float(naive_mae - mae),
        })
    if ma7_mae is not None:
        metrics.update({
            "MA7_MAE": ma7_mae,
            "MA7_MAPE_%": ma7_mape,
            "MAE_GAIN_vs_MA7": float(ma7_mae - mae),
        })

    # Важности фич (gain)
    try:
        booster = model.get_booster()
        fi_gain = booster.get_score(importance_type="gain")
        top_fi = sorted(fi_gain.items(), key=lambda x: x[1], reverse=True)[:50]
        metrics["feature_importance_gain_top50"] = top_fi
    except Exception:
        pass
    with open(Path(args.warehouse_dir) / "metrics_global_xgboost.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"OK: global XGBoost saved → {out_path}\nMAE={mae:.4f}  MAPE={mmape:.2f}%")


if __name__ == "__main__":
    main()
