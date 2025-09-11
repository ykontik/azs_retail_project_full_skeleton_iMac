#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error

# Добавляем корень проекта в sys.path ДО импортов локальных модулей
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_features import make_features


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def main():
    p = argparse.ArgumentParser("Train single (store, family) model")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--data_dir", default="data_raw")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir/"train.csv", parse_dates=["date"])
    trans = pd.read_csv(data_dir/"transactions.csv", parse_dates=["date"])
    oil   = pd.read_csv(data_dir/"oil.csv", parse_dates=["date"])
    hol   = pd.read_csv(data_dir/"holidays_events.csv", parse_dates=["date"])
    stores= pd.read_csv(data_dir/"stores.csv")

    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)

    # категориальные как при обучении
    for c in ["store_nbr","family","type","city","state","cluster","is_holiday"]:
        if c in Xfull.columns:
            Xfull[c] = Xfull[c].astype("category")

    # исключаем datetime и служебные
    dt_cols = Xfull.select_dtypes(include=["datetime64[ns]","datetimetz"]).columns.tolist()
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")
    exclude = {"id","sales","date",*dt_cols}
    feat_cols = [c for c in Xfull.columns if c not in exclude]

    # фильтр по паре
    mask = (Xfull["store_nbr"] == args.store) & (Xfull["family"] == args.family)
    df = Xfull.loc[mask].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"Нет данных для пары ({args.store}, {args.family}). Проверь family в CSV.")
    y = df["sales"].values
    X = df[feat_cols]

    # сплит: последние valid_days в валидацию
    max_date = df["date"].max()
    min_valid_date = max_date - pd.Timedelta(days=args.valid_days-1)
    mask_val = df["date"] >= min_valid_date
    if mask_val.sum() < 7:
        raise SystemExit("Слишком короткий хвост для валидации (<7 точек). Уменьши --valid_days.")

    X_tr, y_tr = X[~mask_val], y[~mask_val]
    X_va, y_va = X[mask_val], y[mask_val]

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        categorical_feature=[c for c in ["store_nbr","family","type","city","state","cluster","is_holiday"] if c in feat_cols],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0),
        ],
    )

    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    mmape = mape(y_va, pred)
    print(f"OK: ({args.store}, {args.family})  MAE={mae:.3f}  MAPE={mmape:.2f}%")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.models_dir) / f"{args.store}__{args.family.replace(' ', '_')}.joblib"
    dump(model, out)
    print("Saved →", out)

if __name__ == "__main__":
    main()
