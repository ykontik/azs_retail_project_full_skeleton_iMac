#!/usr/bin/env python3
"""
SHAP-отчёт по важности признаков для одной модели (per-SKU LGBM/XGBoost).

Пример:
  python scripts/shap_report.py --store 1 --family AUTOMOTIVE \
    --train data_raw/train.csv --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv --holidays data_raw/holidays_events.csv --stores data_raw/stores.csv

Артефакты:
  - data_dw/shap_summary_{store}__{family}.png
  - data_dw/shap_top_{store}__{family}.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap  # типичная версия 0.45+
except Exception:  # pragma: no cover
    raise SystemExit("SHAP не установлен. Добавьте shap в requirements или установите локально.")

from make_features import make_features


def _prepare_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def parse_args():
    p = argparse.ArgumentParser("SHAP report per-SKU")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--transactions", required=True)
    p.add_argument("--oil", required=True)
    p.add_argument("--holidays", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--sample", type=int, default=2000, help="Размер выборки для отчёта")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    model_path = (
        Path(args.models_dir) / f"{args.store}__{str(args.family).replace(' ', '_')}.joblib"
    )
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}")
    model = joblib.load(model_path)

    train = pd.read_csv(args.train, parse_dates=["date"])
    trans = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    hol = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    X, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    df_pair = (
        X[(X["store_nbr"] == int(args.store)) & (X["family"] == str(args.family))]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df_pair.empty:
        raise SystemExit("Нет данных для пары.")

    cat_cols = [
        c
        for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]
        if c in df_pair.columns
    ]
    df_pair = _prepare_categoricals(df_pair, cat_cols)

    # Подготовка признаков
    feat_names = [c for c in df_pair.columns if c not in {"id", "sales", "date"}]
    Xn = df_pair[feat_names]
    if len(Xn) > args.sample:
        Xn = Xn.tail(args.sample).copy()

    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(Xn)

    # Сохранить сводные результаты
    out_png = (
        Path(args.warehouse_dir)
        / f"shap_summary_{args.store}__{str(args.family).replace(' ', '_')}.png"
    )
    import matplotlib.pyplot as plt

    shap.plots.beeswarm(shap_values, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Таблица топ‑фич по средней |SHAP|
    mean_abs = pd.DataFrame(
        {
            "feature": feat_names,
            "mean_abs_shap": np.mean(np.abs(shap_values.values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    out_csv = (
        Path(args.warehouse_dir)
        / f"shap_top_{args.store}__{str(args.family).replace(' ', '_')}.csv"
    )
    mean_abs.to_csv(out_csv, index=False)
    print("Saved:", out_png, out_csv)


if __name__ == "__main__":
    main()
