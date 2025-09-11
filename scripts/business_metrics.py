#!/usr/bin/env python3
"""
Бизнес-метрики: перевод MAPE в деньги и расчет страхового запаса/ROP.

Пример:
  python scripts/business_metrics.py --store 1 --family AUTOMOTIVE \
    --price 3.5 --margin_rate 0.25 --holding_cost 0.05 --lead_time_days 2 --service_level 0.95

Выводит:
  - daily_mean (последние 30 дней)
  - MAPE (из metrics_per_sku.csv, если доступен)
  - sigma ≈ MAPE% * daily_mean
  - safety_stock, reorder_point
  - Приближенная оценка затрат недопоставки/излишков
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", str(WAREHOUSE_DIR / "parquet")))


def parse_args():
    p = argparse.ArgumentParser("MAPE→деньги и ROP")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--price", type=float, required=True, help="Цена за единицу товара")
    p.add_argument("--margin_rate", type=float, default=0.25, help="Маржинальность (доля) для потерь недопродажи")
    p.add_argument("--holding_cost", type=float, default=0.05, help="Дневная стоимость хранения за единицу")
    p.add_argument("--lead_time_days", type=int, default=2)
    p.add_argument("--service_level", type=float, default=0.95)
    return p.parse_args()


def _read_pair_series(store: int, family: str) -> pd.DataFrame:
    # Пытаемся взять features.parquet, иначе train.csv
    feat_path = PARQUET_DIR / "features" / "features.parquet"
    if feat_path.exists():
        X = pd.read_parquet(feat_path)
    else:
        raw = Path("data_raw") / "train.csv"
        if not raw.exists():
            raise SystemExit("Нет данных features.parquet и data_raw/train.csv")
        X = pd.read_csv(raw, parse_dates=["date"])
    df = X[(X["store_nbr"] == store) & (X["family"] == family)].copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    return df


def _read_mape(store: int, family: str) -> float | None:
    mpath = WAREHOUSE_DIR / "metrics_per_sku.csv"
    if not mpath.exists():
        return None
    df = pd.read_csv(mpath)
    sub = df[(df["store_nbr"] == store) & (df["family"] == family)]
    if sub.empty:
        return None
    try:
        return float(sub.iloc[0]["MAPE_%"])
    except Exception:
        return None


def _z_from_service_level(p: float) -> float:
    table = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}
    closest = min(table.keys(), key=lambda x: abs(x - p))
    return table[closest]


def main():
    args = parse_args()
    df = _read_pair_series(args.store, args.family)
    if df.empty:
        raise SystemExit("Нет данных для пары.")

    # Оценка среднего спроса и MAPE
    if "sales" in df.columns:
        tail = df.tail(30)
        daily_mean = float(tail["sales"].mean())
    else:
        daily_mean = float(df.tail(30)["sales_lag_1"].mean()) if "sales_lag_1" in df.columns else 0.0

    mape_pct = _read_mape(args.store, args.family)
    if mape_pct is None:
        mape_pct = 20.0  # допущение, если нет метрик

    sigma = max((mape_pct / 100.0) * daily_mean, 0.0)
    z = _z_from_service_level(args.service_level)
    L = max(int(args.lead_time_days), 1)
    safety_stock = z * sigma * (L ** 0.5)
    rop = daily_mean * L + safety_stock

    # Приближённые денежные оценки
    # Стоимость недопоставки (упущенная маржа): margin * price * ожидаемый дефицит
    expected_under_units = (1 - args.service_level) * daily_mean * L
    under_cost = expected_under_units * (args.margin_rate * args.price)
    # Стоимость излишков (хранение): holding * SS * L_coef(≈1)
    over_cost = args.holding_cost * safety_stock

    print("store/family:", args.store, args.family)
    print(f"daily_mean ≈ {daily_mean:.3f}")
    print(f"MAPE ≈ {mape_pct:.2f}% → sigma ≈ {sigma:.3f}")
    print(f"safety_stock = {safety_stock:.3f}, reorder_point = {rop:.3f}")
    print(f"Underage cost (≈): {under_cost:.2f} per replenishment window")
    print(f"Overage cost (≈):  {over_cost:.2f} per day")


if __name__ == "__main__":
    main()

