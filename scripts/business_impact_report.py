#!/usr/bin/env python3
"""
Оценка экономического эффекта от использования моделей на реальных (обучающих) данных.

Считаем по каждой паре (store_nbr, family):
  - Средний дневной спрос (последние N дней)
  - MAPE наивной модели (скользящее среднее 7 дней)
  - MAPE модели (из data_dw/metrics_per_sku.csv)
  - Safety Stock / ROP для наивной и модельной стратегий
  - Денежные эффекты (недопоставка/излишки) и их разница (экономия)

Выход: data_dw/business_impact_report.csv и консольная сводка.

Пример:
  python scripts/business_impact_report.py \
    --price_csv configs/prices.csv --lead_time_days 2 --service_level 0.95 --tail_days 30 --valid_days 28
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Business impact report")
    p.add_argument("--train", default="data_raw/train.csv")
    p.add_argument("--metrics", default="data_dw/metrics_per_sku.csv")
    p.add_argument(
        "--price_csv",
        default="configs/prices.csv",
        help="CSV с ценами/маржой по family (и опц. store_nbr)",
    )
    p.add_argument("--lead_time_days", type=int, default=2)
    p.add_argument("--service_level", type=float, default=0.95)
    p.add_argument("--tail_days", type=int, default=30, help="Дней для расчёта среднего спроса")
    p.add_argument(
        "--valid_days", type=int, default=28, help="Дней для расчёта MAPE наивной модели"
    )
    p.add_argument("--out_csv", default="data_dw/business_impact_report.csv")
    return p.parse_args()


def _z_from_service_level(p: float) -> float:
    table = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}
    closest = min(table.keys(), key=lambda x: abs(x - p))
    return table[closest]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def naive_mape_last(
    df_pair: pd.DataFrame, valid_days: int = 28, avg_window: int = 7
) -> float | None:
    df = df_pair.sort_values("date").copy()
    if len(df) < (valid_days + avg_window + 1) or "sales" not in df.columns:
        return None
    y = df["sales"].values.astype(float)
    # Для последних valid_days прогнозируем среднее по предыдущим avg_window дням
    preds = []
    actual = []
    for i in range(len(y) - valid_days, len(y)):
        start = i - avg_window
        if start < 0:
            return None
        preds.append(np.mean(y[start:i]))
        actual.append(y[i])
    return mape(np.array(actual), np.array(preds))


def load_prices(price_csv: str) -> pd.DataFrame:
    p = Path(price_csv)
    if not p.exists():
        # Заглушка: одна цена/маржа/хранение для всех
        return pd.DataFrame(
            {
                "family": [],
                "price": [],
                "margin_rate": [],
                "holding_cost": [],
            }
        )
    return pd.read_csv(p)


def main():
    args = parse_args()
    train = pd.read_csv(args.train, parse_dates=["date"])
    if not {"store_nbr", "family", "sales", "date"}.issubset(train.columns):
        raise SystemExit("train.csv должен содержать колонки: date, store_nbr, family, sales")
    metrics_path = Path(args.metrics)
    metrics = None
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)

    prices = load_prices(args.price_csv)
    z = _z_from_service_level(args.service_level)

    rows = []
    for (s, f), dfp in train.groupby(["store_nbr", "family"], sort=False):
        dfp = dfp.sort_values("date")
        tail = dfp.tail(args.tail_days)
        daily_mean = float(tail["sales"].mean()) if not tail.empty else 0.0

        # MAPE модели (если есть в metrics_per_sku)
        model_mape = None
        if metrics is not None:
            sel = metrics[(metrics["store_nbr"] == s) & (metrics["family"] == f)]
            if not sel.empty and ("MAPE_%" in sel.columns):
                try:
                    model_mape = float(sel.iloc[0]["MAPE_%"])
                except Exception:
                    model_mape = None
        # Наивный MAPE
        naive = naive_mape_last(dfp, valid_days=args.valid_days, avg_window=7)

        # Цена/маржа/хранение
        price = 3.5
        margin_rate = 0.25
        holding_cost = 0.05
        if not prices.empty:
            selp = prices[
                (prices.get("family") == f)
                & ((prices.get("store_nbr").isna()) if "store_nbr" in prices.columns else True)
            ]
            if not selp.empty:
                row = selp.iloc[0]
                price = float(row.get("price", price))
                margin_rate = float(row.get("margin_rate", margin_rate))
                holding_cost = float(row.get("holding_cost", holding_cost))

        # Сигмы для наивной и модельной стратегий
        sigma_naive = (naive / 100.0) * daily_mean if naive is not None else None
        sigma_model = (model_mape / 100.0) * daily_mean if model_mape is not None else None

        # ROP / SS
        def _ss_rop(sigma: float | None):
            if sigma is None:
                return None, None
            ss = z * sigma * (args.lead_time_days**0.5)
            rop = daily_mean * args.lead_time_days + ss
            return ss, rop

        ss_naive, rop_naive = _ss_rop(sigma_naive)
        ss_model, rop_model = _ss_rop(sigma_model)

        # Денежные эффекты (грубая оценка)
        def _under_over(sigma: float | None, ss: float | None):
            if sigma is None or ss is None:
                return None, None
            # недопоставка: упущенная маржа за окно пополнения (аппроксимация)
            under_units = (1 - args.service_level) * daily_mean * args.lead_time_days
            under_cost = under_units * (margin_rate * price)
            # излишки: стоимость хранения safety stock (в день)
            over_cost = holding_cost * ss
            return under_cost, over_cost

        under_naive, over_naive = _under_over(sigma_naive, ss_naive)
        under_model, over_model = _under_over(sigma_model, ss_model)

        # Экономия (модель против наивной)
        savings_under = (
            (under_naive - under_model)
            if (under_naive is not None and under_model is not None)
            else None
        )
        savings_over = (
            (over_naive - over_model)
            if (over_naive is not None and over_model is not None)
            else None
        )

        rows.append(
            {
                "store_nbr": s,
                "family": f,
                "daily_mean": daily_mean,
                "naive_MAPE_%": naive,
                "model_MAPE_%": model_mape,
                "sigma_naive": sigma_naive,
                "sigma_model": sigma_model,
                "SS_naive": ss_naive,
                "SS_model": ss_model,
                "ROP_naive": rop_naive,
                "ROP_model": rop_model,
                "price": price,
                "margin_rate": margin_rate,
                "holding_cost": holding_cost,
                "under_cost_naive": under_naive,
                "under_cost_model": under_model,
                "over_cost_naive": over_naive,
                "over_cost_model": over_model,
                "savings_under": savings_under,
                "savings_over_per_day": savings_over,
            }
        )

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Краткая сводка
    total_pairs = len(out)
    with_mape = out[~out["model_MAPE_%"].isna()]
    total_with_model = len(with_mape)
    # Суммарная экономия (учитывая только пары, где доступны обе оценки)
    sav_under = with_mape["savings_under"].dropna().sum()
    sav_over_day = with_mape["savings_over_per_day"].dropna().sum()
    print("--- Business Impact Report ---")
    print(f"Всего пар: {total_pairs}, с модельной MAPE: {total_with_model}")
    print(f"Суммарная экономия на недопоставках (за окно пополнения), ≈: {sav_under:,.2f}")
    print(f"Суммарная экономия на хранении (в день), ≈: {sav_over_day:,.2f}")
    print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
