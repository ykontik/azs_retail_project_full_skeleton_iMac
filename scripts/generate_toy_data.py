#!/usr/bin/env python3
"""
Генерация небольшого toy-датасета в data_raw/ для быстрых демо/тестов.
Создаются: train.csv, transactions.csv, oil.csv, holidays_events.csv, stores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_toy_data(
    out_dir: str | Path = "data_raw",
    start: str = "2017-01-01",
    days: int = 45,
    stores: list[int] | None = None,
    families: list[str] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=start, periods=int(days), freq="D")

    if stores is None:
        stores = [1, 2]
    if families is None:
        families = ["AUTOMOTIVE", "BABY CARE"]

    # train.csv
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for s in stores:
            for f in families:
                base = 10 + 3 * np.sin(2 * np.pi * (d.dayofyear % 30) / 30.0) + 0.5 * s
                noise = rng.normal(0, 1.5)
                sales = max(0.0, base + noise)
                onp = rng.integers(0, 2, dtype=int)
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "store_nbr": s,
                        "family": f,
                        "sales": round(float(sales), 3),
                        "onpromotion": int(onp),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "train.csv", index=False)

    # transactions.csv
    rows = []
    for d in dates:
        for s in stores:
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "store_nbr": s,
                    "transactions": int(100 + 5 * s + 10 * (d.dayofweek in (5, 6))),
                }
            )
    pd.DataFrame(rows).to_csv(out / "transactions.csv", index=False)

    # oil.csv
    oil = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "dcoilwtico": 50.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 20.0),
        }
    )
    oil.to_csv(out / "oil.csv", index=False)

    # holidays_events.csv
    hol = pd.DataFrame(
        {
            "date": [
                dates[0].strftime("%Y-%m-%d"),
                (dates[0] + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
            ],
            "type": ["Holiday", "Holiday"],
            "locale": ["National", "National"],
            "locale_name": ["Ecuador", "Ecuador"],
            "description": ["New Year", "Some Holiday"],
            "transferred": [False, False],
        }
    )
    hol.to_csv(out / "holidays_events.csv", index=False)

    # stores.csv
    stores_df = pd.DataFrame(
        {
            "store_nbr": stores,
            "city": ["Quito", "Guayaquil"][: len(stores)],
            "state": ["Pichincha", "Guayas"][: len(stores)],
            "type": ["A", "B"][: len(stores)],
            "cluster": list(range(1, len(stores) + 1)),
        }
    )
    stores_df.to_csv(out / "stores.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_raw")
    ap.add_argument("--days", type=int, default=45)
    args = ap.parse_args()
    generate_toy_data(args.out, days=args.days)
    print(f"Toy data generated in: {args.out}")


if __name__ == "__main__":
    main()
