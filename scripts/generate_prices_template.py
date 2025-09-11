#!/usr/bin/env python3
"""
Генерация шаблона цен/маржи/стоимости хранения по семьям (и опционально магазинам).

Сканирует data_raw/train.csv, собирает уникальные family и создает CSV:
  configs/prices.csv с колонками: family,price,margin_rate,holding_cost

Можно расширить вручную колонкой store_nbr для переопределений по магазину.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Generate prices template")
    p.add_argument("--train", default="data_raw/train.csv")
    p.add_argument("--out", default="configs/prices.csv")
    p.add_argument("--default_price", type=float, default=3.5)
    p.add_argument("--default_margin", type=float, default=0.25)
    p.add_argument("--default_holding", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"Не найден {train_path}")
    df = pd.read_csv(train_path, usecols=["family"])  # быстрее
    fams = (
        df["family"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "family": fams,
            "price": [args.default_price] * len(fams),
            "margin_rate": [args.default_margin] * len(fams),
            "holding_cost": [args.default_holding] * len(fams),
        }
    ).to_csv(out_path, index=False)
    print(f"OK: шаблон цен записан в {out_path} (строк: {len(fams)})")


if __name__ == "__main__":
    main()

