#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns  # type: ignore


def main() -> None:
    Path("docs").mkdir(exist_ok=True)
    df = pd.read_csv("data_raw/train.csv", parse_dates=["date"])  # expects columns: date,sales,[onpromotion]
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    plt.figure(figsize=(8, 4))
    sns.barplot(x="dow", y="sales", data=df)
    plt.title("Средние продажи по дням недели")
    plt.tight_layout()
    plt.savefig("docs/eda_seasonality_dow.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(x="month", y="sales", data=df)
    plt.title("Средние продажи по месяцам")
    plt.tight_layout()
    plt.savefig("docs/eda_seasonality_month.png")
    plt.close()

    if "onpromotion" in df.columns:
        df["promo"] = (df["onpromotion"] > 0).astype(int)
        g = df.groupby("promo")["sales"].mean().reset_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x="promo", y="sales", data=g)
        plt.title("Продажи: без промо vs промо")
        plt.tight_layout()
        plt.savefig("docs/eda_promo_effect.png")
        plt.close()
    print("OK: EDA графики сохранены в docs/")


if __name__ == "__main__":
    main()

