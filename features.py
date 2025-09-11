
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv  # загрузка .env

from make_features import make_features

load_dotenv()
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data_dw/parquet"))
RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))

OUT_DIR = PARQUET_DIR / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["date"])
    transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    X, y = make_features(train, holidays, transactions, oil, stores, dropna_target=True)
    X.to_parquet(OUT_DIR / "features.parquet", index=False)
    if y is not None:
        pd.DataFrame({"sales": y}).to_parquet(OUT_DIR / "target.parquet", index=False)

    print("features →", OUT_DIR / "features.parquet")

if __name__ == "__main__":
    main()
