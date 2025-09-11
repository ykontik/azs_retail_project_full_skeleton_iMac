
import os
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data_dw/parquet"))
WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
LEAD_TIME_DAYS = int(os.getenv("LEAD_TIME_DAYS", "2"))
SERVICE_LEVEL_Z = float(os.getenv("SERVICE_LEVEL_Z", "1.65"))

OUT_PATH = WAREHOUSE_DIR / "stock_plan.csv"
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    feat_path = PARQUET_DIR / "features" / "features.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
    else:
        raw_train = Path("data_raw") / "train.csv"
        if not raw_train.exists():
            print("Нет ни features.parquet, ни data_raw/train.csv")
            return
        df = pd.read_csv(raw_train, parse_dates=["date"])

    df = df.sort_values(["store_nbr","family","date"])

    res = []
    for (store, fam), grp in df.groupby(["store_nbr","family"]):
        tail = grp.tail(30)
        if "sales" not in tail.columns:
            continue
        daily_mean = tail["sales"].mean()
        daily_std = tail["sales"].std(ddof=1) if len(tail) > 1 else 0.0

        ss = SERVICE_LEVEL_Z * daily_std * np.sqrt(LEAD_TIME_DAYS)
        rop = daily_mean * LEAD_TIME_DAYS + ss

        res.append({
            "store_nbr": store, "family": fam,
            "daily_mean": daily_mean, "daily_std": daily_std,
            "lead_time_days": LEAD_TIME_DAYS,
            "service_z": SERVICE_LEVEL_Z,
            "safety_stock": ss, "reorder_point": rop
        })

    out = pd.DataFrame(res).sort_values(["store_nbr","family"])
    out.to_csv(OUT_PATH, index=False)
    print("stock_plan →", OUT_PATH)

if __name__ == "__main__":
    main()
