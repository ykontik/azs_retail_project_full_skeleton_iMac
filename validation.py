from pathlib import Path

import pandas as pd

REQUIRED = {
    "train.csv": {"date", "store_nbr", "family", "sales", "onpromotion"},
    "transactions.csv": {"date", "store_nbr", "transactions"},
    "oil.csv": {"date", "dcoilwtico"},
    "holidays_events.csv": {"date", "type", "locale"},
    "stores.csv": {"store_nbr", "city", "state", "type", "cluster"},
}


def validate_folder(raw_dir: str = "data_raw") -> bool:
    ok_all = True
    p = Path(raw_dir)
    for fname, cols in REQUIRED.items():
        f = p / fname
        if not f.exists():
            print(f"[MISSING] {fname}")
            ok_all = False
            continue
        try:
            df = pd.read_csv(f, nrows=100)
            miss = cols - set(df.columns)
            if miss:
                print(f"[BAD COLS] {fname} — отсутствуют: {sorted(miss)}")
                ok_all = False
            else:
                print(f"[OK] {fname}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            ok_all = False
    return ok_all


if __name__ == "__main__":
    print("===> VALID:", validate_folder())
