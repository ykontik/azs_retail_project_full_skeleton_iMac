import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", required=True, default="data_raw")
    args = p.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "transactions.csv", "oil.csv", "holidays_events.csv", "stores.csv"]:
        s = src / name
        if s.exists():
            shutil.copy2(s, out / name)
            print("copied →", name)
        else:
            print("skip:", name)


if __name__ == "__main__":
    main()
