#!/usr/bin/env python3
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    train = pd.read_csv("data_raw/train.csv")
    stores = sorted(train["store_nbr"].dropna().astype(int).unique().tolist())
    targets = list(dict.fromkeys(stores[:5] + stores[-5:]))

    # Фильтры через окружение: STORE_LIST="1,2,3"; FAMILY_LIST="BEVERAGES,PRODUCE"; MAX_PAIRS=100
    store_list = os.environ.get("STORE_LIST", "").strip()
    family_list = os.environ.get("FAMILY_LIST", "").strip()
    store_set = {int(x) for x in store_list.split(',') if x.strip().isdigit()} if store_list else None
    family_set = {x.strip() for x in family_list.split(',') if x.strip()} if family_list else None

    pairs: list[tuple[int, str]] = []
    for s in targets:
        if store_set is not None and s not in store_set:
            continue
        fams = sorted(train.loc[train["store_nbr"] == s, "family"].dropna().astype(str).unique().tolist())
        for f in fams:
            if family_set is not None and f not in family_set:
                continue
            pairs.append((s, f))

    max_pairs_env = os.environ.get("MAX_PAIRS")
    if max_pairs_env and max_pairs_env.isdigit():
        pairs = pairs[: int(max_pairs_env)]
    valid_days = int(os.environ.get("VALID_DAYS", 28))
    # Проверим доступность lightgbm: если нет — пропускаем LGBM, но тренируем XGB
    try:
        import lightgbm  # noqa: F401
        have_lgbm = True
    except Exception:
        have_lgbm = False
        print("WARN: LightGBM не установлен. Пропущу LGBM и обучу только XGB. Установите: pip install lightgbm==4.3.0")

    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    stop_file = os.environ.get("STOP_FILE", "STOP_BATCH")

    for i, (s, f) in enumerate(pairs, 1):
        if stop_file and Path(stop_file).exists():
            print(f"STOP requested via {stop_file}. Exiting loop at {i-1}/{len(pairs)}.")
            break
        stem = f"{s}__{str(f).replace(' ', '_')}"
        lgb_path = models_dir / f"{stem}.joblib"
        xgb_path = models_dir / f"{stem}__xgb.joblib"

        # LGBM per-SKU
        if have_lgbm:
            if lgb_path.exists():
                print(f"[{i}/{len(pairs)}] LGBM per-SKU -> ({s}, {f}) SKIP (exists)")
            else:
                print(f"[{i}/{len(pairs)}] LGBM per-SKU -> ({s}, {f})")
                cmd = f"python scripts/train_one_pair.py --store {s} --family {shlex.quote(f)} --valid_days {valid_days}"
                subprocess.run(cmd, shell=True, check=False)
        else:
            print(f"[{i}/{len(pairs)}] LGBM per-SKU -> ({s}, {f}) SKIP (no lightgbm)")

        # XGB per-SKU
        if xgb_path.exists():
            print(f"[{i}/{len(pairs)}] XGB per-SKU -> ({s}, {f}) SKIP (exists)")
        else:
            print(f"[{i}/{len(pairs)}] XGB per-SKU -> ({s}, {f})")
            cmd = f"python scripts/train_one_pair_xgb.py --store {s} --family {shlex.quote(f)} --valid_days {valid_days}"
            subprocess.run(cmd, shell=True, check=False)
    print("OK: batch-train-selected done")


if __name__ == "__main__":
    main()
