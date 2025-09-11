#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    raw = Path("data_raw")
    train = pd.read_csv(raw / "train.csv", parse_dates=["date"])
    trans = pd.read_csv(raw / "transactions.csv", parse_dates=["date"]) if (raw / "transactions.csv").exists() else None
    oil = pd.read_csv(raw / "oil.csv", parse_dates=["date"]) if (raw / "oil.csv").exists() else None
    hol = pd.read_csv(raw / "holidays_events.csv", parse_dates=["date"]) if (raw / "holidays_events.csv").exists() else None
    stores = pd.read_csv(raw / "stores.csv") if (raw / "stores.csv").exists() else None

    stores_all = sorted(train["store_nbr"].dropna().astype(int).unique().tolist())
    targets = list(dict.fromkeys(stores_all[:5] + stores_all[-5:]))
    # Фильтры через окружение: STORE_LIST, FAMILY_LIST, MAX_PAIRS, STOP_FILE
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

    stop_file = os.environ.get("STOP_FILE", "STOP_BATCH")
    for i, (s, f) in enumerate(pairs, 1):
        if stop_file and Path(stop_file).exists():
            print(f"STOP requested via {stop_file}. Exiting loop at {i-1}/{len(pairs)}.")
            break
        print(f"[{i}/{len(pairs)}] Quantiles -> ({s}, {f})")
        tt = train[(train["store_nbr"] == s) & (train["family"].astype(str) == str(f))]
        if tt.empty:
            print("skip(empty)")
            continue
        # если квантильные модели уже есть — пропустим
        from pathlib import Path as _P
        models_dir = _P(os.environ.get("MODELS_DIR", "models"))
        stem = f"{s}__{str(f).replace(' ', '_')}"
        q50 = models_dir / f"{stem}__q50.joblib"
        q90 = models_dir / f"{stem}__q90.joblib"
        if q50.exists() and q90.exists():
            print(f"[{i}/{len(pairs)}] Quantiles -> ({s}, {f}) SKIP (exists)")
            continue
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            tt.to_csv(tdp / "train.csv", index=False)
            (trans if trans is not None else pd.DataFrame(columns=["date", "transactions"]))\
                .to_csv(tdp / "transactions.csv", index=False)
            (oil if oil is not None else pd.DataFrame(columns=["date", "dcoilwtico"]))\
                .to_csv(tdp / "oil.csv", index=False)
            (hol if hol is not None else pd.DataFrame(columns=["date", "locale"]))\
                .to_csv(tdp / "holidays_events.csv", index=False)
            (stores if stores is not None else pd.DataFrame(columns=["store_nbr"]))\
                .to_csv(tdp / "stores.csv", index=False)
            cmd = [
                "python", "train_forecast.py",
                "--train", str(tdp / "train.csv"),
                "--transactions", str(tdp / "transactions.csv"),
                "--oil", str(tdp / "oil.csv"),
                "--holidays", str(tdp / "holidays_events.csv"),
                "--stores", str(tdp / "stores.csv"),
                "--models_dir", "models",
                "--warehouse_dir", "data_dw",
                "--top_n_sku", "1",
                "--top_recent_days", "99999",
                "--valid_days", os.environ.get("VALID_DAYS", "28"),
                "--quantiles", "0.5", "0.9",
            ]
            subprocess.run(cmd, check=False)
    print("OK: train-quantiles-selected done")


if __name__ == "__main__":
    main()
