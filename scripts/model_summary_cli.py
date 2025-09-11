#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    rows = []
    mp = Path("data_dw/metrics_per_sku.csv")
    if mp.exists():
        m = pd.read_csv(mp)
        rows.append({
            "model": "LGBM per-SKU",
            "MAE": float(m["MAE"].mean()),
            "MAPE": float(m["MAPE_%"].mean()),
            "size_hint": "joblib per-pair",
        })
    for name, fn in [
        ("CatBoost global", Path("data_dw/metrics_global_catboost.json")),
        ("XGBoost global", Path("data_dw/metrics_global_xgboost.json")),
    ]:
        if fn.exists():
            try:
                d = json.loads(fn.read_text(encoding="utf-8"))
                rows.append({
                    "model": name,
                    "MAE": d.get("MAE"),
                    "MAPE": d.get("MAPE_%"),
                    "size_hint": "global",
                })
            except Exception:
                pass
    out = pd.DataFrame(rows)
    Path("docs").mkdir(exist_ok=True)
    out.to_csv("docs/model_comparison.csv", index=False)
    print("OK: docs/model_comparison.csv")


if __name__ == "__main__":
    main()

