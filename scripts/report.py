from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

WAREHOUSE_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
METRICS_PATH = WAREHOUSE_DIR / "metrics_per_sku.csv"
OUT_HTML = WAREHOUSE_DIR / "report.html"


def _fmt_pct(x: float | int | None) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return ""


def build_report() -> str:
    if not METRICS_PATH.exists():
        raise SystemExit(f"Не найден файл метрик: {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH)

    # Базовые агрегаты
    agg_store = (
        df.groupby("store_nbr", dropna=True)[["MAE", "MAPE_%"]]
        .mean()
        .reset_index()
        .sort_values("MAE")
    )
    agg_family = (
        df.groupby("family", dropna=True)[["MAE", "MAPE_%"]]
        .mean()
        .reset_index()
        .sort_values("MAE")
    )

    # Топ худших пар по MAE
    worst_pairs = df.sort_values("MAE", ascending=False).head(50)

    # Сравнение с baseline (если есть колонки NAIVE_*)
    cmp_cols = [c for c in df.columns if c.startswith("NAIVE_")]
    cmp_section = (
        df[["store_nbr", "family", "MAE", *cmp_cols]].sort_values("MAE").head(50)
        if cmp_cols
        else None
    )

    # HTML
    css = """
    <style>
    body{font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px;}
    h1{margin-top:0}
    table{border-collapse: collapse; width: 100%; margin: 10px 0 30px;}
    th, td{border: 1px solid #ddd; padding: 6px 8px; font-size: 14px;}
    th{background:#f3f3f3; text-align:left}
    .grid{display:grid; grid-template-columns: 1fr 1fr; gap:24px}
    .small{font-size:12px; color:#666}
    </style>
    """
    h1 = "<h1>Отчёт по метрикам (per-SKU)</h1>"

    summary = (
        f"<p class='small'>Строк: {len(df)} | Средний MAE: {df['MAE'].mean():.3f} | "
        f"Средний MAPE: {df['MAPE_%'].mean():.2f}%</p>"
    )

    html = [css, h1, summary]
    html.append("<h2>Агрегаты по магазинам</h2>")
    html.append(agg_store.to_html(index=False))
    html.append("<h2>Агрегаты по семействам</h2>")
    html.append(agg_family.to_html(index=False))
    html.append("<h2>Топ 50 худших пар по MAE</h2>")
    html.append(worst_pairs[["store_nbr", "family", "MAE", "MAPE_%"]].to_html(index=False))
    if cmp_section is not None:
        html.append("<h2>Сравнение с baseline</h2>")
        html.append(cmp_section.to_html(index=False))

    return "\n".join(html)


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    content = build_report()
    OUT_HTML.write_text(content, encoding="utf-8")
    print(f"report → {OUT_HTML}")


if __name__ == "__main__":
    main()

