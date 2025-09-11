import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Лидерборд ошибок", layout="wide")
st.title("🥇 Лидерборд ошибок (MAE/MAPE)")

DW_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
METRICS_PATH = DW_DIR / "metrics_per_sku.csv"
RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    df = pd.read_csv(METRICS_PATH)
    return df

df = load_metrics()
if df is None or df.empty:
    st.info("Файл metrics_per_sku.csv не найден. Запустите обучение, чтобы увидеть метрики.")
    st.stop()

cols = st.columns(5)
with cols[0]:
    metric = st.selectbox("Метрика", ["MAE", "MAPE_%"], index=0)
with cols[1]:
    top_n = st.number_input("Показать топ", min_value=10, max_value=500, value=100, step=10)
with cols[2]:
    store_filter = st.multiselect("Фильтр по магазинам", sorted(df["store_nbr"].dropna().unique().tolist()))
with cols[3]:
    family_filter = st.multiselect("Фильтр по семействам", sorted(df["family"].dropna().astype(str).unique().tolist()))
with cols[4]:
    valid_days = st.number_input("valid_days для wMAPE", min_value=7, max_value=90, value=28, step=1, help="Окно holdout, использованное при обучении. Используется для оценки wMAPE.")

sub = df.copy()
if store_filter:
    sub = sub[sub["store_nbr"].isin(store_filter)]
if family_filter:
    sub = sub[sub["family"].astype(str).isin(family_filter)]

# wMAPE (оценка) на лету: wMAPE ≈ (MAE * N) / sum(y_true) * 100
# где N — число точек в хвосте; sum(y_true) берём из RAW_DIR/train.csv за последние valid_days
@st.cache_data(show_spinner=False)
def _wmape_by_pair(valid_days_: int) -> Optional[pd.DataFrame]:
    try:
        train_path = RAW_DIR / "train.csv"
        if not train_path.exists():
            return None
        tr = pd.read_csv(train_path, parse_dates=["date"])  # ожидаются колонки: date, store_nbr, family, sales
        max_date = pd.to_datetime(tr["date"]).max()
        threshold = max_date - pd.Timedelta(days=int(valid_days_) - 1)
        tail = tr[pd.to_datetime(tr["date"]) >= threshold]
        grp = tail.groupby(["store_nbr", "family"], dropna=False).agg(
            N=("sales", "count"),
            SUM_Y=("sales", "sum"),
        ).reset_index()
        return grp
    except Exception:
        return None

wm = _wmape_by_pair(int(valid_days))
if wm is not None and not wm.empty:
    sub = sub.merge(wm, on=["store_nbr", "family"], how="left")
    sub["wMAPE_%_est"] = (sub["MAE"] * sub["N"]) / sub["SUM_Y"].replace(0, np.nan) * 100.0
else:
    sub["wMAPE_%_est"] = np.nan

st.subheader("Худшие пары (по выбранной метрике)")
lead = sub.sort_values(metric, ascending=False).head(int(top_n))
cols_show = ["store_nbr", "family", "MAE", "MAPE_%", "wMAPE_%_est", "NAIVE_LAG7_MAE", "NAIVE_MA7_MAE"]
cols_show = [c for c in cols_show if c in lead.columns]
st.dataframe(lead[cols_show].fillna(""), use_container_width=True)
st.download_button(
    "⬇️ Скачать лидерборд (CSV)",
    data=lead.to_csv(index=False).encode("utf-8"),
    file_name="leaderboard_filtered.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Агрегаты")
c1, c2 = st.columns(2)
with c1:
    st.caption("Средние по магазинам")
    agg_store = (
        sub.groupby("store_nbr", dropna=True)[["MAE", "MAPE_%"]]
        .mean()
        .reset_index()
        .sort_values("MAE")
    )
    st.dataframe(agg_store, use_container_width=True)
    try:
        st.bar_chart(agg_store.set_index("store_nbr")["MAE"], height=200)
    except Exception:
        pass
with c2:
    st.caption("Средние по семействам")
    agg_family = (
        sub.groupby("family", dropna=True)[["MAE", "MAPE_%"]]
        .mean()
        .reset_index()
        .sort_values("MAE")
    )
    st.dataframe(agg_family, use_container_width=True)
    try:
        st.bar_chart(agg_family.set_index("family")["MAE"], height=200)
    except Exception:
        pass
