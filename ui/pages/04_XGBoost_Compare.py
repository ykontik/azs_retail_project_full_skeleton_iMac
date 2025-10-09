import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from make_features import make_features

st.set_page_config(page_title="Сравнение: XGBoost vs LGBM", layout="wide")
st.title("🆚 Сравнение: Global XGBoost vs per-SKU LGBM")

RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

@st.cache_data(show_spinner=False)
def load_raw():
    paths = {k: RAW_DIR / f"{k}.csv" for k in ["train", "transactions", "oil", "holidays_events", "stores"]}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return None, missing
    train = pd.read_csv(paths["train"], parse_dates=["date"])
    trans = pd.read_csv(paths["transactions"], parse_dates=["date"])
    oil = pd.read_csv(paths["oil"], parse_dates=["date"])
    hol = pd.read_csv(paths["holidays_events"], parse_dates=["date"])
    stores = pd.read_csv(paths["stores"])
    return (train, hol, trans, oil, stores), []

@st.cache_data(show_spinner=True)
def build_features():
    data, missing = load_raw()
    if data is None:
        return None, missing
    train, hol, trans, oil, stores = data
    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    for c in ["store_nbr","family","type","city","state","cluster","is_holiday"]:
        if c in Xfull.columns:
            Xfull[c] = Xfull[c].astype("category")
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")
    return Xfull, []

def model_feature_names(model) -> Optional[List[str]]:
    names = getattr(model, "feature_name_", None)
    if names is None and hasattr(model, "booster_"):
        try:
            names = list(model.booster_.feature_name())
        except Exception:
            names = None
    return names

def xgb_feature_names(model) -> Optional[List[str]]:
    """Достаёт имена фич из XGBRegressor.
    Порядок столбцов критичен, поэтому возвращаем список или None.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return list(names)
        except Exception:
            pass
    try:
        booster = model.get_booster()
        if hasattr(booster, "feature_names"):
            return list(booster.feature_names)
    except Exception:
        return None
    return None

from collections.abc import Sequence


def align_features(df: pd.DataFrame, required: Sequence[str], *, numeric: bool = False) -> pd.DataFrame:
    """Добавляет недостающие колонки (0.0) и переупорядочивает столбцы под модель.

    numeric=False — сохраняет типы (для LGBM с категор. фичами)
    numeric=True  — приводит все столбцы к числу (для XGB и др.)
    """
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = 0.0
    out = out[required].copy()
    if numeric:
        try:
            out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception:
            pass
    return out

models = sorted([p for p in MODELS_DIR.glob("*.joblib") if ("__q" not in p.name) and ("global_xgboost" not in p.name)])
if not models:
    st.warning("Нет per-SKU моделей. Запустите обучение.")
    st.stop()

# load global XGB
xgb_path = MODELS_DIR / "global_xgboost.joblib"
if not xgb_path.exists():
    st.warning("Глобальная XGBoost не найдена. Запустите: make train_global_xgb")
    st.stop()
mdl_xgb = joblib.load(xgb_path)

pairs = []
for mp in models:
    try:
        s, f = mp.stem.split("__", 1)
        pairs.append((int(s), f.replace("_", " ")))
    except Exception:
        continue
pairs = sorted(set(pairs))

st.sidebar.header("Параметры")
store_sel = st.sidebar.selectbox("store_nbr", sorted(set(s for s, _ in pairs)))
fam_opts = sorted([f for s, f in pairs if s == store_sel])
family_sel = st.sidebar.selectbox("family", fam_opts)
back_days = st.sidebar.slider("Дней в хвосте", min_value=28, max_value=180, value=90, step=7)
period = st.sidebar.radio("Период", ["День", "Неделя", "Месяц"], index=0)
min_tail_sales = st.sidebar.number_input("Мин. сумма продаж в хвосте (фильтр пар)", min_value=0, value=0, step=10)

Xfull, missing = build_features()
if Xfull is None:
    st.error("Нет данных: " + ", ".join(missing))
    st.stop()

mask = (Xfull["store_nbr"] == int(store_sel)) & (Xfull["family"] == family_sel)
df_pair = Xfull.loc[mask].sort_values("date").copy()
if df_pair.empty:
    st.error("Нет данных для выбранной пары.")
    st.stop()

# per-SKU LGBM
lgb_path = MODELS_DIR / f"{int(store_sel)}__{str(family_sel).replace(' ', '_')}.joblib"
try:
    mdl_lgb = joblib.load(lgb_path)
except Exception:
    st.error(f"Per-SKU LGBM не найдена: {lgb_path.name}")
    st.stop()
feat_lgb = model_feature_names(mdl_lgb)
for f in feat_lgb:
    if f not in df_pair.columns:
        df_pair[f] = 0.0

tail = df_pair.tail(int(back_days)).copy()
y_true = tail["sales"].values

# LGBM: свой список признаков
X_lgb = align_features(tail, feat_lgb, numeric=False)
y_lgb = mdl_lgb.predict(X_lgb)

# XGB: свой список признаков
feat_xgb = xgb_feature_names(mdl_xgb) or feat_lgb
X_xgb = align_features(tail, feat_xgb, numeric=True)
y_xgb = mdl_xgb.predict(X_xgb)

def agg_series(dates: pd.Series, y: np.ndarray, how: str):
    if how == "День":
        return dates, y
    freq = "W" if how == "Неделя" else "M"
    df = pd.DataFrame({"date": dates.values, "y": y}).set_index("date").groupby(pd.Grouper(freq=freq)).sum().reset_index()
    return df["date"].values, df["y"].values

st.subheader("Метрики (tail)")
def _metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.where(y_true == 0, 1, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, mape
mae_l, mape_l = _metrics(y_true, y_lgb)
mae_x, mape_x = _metrics(y_true, y_xgb)
col1, col2 = st.columns(2)
with col1:
    st.metric("LGBM — MAE", f"{mae_l:.2f}")
    st.metric("LGBM — MAPE %", f"{mape_l:.2f}%")
with col2:
    st.metric("XGBoost — MAE", f"{mae_x:.2f}")
    st.metric("XGBoost — MAPE %", f"{mape_x:.2f}%")

st.subheader("График (tail)")
import matplotlib.pyplot as plt

x_dates, y_true_agg = agg_series(tail["date"], y_true, period)
_, y_lgb_agg = agg_series(tail["date"], y_lgb, period)
_, y_xgb_agg = agg_series(tail["date"], y_xgb, period)
fig = plt.figure(figsize=(12,4))
plt.plot(x_dates, y_true_agg, label="Факт")
plt.plot(x_dates, y_lgb_agg, label="LGBM")
plt.plot(x_dates, y_xgb_agg, label="XGBoost")
plt.title(f"Хвост {int(back_days)} дн., период: {period.lower()}")
plt.xlabel("Дата/Период"); plt.ylabel("Продажи")
plt.legend(); plt.grid()
st.pyplot(fig)

st.markdown("---")
st.subheader("Heatmap: XGBoost vs LGBM (положительное = XGB лучше)")
max_pairs = st.slider("Пары для heatmap (ограничение)", min_value=10, max_value=300, value=100, step=10)
pairs_all = pairs[:max_pairs]
rows = []
skipped = []
for (s, f) in pairs_all:
    try:
        mp = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}.joblib"
        try:
            m = joblib.load(mp)
        except Exception as e:
            skipped.append({"store_nbr": s, "family": f, "reason": f"load per-SKU model failed: {e}"})
            continue
        feats = model_feature_names(m)
        df_p = Xfull[(Xfull["store_nbr"] == int(s)) & (Xfull["family"] == f)].sort_values("date").tail(int(back_days)).copy()
        if df_p.empty:
            skipped.append({"store_nbr": s, "family": f, "reason": "empty tail"}); continue
        if not feats:
            skipped.append({"store_nbr": s, "family": f, "reason": "no LGBM feature names"}); continue
        # LGBM
        Xl = align_features(df_p, feats, numeric=False)
        # XGB по своему списку признаков
        feats_x = xgb_feature_names(mdl_xgb) or feats
        Xx = align_features(df_p, feats_x, numeric=True)
        y_t = df_p["sales"].values
        try:
            y_l = m.predict(Xl)
        except Exception as e:
            skipped.append({"store_nbr": s, "family": f, "reason": f"LGBM predict failed: {e}"}); continue
        try:
            y_x = mdl_xgb.predict(Xx)
        except Exception as e:
            skipped.append({"store_nbr": s, "family": f, "reason": f"XGB predict failed: {e}"}); continue
        # фильтр по минимальной сумме продаж в хвосте
        tail_sum = float(np.sum(y_t))
        if tail_sum < float(min_tail_sales):
            skipped.append({"store_nbr": s, "family": f, "reason": f"tail_sum<{min_tail_sales}"}); continue
        mae = float(np.mean(np.abs(y_t - y_l)))
        mape = float(np.mean(np.abs((y_t - y_l) / np.where(y_t == 0, 1, y_t))) * 100.0)
        mae_x = float(np.mean(np.abs(y_t - y_x)))
        mape_x = float(np.mean(np.abs((y_t - y_x) / np.where(y_t == 0, 1, y_t))) * 100.0)
        rows.append({
            "store_nbr": int(s),
            "family": f,
            "LGBM_MAE": round(mae, 2),
            "LGBM_MAPE": round(mape, 2),
            "XGB_MAE": round(mae_x, 2),
            "XGB_MAPE": round(mape_x, 2),
            "GAIN_XGB_vs_LGBM_MAE": round(mae - mae_x, 2),
            "GAIN_XGB_vs_LGBM_MAPE": round(mape - mape_x, 2),
        })
    except Exception:
        skipped.append({"store_nbr": s, "family": f, "reason": "unexpected exception"}); continue

if rows:
    dfm = pd.DataFrame(rows)
    dim_choice = st.radio("Разрез", ["По магазинам", "По семействам"], horizontal=True)
    metric_choice = st.radio("Метрика", ["MAE", "MAPE"], horizontal=True)
    dim_col = "store_nbr" if dim_choice == "По магазинам" else "family"
    val_xgb = "GAIN_XGB_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_XGB_vs_LGBM_MAPE"
    pv_xgb = dfm.pivot_table(index=dim_col, columns="family" if dim_col=="store_nbr" else "store_nbr", values=val_xgb, aggfunc="mean").fillna(0.0)
    pv_display = pv_xgb.round(2)
    st.dataframe(pv_display.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
    st.download_button(
        "⬇️ CSV: heatmap XGB vs LGBM",
        data=pv_display.to_csv().encode("utf-8"),
        file_name="heatmap_xgb_vs_lgbm.csv",
        mime="text/csv",
    )
else:
    st.info("Недостаточно данных для heatmap.")

# Показать причины пропусков (диагностика)
if skipped:
    with st.expander("Пары, исключённые из сравнения (диагностика)", expanded=False):
        skipped_df = pd.DataFrame(skipped)
        st.dataframe(skipped_df.round(2), use_container_width=True)
