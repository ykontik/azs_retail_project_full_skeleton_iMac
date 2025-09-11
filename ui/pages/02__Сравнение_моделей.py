import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from make_features import make_features

st.set_page_config(page_title="Сравнение моделей (строгое)", layout="wide")
st.title("🧪 Сравнение моделей: LGBM per‑SKU vs Global CatBoost/XGB (строгое соответствие фич)")

RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DW_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))


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
    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=False)
    # базовая нормализация типов
    for c in ["store_nbr", "family", "type", "city", "state", "cluster", "is_holiday"]:
        if c in Xfull.columns:
            try:
                Xfull[c] = Xfull[c].astype("category")
            except Exception:
                pass
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
    if names is None and hasattr(model, "feature_names_in_"):
        try:
            names = list(model.feature_names_in_)
        except Exception:
            names = None
    if names is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            fn = getattr(booster, "feature_names", None)
            if fn:
                names = list(fn)
        except Exception:
            names = None
    return names


from typing import Tuple


def load_meta_features() -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
    cb_feats = None
    cb_cats = None
    xgb_feats = None
    try:
        mcb = DW_DIR / "metrics_global_catboost.json"
        if mcb.exists():
            import json as _json
            data = _json.loads(mcb.read_text(encoding="utf-8"))
            cb_feats = data.get("features") if isinstance(data.get("features"), list) else None
            cats = data.get("categoricals")
            cb_cats = cats if isinstance(cats, list) else None
    except Exception:
        pass
    try:
        mxg = DW_DIR / "metrics_global_xgboost.json"
        if mxg.exists():
            import json as _json
            data = _json.loads(mxg.read_text(encoding="utf-8"))
            xgb_feats = data.get("features") if isinstance(data.get("features"), list) else None
    except Exception:
        pass
    return cb_feats, cb_cats, xgb_feats


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[cols]


def predict_catboost(model, df: pd.DataFrame, feature_list: List[str], categoricals: Optional[List[str]]):
    from catboost import Pool
    X = ensure_columns(df, feature_list).copy()
    # CatBoost любит строковые категориальные
    if categoricals:
        for c in categoricals:
            if c in X.columns:
                try:
                    X[c] = X[c].astype(str)
                except Exception:
                    pass
        cat_idx = [i for i, c in enumerate(feature_list) if categoricals and c in set(categoricals)]
    else:
        cat_idx = []
    pool = Pool(X, feature_names=feature_list, cat_features=cat_idx if cat_idx else None)
    return model.predict(pool)


def predict_xgb(model, df: pd.DataFrame, feature_list: List[str]):
    # XGB безопаснее подавать как DataFrame с нужными именами
    X = ensure_columns(df, feature_list).copy()
    # привести к числам
    try:
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    except Exception:
        pass
    return model.predict(X)


# Список доступных per‑SKU моделей
models = sorted([p for p in MODELS_DIR.glob("*.joblib") if "__q" not in p.name and "global_xgboost" not in p.name])
if not models:
    st.warning("Нет per‑SKU моделей в каталоге models/. Запустите обучение.")
    st.stop()

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

# Переключатели линий — над графиком
st.subheader("Показать линии")
col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
with col_t1:
    show_fact = st.checkbox("Факт", value=True, key="t_fact_strict")
with col_t2:
    show_lgbm = st.checkbox("LGBM per‑SKU", value=True, key="t_lgbm_strict")
with col_t3:
    show_cb = st.checkbox("CatBoost (global)", value=True, key="t_cb_strict")
with col_t4:
    show_xgb = st.checkbox("XGB (global)", value=True, key="t_xgb_strict")
with col_t5:
    show_xgbps = st.checkbox("XGB per‑SKU", value=True, key="t_xgbps_strict")

Xfull, missing = build_features()
if Xfull is None:
    st.error("Нет данных: " + ", ".join(missing))
    st.stop()

# Данные по паре
mask = (Xfull["store_nbr"] == int(store_sel)) & (Xfull["family"].astype(str) == str(family_sel))
df_pair = Xfull.loc[mask].sort_values("date").copy()
if df_pair.empty:
    st.error("Нет данных по выбранной паре. Проверьте data_raw/ и make_features.")
    st.stop()

tail = df_pair.tail(int(back_days)).copy()
y_true = tail["sales"].values if "sales" in tail.columns else None

# Per‑SKU LGBM
base_stem = f"{int(store_sel)}__{str(family_sel).replace(' ', '_')}"
lgb_path = MODELS_DIR / f"{base_stem}.joblib"
if not lgb_path.exists():
    st.error(f"Per-SKU модель не найдена: {lgb_path.name}")
    st.stop()
mdl_lgb = joblib.load(lgb_path)
feat_lgb = model_feature_names(mdl_lgb)
if not feat_lgb:
    st.error("Не удалось извлечь список фич per‑SKU модели.")
    st.stop()

# Global CatBoost / XGB
cb_feats, cb_cats, xgb_feats = load_meta_features()
mdl_cb = None
cb_err = None
try:
    cb_path = MODELS_DIR / "global_catboost.cbm"
    if cb_path.exists():
        from catboost import CatBoostRegressor
        mdl_cb = CatBoostRegressor()
        mdl_cb.load_model(str(cb_path))
except Exception as e:
    mdl_cb = None
    cb_err = str(e)

xgb_global = None
xgb_err = None
try:
    xgb_path = MODELS_DIR / "global_xgboost.joblib"
    if xgb_path.exists():
        xgb_global = joblib.load(xgb_path)
except Exception as e:
    xgb_global = None
    xgb_err = str(e)

# XGB per‑SKU (если есть)
mdl_xgb_ps = None
ps_feats = None
try:
    xgbps_path = MODELS_DIR / f"{base_stem}__xgb.joblib"
    if xgbps_path.exists():
        mdl_xgb_ps = joblib.load(xgbps_path)
        ps_feats = getattr(mdl_xgb_ps, "feature_names_in_", None)
        if ps_feats is None:
            # fallback: попробуем взять список фич из .features.json рядом с LGBM
            fj = MODELS_DIR / f"{base_stem}.features.json"
            if fj.exists():
                import json as _json
                data = _json.loads(fj.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    ps_feats = [c for c in data if c in tail.columns]
except Exception:
    mdl_xgb_ps = None


def agg_series(dates: pd.Series, y: np.ndarray, how: str):
    df = pd.DataFrame({"date": dates.values, "y": y})
    if how == "День":
        return df["date"].values, df["y"].values
    rule = "W" if how == "Неделя" else "M"
    g = df.set_index("date").groupby(pd.Grouper(freq=rule)).sum().reset_index()
    return g["date"].values, g["y"].values


# Предсказания
X_tail_lgb = ensure_columns(tail, feat_lgb)
y_lgb = mdl_lgb.predict(X_tail_lgb)

y_cb = None
err_cb_pred = None
if mdl_cb is not None and cb_feats:
    try:
        y_cb = predict_catboost(mdl_cb, tail, cb_feats, cb_cats)
    except Exception as e:
        err_cb_pred = str(e)

y_xgb = None
err_xgb_pred = None
if xgb_global is not None and xgb_feats:
    try:
        y_xgb = predict_xgb(xgb_global, tail, xgb_feats)
    except Exception as e:
        err_xgb_pred = str(e)

y_xgbps = None
err_xgbps_pred = None
if mdl_xgb_ps is not None:
    try:
        feats = list(ps_feats) if ps_feats else feat_lgb
        y_xgbps = predict_xgb(mdl_xgb_ps, tail, feats)
    except Exception as e:
        err_xgbps_pred = str(e)


# График
import matplotlib.pyplot as plt

x_dates, y_true_agg = agg_series(tail["date"], y_true, period)
_, y_lgb_agg = agg_series(tail["date"], y_lgb, period)
fig, ax = plt.subplots(figsize=(18, 7))
colors = {"fact": "#1f77b4", "lgbm": "#ff7f0e", "cat": "#2ca02c", "xgb": "#d62728", "xgbps": "#17becf"}
if show_fact and y_true is not None:
    ax.plot(x_dates, y_true_agg, label="Факт", color=colors["fact"], linewidth=2.0)
if show_lgbm:
    ax.plot(x_dates, y_lgb_agg, label="LGBM", color=colors["lgbm"], linewidth=2.0)
if show_cb and (y_cb is not None):
    _, y_cb_agg = agg_series(tail["date"], y_cb, period)
    ax.plot(x_dates, y_cb_agg, label="CatBoost (global)", color=colors["cat"], linewidth=1.7)
if show_xgb and (y_xgb is not None):
    _, y_xgb_agg = agg_series(tail["date"], y_xgb, period)
    ax.plot(x_dates, y_xgb_agg, label="XGBoost (global)", color=colors["xgb"], linewidth=1.7)
if show_xgbps and (y_xgbps is not None):
    _, y_xgbps_agg = agg_series(tail["date"], y_xgbps, period)
    ax.plot(x_dates, y_xgbps_agg, label="XGBoost (per‑SKU)", color=colors["xgbps"], linewidth=1.7)

ax.set_title(f"Хвост {int(back_days)} дн., период: {period.lower()}")
ax.set_xlabel("Дата/Период"); ax.set_ylabel("Продажи")
if ax.lines:
    ax.legend(loc="upper right")
ax.grid(True)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)
try:
    import matplotlib.pyplot as _plt
    _plt.close(fig)
except Exception:
    pass


with st.expander("Диагностика", expanded=False):
    st.write({
        "pair": {"store_nbr": int(store_sel), "family": family_sel},
        "tail_len": len(tail),
        "tail_dates": {
            "min": str(tail["date"].min()) if "date" in tail.columns else None,
            "max": str(tail["date"].max()) if "date" in tail.columns else None,
        },
        "lgbm": {"feat_count": len(feat_lgb) if feat_lgb else 0},
        "catboost": {
            "available": mdl_cb is not None,
            "predicted": y_cb is not None,
            "features_from_meta": bool(cb_feats),
            "errors": cb_err or err_cb_pred,
        },
        "xgb_global": {
            "available": xgb_global is not None,
            "predicted": y_xgb is not None,
            "features_from_meta": bool(xgb_feats),
            "errors": xgb_err or err_xgb_pred,
        },
        "xgb_per_sku": {
            "available": mdl_xgb_ps is not None,
            "predicted": y_xgbps is not None,
            "errors": err_xgbps_pred,
        },
    })

# Метрики по хвосту (дневные)
st.subheader("Метрики по хвосту (дневные)")

def _mae_mape(y_true_arr, y_pred_arr):
    if y_true_arr is None or y_pred_arr is None:
        return None, None
    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
        return None, None
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    denom = np.where(y_true_arr == 0, 1, y_true_arr)
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)
    return mae, mape

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
mae_l, mape_l = _mae_mape(y_true, y_lgb)
if mae_l is not None:
    with col_m1:
        st.metric("LGBM — MAE", f"{mae_l:.3f}")
        st.metric("LGBM — MAPE %", f"{mape_l:.2f}%")
mae_cb, mape_cb = _mae_mape(y_true, y_cb)
if mae_cb is not None:
    with col_m2:
        st.metric("CatBoost — MAE", f"{mae_cb:.3f}")
        st.metric("CatBoost — MAPE %", f"{mape_cb:.2f}%")
mae_xgb, mape_xgb = _mae_mape(y_true, y_xgb)
if mae_xgb is not None:
    with col_m3:
        st.metric("XGBoost — MAE", f"{mae_xgb:.3f}")
        st.metric("XGBoost — MAPE %", f"{mape_xgb:.2f}%")
mae_xps, mape_xps = _mae_mape(y_true, y_xgbps)
if mae_xps is not None:
    with col_m4:
        st.metric("XGB per‑SKU — MAE", f"{mae_xps:.3f}")
        st.metric("XGB per‑SKU — MAPE %", f"{mape_xps:.2f}%")

st.markdown("---")
st.subheader("Сводные сравнения и выгрузка")
metric_choice = st.radio("Метрика для сравнения (heatmap)", ["MAE", "MAPE"], horizontal=True)
dim_choice = st.radio("Разрез Heatmap", ["По магазинам", "По семействам"], horizontal=True)
max_pairs = st.slider("Ограничить количество пар для расчёта", min_value=10, max_value=300, value=100, step=10)

# Фильтры по семействам и сумме продаж хвоста (как в 03-й странице)
all_fams = sorted(set(f for _, f in pairs))
fam_filter = st.multiselect("Фильтр по семействам (для сводной таблицы)", options=all_fams, default=[])
min_tail_sales = st.number_input("Мин. сумма продаж в хвосте (для включения пары)", min_value=0, value=0, step=10)

pairs_all = pairs[:max_pairs]
if fam_filter:
    pairs_all = [p for p in pairs_all if p[1] in fam_filter]
rows = []
for (s, f) in pairs_all:
    try:
        mp = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}.joblib"
        if not mp.exists():
            continue
        m = joblib.load(mp)
        feats = model_feature_names(m)
        df_p = Xfull[(Xfull["store_nbr"] == int(s)) & (Xfull["family"] == f)].sort_values("date").tail(int(back_days)).copy()
        if df_p.empty or not feats:
            continue
        for ff in feats:
            if ff not in df_p.columns:
                df_p[ff] = 0.0
        y_t = df_p["sales"].values
        tail_sum = float(np.sum(y_t))
        if tail_sum < float(min_tail_sales):
            continue
        y_l = m.predict(df_p[feats])
        row = {"store_nbr": int(s), "family": f}
        # CatBoost
        y_c = None
        if mdl_cb is not None and cb_feats:
            try:
                y_c = predict_catboost(mdl_cb, df_p, cb_feats, cb_cats)
            except Exception:
                y_c = None
        # XGB global
        y_x = None
        if xgb_global is not None and xgb_feats:
            try:
                y_x = predict_xgb(xgb_global, df_p, xgb_feats)
            except Exception:
                y_x = None
        # XGB per‑SKU
        y_xps = None
        xgbps_path = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}__xgb.joblib"
        if xgbps_path.exists():
            try:
                mx = joblib.load(xgbps_path)
                fx = getattr(mx, 'feature_names_in_', None) or feats
                for ff in fx:
                    if ff not in df_p.columns:
                        df_p[ff] = 0.0
                y_xps = predict_xgb(mx, df_p, list(fx))
            except Exception:
                y_xps = None

        # метрики
        denom = np.where(y_t == 0, 1, y_t)
        mae_l = float(np.mean(np.abs(y_t - y_l)))
        mape_l = float(np.mean(np.abs((y_t - y_l) / denom)) * 100.0)
        wmape_l = float(np.sum(np.abs(y_t - y_l)) / max(tail_sum, 1.0) * 100.0)
        # Weekly MAPE по недельной агрегации
        dfw = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_l})
        gw = dfw.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
        denom_w = np.where(gw["y_true"].values == 0, 1, gw["y_true"].values)
        weekly_mape_l = float(np.mean(np.abs((gw["y_true"].values - gw["y_pred"].values) / denom_w)) * 100.0)
        row.update({
            "LGBM_MAE": mae_l, "LGBM_MAPE": mape_l,
            "LGBM_wMAPE": wmape_l, "LGBM_wkMAPE": weekly_mape_l,
        })
        if y_c is not None:
            mae_c = float(np.mean(np.abs(y_t - y_c)))
            mape_c = float(np.mean(np.abs((y_t - y_c) / denom)) * 100.0)
            wmape_c = float(np.sum(np.abs(y_t - y_c)) / max(tail_sum, 1.0) * 100.0)
            # weekly для CB
            dfw_c = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_c})
            gw_c = dfw_c.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_c = np.where(gw_c["y_true"].values == 0, 1, gw_c["y_true"].values)
            weekly_mape_c = float(np.mean(np.abs((gw_c["y_true"].values - gw_c["y_pred"].values) / denom_w_c)) * 100.0)
            row.update({
                "CB_MAE": mae_c, "CB_MAPE": mape_c,
                "GAIN_CB_vs_LGBM_MAE": mae_l - mae_c,
                "GAIN_CB_vs_LGBM_MAPE": mape_l - mape_c,
                "CB_wMAPE": wmape_c, "CB_wkMAPE": weekly_mape_c,
            })
        if y_x is not None:
            mae_x = float(np.mean(np.abs(y_t - y_x)))
            mape_x = float(np.mean(np.abs((y_t - y_x) / denom)) * 100.0)
            wmape_x = float(np.sum(np.abs(y_t - y_x)) / max(tail_sum, 1.0) * 100.0)
            dfw_x = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_x})
            gw_x = dfw_x.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_x = np.where(gw_x["y_true"].values == 0, 1, gw_x["y_true"].values)
            weekly_mape_x = float(np.mean(np.abs((gw_x["y_true"].values - gw_x["y_pred"].values) / denom_w_x)) * 100.0)
            row.update({
                "XGB_MAE": mae_x, "XGB_MAPE": mape_x,
                "GAIN_XGB_vs_LGBM_MAE": mae_l - mae_x,
                "GAIN_XGB_vs_LGBM_MAPE": mape_l - mape_x,
                "XGB_wMAPE": wmape_x, "XGB_wkMAPE": weekly_mape_x,
            })
        if y_xps is not None:
            mae_xps = float(np.mean(np.abs(y_t - y_xps)))
            mape_xps = float(np.mean(np.abs((y_t - y_xps) / denom)) * 100.0)
            wmape_xps = float(np.sum(np.abs(y_t - y_xps)) / max(tail_sum, 1.0) * 100.0)
            dfw_xps = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_xps})
            gw_xps = dfw_xps.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_xps = np.where(gw_xps["y_true"].values == 0, 1, gw_xps["y_true"].values)
            weekly_mape_xps = float(np.mean(np.abs((gw_xps["y_true"].values - gw_xps["y_pred"].values) / denom_w_xps)) * 100.0)
            row.update({
                "XGBps_MAE": mae_xps, "XGBps_MAPE": mape_xps,
                "GAIN_XGBps_vs_LGBM_MAE": mae_l - mae_xps,
                "GAIN_XGBps_vs_LGBM_MAPE": mape_l - mape_xps,
                "XGBps_wMAPE": wmape_xps, "XGBps_wkMAPE": weekly_mape_xps,
            })
        rows.append(row)
    except Exception:
        continue

if rows:
    dfm = pd.DataFrame(rows)
    # Сводка (по магазинам/семействам) с wMAPE и Weekly MAPE + сортировка и выгрузки
    st.subheader("Сводка: агрегаты и сортировка")
    cols = [c for c in dfm.columns if c.endswith(("_MAE", "_MAPE")) or ("wMAPE" in c) or ("wkMAPE" in c)]
    if cols:
        group_mode = st.radio("Группировать сводку", ["По магазинам", "По семьям"], horizontal=True)
        group_col = "store_nbr" if group_mode == "По магазинам" else "family"
        summary = dfm.groupby(group_col)[cols].mean().reset_index()

        # Сортировка
        sort_metric = st.selectbox("Сортировать по метрике", options=cols, index=0, key="sum_sort_metric")
        sort_order = st.radio("Порядок", ["по убыванию", "по возрастанию"], horizontal=True, index=0, key="sum_sort_order")
        asc = True if sort_order == "по возрастанию" else False
        summary_sorted = summary.sort_values(sort_metric, ascending=asc)

        # Фильтрация по диапазону выбранной метрики
        min_v = float(np.nanmin(summary_sorted[sort_metric].values))
        max_v = float(np.nanmax(summary_sorted[sort_metric].values))
        # шаг подбираем грубо
        step = max((max_v - min_v) / 100.0, 1e-6)
        rng = st.slider(
            "Диапазон метрики",
            min_value=float(min_v),
            max_value=float(max_v),
            value=(float(min_v), float(max_v)),
            step=step,
            key="sum_range_slider",
        )
        summary_filtered = summary_sorted[(summary_sorted[sort_metric] >= rng[0]) & (summary_sorted[sort_metric] <= rng[1])]

        # Top‑N после фильтрации
        top_n = st.number_input("Показать топ‑N строк", min_value=1, max_value=1000, value=min(50, len(summary_filtered)), step=1, key="sum_top_n")
        summary_view = summary_filtered.head(int(top_n))
        try:
            styled = summary_view.style.background_gradient(cmap="YlGnBu", subset=[c for c in cols if ("wMAPE" in c) or ("wkMAPE" in c)])
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(summary_view, use_container_width=True)
        # Выгрузки
        st.download_button(
            f"⬇️ CSV: сводка ({'stores' if group_col=='store_nbr' else 'families'})",
            data=summary_view.to_csv(index=False).encode("utf-8"),
            file_name=("summary_per_store_extended.csv" if group_col == "store_nbr" else "summary_per_family_extended.csv"),
            mime="text/csv",
        )

    dim_col = "store_nbr" if dim_choice == "По магазинам" else "family"
    # CatBoost vs LGBM
    if {"GAIN_CB_vs_LGBM_MAE", "GAIN_CB_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_CB_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_CB_vs_LGBM_MAPE"
        pv = dfm.pivot_table(index=dim_col, columns="family" if dim_col=="store_nbr" else "store_nbr", values=val, aggfunc="mean").fillna(0.0)
        st.subheader("Heatmap: CatBoost vs LGBM (положительное = CatBoost лучше)")
        st.dataframe(pv.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button("⬇️ CSV: heatmap CB vs LGBM", data=pv.to_csv().encode("utf-8"), file_name="heatmap_cb_vs_lgbm.csv", mime="text/csv")
    # XGB vs LGBM
    if {"GAIN_XGB_vs_LGBM_MAE", "GAIN_XGB_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_XGB_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_XGB_vs_LGBM_MAPE"
        pv = dfm.pivot_table(index=dim_col, columns="family" if dim_col=="store_nbr" else "store_nbr", values=val, aggfunc="mean").fillna(0.0)
        st.subheader("Heatmap: XGB (global) vs LGBM (положительное = XGB лучше)")
        st.dataframe(pv.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button("⬇️ CSV: heatmap XGB vs LGBM", data=pv.to_csv().encode("utf-8"), file_name="heatmap_xgb_vs_lgbm.csv", mime="text/csv")
    # XGB per‑SKU vs LGBM
    if {"GAIN_XGBps_vs_LGBM_MAE", "GAIN_XGBps_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_XGBps_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_XGBps_vs_LGBM_MAPE"
        pv = dfm.pivot_table(index=dim_col, columns="family" if dim_col=="store_nbr" else "store_nbr", values=val, aggfunc="mean").fillna(0.0)
        st.subheader("Heatmap: XGB per‑SKU vs LGBM (положительное = XGB per‑SKU лучше)")
        st.dataframe(pv.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button("⬇️ CSV: heatmap XGB per‑SKU vs LGBM", data=pv.to_csv().encode("utf-8"), file_name="heatmap_xgbps_vs_lgbm.csv", mime="text/csv")
else:
    st.info("Недостаточно данных для сводного сравнения. Убедитесь, что модели обучены.")
