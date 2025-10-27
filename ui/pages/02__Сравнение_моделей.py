import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import CategoricalDtype, is_bool_dtype, is_object_dtype

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch опционален для LSTM
    torch = None
    nn = None

pd.options.display.float_format = "{:.2f}".format

from make_features import make_features

st.set_page_config(page_title="Сравнение моделей (строгое)", layout="wide")
st.title("🧪 Сравнение моделей: LGBM per‑SKU vs Global CatBoost/XGB (строгое соответствие фич)")

RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DW_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))


@st.cache_data(show_spinner=False)
def load_raw():
    paths = {
        k: RAW_DIR / f"{k}.csv"
        for k in ["train", "transactions", "oil", "holidays_events", "stores"]
    }
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


def predict_catboost(
    model, df: pd.DataFrame, feature_list: List[str], categoricals: Optional[List[str]]
):
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


def prepare_rf_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    X = ensure_columns(df, feature_list).copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")
    obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in obj_cols:
        X[col] = X[col].astype("category").cat.codes.astype("int16")
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].cat.codes.astype("int16")
    return X.fillna(0.0)


def predict_rf(model, df: pd.DataFrame, feature_list: List[str]):
    X = prepare_rf_features(df, feature_list)
    return model.predict(X)


def prepare_lstm_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    X = ensure_columns(df, feature_list).copy()
    for col in feature_list:
        if col not in X.columns:
            continue
        series = X[col]
        if is_bool_dtype(series):
            X[col] = series.astype("int8")
        elif isinstance(series.dtype, CategoricalDtype):
            X[col] = series.cat.codes.astype("int16")
        elif is_object_dtype(series):
            X[col] = series.astype("category").cat.codes.astype("int16")
    return X.fillna(0.0)


if torch is not None:

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, max(1, hidden_size // 2)),
                nn.ReLU(),
                nn.Linear(max(1, hidden_size // 2), 1),
            )

        def forward(self, x):  # type: ignore[override]
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            pred = self.head(last)
            return pred.squeeze(-1)


def _safe_family(family: str) -> str:
    return str(family).replace(" ", "_")


def load_lstm_artifacts(store: int, family: str):
    if torch is None or nn is None:
        return None, None, None, "PyTorch не установлен"
    safe_family = _safe_family(family)
    model_path = MODELS_DIR / f"{store}__{safe_family}__lstm.pt"
    meta_path = MODELS_DIR / f"{store}__{safe_family}__lstm.lstm.json"
    scaler_path = MODELS_DIR / f"{store}__{safe_family}__lstm.scaler.pkl"
    if not model_path.exists() or not meta_path.exists() or not scaler_path.exists():
        return None, None, None, "Артефакты LSTM не найдены для пары"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, None, f"Ошибка чтения meta: {exc}"
    feature_cols = meta.get("feature_columns")
    if not isinstance(feature_cols, list) or not feature_cols:
        return None, None, None, "В meta отсутствует список признаков"
    window = int(meta.get("window", 30))
    hidden_size = int(meta.get("hidden_size", 128))
    num_layers = int(meta.get("num_layers", 2))
    dropout = float(meta.get("dropout", 0.2))
    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        return None, None, None, f"Ошибка загрузки scaler: {exc}"
    try:
        model = LSTMRegressor(len(feature_cols), hidden_size, num_layers, dropout)  # type: ignore[arg-type]
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:  # pragma: no cover - зависит от torch
        return None, None, None, f"Ошибка загрузки модели LSTM: {exc}"
    meta_out = {
        "feature_columns": feature_cols,
        "window": window,
    }
    return model, meta_out, scaler, None


def predict_lstm_series(
    store: int,
    family: str,
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    model, meta, scaler, err = load_lstm_artifacts(store, family)
    if err:
        return None, err
    assert meta is not None and model is not None and scaler is not None
    feature_cols = list(meta["feature_columns"])
    window = int(meta["window"])
    prepared = prepare_lstm_features(df, feature_cols)
    if len(prepared) <= window:
        return None, "Недостаточно данных для LSTM (короткий хвост)"
    try:
        scaled = scaler.transform(prepared[feature_cols])
    except Exception as exc:
        return None, f"Ошибка масштабирования LSTM: {exc}"
    sequences = []
    for idx in range(window, len(scaled)):
        sequences.append(scaled[idx - window : idx])
    if not sequences:
        return None, "Не удалось сформировать последовательности для LSTM"
    x_tensor = torch.from_numpy(np.stack(sequences).astype(np.float32))
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()
    full = np.full(len(prepared), np.nan, dtype=float)
    full[window:] = preds
    return full, None


# Список доступных per‑SKU моделей (берём базовые LGBM-файлы вида store__family.joblib)
models = sorted(
    [
        p
        for p in MODELS_DIR.glob("*.joblib")
        if "__q" not in p.name and not p.name.startswith("global_") and len(p.stem.split("__")) == 2
    ]
)
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
col_t1, col_t2, col_t3, col_t4, col_t5, col_t6, col_t7 = st.columns(7)
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
with col_t6:
    show_rf = st.checkbox("RandomForest per‑SKU", value=True, key="t_rf_strict")
with col_t7:
    show_lstm = st.checkbox("LSTM per‑SKU", value=False, key="t_lstm_strict")

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

st.subheader("Опции визуализации")
opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
with opt_col1:
    normalize_minmax = st.checkbox(
        "Нормализовать ряды (min-max)",
        value=False,
        help="Масштабировать значения от 0 до 1 относительно выбранного хвоста.",
        key="viz_norm_strict",
    )
with opt_col2:
    show_residuals = st.checkbox(
        "Показать остатки (y_true − y_pred)",
        value=False,
        disabled=y_true is None,
        key="viz_resid_strict",
    )
with opt_col3:
    separate_axes = st.checkbox(
        "Отдельные шкалы (подграфики)", value=False, key="viz_subplots_strict"
    )
with opt_col4:
    show_quantile_band = st.checkbox(
        "Показать коридор P50–P90 (если есть)",
        value=False,
        key="viz_quantiles_strict",
    )

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

mdl_rf = None
rf_feats: Optional[List[str]] = None
rf_err: Optional[str] = None
try:
    rf_path = MODELS_DIR / f"{base_stem}__rf.joblib"
    if rf_path.exists():
        mdl_rf = joblib.load(rf_path)
        rf_feats = getattr(mdl_rf, "feature_names_in_", None)
        if rf_feats is None:
            rf_json = MODELS_DIR / f"{base_stem}__rf.features.json"
            if rf_json.exists():
                import json as _json

                data = _json.loads(rf_json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rf_feats = [c for c in data if isinstance(c, str)]
        if rf_feats is not None:
            rf_feats = list(dict.fromkeys(rf_feats))
except Exception as exc:
    mdl_rf = None
    rf_err = str(exc)


def agg_series(dates: pd.Series, y: np.ndarray, how: str):
    df = pd.DataFrame({"date": dates.values, "y": y})
    if how == "День":
        return df["date"].values, df["y"].values
    rule = "W" if how == "Неделя" else "M"
    g = df.set_index("date").groupby(pd.Grouper(freq=rule)).sum().reset_index()
    return g["date"].values, g["y"].values


def agg_series_to_series(dates: pd.Series, y: np.ndarray | None, how: str) -> Optional[pd.Series]:
    if y is None:
        return None
    x_vals, y_vals = agg_series(dates, y, how)
    if len(x_vals) == 0:
        return pd.Series(dtype=float)
    index = pd.to_datetime(x_vals)
    return pd.Series(y_vals, index=index)


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

y_rf = None
err_rf_pred = None
if mdl_rf is not None:
    try:
        feats = list(rf_feats) if rf_feats else feat_lgb
        y_rf = predict_rf(mdl_rf, tail, feats)
    except Exception as e:
        err_rf_pred = str(e)

y_lstm = None
err_lstm_pred: Optional[str] = None
if show_lstm:
    try:
        y_lstm, err_lstm_pred = predict_lstm_series(int(store_sel), str(family_sel), tail)
    except Exception as e:  # pragma: no cover - зависит от torch
        err_lstm_pred = str(e)

q50_series = None
q90_series = None
quantile_note: Optional[str] = None
if show_quantile_band:
    try:
        q50_path = MODELS_DIR / f"{base_stem}__q50.joblib"
        q90_path = MODELS_DIR / f"{base_stem}__q90.joblib"
        if q50_path.exists() and q90_path.exists():
            mdl_q50 = joblib.load(q50_path)
            mdl_q90 = joblib.load(q90_path)
            q50_raw = mdl_q50.predict(X_tail_lgb)
            q90_raw = mdl_q90.predict(X_tail_lgb)
            q50_series = agg_series_to_series(tail["date"], q50_raw, period)
            q90_series = agg_series_to_series(tail["date"], q90_raw, period)
        else:
            quantile_note = "Квантильные модели (P50/P90) не найдены для выбранной пары."
    except Exception as exc:
        quantile_note = f"Ошибка применения квантильных моделей: {exc}"


# График
colors = {
    "Факт": "#1f77b4",
    "LGBM": "#ff7f0e",
    "CatBoost (global)": "#2ca02c",
    "XGBoost (global)": "#d62728",
    "XGBoost (per‑SKU)": "#17becf",
    "RandomForest (per‑SKU)": "#9467bd",
    "LSTM (per‑SKU)": "#8c564b",
}


def add_series_entry(
    entries: List[dict],
    label: str,
    values: Optional[np.ndarray],
    color: str,
    kind: str,
) -> None:
    series = agg_series_to_series(tail["date"], values, period)
    if series is None or series.empty:
        return
    entries.append({"label": label, "series": series, "color": color, "kind": kind})


series_entries: List[dict] = []
actual_series = agg_series_to_series(tail["date"], y_true, period)
if show_fact and actual_series is not None:
    series_entries.append(
        {"label": "Факт", "series": actual_series, "color": colors["Факт"], "kind": "actual"}
    )

if show_lgbm:
    add_series_entry(series_entries, "LGBM", y_lgb, colors["LGBM"], "pred")
if show_cb and (y_cb is not None):
    add_series_entry(series_entries, "CatBoost (global)", y_cb, colors["CatBoost (global)"], "pred")
if show_xgb and (y_xgb is not None):
    add_series_entry(series_entries, "XGBoost (global)", y_xgb, colors["XGBoost (global)"], "pred")
if show_xgbps and (y_xgbps is not None):
    add_series_entry(
        series_entries, "XGBoost (per‑SKU)", y_xgbps, colors["XGBoost (per‑SKU)"], "pred"
    )
if show_rf and (y_rf is not None):
    add_series_entry(
        series_entries, "RandomForest (per‑SKU)", y_rf, colors["RandomForest (per‑SKU)"], "pred"
    )
if show_lstm and (y_lstm is not None):
    add_series_entry(
        series_entries, "LSTM (per‑SKU)", y_lstm, colors["LSTM (per‑SKU)"], "pred"
    )

all_value_arrays: List[np.ndarray] = []
for entry in series_entries:
    if not entry["series"].empty:
        all_value_arrays.append(entry["series"].dropna().values)
if show_quantile_band and q50_series is not None and q90_series is not None:
    all_value_arrays.append(q50_series.dropna().values)
    all_value_arrays.append(q90_series.dropna().values)

if all_value_arrays:
    concat_values = np.concatenate([arr for arr in all_value_arrays if arr.size > 0])
    if concat_values.size > 0:
        global_min = float(np.nanmin(concat_values))
        global_max = float(np.nanmax(concat_values))
    else:
        global_min, global_max = 0.0, 1.0
else:
    global_min, global_max = 0.0, 1.0
global_range = global_max - global_min


def normalize_series(series: pd.Series) -> pd.Series:
    if not normalize_minmax or global_range <= 0:
        return series
    values = series.values
    norm = (values - global_min) / global_range if global_range > 0 else np.zeros_like(values)
    return pd.Series(norm, index=series.index)


plot_entries: List[dict] = []
for entry in series_entries:
    series = entry["series"]
    if series.empty:
        continue
    plot_entries.append(
        {
            **entry,
            "plot_series": normalize_series(series),
        }
    )

quantile_plot_pair: Optional[tuple[pd.Series, pd.Series]] = None
if show_quantile_band:
    if q50_series is not None and q90_series is not None:
        quantile_plot_pair = (normalize_series(q50_series), normalize_series(q90_series))
    elif quantile_note:
        st.info(quantile_note)

if not plot_entries:
    st.warning("Нет выбранных рядов для отображения. Включите хотя бы одну серию.")
else:
    ylabel = "Нормализованные продажи" if normalize_minmax else "Продажи"
    title = f"Хвост {int(back_days)} дн., период: {period.lower()}"

    if separate_axes and len(plot_entries) > 0:
        fig, axes = plt.subplots(
            len(plot_entries),
            1,
            sharex=True,
            figsize=(18, max(3.0 * len(plot_entries), 6.0)),
        )
        if len(plot_entries) == 1:
            axes = [axes]
        for idx, (ax, entry) in enumerate(zip(axes, plot_entries)):
            series = entry["plot_series"]
            ax.plot(
                series.index,
                series.values,
                color=entry["color"],
                linewidth=2.0,
                label=entry["label"],
            )
            ax.set_ylabel(ylabel)
            ax.set_title(entry["label"], loc="left")
            ax.grid(True, alpha=0.3)
            if idx == len(plot_entries) - 1:
                ax.set_xlabel("Дата/Период")
        target_ax = axes[0]
        if quantile_plot_pair is not None:
            q_df = pd.concat(quantile_plot_pair, axis=1).dropna()
            q_df.columns = ["P50", "P90"]
            base_index = plot_entries[0]["plot_series"].index
            q_df = q_df.reindex(base_index).dropna()
            if not q_df.empty:
                target_ax.fill_between(
                    q_df.index,
                    q_df["P50"].values,
                    q_df["P90"].values,
                    color="#ffd166",
                    alpha=0.25,
                    label="P50–P90",
                )
                target_ax.legend(loc="upper right")
        axes[0].set_title(title)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(18, 7))
        for entry in plot_entries:
            series = entry["plot_series"]
            ax.plot(
                series.index,
                series.values,
                label=entry["label"],
                color=entry["color"],
                linewidth=2.0 if entry["kind"] == "actual" else 1.7,
            )
        if quantile_plot_pair is not None:
            q_df = pd.concat(quantile_plot_pair, axis=1).dropna()
            q_df.columns = ["P50", "P90"]
            base_index = plot_entries[0]["plot_series"].index
            q_df = q_df.reindex(base_index).dropna()
            if not q_df.empty:
                ax.fill_between(
                    q_df.index,
                    q_df["P50"].values,
                    q_df["P90"].values,
                    color="#ffd166",
                    alpha=0.25,
                    label="P50–P90",
                )
        ax.set_title(title)
        ax.set_xlabel("Дата/Период")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

residual_entries: List[dict] = []
if show_residuals and actual_series is not None:
    for entry in series_entries:
        if entry["kind"] != "pred":
            continue
        aligned_actual, aligned_pred = actual_series.align(entry["series"], join="inner")
        if aligned_actual.empty or aligned_pred.empty:
            continue
        residual_series = aligned_actual - aligned_pred
        residual_entries.append(
            {"label": entry["label"], "series": residual_series, "color": entry["color"]}
        )

if residual_entries:
    fig_res, ax_res = plt.subplots(figsize=(18, 4))
    for entry in residual_entries:
        ax_res.plot(
            entry["series"].index,
            entry["series"].values,
            label=entry["label"],
            color=entry["color"],
            linewidth=1.6,
        )
    ax_res.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax_res.set_title("Остатки (y_true − y_pred)")
    ax_res.set_xlabel("Дата/Период")
    ax_res.set_ylabel("Продажи")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc="upper right")
    fig_res.tight_layout()
    st.pyplot(fig_res, clear_figure=True)
    plt.close(fig_res)


with st.expander("Диагностика", expanded=False):
    st.write(
        {
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
            "rf_per_sku": {
                "available": mdl_rf is not None,
                "predicted": y_rf is not None,
                "errors": rf_err or err_rf_pred,
            },
            "lstm_per_sku": {
                "requested": show_lstm,
                "predicted": y_lstm is not None,
                "errors": err_lstm_pred,
                "torch_available": torch is not None,
            },
        }
    )

# Метрики по хвосту (дневные)
st.subheader("Метрики по хвосту (дневные)")


def _mae_mape(y_true_arr, y_pred_arr):
    if y_true_arr is None or y_pred_arr is None:
        return None, None
    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) == 0:
        return None, None
    mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    if not np.any(mask):
        return None, None
    y_true_f = y_true_arr[mask]
    y_pred_f = y_pred_arr[mask]
    mae = float(np.mean(np.abs(y_true_f - y_pred_f)))
    denom = np.where(y_true_f == 0, 1, y_true_f)
    mape = float(np.mean(np.abs((y_true_f - y_pred_f) / denom)) * 100.0)
    return mae, mape


col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
mae_l, mape_l = _mae_mape(y_true, y_lgb)
if mae_l is not None:
    with col_m1:
        st.metric("LGBM — MAE", f"{mae_l:.2f}")
        st.metric("LGBM — MAPE %", f"{mape_l:.2f}%")
mae_cb, mape_cb = _mae_mape(y_true, y_cb)
if mae_cb is not None:
    with col_m2:
        st.metric("CatBoost — MAE", f"{mae_cb:.2f}")
        st.metric("CatBoost — MAPE %", f"{mape_cb:.2f}%")
mae_xgb, mape_xgb = _mae_mape(y_true, y_xgb)
if mae_xgb is not None:
    with col_m3:
        st.metric("XGBoost — MAE", f"{mae_xgb:.2f}")
        st.metric("XGBoost — MAPE %", f"{mape_xgb:.2f}%")
mae_xps, mape_xps = _mae_mape(y_true, y_xgbps)
if mae_xps is not None:
    with col_m4:
        st.metric("XGB per‑SKU — MAE", f"{mae_xps:.2f}")
        st.metric("XGB per‑SKU — MAPE %", f"{mape_xps:.2f}%")
mae_rf, mape_rf = _mae_mape(y_true, y_rf)
if mae_rf is not None:
    with col_m5:
        st.metric("RandomForest — MAE", f"{mae_rf:.2f}")
        st.metric("RandomForest — MAPE %", f"{mape_rf:.2f}%")

mae_lstm, mape_lstm = _mae_mape(y_true, y_lstm)
if mae_lstm is not None:
    with col_m6:
        st.metric("LSTM — MAE", f"{mae_lstm:.2f}")
        st.metric("LSTM — MAPE %", f"{mape_lstm:.2f}%")

st.markdown("---")
st.subheader("Сводные сравнения и выгрузка")
metric_choice = st.radio("Метрика для сравнения (heatmap)", ["MAE", "MAPE"], horizontal=True)
dim_choice = st.radio("Разрез Heatmap", ["По магазинам", "По семействам"], horizontal=True)
max_pairs = st.slider(
    "Ограничить количество пар для расчёта", min_value=10, max_value=300, value=100, step=10
)

# Фильтры по семействам и сумме продаж хвоста (как в 03-й странице)
all_fams = sorted(set(f for _, f in pairs))
fam_filter = st.multiselect(
    "Фильтр по семействам (для сводной таблицы)", options=all_fams, default=[]
)
min_tail_sales = st.number_input(
    "Мин. сумма продаж в хвосте (для включения пары)", min_value=0, value=0, step=10
)

pairs_all = pairs[:max_pairs]
if fam_filter:
    pairs_all = [p for p in pairs_all if p[1] in fam_filter]
rows = []
for s, f in pairs_all:
    try:
        mp = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}.joblib"
        if not mp.exists():
            continue
        m = joblib.load(mp)
        feats = model_feature_names(m)
        df_p = (
            Xfull[(Xfull["store_nbr"] == int(s)) & (Xfull["family"] == f)]
            .sort_values("date")
            .tail(int(back_days))
            .copy()
        )
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
                fx = getattr(mx, "feature_names_in_", None) or feats
                for ff in fx:
                    if ff not in df_p.columns:
                        df_p[ff] = 0.0
                y_xps = predict_xgb(mx, df_p, list(fx))
            except Exception:
                y_xps = None
        # RandomForest per‑SKU
        y_rf_pair = None
        rf_path = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}__rf.joblib"
        if rf_path.exists():
            try:
                mr = joblib.load(rf_path)
                fr = getattr(mr, "feature_names_in_", None)
                if fr is None:
                    rf_json = MODELS_DIR / f"{int(s)}__{str(f).replace(' ', '_')}__rf.features.json"
                    if rf_json.exists():
                        import json as _json

                        data = _json.loads(rf_json.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            fr = [c for c in data if isinstance(c, str)]
                fr_list = list(fr) if fr else feats
                y_rf_pair = predict_rf(mr, df_p, fr_list)
            except Exception:
                y_rf_pair = None

        # метрики
        denom = np.where(y_t == 0, 1, y_t)
        mae_l = float(np.mean(np.abs(y_t - y_l)))
        mape_l = float(np.mean(np.abs((y_t - y_l) / denom)) * 100.0)
        wmape_l = float(np.sum(np.abs(y_t - y_l)) / max(tail_sum, 1.0) * 100.0)
        # Weekly MAPE по недельной агрегации
        dfw = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_l})
        gw = dfw.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
        denom_w = np.where(gw["y_true"].values == 0, 1, gw["y_true"].values)
        weekly_mape_l = float(
            np.mean(np.abs((gw["y_true"].values - gw["y_pred"].values) / denom_w)) * 100.0
        )
        row.update(
            {
                "LGBM_MAE": round(mae_l, 2),
                "LGBM_MAPE": round(mape_l, 2),
                "LGBM_wMAPE": round(wmape_l, 2),
                "LGBM_wkMAPE": round(weekly_mape_l, 2),
            }
        )
        if y_c is not None:
            mae_c = float(np.mean(np.abs(y_t - y_c)))
            mape_c = float(np.mean(np.abs((y_t - y_c) / denom)) * 100.0)
            wmape_c = float(np.sum(np.abs(y_t - y_c)) / max(tail_sum, 1.0) * 100.0)
            # weekly для CB
            dfw_c = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_c})
            gw_c = dfw_c.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_c = np.where(gw_c["y_true"].values == 0, 1, gw_c["y_true"].values)
            weekly_mape_c = float(
                np.mean(np.abs((gw_c["y_true"].values - gw_c["y_pred"].values) / denom_w_c)) * 100.0
            )
            row.update(
                {
                    "CB_MAE": round(mae_c, 2),
                    "CB_MAPE": round(mape_c, 2),
                    "GAIN_CB_vs_LGBM_MAE": round(mae_l - mae_c, 2),
                    "GAIN_CB_vs_LGBM_MAPE": round(mape_l - mape_c, 2),
                    "CB_wMAPE": round(wmape_c, 2),
                    "CB_wkMAPE": round(weekly_mape_c, 2),
                }
            )
        if y_x is not None:
            mae_x = float(np.mean(np.abs(y_t - y_x)))
            mape_x = float(np.mean(np.abs((y_t - y_x) / denom)) * 100.0)
            wmape_x = float(np.sum(np.abs(y_t - y_x)) / max(tail_sum, 1.0) * 100.0)
            dfw_x = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_x})
            gw_x = dfw_x.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_x = np.where(gw_x["y_true"].values == 0, 1, gw_x["y_true"].values)
            weekly_mape_x = float(
                np.mean(np.abs((gw_x["y_true"].values - gw_x["y_pred"].values) / denom_w_x)) * 100.0
            )
            row.update(
                {
                    "XGB_MAE": round(mae_x, 2),
                    "XGB_MAPE": round(mape_x, 2),
                    "GAIN_XGB_vs_LGBM_MAE": round(mae_l - mae_x, 2),
                    "GAIN_XGB_vs_LGBM_MAPE": round(mape_l - mape_x, 2),
                    "XGB_wMAPE": round(wmape_x, 2),
                    "XGB_wkMAPE": round(weekly_mape_x, 2),
                }
            )
        if y_xps is not None:
            mae_xps = float(np.mean(np.abs(y_t - y_xps)))
            mape_xps = float(np.mean(np.abs((y_t - y_xps) / denom)) * 100.0)
            wmape_xps = float(np.sum(np.abs(y_t - y_xps)) / max(tail_sum, 1.0) * 100.0)
            dfw_xps = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_xps})
            gw_xps = dfw_xps.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_xps = np.where(gw_xps["y_true"].values == 0, 1, gw_xps["y_true"].values)
            weekly_mape_xps = float(
                np.mean(np.abs((gw_xps["y_true"].values - gw_xps["y_pred"].values) / denom_w_xps))
                * 100.0
            )
            row.update(
                {
                    "XGBps_MAE": round(mae_xps, 2),
                    "XGBps_MAPE": round(mape_xps, 2),
                    "GAIN_XGBps_vs_LGBM_MAE": round(mae_l - mae_xps, 2),
                    "GAIN_XGBps_vs_LGBM_MAPE": round(mape_l - mape_xps, 2),
                    "XGBps_wMAPE": round(wmape_xps, 2),
                    "XGBps_wkMAPE": round(weekly_mape_xps, 2),
                }
            )
        if y_rf_pair is not None:
            mae_rf_pair = float(np.mean(np.abs(y_t - y_rf_pair)))
            mape_rf_pair = float(np.mean(np.abs((y_t - y_rf_pair) / denom)) * 100.0)
            wmape_rf = float(np.sum(np.abs(y_t - y_rf_pair)) / max(tail_sum, 1.0) * 100.0)
            dfw_rf = pd.DataFrame({"date": df_p["date"].values, "y_true": y_t, "y_pred": y_rf_pair})
            gw_rf = dfw_rf.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
            denom_w_rf = np.where(gw_rf["y_true"].values == 0, 1, gw_rf["y_true"].values)
            weekly_mape_rf = float(
                np.mean(np.abs((gw_rf["y_true"].values - gw_rf["y_pred"].values) / denom_w_rf))
                * 100.0
            )
            row.update(
                {
                    "RF_MAE": round(mae_rf_pair, 2),
                    "RF_MAPE": round(mape_rf_pair, 2),
                    "GAIN_RF_vs_LGBM_MAE": round(mae_l - mae_rf_pair, 2),
                    "GAIN_RF_vs_LGBM_MAPE": round(mape_l - mape_rf_pair, 2),
                    "RF_wMAPE": round(wmape_rf, 2),
                    "RF_wkMAPE": round(weekly_mape_rf, 2),
                }
            )
        y_lstm_pair = None
        try:
            y_lstm_pair, lstm_pair_err = predict_lstm_series(int(s), str(f), df_p)
            if lstm_pair_err:
                y_lstm_pair = None
        except Exception:
            y_lstm_pair = None
        if y_lstm_pair is not None:
            mask_lstm = ~np.isnan(y_lstm_pair)
            if np.any(mask_lstm):
                y_true_lstm = y_t[mask_lstm]
                y_pred_lstm = y_lstm_pair[mask_lstm]
                mae_lstm_pair = float(np.mean(np.abs(y_true_lstm - y_pred_lstm)))
                mape_lstm_pair = float(
                    np.mean(
                        np.abs(
                            (y_true_lstm - y_pred_lstm)
                            / np.where(y_true_lstm == 0, 1, y_true_lstm)
                        )
                    )
                    * 100.0
                )
                wmape_lstm = float(
                    np.sum(np.abs(y_true_lstm - y_pred_lstm)) / max(np.sum(y_true_lstm), 1.0) * 100.0
                )
                dfw_lstm = pd.DataFrame(
                    {
                        "date": df_p["date"].values[mask_lstm],
                        "y_true": y_true_lstm,
                        "y_pred": y_pred_lstm,
                    }
                )
                gw_lstm = (
                    dfw_lstm.set_index("date").groupby(pd.Grouper(freq="W")).sum().reset_index()
                )
                denom_w_lstm = np.where(
                    gw_lstm["y_true"].values == 0, 1, gw_lstm["y_true"].values
                )
                weekly_mape_lstm = float(
                    np.mean(
                        np.abs((gw_lstm["y_true"].values - gw_lstm["y_pred"].values) / denom_w_lstm)
                    )
                    * 100.0
                )
                row.update(
                    {
                        "LSTM_MAE": round(mae_lstm_pair, 2),
                        "LSTM_MAPE": round(mape_lstm_pair, 2),
                        "GAIN_LSTM_vs_LGBM_MAE": round(mae_l - mae_lstm_pair, 2),
                        "GAIN_LSTM_vs_LGBM_MAPE": round(mape_l - mape_lstm_pair, 2),
                        "LSTM_wMAPE": round(wmape_lstm, 2),
                        "LSTM_wkMAPE": round(weekly_mape_lstm, 2),
                    }
                )
        rows.append(row)
    except Exception:
        continue

if rows:
    dfm = pd.DataFrame(rows)
    # Сводка (по магазинам/семействам) с wMAPE и Weekly MAPE + сортировка и выгрузки
    st.subheader("Сводка: агрегаты и сортировка")
    cols = [
        c for c in dfm.columns if c.endswith(("_MAE", "_MAPE")) or ("wMAPE" in c) or ("wkMAPE" in c)
    ]
    if cols:
        group_mode = st.radio("Группировать сводку", ["По магазинам", "По семьям"], horizontal=True)
        group_col = "store_nbr" if group_mode == "По магазинам" else "family"
        summary = dfm.groupby(group_col)[cols].mean().reset_index()

        # Сортировка
        sort_metric = st.selectbox(
            "Сортировать по метрике", options=cols, index=0, key="sum_sort_metric"
        )
        sort_order = st.radio(
            "Порядок",
            ["по убыванию", "по возрастанию"],
            horizontal=True,
            index=0,
            key="sum_sort_order",
        )
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
        summary_filtered = summary_sorted[
            (summary_sorted[sort_metric] >= rng[0]) & (summary_sorted[sort_metric] <= rng[1])
        ]
        count_filtered = len(summary_filtered)
        if count_filtered == 0:
            st.info("Нет строк, удовлетворяющих выбранному диапазону.")
        else:
            top_n = st.number_input(
                "Показать топ‑N строк",
                min_value=1,
                max_value=min(1000, max(1, summary_sorted.shape[0])),
                value=min(50, max(1, summary_sorted.shape[0])),
                step=1,
                key="sum_top_n",
            )
            summary_view = summary_filtered.head(int(top_n))
            summary_display = summary_view.copy()
            for col in cols:
                if col in summary_display.columns:
                    summary_display[col] = summary_display[col].round(2)
            try:
                styled = summary_display.style.background_gradient(
                    cmap="YlGnBu", subset=[c for c in cols if ("wMAPE" in c) or ("wkMAPE" in c)]
                )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(summary_display, use_container_width=True)
            # Выгрузки
            st.download_button(
                f"⬇️ CSV: сводка ({'stores' if group_col=='store_nbr' else 'families'})",
                data=summary_display.to_csv(index=False).encode("utf-8"),
                file_name=(
                    "summary_per_store_extended.csv"
                    if group_col == "store_nbr"
                    else "summary_per_family_extended.csv"
                ),
                mime="text/csv",
            )

    dim_col = "store_nbr" if dim_choice == "По магазинам" else "family"
    # CatBoost vs LGBM
    if {"GAIN_CB_vs_LGBM_MAE", "GAIN_CB_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_CB_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_CB_vs_LGBM_MAPE"
        pv = dfm.pivot_table(
            index=dim_col,
            columns="family" if dim_col == "store_nbr" else "store_nbr",
            values=val,
            aggfunc="mean",
        ).fillna(0.0)
        pv_display = pv.round(2)
        st.subheader("Heatmap: CatBoost vs LGBM (положительное = CatBoost лучше)")
        st.dataframe(pv_display.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button(
            "⬇️ CSV: heatmap CB vs LGBM",
            data=pv_display.to_csv().encode("utf-8"),
            file_name="heatmap_cb_vs_lgbm.csv",
            mime="text/csv",
        )
    # XGB vs LGBM
    if {"GAIN_XGB_vs_LGBM_MAE", "GAIN_XGB_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_XGB_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_XGB_vs_LGBM_MAPE"
        pv = dfm.pivot_table(
            index=dim_col,
            columns="family" if dim_col == "store_nbr" else "store_nbr",
            values=val,
            aggfunc="mean",
        ).fillna(0.0)
        pv_display = pv.round(2)
        st.subheader("Heatmap: XGB (global) vs LGBM (положительное = XGB лучше)")
        st.dataframe(pv_display.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button(
            "⬇️ CSV: heatmap XGB vs LGBM",
            data=pv_display.to_csv().encode("utf-8"),
            file_name="heatmap_xgb_vs_lgbm.csv",
            mime="text/csv",
        )
    # XGB per‑SKU vs LGBM
    if {"GAIN_XGBps_vs_LGBM_MAE", "GAIN_XGBps_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_XGBps_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_XGBps_vs_LGBM_MAPE"
        pv = dfm.pivot_table(
            index=dim_col,
            columns="family" if dim_col == "store_nbr" else "store_nbr",
            values=val,
            aggfunc="mean",
        ).fillna(0.0)
        pv_display = pv.round(2)
        st.subheader("Heatmap: XGB per‑SKU vs LGBM (положительное = XGB per‑SKU лучше)")
        st.dataframe(pv_display.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button(
            "⬇️ CSV: heatmap XGB per‑SKU vs LGBM",
            data=pv_display.to_csv().encode("utf-8"),
            file_name="heatmap_xgbps_vs_lgbm.csv",
            mime="text/csv",
        )
    if {"GAIN_RF_vs_LGBM_MAE", "GAIN_RF_vs_LGBM_MAPE"}.issubset(dfm.columns):
        val = "GAIN_RF_vs_LGBM_MAE" if metric_choice == "MAE" else "GAIN_RF_vs_LGBM_MAPE"
        pv = dfm.pivot_table(
            index=dim_col,
            columns="family" if dim_col == "store_nbr" else "store_nbr",
            values=val,
            aggfunc="mean",
        ).fillna(0.0)
        pv_display = pv.round(2)
        st.subheader("Heatmap: RandomForest vs LGBM (положительное = RandomForest лучше)")
        st.dataframe(pv_display.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        st.download_button(
            "⬇️ CSV: heatmap RF vs LGBM",
            data=pv_display.to_csv().encode("utf-8"),
            file_name="heatmap_rf_vs_lgbm.csv",
            mime="text/csv",
        )
else:
    st.info("Недостаточно данных для сводного сравнения. Убедитесь, что модели обучены.")
