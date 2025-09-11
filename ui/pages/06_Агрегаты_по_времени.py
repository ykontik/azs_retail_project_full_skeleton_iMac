import os
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import streamlit as st

from make_features import make_features

st.set_page_config(page_title="Агрегаты по времени", layout="wide")
st.title("🗓️ Агрегаты ошибок по времени (недели/месяцы)")

DATA_DIR = Path(os.getenv("RAW_DIR", "data_raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


@st.cache_data(show_spinner=False)
def load_raw():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    trans = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"]) if (DATA_DIR/"transactions.csv").exists() else None
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"]) if (DATA_DIR/"oil.csv").exists() else None
    hol = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"]) if (DATA_DIR/"holidays_events.csv").exists() else None
    stores = pd.read_csv(DATA_DIR / "stores.csv") if (DATA_DIR/"stores.csv").exists() else None
    return train, hol, trans, oil, stores


@st.cache_data(show_spinner=True)
def build_full_features():
    train, hol, trans, oil, stores = load_raw()
    Xfull, _ = make_features(train, hol, trans, oil, stores, dropna_target=True)
    # категориальные — как в обучении
    for c in ["store_nbr","family","type","city","state","cluster","is_holiday"]:
        if c in Xfull.columns:
            Xfull[c] = Xfull[c].astype("category")
    # булевые в int
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")
    return Xfull


def model_feature_names(model) -> Optional[List[str]]:
    names = getattr(model, "feature_name_", None)
    if names is None and hasattr(model, "booster_"):
        try:
            names = list(model.booster_.feature_name())
        except Exception:
            names = None
    return names


st.sidebar.header("Параметры агрегации")
period = st.sidebar.radio("Период", ["Неделя", "Месяц"], index=0)
agg_mode = st.sidebar.radio("Агрегирование", ["Сумма", "Среднее"], index=0)
metric = st.sidebar.radio("Метрика", ["MAE", "MAPE"], index=1)
dimension = st.sidebar.radio("Разрез", ["По магазинам", "По семействам"], index=0)
back_days = st.sidebar.slider("Дней в хвосте", min_value=28, max_value=180, value=90, step=7)
max_models = st.sidebar.slider("Ограничить кол-во моделей", min_value=10, max_value=300, value=100, step=10)

freq = "W" if period == "Неделя" else "M"
agg_fn = "sum" if agg_mode == "Сумма" else "mean"
dim_key = "store_nbr" if dimension == "По магазинам" else "family"

models = sorted([p for p in MODELS_DIR.glob("*.joblib") if "__q" not in p.name])
if not models:
    st.warning("Нет моделей в каталоге models/. Запустите обучение.")
    st.stop()

Xfull = build_full_features()

# Фильтры по магазинам/семействам
all_stores = sorted(Xfull["store_nbr"].dropna().astype(int).unique().tolist()) if "store_nbr" in Xfull.columns else []
all_fams = sorted(Xfull["family"].dropna().astype(str).unique().tolist()) if "family" in Xfull.columns else []
f1, f2 = st.columns(2)
with f1:
    store_filter = st.multiselect("Фильтр магазинов", options=all_stores, default=[])
with f2:
    family_filter = st.multiselect("Фильтр семейств", options=all_fams, default=[])

progress = st.progress(0)
status = st.empty()

# Аккумуляторы по (dim, period)
acc = {}
total = min(len(models), max_models)
for i, mp in enumerate(models[:total], start=1):
    try:
        stem = mp.stem
        store_str, fam_str = stem.split("__", 1)
        store = int(store_str)
        fam = fam_str.replace("_", " ")
    except Exception:
        continue

    # применим фильтры по магазинам и семействам, если заданы
    if store_filter and store not in set(store_filter):
        continue
    if family_filter and fam not in set(family_filter):
        continue

    status.text(f"Обрабатываю модель {i}/{total}: ({store}, {fam})")
    try:
        model = joblib.load(mp)
    except Exception:
        continue
    feat_names = model_feature_names(model)
    if not feat_names:
        continue

    df_pair = Xfull[(Xfull["store_nbr"] == store) & (Xfull["family"] == fam)].copy().sort_values("date")
    if df_pair.empty:
        continue
    tail = df_pair.tail(int(back_days)).copy()
    if tail.empty or "sales" not in tail.columns:
        continue
    # гарантия наличия колонок фич
    for f in feat_names:
        if f not in tail.columns:
            tail[f] = 0.0
    X_tail = tail[feat_names]
    try:
        y_pred = model.predict(X_tail)
    except Exception:
        continue

    dfp = pd.DataFrame({
        "date": tail["date"].values,
        "y_true": tail["sales"].values,
        "y_pred": y_pred,
        dim_key: tail[dim_key].values,
    })
    # агрегируем по периоду и разрезу
    g = (dfp.set_index("date")
             .groupby([pd.Grouper(freq=freq), dim_key])
             .agg({"y_true": agg_fn, "y_pred": agg_fn})
             .reset_index())

    for _, row in g.iterrows():
        period_key = pd.to_datetime(row["date"]).date()
        dim_val = row[dim_key]
        key = (dim_val, period_key)
        if key not in acc:
            acc[key] = {"y_true": 0.0, "y_pred": 0.0}
        acc[key]["y_true"] += float(row["y_true"]) if agg_fn == "sum" else float(row["y_true"])  # mean уже учтён на шаге выше
        acc[key]["y_pred"] += float(row["y_pred"]) if agg_fn == "sum" else float(row["y_pred"])  # mean уже учтён на шаге выше

    progress.progress(i / total)

# Преобразуем аккумулированные данные в таблицу метрик
if not acc:
    st.warning("Недостаточно данных для построения агрегатов.")
    st.stop()

rows = []
for (dim_val, period_key), vals in acc.items():
    y_t = vals["y_true"]
    y_p = vals["y_pred"]
    ae = abs(y_t - y_p)
    me = 0.0 if y_t == 0 else abs((y_t - y_p) / y_t) * 100.0
    rows.append({dim_key: dim_val, "period": period_key, "MAE": ae, "MAPE": me})

dfm = pd.DataFrame(rows)
piv = dfm.pivot_table(index=dim_key, columns="period", values=metric, aggfunc="mean").fillna(0.0)

st.subheader(f"Тепловая карта: {metric} ({'сумма' if agg_fn=='sum' else 'среднее'}) по {dimension.lower()} и периодам")
st.caption("Ось Y — %s; ось X — %s. Подсказка: горизонтальная прокрутка для длинных периодов" %
           ("магазины" if dimension=="По магазинам" else "семейства", "недели" if period=="Неделя" else "месяцы"))
st.dataframe(piv.style.background_gradient(cmap="YlOrRd"), use_container_width=True)
st.download_button(
    "⬇️ Скачать тепловую карту (CSV)",
    data=piv.to_csv().encode("utf-8"),
    file_name=f"heatmap_{metric}_{'stores' if dimension=='По магазинам' else 'families'}_{'W' if period=='Неделя' else 'M'}.csv",
    mime="text/csv",
)
