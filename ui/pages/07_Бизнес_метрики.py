import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Бизнес‑метрики", layout="wide")
st.title("💼 Бизнес‑метрики: MAPE → деньги, SS/ROP, влияние")

RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))
DW_DIR = Path(os.getenv("WAREHOUSE_DIR", "data_dw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


@st.cache_data(show_spinner=False)
def load_train():
    p = RAW_DIR / "train.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, parse_dates=["date"])  # ожидаются date, store_nbr, family, sales
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    p = DW_DIR / "metrics_per_sku.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_prices():
    p = Path("configs/prices.csv")
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        # ожидаемые колонки: family[, store_nbr, price, margin_rate, holding_cost]
        return df
    except Exception:
        return None


train = load_train()
metrics = load_metrics()
prices = load_prices()

if train is None:
    st.error("Не найден data_raw/train.csv — для расчётов нужен объём продаж.")
    st.stop()

# Списки выбора
pairs = (
    train[["store_nbr", "family"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["store_nbr", "family"])  # type: ignore[arg-type]
    .values.tolist()
)

st.sidebar.header("Параметры")
store_sel = st.sidebar.selectbox("store_nbr", sorted(sorted(set(int(s) for s, _ in pairs))))
fam_opts = sorted([str(f) for s, f in pairs if int(s) == int(store_sel)])
family_sel = st.sidebar.selectbox("family", fam_opts)
valid_days = st.sidebar.slider("Дней в хвосте (для средних)", min_value=14, max_value=90, value=30, step=1)

# Подставим прайс/маржу/хранение из configs/prices.csv при наличии
def _defaults_from_prices(sto: int, fam: str):
    if prices is None or prices.empty:
        return None, None, None
    df = prices.copy()
    cand = df[df["family"].astype(str) == str(fam)]
    # приоритет: запись с нужным store_nbr, затем без store_nbr
    if "store_nbr" in df.columns:
        c2 = df[(df["family"].astype(str) == str(fam)) & (df["store_nbr"].astype(float) == float(sto))]
        if not c2.empty:
            cand = c2
    if cand.empty:
        return None, None, None
    row = cand.iloc[0].to_dict()
    return row.get("price"), row.get("margin_rate"), row.get("holding_cost")

price_def, margin_def, holding_def = _defaults_from_prices(int(store_sel), str(family_sel))

st.subheader("Ввод параметров")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    price = st.number_input("Цена", min_value=0.0, value=float(price_def or 3.5), step=0.1)
with col2:
    margin_rate = st.number_input("Маржинальность", min_value=0.0, max_value=1.0, value=float(margin_def or 0.25), step=0.01)
with col3:
    holding_cost = st.number_input("Хранение/день", min_value=0.0, value=float(holding_def or 0.05), step=0.01)
with col4:
    lead_time_days = st.number_input("Lead time (дней)", min_value=1, value=2, step=1)
with col5:
    service_level = st.number_input("Уровень сервиса", min_value=0.80, max_value=0.99, value=0.95, step=0.01)


def _z_from_service_level(p: float) -> float:
    table = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}
    closest = min(table.keys(), key=lambda x: abs(x - p))
    return table[closest]


st.subheader("Расчёт по паре")
df_pair = train[(train["store_nbr"] == int(store_sel)) & (train["family"].astype(str) == str(family_sel))].copy()
df_pair = df_pair.sort_values("date")
tail = df_pair.tail(int(valid_days)).copy()
daily_mean = float(tail["sales"].mean()) if not tail.empty else 0.0

mape_pct = None
if metrics is not None and not metrics.empty:
    sub = metrics[(metrics["store_nbr"] == int(store_sel)) & (metrics["family"].astype(str) == str(family_sel))]
    if not sub.empty and "MAPE_%" in sub.columns:
        try:
            mape_pct = float(sub.iloc[0]["MAPE_%"])
        except Exception:
            mape_pct = None

if mape_pct is None:
    mape_pct = 20.0  # запасной вариант

sigma = max((mape_pct / 100.0) * daily_mean, 0.0)
z = _z_from_service_level(float(service_level))
L = max(int(lead_time_days), 1)
safety_stock = z * sigma * (L ** 0.5)
rop = daily_mean * L + safety_stock

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Средний спрос/день", f"{daily_mean:.3f}")
with colB:
    st.metric("MAPE %", f"{mape_pct:.2f}%")
with colC:
    st.metric("Safety Stock", f"{safety_stock:.1f}")
with colD:
    st.metric("Reorder Point", f"{rop:.1f}")

# Денежные оценки: упущенная маржа (недопоставка) и хранение (излишки)
expected_under_units = (1 - float(service_level)) * daily_mean * L
under_cost = expected_under_units * (float(margin_rate) * float(price))
over_cost = float(holding_cost) * float(safety_stock)

st.caption("Денежные оценки (приближённо)")
c1, c2 = st.columns(2)
with c1:
    st.metric("Underage (≈ упущенная маржа за окно поставки)", f"{under_cost:.2f}")
with c2:
    st.metric("Overage (≈ хранение/день)", f"{over_cost:.2f}")

st.markdown("---")
st.subheader("MAPE → Деньги: дневной и месячный эффект")
rev_loss_day = (mape_pct / 100.0) * daily_mean * float(price)
rev_loss_month = rev_loss_day * 30.0
margin_loss_month = rev_loss_month * float(margin_rate)

colE, colF, colG = st.columns(3)
with colE:
    st.metric("Потеря выручки/день (оценка)", f"{rev_loss_day:.2f}")
with colF:
    st.metric("Потеря выручки/мес (оценка)", f"{rev_loss_month:.2f}")
with colG:
    st.metric("Потеря маржи/мес (оценка)", f"{margin_loss_month:.2f}")

# Расчёт SS/ROP через API (FastAPI /reorder_point)
st.markdown("---")
st.subheader("Расчёт через API (FastAPI)")

def _resolve_api_url() -> str:
    api = os.getenv("API_URL")
    if api:
        return api
    # Попробуем secrets.toml (как в дашборде)
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if p.exists():
            try:
                import tomllib
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                return data.get("API_URL", "http://127.0.0.1:8000")
            except Exception:
                pass
    return "http://127.0.0.1:8000"

api_url = st.text_input("API URL", value=_resolve_api_url(), help="Адрес FastAPI сервиса")

def build_features_dict_for_pair(df_all: pd.DataFrame, s: int, f: str) -> Optional[Dict[str, float]]:
    # Возьмём последнюю строку по паре из train и сформируем фичи, ориентируясь на признаки per‑SKU модели (если есть)
    try:
        from make_features import make_features as _mf
        # Попробуем собрать фичи из сырых данных (если есть подключение к остальным CSV)
        # Используем train только для выбора последнего дня — более точные фичи потребуют полноценного пайплайна
        last_row = df_all[(df_all["store_nbr"] == int(s)) & (df_all["family"].astype(str) == str(f))].sort_values("date").tail(1)
        if last_row.empty:
            return None
        # Фallback: загрузим остальные источники, если доступны, и пересоберём фичи корректно
        paths = {k: RAW_DIR / f"{k}.csv" for k in ["transactions", "oil", "holidays_events", "stores"]}
        trans = pd.read_csv(paths["transactions"], parse_dates=["date"]) if paths["transactions"].exists() else None
        oil = pd.read_csv(paths["oil"], parse_dates=["date"]) if paths["oil"].exists() else None
        hol = pd.read_csv(paths["holidays_events"], parse_dates=["date"]) if paths["holidays_events"].exists() else None
        stores = pd.read_csv(paths["stores"]) if paths["stores"].exists() else None
        Xf, _ = _mf(df_all, hol, trans, oil, stores, dropna_target=False)
        sub = Xf[(Xf["store_nbr"] == int(s)) & (Xf["family"].astype(str) == str(f))].sort_values("date")
        if sub.empty:
            return None
        # Список фич из модели (если есть)
        base = f"{int(s)}__{str(f).replace(' ', '_')}"
        feat_json = MODELS_DIR / f"{base}.features.json"
        feat_cols = None
        if feat_json.exists():
            try:
                data = json.loads(feat_json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    feat_cols = [c for c in data if c in sub.columns]
            except Exception:
                feat_cols = None
        if feat_cols is None:
            # Общий список: все числовые кроме служебных
            exclude = {"id", "sales", "date"}
            feat_cols = [c for c in sub.columns if (c not in exclude) and pd.api.types.is_numeric_dtype(sub[c])]
        last = sub.iloc[-1]
        feats = {}
        for name in feat_cols:
            val = last.get(name, 0.0)
            try:
                feats[name] = float(val)
            except Exception:
                feats[name] = 0.0
        return feats
    except Exception:
        return None

col_api1, col_api2 = st.columns(2)
with col_api1:
    if st.button("Рассчитать через API /reorder_point"):
        feats = build_features_dict_for_pair(train, int(store_sel), str(family_sel))
        if feats is None or not feats:
            st.error("Не удалось собрать фичи для пары. Убедитесь, что данные доступны (transactions/oil/holidays/stores).")
        else:
            payload = {
                "store_nbr": int(store_sel),
                "family": str(family_sel),
                "features": feats,
                "lead_time_days": int(lead_time_days),
                "service_level": float(service_level),
            }
            try:
                r = requests.post(f"{api_url}/reorder_point", json=payload, timeout=8)
                if not r.ok:
                    st.error(f"API ошибка: {r.status_code} {r.text}")
                else:
                    data = r.json()
                    st.success("SS/ROP получены из API")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Daily mean (API)", f"{data.get('daily_mean', 0):.3f}")
                    with c2: st.metric("Sigma (API)", f"{data.get('sigma_daily', 0):.3f}")
                    with c3: st.metric("Safety Stock (API)", f"{data.get('safety_stock', 0):.1f}")
                    with c4: st.metric("ROP (API)", f"{data.get('reorder_point', 0):.1f}")
            except Exception as e:
                st.error(f"Не удалось обратиться к API: {e}")

# Массовый бизнес‑отчёт (вызов scripts/business_impact_report.py)
st.markdown("---")
st.subheader("Сформировать сводный бизнес‑отчёт (скрипт)")
colr1, colr2, colr3 = st.columns(3)
with colr1:
    tail_days = st.number_input("tail_days (средний спрос)", min_value=7, max_value=90, value=30, step=1)
with colr2:
    valid_days_rep = st.number_input("valid_days (наивный MAPE)", min_value=7, max_value=90, value=28, step=1)
with colr3:
    price_csv = st.text_input("prices.csv", value=str(Path("configs/prices.csv").resolve()))

run_rep = st.button("Сформировать отчёт (scripts/business_impact_report.py)")
if run_rep:
    import subprocess
    import sys
    cmd = [sys.executable, "scripts/business_impact_report.py",
           "--price_csv", price_csv,
           "--lead_time_days", str(int(lead_time_days)),
           "--service_level", str(float(service_level)),
           "--tail_days", str(int(tail_days)),
           "--valid_days", str(int(valid_days_rep)),
           "--out_csv", str(DW_DIR / "business_impact_report.csv")]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            st.error(f"Ошибка при формировании отчёта: {res.stderr or res.stdout}")
        else:
            st.success("Отчёт сформирован: data_dw/business_impact_report.csv")
            if (DW_DIR / "business_impact_report.csv").exists():
                dfrep = pd.read_csv(DW_DIR / "business_impact_report.csv")
                st.dataframe(dfrep.head(200), use_container_width=True)
                st.download_button("⬇️ Скачать отчёт (CSV)", data=dfrep.to_csv(index=False).encode("utf-8"), file_name="business_impact_report.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Не удалось запустить скрипт: {e}")

st.markdown("---")
st.subheader("Сводный отчёт по влиянию (если есть)")
impact_csv = DW_DIR / "business_impact_report.csv"
if impact_csv.exists():
    try:
        imp = pd.read_csv(impact_csv)
        # простые фильтры
        cf1, cf2 = st.columns(2)
        with cf1:
            sel_stores = st.multiselect("Магазины", options=sorted(imp["store_nbr"].dropna().astype(int).unique().tolist()), default=[])
        with cf2:
            sel_fams = st.multiselect("Семейства", options=sorted(imp["family"].dropna().astype(str).unique().tolist()), default=[])
        sub = imp.copy()
        if sel_stores:
            sub = sub[sub["store_nbr"].isin(sel_stores)]
        if sel_fams:
            sub = sub[sub["family"].astype(str).isin(sel_fams)]
        st.dataframe(sub, use_container_width=True)
        st.download_button("⬇️ Скачать impact CSV", data=sub.to_csv(index=False).encode("utf-8"), file_name="business_impact_report_filtered.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Не удалось прочитать business_impact_report.csv: {e}")
else:
    st.info("Файл data_dw/business_impact_report.csv не найден. Сгенерируйте через make impact и configs/prices.csv.")

# Быстрый массовый расчёт SS/ROP (без скрипта)
st.markdown("---")
st.subheader("Быстрый массовый расчёт SS/ROP (без скрипта)")

# Контролы
colm1, colm2, colm3, colm4 = st.columns(4)
with colm1:
    mass_tail_days = st.number_input("tail_days (средний спрос)", min_value=7, max_value=90, value=30, step=1, key="mass_tail")
with colm2:
    sigma_method = st.radio("Метод σ", ["По MAPE (метрики)", "По квантилям P50/P90 (если есть)"] , index=0)
with colm3:
    mass_max_pairs = st.number_input("Ограничить пары", min_value=10, max_value=2000, value=200, step=10)
with colm4:
    mass_filters = st.checkbox("Фильтры по парам", value=False)

stores_all = sorted(train["store_nbr"].dropna().astype(int).unique().tolist())
fams_all = sorted(train["family"].dropna().astype(str).unique().tolist())
sel_stores = []
sel_fams = []
if mass_filters:
    cfs1, cfs2 = st.columns(2)
    with cfs1:
        sel_stores = st.multiselect("Магазины", options=stores_all, default=[])
    with cfs2:
        sel_fams = st.multiselect("Семейства", options=fams_all, default=[])

def _build_all_features():
    # Готовим фичи для всех пар (нужно для метода по квантилям)
    try:
        paths = {k: RAW_DIR / f"{k}.csv" for k in ["transactions", "oil", "holidays_events", "stores"]}
        trans = pd.read_csv(paths["transactions"], parse_dates=["date"]) if paths["transactions"].exists() else None
        oil = pd.read_csv(paths["oil"], parse_dates=["date"]) if paths["oil"].exists() else None
        hol = pd.read_csv(paths["holidays_events"], parse_dates=["date"]) if paths["holidays_events"].exists() else None
        stores = pd.read_csv(paths["stores"]) if paths["stores"].exists() else None
        from make_features import make_features as _mf
        Xf, _ = _mf(train, hol, trans, oil, stores, dropna_target=False)
        return Xf
    except Exception:
        return None

def _sigma_from_quantiles(xrow: pd.Series, s: int, f: str) -> Tuple[Optional[float], Optional[float]]:
    base = f"{int(s)}__{str(f).replace(' ', '_')}"
    p50 = MODELS_DIR / f"{base}__q50.joblib"
    p90 = MODELS_DIR / f"{base}__q90.joblib"
    if (not p50.exists()) or (not p90.exists()):
        return None, None
    # список фич — сначала из features.json, иначе все числовые
    feat_cols = None
    fj = MODELS_DIR / f"{base}.features.json"
    if fj.exists():
        try:
            data = json.loads(fj.read_text(encoding="utf-8"))
            if isinstance(data, list):
                feat_cols = data
        except Exception:
            feat_cols = None
    if feat_cols is None:
        exclude = {"id", "sales", "date"}
        feat_cols = [c for c in xrow.index if (c not in exclude) and pd.api.types.is_numeric_dtype(type(xrow[c]))]
    try:
        mdl50 = joblib.load(p50)
        mdl90 = joblib.load(p90)
        X = np.array([[float(xrow.get(c, 0.0)) for c in feat_cols]], dtype=float)
        q50 = float(mdl50.predict(X)[0])
        q90 = float(mdl90.predict(X)[0])
        sigma = max((q90 - q50) / 1.2816, 0.0)
        return q50, sigma
    except Exception:
        return None, None

mass_btn = st.button("Посчитать SS/ROP для всех (быстро)")
if mass_btn:
    z = _z_from_service_level(float(service_level))
    # Предподсчёты
    Xall = _build_all_features() if sigma_method.startswith("По квантилям") else None
    rows = []
    seen = 0
    for (s, f), dfp in train.groupby(["store_nbr", "family"], sort=False):
        if sel_stores and int(s) not in set(sel_stores):
            continue
        if sel_fams and str(f) not in set(sel_fams):
            continue
        seen += 1
        if seen > int(mass_max_pairs):
            break
        dfp = dfp.sort_values("date")
        tailp = dfp.tail(int(mass_tail_days))
        daily_mean_p = float(tailp["sales"].mean()) if not tailp.empty else 0.0
        sigma_p = None
        daily_from_q = None
        method_used = "mape"
        if sigma_method.startswith("По квантилям") and Xall is not None:
            sub = Xall[(Xall["store_nbr"] == int(s)) & (Xall["family"].astype(str) == str(f))].sort_values("date")
            if not sub.empty:
                q50, sig = _sigma_from_quantiles(sub.iloc[-1], int(s), str(f))
                if (q50 is not None) and (sig is not None):
                    daily_from_q = q50
                    sigma_p = sig
                    method_used = "quantiles"
        if sigma_p is None:
            # Фоллбек на MAPE
            mape_pct_p = None
            if metrics is not None and not metrics.empty and "MAPE_%" in metrics.columns:
                sel = metrics[(metrics["store_nbr"] == s) & (metrics["family"].astype(str) == str(f))]
                if not sel.empty:
                    try:
                        mape_pct_p = float(sel.iloc[0]["MAPE_%"])
                    except Exception:
                        mape_pct_p = None
            if mape_pct_p is None:
                mape_pct_p = 20.0
            sigma_p = max((mape_pct_p / 100.0) * daily_mean_p, 0.0)
        mean_used = daily_from_q if (daily_from_q is not None) else daily_mean_p
        ss = z * sigma_p * (int(lead_time_days) ** 0.5)
        rop_val = mean_used * int(lead_time_days) + ss
        rows.append({
            "store_nbr": int(s),
            "family": str(f),
            "daily_mean": mean_used,
            "sigma": sigma_p,
            "SS": ss,
            "ROP": rop_val,
            "method": method_used,
        })
    if rows:
        out = pd.DataFrame(rows)
        st.success(f"Рассчитано пар: {len(out)} (из {seen} просмотренных)")
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇️ Скачать SS/ROP (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="mass_ss_rop.csv", mime="text/csv")
    else:
        st.info("Нет данных для расчёта (проверьте фильтры или наличие квантильных моделей).")
