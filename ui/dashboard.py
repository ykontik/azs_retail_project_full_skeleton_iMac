
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # один уровень вверх от ui/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_features import make_features

st.set_page_config(page_title="AZS + Retail MVP", layout="wide")
# ---- безопасное определение API_URL без обращения к st.secrets ----
def _resolve_api_url() -> str:
    api = os.getenv("API_URL")
    if api:
        return api
    # читаем secrets.toml вручную, только если он существует
    candidate_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",  # корень проекта
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",      # рядом с ui/
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                import tomllib  # Python 3.11+
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                return data.get("API_URL", "http://127.0.0.1:8000")
            except Exception:
                pass
    return "http://127.0.0.1:8000"

API_URL = _resolve_api_url()
st.title("⛽ AZS + Retail — MVP Dashboard")
with st.expander("⚙️ Подключение к API", expanded=False):
    api_url_input = st.text_input("API URL", value=API_URL, help="Адрес FastAPI сервиса")
    if api_url_input:
        API_URL = api_url_input

st.subheader("📈 Метрики по SKU (MAE / MAPE)")
metrics_path = Path("data_dw/metrics_per_sku.csv")
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True)
    # кнопка скачать CSV
    st.download_button(
        "⬇️ Скачать metrics_per_sku.csv",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="metrics_per_sku.csv",
        mime="text/csv",
    )
    # показать summary_metrics.txt, если есть
    sum_path = Path("data_dw/summary_metrics.txt")
    if sum_path.exists():
        st.subheader("📄 Summary (MAE/MAPE)")
        st.code(sum_path.read_text(encoding="utf-8"))
    # Агрегаты по магазинам/семействам
    st.subheader("📊 Агрегаты метрик")
    colm, colf = st.columns(2)
    # агрегирование по номеру магазина
    with colm:
        st.caption("Средние по магазинам")
        try:
            agg_store = (metrics.groupby("store_nbr", dropna=True)[["MAE","MAPE_%"]]
                                  .mean()
                                  .round(2)
                                  .reset_index()
                                  .sort_values("MAE"))
            st.dataframe(agg_store, use_container_width=True)
            st.bar_chart(agg_store.set_index("store_nbr")["MAE"], height=160)
        except Exception as e:
            st.warning(f"Не удалось посчитать агрегаты по магазинам: {e}")
    # агрегирование по семействам
    with colf:
        st.caption("Средние по семействам")
        try:
            agg_family = (metrics.groupby("family", dropna=True)[["MAE","MAPE_%"]]
                                  .mean()
                                  .round(2)
                                  .reset_index()
                                  .sort_values("MAE"))
            st.dataframe(agg_family, use_container_width=True)
            st.bar_chart(agg_family.set_index("family")["MAE"], height=160)
        except Exception as e:
            st.warning(f"Не удалось посчитать агрегаты по семействам: {e}")
else:
    st.info("Файл metrics_per_sku.csv не найден. Запусти обучение, чтобы увидеть метрики.")

st.subheader("🧠 Доступные модели (из API)")
available_models_df = None
try:
    r = requests.get(f"{API_URL}/models", timeout=5)
    if r.ok:
        models_json = r.json()["models"]
        models_df = pd.DataFrame(models_json)
        if not models_df.empty:
            available_models_df = models_df.copy()
            store_opts = sorted(models_df["store_nbr"].dropna().astype(int).unique())
            store_sel = st.selectbox("store_nbr", store_opts, index=0, key="store_sel")
            fam_opts = sorted(models_df.loc[models_df["store_nbr"] == store_sel, "family"].dropna().unique().tolist())
            family_sel = st.selectbox("family", fam_opts, index=0, key="family_sel")
        else:
            store_sel = st.number_input("store_nbr", min_value=1, step=1, value=1)
            family_sel = st.text_input("family", value="AUTOMOTIVE")
        st.dataframe(models_df, use_container_width=True)
    else:
        st.warning("Не удалось получить список моделей из API.")
except Exception as e:
    st.warning(f"API недоступен: {e}")

st.subheader("⚡ Быстрый прогноз (через API)")
col1, col2 = st.columns(2)
with col1:
    store_nbr = st.number_input("store_nbr", min_value=1, step=1, value=1)
    family = st.text_input("family", value="AUTOMOTIVE")
with col2:
    st.caption("Вставь JSON с фичами. Отсутствующие будут заполнены 0.")
    default_features = {
        "year": 2017, "month": 8, "week": 33, "day": 15, "dayofweek": 2,
        "is_weekend": 0, "is_month_start": 0, "is_month_end": 0,
        "dow_sin": 0.0, "dow_cos": 0.0, "month_sin": 0.0, "month_cos": 0.0,
        "trend": 1600, "is_holiday": 0, "is_christmas": 0, "is_newyear": 0, "is_black_friday": 0,
        "transactions": 500.0, "oil_price": 50.0,
        # промо-признаки (если модель их ждёт)
        "onpromotion": 0.0,
        "onpromotion_lag_7": 0.0, "onpromotion_lag_14": 0.0, "onpromotion_lag_28": 0.0,
        "onpromotion_rollmean_7": 0.0, "onpromotion_rollstd_7": 0.0,
        "onpromotion_rollmean_30": 0.0, "onpromotion_rollstd_30": 0.0,
        "sales_lag_7": 5.0, "sales_lag_14": 4.0, "sales_lag_28": 6.0,
        "sales_rollmean_7": 5.0, "sales_rollstd_7": 1.2,
        "sales_rollmean_30": 5.3, "sales_rollstd_30": 1.5,
        "cluster": 13
    }
    # Держим буфер текста фич отдельно от ключа виджета, чтобы можно было программно обновлять
    if 'features_text_buf' not in st.session_state:
        st.session_state['features_text_buf'] = json.dumps(default_features, indent=2)
    features_text = st.text_area("features (JSON)", value=st.session_state['features_text_buf'], height=240)
    if features_text != st.session_state['features_text_buf']:
        st.session_state['features_text_buf'] = features_text

    if st.button("Автозаполнить фичи по последней дате", help="Считать data_raw/*, сделать make_features и подставить последнюю строку для выбранной пары"):
        try:
            # Загрузка данных
            paths = {k: Path("data_raw")/f"{k}.csv" for k in ["train","transactions","oil","holidays_events","stores"]}
            if not all(p.exists() for p in paths.values()):
                st.warning("Не найдены все файлы из data_raw. Нужны: train, transactions, oil, holidays_events, stores")
            else:
                train_df = pd.read_csv(paths["train"], parse_dates=["date"])
                trans_df = pd.read_csv(paths["transactions"], parse_dates=["date"])
                oil_df = pd.read_csv(paths["oil"], parse_dates=["date"])
                hol_df = pd.read_csv(paths["holidays_events"], parse_dates=["date"])
                stores_df = pd.read_csv(paths["stores"])
                Xf, _ = make_features(train_df, hol_df, trans_df, oil_df, stores_df, dropna_target=False)
                mask = (Xf["store_nbr"] == int(store_nbr)) & (Xf["family"] == family)
                sub = Xf.loc[mask].sort_values("date")
                if sub.empty:
                    st.warning("Нет данных для такой пары store/family.")
                else:
                    # Загружаем модель, чтобы знать список фич
                    model_path = Path("models") / f"{int(store_nbr)}__{str(family).replace(' ', '_')}.joblib"
                    if not model_path.exists():
                        st.warning("Модель не найдена, но подставлю все доступные фичи.")
                        feat_names = [c for c in sub.columns if c not in ("id","sales","date")]
                    else:
                        mdl = joblib.load(model_path)
                        feat_names = getattr(mdl, "feature_name_", None)
                        if feat_names is None and hasattr(mdl, "booster_"):
                            try:
                                feat_names = list(mdl.booster_.feature_name())
                            except Exception:
                                feat_names = None
                        if not feat_names:
                            feat_names = [c for c in sub.columns if c not in ("id","sales","date")]
                    # Готовим словарь фич для API: только числовые значения, строки → 0.0
                    last_row = sub.iloc[-1]
                    feats = {}
                    for name in feat_names:
                        val = last_row.get(name, 0.0)
                        try:
                            if isinstance(val, bool):
                                feats[name] = 1.0 if val else 0.0
                            elif isinstance(val, (int, float, np.floating, np.integer)):
                                feats[name] = float(val)
                            else:
                                # строковые/категориальные признаки не кодируем — оставляем 0.0
                                feats[name] = float(str(val)) if str(val).replace('.', '', 1).isdigit() else 0.0
                        except Exception:
                            feats[name] = 0.0
                    st.session_state['features_text_buf'] = json.dumps(feats, ensure_ascii=False, indent=2)
                    st.success("Фичи подставлены из последней доступной даты.")
                    # Перерисовать, чтобы отобразить новые фичи в text_area
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"Не удалось автозаполнить фичи: {e}")

if st.button("Спрогнозировать через API", type="primary"):
    try:
        feats = json.loads(st.session_state.get('features_text_buf', features_text))
        payload = {"store_nbr": int(store_nbr), "family": family, "features": feats}
        r = requests.post(f"{API_URL}/predict_demand", json=payload, timeout=10)
        if r.ok:
            out = r.json()
            st.success(f"Прогноз: {out['pred_qty']:.3f}")
            with st.expander("Использованные фичи"):
                st.code(json.dumps(out["used_features"], ensure_ascii=False, indent=2))
        else:
            st.error(f"Ошибка API: {r.status_code} — {r.text}")
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.markdown("---")
st.subheader("📉 Графики: прогноз vs факт (локальный backtest)")
st.caption("Нужны train/transactions/oil/holidays/stores в data_raw/ и обученная модель.")
colA, colB, colC, colD = st.columns(4)
with colA:
    if (available_models_df is not None) and (not available_models_df.empty):
        store_opts_bt = sorted(available_models_df["store_nbr"].dropna().astype(int).unique())
        store_bt = st.selectbox("store_nbr (bt)", store_opts_bt, index=0, key="store_bt_sel")
    else:
        store_bt = st.number_input("store_nbr (bt)", min_value=1, step=1, value=1, key="store_bt")
with colB:
    if (available_models_df is not None) and (not available_models_df.empty):
        fam_opts_bt = sorted(available_models_df.loc[available_models_df["store_nbr"] == int(store_bt), "family"].dropna().unique().tolist())
        if not fam_opts_bt:
            fam_opts_bt = ["AUTOMOTIVE"]
        family_bt = st.selectbox("family (bt)", fam_opts_bt, index=0, key="family_bt_sel")
    else:
        family_bt = st.text_input("family (bt)", value="AUTOMOTIVE", key="family_bt")
with colC:
    back_days = st.number_input("Дней в хвосте", min_value=14, max_value=180, value=60, step=7)
show_xgbps_bt = st.checkbox("Показывать XGB per-SKU (если есть)", value=True)

paths = {k: Path("data_raw")/f"{k}.csv" for k in ["train","transactions","oil","holidays_events","stores"]}
missing = [k for k,p in paths.items() if not p.exists()]
if missing:
    st.warning(f"Нет файлов: {', '.join(missing)}")
else:
    model_path = Path("models") / f"{int(store_bt)}__{str(family_bt).replace(' ', '_')}.joblib"
    xgb_ps_path = Path("models") / f"{int(store_bt)}__{str(family_bt).replace(' ', '_')}__xgb.joblib"
    use_catboost_fallback = False
    if not model_path.exists():
        # Фолбэк: глобальная CatBoost
        cb_path = Path("models") / "global_catboost.cbm"
        if cb_path.exists():
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(str(cb_path))
                use_catboost_fallback = True
                st.info(f"Per-SKU модель не найдена ({model_path.name}). Использую глобальную CatBoost.")
            except Exception as e:
                st.error(f"Модель не найдена: {model_path} и не удалось загрузить CatBoost: {e}")
                model = None
        else:
            st.error(f"Модель не найдена: {model_path}")
            model = None
    else:
        model = joblib.load(model_path)
    train_df = pd.read_csv(paths["train"], parse_dates=["date"])
    trans_df = pd.read_csv(paths["transactions"], parse_dates=["date"])
    oil_df = pd.read_csv(paths["oil"], parse_dates=["date"])
    hol_df = pd.read_csv(paths["holidays_events"], parse_dates=["date"])
    stores_df = pd.read_csv(paths["stores"])
    Xfull, yfull = make_features(train_df, hol_df, trans_df, oil_df, stores_df, dropna_target=True)
    for c in ["store_nbr","family","type","city","state","cluster","is_holiday"]:
        if c in Xfull.columns:
            Xfull[c] = Xfull[c].astype("category")
    mask_pair = (Xfull["store_nbr"] == int(store_bt)) & (Xfull["family"] == family_bt)
    df_pair = Xfull.loc[mask_pair].copy().sort_values("date")
    if df_pair.empty:
        st.error("Для этой пары нет данных.")
    else:
        feat_names = getattr(model, "feature_name_", None)
        if feat_names is None and hasattr(model, "booster_"):
            try: feat_names = list(model.booster_.feature_name())
            except: feat_names = None
            # Для CatBoost фолбэка пытаемся использовать список фич из metrics_global_catboost.json
            if use_catboost_fallback and (feat_names is None):
                cb_feat = None
                meta_path = Path("data_dw") / "metrics_global_catboost.json"
                if meta_path.exists():
                    try:
                        cb_feat = json.loads(meta_path.read_text(encoding="utf-8")).get("features")
                    except Exception:
                        cb_feat = None
                feat_names = cb_feat or [c for c in df_pair.columns if c not in ("id","sales","date") and not np.issubdtype(df_pair[c].dtype, np.datetime64)]

            if feat_names is None:
                st.error("Не удалось получить список фич модели.")
            else:
                for f in feat_names:
                    if f not in df_pair.columns: df_pair[f] = 0.0
                tail = df_pair.tail(int(back_days)).copy()
                X_tail = tail[feat_names]; y_tail = tail["sales"].values
                # Прогноз базовой (point) модели или CatBoost
                try:
                    y_pred = model.predict(X_tail)
                except Exception:
                    # CatBoost может требовать Pool, но predict по DataFrame тоже работает; fallback на numpy
                    y_pred = model.predict(X_tail.values)

                # Попробуем также построить линию XGBoost per-SKU, если есть модель
                y_pred_xgb = None
                if xgb_ps_path.exists():
                    try:
                        mdl_xgb_ps = joblib.load(xgb_ps_path)
                        feat_xps = getattr(mdl_xgb_ps, 'feature_names_in_', None)
                        cols_xps = list(feat_xps) if feat_xps is not None else feat_names
                        for f in cols_xps:
                            if f not in tail.columns:
                                tail[f] = 0.0
                        X_tail_xps = tail[cols_xps]
                        y_pred_xgb = mdl_xgb_ps.predict(X_tail_xps)
                    except Exception:
                        y_pred_xgb = None
                # Попытка загрузить квантильные модели P50 и P90 для построения коридора
                q50_path = Path("models") / f"{int(store_bt)}__{str(family_bt).replace(' ', '_')}__q50.joblib"
                q90_path = Path("models") / f"{int(store_bt)}__{str(family_bt).replace(' ', '_')}__q90.joblib"
                q50_pred = q90_pred = None
                if not use_catboost_fallback:  # квантили есть только у per-SKU моделей
                    try:
                        if q50_path.exists():
                            mdl_q50 = joblib.load(q50_path)
                            q50_pred = mdl_q50.predict(X_tail)
                        if q90_path.exists():
                            mdl_q90 = joblib.load(q90_path)
                            q90_pred = mdl_q90.predict(X_tail)
                    except Exception as e:
                        st.warning(f"Квантильные модели не удалось применить: {e}")
                import matplotlib.pyplot as plt
                # Вкладка с графиком по дням и вкладка с агрегированием по времени
                tabs = st.tabs(["По дням", "Агрегации (недели/месяцы)"])

                # Вкладка 1: По дням
                with tabs[0]:
                    fig1 = plt.figure(figsize=(12,4))
                    plt.plot(tail["date"], y_tail, label="Факт")
                    plt.plot(tail["date"], y_pred, label="Прогноз (LGBM/CatBoost)")
                    if y_pred_xgb is not None and show_xgbps_bt:
                        plt.plot(tail["date"], y_pred_xgb, label="XGBoost per-SKU")
                    # Если доступны квантильные прогнозы — рисуем коридор
                    if q50_pred is not None and q90_pred is not None:
                        plt.fill_between(tail["date"], q50_pred, q90_pred, color="orange", alpha=0.25, label="P50–P90")
                    title = "Продажи: факт vs прогноз"
                    if 'use_catboost_fallback' in locals() and use_catboost_fallback:
                        title += " — CatBoost fallback"
                        # Визуальная пометка на графике
                        plt.text(0.01, 0.98, "CatBoost fallback", transform=plt.gca().transAxes,
                                 fontsize=10, va='top', ha='left', color='white',
                                 bbox=dict(facecolor='#6c757d', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                    plt.title(title); plt.xlabel("Дата"); plt.ylabel("Продажи")
                    plt.legend(); plt.grid()
                    st.pyplot(fig1)
                mae = np.mean(np.abs(y_tail - y_pred))
                denom = np.where(y_tail == 0, 1, y_tail)
                mape = np.mean(np.abs((y_tail - y_pred) / denom)) * 100.0
                st.metric("MAE (tail)", f"{mae:.3f} шт.")
                st.metric("MAPE (tail, %)", f"{mape:.2f}%")
                if y_pred_xgb is not None and show_xgbps_bt:
                        mae_xgb = np.mean(np.abs(y_tail - y_pred_xgb))
                        mape_xgb = np.mean(np.abs((y_tail - y_pred_xgb) / denom)) * 100.0
                        st.metric("MAE XGBoost (tail)", f"{mae_xgb:.3f} шт.")
                        st.metric("MAPE XGBoost (tail, %)", f"{mape_xgb:.2f}%")
                # Метрики для квантильного коридора (по дням), если есть
                if q50_pred is not None and q90_pred is not None:
                    # доля точек факта внутри [P50, P90]
                    inside = np.mean((y_tail >= np.minimum(q50_pred, q90_pred)) & (y_tail <= np.maximum(q50_pred, q90_pred)))
                    avg_width = float(np.mean(np.abs(q90_pred - q50_pred)))
                    st.metric("Доля покрытия факта (P50–P90)", f"{inside*100:.1f}%")
                    st.metric("Средняя ширина коридора", f"{avg_width:.3f} шт.")

                # Вкладка 2: Агрегации по времени (недели/месяцы)
                with tabs[1]:
                    st.caption("Агрегирование по выбранному периоду (сумма по дням)")
                    period = st.radio("Период", ["Неделя", "Месяц"], horizontal=True)
                    agg_mode = st.radio("Агрегирование", ["Сумма", "Среднее"], horizontal=True)
                    freq = "W" if period == "Неделя" else "M"
                    # Готовим DataFrame для агрегации
                    df_plot = pd.DataFrame({
                        "date": tail["date"].values,
                        "y_true": y_tail,
                        "y_pred": y_pred,
                    })
                    if q50_pred is not None and q90_pred is not None:
                        df_plot["p50"] = q50_pred
                        df_plot["p90"] = q90_pred
                    # Агрегирование суммой по периоду
                    grouped = df_plot.set_index("date").groupby(pd.Grouper(freq=freq))
                    if agg_mode == "Сумма":
                        g = grouped.sum().reset_index()
                    else:
                        g = grouped.mean().reset_index()
                    # График агрегатов
                    fig2 = plt.figure(figsize=(12,4))
                    plt.plot(g["date"], g["y_true"], label="Факт (agg)")
                    plt.plot(g["date"], g["y_pred"], label="Прогноз (agg)")
                    if y_pred_xgb is not None and show_xgbps_bt:
                        # агрегируем XGB по выбранному периоду
                        g_x = (pd.DataFrame({"date": tail["date"].values, "y": y_pred_xgb})
                                 .set_index("date").groupby(pd.Grouper(freq=freq)).sum().reset_index())
                        plt.plot(g_x["date"], g_x["y"], label="XGBoost per-SKU (agg)")
                    if "p50" in g.columns and "p90" in g.columns:
                        plt.fill_between(g["date"], g["p50"], g["p90"], color="orange", alpha=0.25, label="P50–P90 (agg)")
                    ylabel = "Продажи (сумма)" if agg_mode == "Сумма" else "Продажи (среднее)"
                    agg_title = f"Агрегации по времени: {period.lower()}"
                    if 'use_catboost_fallback' in locals() and use_catboost_fallback:
                        agg_title += " — CatBoost fallback"
                        plt.text(0.01, 0.98, "CatBoost fallback", transform=plt.gca().transAxes,
                                 fontsize=10, va='top', ha='left', color='white',
                                 bbox=dict(facecolor='#6c757d', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                    plt.title(agg_title); plt.xlabel("Период"); plt.ylabel(ylabel)
                    plt.legend(); plt.grid()
                    st.pyplot(fig2)

                    # Метрики на агрегированном ряду
                    y_true_agg = g["y_true"].values
                    y_pred_agg = g["y_pred"].values
                    mae_agg = np.mean(np.abs(y_true_agg - y_pred_agg))
                    denom_agg = np.where(y_true_agg == 0, 1, y_true_agg)
                    mape_agg = np.mean(np.abs((y_true_agg - y_pred_agg) / denom_agg)) * 100.0
                    st.metric("MAE (agg)", f"{mae_agg:.3f} шт.")
                    st.metric("MAPE (agg, %)", f"{mape_agg:.2f}%")

st.markdown("---")
st.subheader("🔍 Важность признаков (feature importance)")

colF1, colF2 = st.columns(2)
with colF1:
    store_imp = st.number_input("store_nbr (FI)", min_value=1, step=1, value=int(store_bt))
with colF2:
    family_imp = st.text_input("family (FI)", value=family_bt)

imp_path = Path("models") / f"{int(store_imp)}__{str(family_imp).replace(' ', '_')}.joblib"
if not imp_path.exists():
    st.info(f"Нет модели для FI: {imp_path.name}")
else:
    try:
        mdl = joblib.load(imp_path)
        # Получаем имена фич и их важности (gain)
        if hasattr(mdl, "booster_"):
            names = list(mdl.booster_.feature_name())
            gains = mdl.booster_.feature_importance(importance_type="gain")
        else:
            # fallback: если нет booster_ (редко)
            names = getattr(mdl, "feature_name_", [])
            gains = getattr(mdl, "feature_importances_", [])
        import pandas as pd
        df_fi = pd.DataFrame({"feature": names, "gain": gains}).sort_values("gain", ascending=False).head(15)
        st.dataframe(df_fi, use_container_width=True)
        st.bar_chart(df_fi.set_index("feature"))
    except Exception as e:
        st.error(f"Не удалось рассчитать важности: {e}")

st.markdown("---")

# ----------------------- Бизнес: ROP / Safety Stock -----------------------
st.subheader("📦 Расчёт ROP / Safety Stock")
colp1, colp2 = st.columns(2)
with colp1:
    lead_time_days = st.number_input("Lead time (дней)", min_value=1, value=2, step=1, key="lead_time_days")
    service_level = st.slider("Уровень сервиса", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="service_level")
    calc = st.button("Рассчитать ROP/SS", key="calc_rop")
with colp2:
    st.caption("Используются квантильные модели P50/P90, если обучены. Иначе — эвристика σ ≈ 0.25·mean.")

if 'features_text_buf' in st.session_state and st.session_state.get('features_text_buf') and st.session_state.get('calc_rop'):
    try:
        feats_body = json.loads(st.session_state['features_text_buf'])
        body = {
            "store_nbr": int(store_nbr),
            "family": str(family),
            "features": feats_body,
            "lead_time_days": int(st.session_state.get('lead_time_days', 2)),
            "service_level": float(st.session_state.get('service_level', 0.95)),
        }
        r = requests.post(f"{API_URL}/reorder_point", json=body, timeout=10)
        if r.ok:
            data = r.json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Daily mean (API), шт.", f"{data['daily_mean']:.2f}")
            m2.metric("Sigma (API), шт.", f"{data['sigma_daily']:.2f}")
            m3.metric("Safety Stock (API), шт.", f"{data['safety_stock']:.2f}")
            m4.metric("ROP (API), шт.", f"{data['reorder_point']:.2f}")
            st.caption(f"quantiles_used={data.get('quantiles_used', False)} | z={data.get('service_level_z')}")
        else:
            st.error(f"Ошибка API: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Ошибка расчёта: {e}")

# ----------------------- План запасов (офлайн скрипт) -----------------------
st.subheader("📦 План запасов (бейслайн из последних продаж)")
stock_csv = Path("data_dw/stock_plan.csv")
if stock_csv.exists():
    try:
        df_stock = pd.read_csv(stock_csv)
        st.dataframe(df_stock, use_container_width=True)
        st.download_button(
            "⬇️ Скачать stock_plan.csv",
            data=df_stock.to_csv(index=False).encode("utf-8"),
            file_name="stock_plan.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Не удалось прочитать stock_plan.csv: {e}")
else:
    st.info("Файл stock_plan.csv не найден. Запусти: make stock")

# ----------------------- SHAP Предпросмотр -----------------------
st.subheader("🔍 SHAP — важность признаков (предпросмотр)")
try:
    store_cur = int(store_nbr)
    family_cur = str(family)
except Exception:
    store_cur, family_cur = 1, "AUTOMOTIVE"
shap_png = Path("data_dw") / f"shap_summary_{store_cur}__{family_cur}.png"
shap_csv = Path("data_dw") / f"shap_top_{store_cur}__{family_cur}.csv"
cols_sh = st.columns(2)
with cols_sh[0]:
    if shap_png.exists():
        st.image(str(shap_png), caption=shap_png.name, use_column_width=True)
    else:
        st.info("SHAP картинка не найдена. Сгенерируй через scripts/shap_report.py")
with cols_sh[1]:
    if shap_csv.exists():
        df_shap = pd.read_csv(shap_csv)
        st.dataframe(df_shap.head(25), use_container_width=True)
        st.download_button("⬇️ Скачать shap_top.csv", data=df_shap.to_csv(index=False).encode('utf-8'), file_name=shap_csv.name, mime='text/csv')
    else:
        st.info("Таблица SHAP не найдена.")

st.subheader("📂 Просмотр data_raw (первые строки)")
raw_dir = Path("data_raw")
if not raw_dir.exists():
    st.info("Папка data_raw не найдена. Создай и положи CSV: train.csv, transactions.csv, oil.csv, holidays_events.csv, stores.csv")
else:
    cols = st.columns(5)
    files = ["train.csv", "transactions.csv", "oil.csv", "holidays_events.csv", "stores.csv"]
    for i, fname in enumerate(files):
        p = raw_dir / fname
        with cols[i]:
            st.write(f"**{fname}**")
            if p.exists():
                try:
                    df = pd.read_csv(p, nrows=5)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка чтения: {e}")
            else:
                st.warning("Нет файла")
