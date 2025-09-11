
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv  # загрузка .env
from joblib import dump
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm  # прогрессбар в обучении

from make_features import make_features

# Загружаем переменные окружения из .env, если он есть
load_dotenv()

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """MAE, MAPE c защитой от пустых выборок."""
    if len(y_true) == 0:
        return float("nan"), float("nan")
    return float(mean_absolute_error(y_true, y_pred)), float(mape(y_true, y_pred))

def parse_args():
    p = argparse.ArgumentParser(description="Train per-SKU LightGBM models")
    p.add_argument("--config", default=None, help="Путь к YAML-конфигу с параметрами тренировки")
    # Пути делаем необязательными: их можно задать в configs/train.yaml или через RAW_DIR
    p.add_argument("--train", default=None)
    p.add_argument("--transactions", default=None)
    p.add_argument("--oil", default=None)
    p.add_argument("--holidays", default=None)
    p.add_argument("--stores", default=None)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--warehouse_dir", default="data_dw")
    p.add_argument("--top_n_sku", type=int, default=20)
    p.add_argument("--top_recent_days", type=int, default=90, help="Отбирать топ-N по сумме продаж за последние N дней")
    p.add_argument("--valid_days", type=int, default=28)
    p.add_argument("--cv_folds", type=int, default=0, help="Количество временных фолдов (0 = без CV)")
    p.add_argument("--cv_step_days", type=int, default=14, help="Шаг между фолдами по времени")
    p.add_argument("--quantiles", nargs='*', type=float, default=[], help="Список квантилей для обучения (например 0.5 0.9)")
    p.add_argument("--random_state", type=int, default=42)
    # Переключатель цели LightGBM: l1/tweedie/poisson
    p.add_argument("--lgb_objective", type=str, default="l1", choices=["l1", "tweedie", "poisson"],
                   help="Целевая функция LGBM: l1 (MAE), tweedie, poisson")
    p.add_argument("--tweedie_variance_power", type=float, default=1.2,
                   help="Параметр Tweedie variance power (обычно 1.1–1.4)")
    return p.parse_args()

def merge_config_with_args(args: argparse.Namespace) -> argparse.Namespace:
    """Поддержка YAML‑конфига: значения из файла переопределяют дефолты,
    но параметры, явно переданные через CLI, остаются главными.
    """
    if not args.config:
        # Попробуем взять конфиг по умолчанию
        default_cfg = Path("configs/train.yaml")
        if not default_cfg.exists():
            return args
        cfg_path = default_cfg
    else:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            return args
    try:
        import yaml
        data = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
        # Пробуем присвоить только те поля, что не заданы явно через CLI
        for key, val in data.items():
            if not hasattr(args, key):
                continue
            cur = getattr(args, key)
            # Определим, меняли ли параметр через CLI: грубо — сравним с значением по умолчанию ArgumentParser не хранит
            # Поэтому применяем простую логику: если cur пустое по типу (например []/None/0) и в конфиге есть — подставим
            if (cur in (None, [], "") or (isinstance(cur, (int, float)) and cur == 0)) and val is not None:
                setattr(args, key, val)
    except Exception:
        pass
    return args

def fill_paths_from_env(args: argparse.Namespace) -> argparse.Namespace:
    """Заполняет отсутствующие пути к данным из переменной окружения `RAW_DIR` (по умолчанию `data_raw`)."""
    raw_dir = os.getenv("RAW_DIR", "data_raw")

    def _need(x: Optional[str]) -> bool:
        return (x is None) or (str(x).strip() == "")

    if _need(args.train):
        args.train = str(Path(raw_dir) / "train.csv")
    if _need(args.transactions):
        args.transactions = str(Path(raw_dir) / "transactions.csv")
    if _need(args.oil):
        args.oil = str(Path(raw_dir) / "oil.csv")
    if _need(args.holidays):
        args.holidays = str(Path(raw_dir) / "holidays_events.csv")
    if _need(args.stores):
        args.stores = str(Path(raw_dir) / "stores.csv")

    return args

def pick_top_sku(df: pd.DataFrame, top_n: int, recent_days: int) -> pd.DataFrame:
    df = df.copy()
    if recent_days and recent_days > 0 and "date" in df.columns:
        max_date = pd.to_datetime(df["date"]).max()
        min_date = max_date - pd.Timedelta(days=recent_days-1)
        df = df[pd.to_datetime(df["date"]) >= min_date]
    return (
        df.groupby(["store_nbr", "family"])['sales']
          .sum()
          .reset_index()
          .sort_values('sales', ascending=False)
          .head(top_n)
    )

def time_folds(dates: pd.Series, valid_days: int, cv_folds: int, cv_step_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Возвращает список (start_valid, end_valid) для роллинг-валидации."""
    if cv_folds <= 0:
        return []
    dmax = dates.max()
    folds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(cv_folds):
        end_valid = dmax - pd.Timedelta(days=(cv_folds-1-i)*cv_step_days)
        start_valid = end_valid - pd.Timedelta(days=valid_days-1)
        folds.append((start_valid, end_valid))
    return folds

def prepare_categoricals(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X

def main():
    args = parse_args()
    args = merge_config_with_args(args)
    args = fill_paths_from_env(args)

    # MLflow отключён: вычищено по запросу

    # Проверка, что все входные файлы существуют.
    missing = [
        ("train", args.train),
        ("transactions", args.transactions),
        ("oil", args.oil),
        ("holidays", args.holidays),
        ("stores", args.stores),
    ]
    missing = [k for k, p in missing if (p is None) or (not Path(p).exists())]
    if missing:
        raise SystemExit(
            "Отсутствуют входные файлы: " + ", ".join(missing) + \
            ". Задайте пути через CLI флаги, configs/train.yaml или переменную окружения RAW_DIR."
        )
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.warehouse_dir).mkdir(parents=True, exist_ok=True)

    # данные
    train = pd.read_csv(args.train, parse_dates=["date"])
    transactions = pd.read_csv(args.transactions, parse_dates=["date"])
    oil = pd.read_csv(args.oil, parse_dates=["date"])
    holidays = pd.read_csv(args.holidays, parse_dates=["date"])
    stores = pd.read_csv(args.stores)

    # фичи
    Xfull, yfull = make_features(train, holidays, transactions, oil, stores, dropna_target=True)

    # top-N sku
    top_pairs = set(map(tuple, pick_top_sku(train, args.top_n_sku, args.top_recent_days)[["store_nbr","family"]].values.tolist()))

    cat_cols = [c for c in ["store_nbr","family","type","city","state","cluster","is_holiday"] if c in Xfull.columns]
    Xfull = prepare_categoricals(Xfull, cat_cols)
    # уберем datetime-колонки из признаков
    dt_cols = Xfull.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    bool_cols = Xfull.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xfull[bool_cols] = Xfull[bool_cols].astype("int8")  # опционально

    # список признаков без target/id/date и без datetime
    exclude_cols = {"id", "sales", "date", *dt_cols}
    feat_cols = [c for c in Xfull.columns if c not in exclude_cols]

    metrics = []
    total_trained = 0

    # Формируем список пар для обучения и прогрессбар
    present_pairs_df = Xfull[["store_nbr", "family"]].drop_duplicates()
    present_pairs = set(map(tuple, present_pairs_df.values.tolist()))
    train_pairs = [(int(s), str(f)) for (s, f) in top_pairs if (s, f) in present_pairs]

    # прогрессбар показывает тип цели (l1/tweedie/poisson), чтобы альтернативные запуски были наглядны
    total = len(train_pairs)
    t0 = time.perf_counter()
    bar = tqdm(train_pairs, desc=f"Training per-SKU ({args.lgb_objective})", unit="pair", total=total, smoothing=0.3, mininterval=0.5)
    for idx, (store, fam) in enumerate(bar, start=1):
        df_grp = (
            Xfull[(Xfull["store_nbr"] == store) & (Xfull["family"] == fam)]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if df_grp.empty:
            continue
        y, X = df_grp["sales"].values, df_grp[feat_cols]

        max_date = df_grp["date"].max()
        min_valid_date = max_date - pd.Timedelta(days=args.valid_days-1)
        mask_val = df_grp["date"] >= min_valid_date
        if mask_val.sum() < 7: continue

        X_train, y_train = X[~mask_val], y[~mask_val]
        X_val, y_val = X[mask_val], y[mask_val]

        # Параметры цели для LightGBM
        lgb_params = dict(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state,
            n_jobs=-1,
            verbosity=-1,  # отключаем логи LightGBM в консоли
        )
        if args.lgb_objective == "l1":
            lgb_params.update(objective="l1")
        elif args.lgb_objective == "tweedie":
            lgb_params.update(objective="tweedie", tweedie_variance_power=args.tweedie_variance_power)
        else:  # poisson
            lgb_params.update(objective="poisson")

        model = lgb.LGBMRegressor(**lgb_params)
        # Доп. роллинг-валидация по времени (опционально)
        cv_maes: List[float] = []
        cv_mapes: List[float] = []
        folds = time_folds(df_grp["date"], args.valid_days, args.cv_folds, args.cv_step_days)

        # Основное обучение по последнему holdout-окну
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            categorical_feature=[c for c in cat_cols if c in feat_cols],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0),
            ],
        )

        pred = model.predict(X_val)
        mae, mmape = mean_absolute_error(y_val, pred), mape(y_val, pred)

        # MLflow logging removed

        # Базовые (baseline) прогнозы для сравнения на валидации
        # 1) Naive lag-7 (значение 7 дней назад)
        y_series = df_grp["sales"].reset_index(drop=True)
        naive_lag7 = y_series.shift(7).values[mask_val.values]
        mask_lag = ~np.isnan(naive_lag7)
        lag7_mae, lag7_mape = _safe_metrics(y_val[mask_lag], naive_lag7[mask_lag])

        # 2) Скользящее среднее 7 (по предшествующим дням)
        naive_ma7_full = y_series.shift(1).rolling(7, min_periods=1).mean().values
        naive_ma7 = naive_ma7_full[mask_val.values]
        ma7_mae, ma7_mape = _safe_metrics(y_val, naive_ma7)

        # CV метрики (если запрошено)
        if folds:
            for (start_v, end_v) in folds:
                m = (df_grp["date"] >= start_v) & (df_grp["date"] <= end_v)
                if m.sum() < max(7, int(0.05*len(df_grp))):
                    continue
                X_tr, y_tr = X[~m], y[~m]
                X_va, y_va = X[m], y[m]
                # Модель CV с той же целью
                lgb_cv_params = dict(
                    n_estimators=model.best_iteration_ or 1000,
                    learning_rate=0.05,
                    num_leaves=64,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=args.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
                if args.lgb_objective == "l1":
                    lgb_cv_params.update(objective="l1")
                elif args.lgb_objective == "tweedie":
                    lgb_cv_params.update(objective="tweedie", tweedie_variance_power=args.tweedie_variance_power)
                else:
                    lgb_cv_params.update(objective="poisson")
                mdl_cv = lgb.LGBMRegressor(**lgb_cv_params)
                mdl_cv.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="l1",
                    categorical_feature=[c for c in cat_cols if c in feat_cols],
                    callbacks=[lgb.log_evaluation(period=0)],
                )
                p = mdl_cv.predict(X_va)
                cv_maes.append(mean_absolute_error(y_va, p))
                cv_mapes.append(mape(y_va, p))

        metrics.append({
            "store_nbr": store,
            "family": fam,
            "MAE": float(mae),
            "MAPE_%": float(mmape),
            "CV_MAE": float(np.mean(cv_maes)) if cv_maes else None,
            "CV_MAPE_%": float(np.mean(cv_mapes)) if cv_mapes else None,
            # baseline сравнения
            "NAIVE_LAG7_MAE": lag7_mae,
            "NAIVE_LAG7_MAPE_%": lag7_mape,
            "NAIVE_MA7_MAE": ma7_mae,
            "NAIVE_MA7_MAPE_%": ma7_mape,
            "MAE_GAIN_vs_LAG7": (lag7_mae - mae) if (lag7_mae == lag7_mae) else None,
            "MAE_GAIN_vs_MA7": (ma7_mae - mae) if (ma7_mae == ma7_mae) else None,
        })
        model_stem = f"{store}__{str(fam).replace(' ', '_')}"
        model_name = f"{model_stem}.joblib"
        dump(model, os.path.join(args.models_dir, model_name))
        # Дополнительно сохраняем с суффиксом для альтернативных целей (tweedie/poisson)
        if args.lgb_objective in ("tweedie", "poisson"):
            alt_name = f"{model_stem}__{args.lgb_objective}.joblib"
            try:
                dump(model, os.path.join(args.models_dir, alt_name))
            except Exception:
                pass
        # сохраняем список фич для модели
        try:
            feat_names = model.feature_name_ if hasattr(model, 'feature_name_') else list(model.booster_.feature_name())
        except Exception:
            feat_names = feat_cols
        try:
            with open(os.path.join(args.models_dir, f"{Path(model_name).stem}.features.json"), "w", encoding="utf-8") as f:
                json.dump(feat_names, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        total_trained += 1

        # Обучим дополнительные квантильные модели (например, P50/P90) для управления запасами
        # Используем ту же разбивку (train/val), objective='quantile' с alpha
        if args.quantiles:
            quants = list(args.quantiles)
            iter_quants = quants if len(quants) <= 1 else __import__('tqdm').tqdm(quants, desc=f"Quantiles {store}/{fam}", unit="q", leave=False)
            for q in iter_quants:
                try:
                    q = float(q)
                except Exception:
                    continue
                # Пропускаем некорректные квантили
                if not (0.0 < q < 1.0):
                    continue
                model_q = lgb.LGBMRegressor(
                    objective='quantile', alpha=q,
                    n_estimators=model.best_iteration_ or 1000,
                    learning_rate=0.05, num_leaves=64,
                    subsample=0.9, colsample_bytree=0.9, random_state=args.random_state, n_jobs=-1,
                    verbosity=-1,
                )
                model_q.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l1",
                    categorical_feature=[c for c in cat_cols if c in feat_cols],
                    callbacks=[lgb.log_evaluation(period=0)],
                )
                qname = int(round(q * 100))
                q_model_name = f"{store}__{str(fam).replace(' ', '_')}__q{qname}.joblib"
                dump(model_q, os.path.join(args.models_dir, q_model_name))

        # Обновляем postfix прогресса: скорость и ETA
        elapsed = max(time.perf_counter() - t0, 1e-6)
        speed = idx / elapsed
        remaining = max(total - idx, 0)
        eta = remaining / speed if speed > 0 else float('inf')
        bar.set_postfix_str(f"{speed:.2f} pair/s, ETA {eta:.1f}s")

    if metrics:
        dfm = pd.DataFrame(metrics).sort_values("MAE")
        dfm.to_csv(os.path.join(args.warehouse_dir, "metrics_per_sku.csv"), index=False)
        with open(os.path.join(args.warehouse_dir,"summary_metrics.txt"),"w",encoding="utf-8") as f:
            f.write(f"Моделей обучено: {total_trained}\n")
            f.write(f"Средний MAE: {dfm['MAE'].mean():.3f}\n")
            f.write(f"Средний MAPE: {dfm['MAPE_%'].mean():.2f}%\n")
            if 'CV_MAE' in dfm.columns and dfm['CV_MAE'].notna().any():
                f.write(f"Средний CV_MAE: {dfm['CV_MAE'].dropna().mean():.3f}\n")
            if 'CV_MAPE_%' in dfm.columns and dfm['CV_MAPE_%'].notna().any():
                f.write(f"Средний CV_MAPE: {dfm['CV_MAPE_%'].dropna().mean():.2f}%\n")
        print("OK, обучено:", total_trained)
    else:
        print("Метрик нет — слишком короткие ряды?")


if __name__ == "__main__":
    main()
