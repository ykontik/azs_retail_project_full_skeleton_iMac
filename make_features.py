
from typing import List, Optional

import numpy as np
import pandas as pd


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

def _add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    # Для совместимости с тестами ожидается колонка quarter
    try:
        df["quarter"] = df[date_col].dt.quarter
    except Exception:
        df["quarter"] = 0
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df = df.sort_values(date_col)
    df["trend"] = (df[date_col] - df[date_col].min()).dt.days
    return df

def _merge_holidays(base: pd.DataFrame, holidays_events: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Добавляем признаки праздников с учётом локали (national/regional/local).

    Правила:
    - National: действует для всех магазинов в указанную дату.
    - Regional: сопоставляем по state (колонка stores.csv) с holidays.locale_name.
    - Local: сопоставляем по city с holidays.locale_name.
    """
    required = {"date", "locale", "locale_name"}
    if holidays_events is None or holidays_events.empty or not required.intersection(set(holidays_events.columns)):
        base["is_holiday"] = 0
        base["is_holiday_national"] = 0
        base["is_holiday_regional"] = 0
        base["is_holiday_local"] = 0
        return base

    df = base.copy()
    hol = holidays_events.copy()
    hol = _ensure_datetime(hol, "date")
    # Исключаем перенесённые
    if "transferred" in hol.columns:
        hol = hol[hol["transferred"] == False]

    # Универсальные флаги по видам локали
    nat = hol[hol.get("locale", pd.Series(dtype=str)).str.lower() == "national"][ ["date"] ].drop_duplicates().copy()
    nat["is_holiday_national"] = 1

    reg = hol[hol.get("locale", pd.Series(dtype=str)).str.lower() == "regional"].copy()
    loc = hol[hol.get("locale", pd.Series(dtype=str)).str.lower() == "local"].copy()

    # Сливаем national без условий
    df = df.merge(nat, on="date", how="left")

    # Для regional нужен столбец state
    if "state" in df.columns and "locale_name" in reg.columns:
        reg_simple = reg[["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "state"})
        reg_simple["is_holiday_regional"] = 1
        df = df.merge(reg_simple, on=["date", "state"], how="left")
    else:
        df["is_holiday_regional"] = np.nan

    # Для local нужен столбец city
    if "city" in df.columns and "locale_name" in loc.columns:
        loc_simple = loc[["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "city"})
        loc_simple["is_holiday_local"] = 1
        df = df.merge(loc_simple, on=["date", "city"], how="left")
    else:
        df["is_holiday_local"] = np.nan

    # Финальный бинарный флаг праздника
    for c in ["is_holiday_national", "is_holiday_regional", "is_holiday_local"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)
    df["is_holiday"] = (df["is_holiday_national"] | df["is_holiday_regional"] | df["is_holiday_local"]).astype(int)
    return df

def _merge_transactions(base: pd.DataFrame, transactions: Optional[pd.DataFrame]) -> pd.DataFrame:
    if transactions is None or transactions.empty:
        base["transactions"] = np.nan
        return base
    tr = transactions.copy()
    tr = _ensure_datetime(tr, "date")
    if "store_nbr" in tr.columns:
        out = base.merge(tr[["date", "store_nbr", "transactions"]], on=["date", "store_nbr"], how="left")
    else:
        out = base.merge(tr[["date", "transactions"]], on="date", how="left")
    return out

def _merge_oil(base: pd.DataFrame, oil: Optional[pd.DataFrame], price_col: str = "dcoilwtico") -> pd.DataFrame:
    if oil is None or oil.empty:
        base["oil_price"] = np.nan
        return base
    oi = oil.copy()
    oi = _ensure_datetime(oi, "date")
    oi = oi.rename(columns={price_col: "oil_price"})
    return base.merge(oi[["date", "oil_price"]], on="date", how="left")

def _add_txn_oil_lags_rolls(
    df: pd.DataFrame,
    group_cols: List[str],
    txn_col: str = "transactions",
    oil_col: str = "oil_price",
    lags: List[int] = [7],
    roll_windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Добавляет лаги/скользящие по транзакциям и нефти.

    Транзакции обычно заданы на уровне магазина (store_nbr), нефть — глобальная. Мы считаем
    по тем же группам, что и продажи, чтобы не ломать форму данных; значения просто копируются
    для каждой family внутри магазина.
    """
    out = df.copy()
    out = out.sort_values(group_cols + ["date"])  # стабильный порядок
    g = out.groupby(group_cols, sort=False)
    if txn_col in out.columns:
        for l in lags:
            out[f"{txn_col}_lag_{l}"] = g[txn_col].shift(l)
        for w in roll_windows:
            out[f"{txn_col}_rollmean_{w}"] = g[txn_col].shift(1).rolling(w, min_periods=1).mean()
            out[f"{txn_col}_rollstd_{w}"] = g[txn_col].shift(1).rolling(w, min_periods=1).std()
    if oil_col in out.columns:
        for l in lags:
            out[f"{oil_col}_lag_{l}"] = g[oil_col].shift(l)
        for w in roll_windows:
            out[f"{oil_col}_rollmean_{w}"] = g[oil_col].shift(1).rolling(w, min_periods=1).mean()
            out[f"{oil_col}_rollstd_{w}"] = g[oil_col].shift(1).rolling(w, min_periods=1).std()
    return out

def _add_onpromotion_features(
    df: pd.DataFrame,
    group_cols: List[str],
    promo_col: str = "onpromotion",
    lags: List[int] = [1, 7, 14, 28],
    roll_windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Добавляет признаки по промо-акциям (onpromotion): лаги и скользящие статистики.

    Ожидается, что столбец `onpromotion` есть в исходном наборе (train/test Favorita).
    Если столбца нет — создаём его с нулями, чтобы не ломать схему фичей.
    """
    out = df.copy()
    if promo_col not in out.columns:
        out[promo_col] = 0.0
    # приведение к числу и заполнение пропусков
    out[promo_col] = out[promo_col].fillna(0.0).astype(float)
    out = out.sort_values(group_cols + ["date"])  # стабильный порядок
    g = out.groupby(group_cols, sort=False)
    for l in lags:
        out[f"{promo_col}_lag_{l}"] = g[promo_col].shift(l)
    for w in roll_windows:
        out[f"{promo_col}_rollmean_{w}"] = g[promo_col].shift(1).rolling(w, min_periods=1).mean()
        out[f"{promo_col}_rollstd_{w}"] = g[promo_col].shift(1).rolling(w, min_periods=1).std()
    return out

def _add_holiday_distance_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Добавляет признаки расстояния до/от праздников и флаги соседних дней.

    Использует сводный флаг df['is_holiday'] (0/1). Считается по датам (без учета магазина/товара).
    """
    if "is_holiday" not in df.columns:
        return df
    out = df.copy()
    # Агрегация по датам: есть ли праздник в этот день в целом
    cal = out[[date_col, "is_holiday"]].drop_duplicates().sort_values(date_col)
    cal = cal.groupby(date_col, as_index=True)["is_holiday"].max().reset_index()
    dates = cal[date_col].values
    is_h = cal["is_holiday"].astype(bool).values
    # предыдущий праздник
    prev_idx = -1
    prev_days = []
    for i in range(len(dates)):
        if is_h[i]:
            prev_idx = i
            prev_days.append(0)
        else:
            if prev_idx == -1:
                prev_days.append(np.nan)
            else:
                prev_days.append((dates[i] - dates[prev_idx]).astype('timedelta64[D]').astype(float))
    # следующий праздник
    next_idx = -1
    next_days = [None] * len(dates)
    for i in range(len(dates) - 1, -1, -1):
        if is_h[i]:
            next_idx = i
            next_days[i] = 0.0
        else:
            if next_idx == -1:
                next_days[i] = np.nan
            else:
                next_days[i] = (dates[next_idx] - dates[i]).astype('timedelta64[D]').astype(float)
    cal["days_from_prev_holiday"] = prev_days
    cal["days_to_next_holiday"] = next_days
    cal["is_day_before_holiday"] = (cal["days_to_next_holiday"] == 1).astype(int)
    cal["is_day_after_holiday"] = (cal["days_from_prev_holiday"] == 1).astype(int)
    out = out.merge(
        cal[[date_col, "days_from_prev_holiday", "days_to_next_holiday", "is_day_before_holiday", "is_day_after_holiday"]],
        on=date_col,
        how="left",
    )
    return out

def _merge_stores_meta(base: pd.DataFrame, stores: Optional[pd.DataFrame]) -> pd.DataFrame:
    if stores is None or stores.empty:
        return base
    st = stores.copy()
    cols = [c for c in ["store_nbr", "city", "state", "type", "cluster"] if c in st.columns]
    return base.merge(st[cols], on="store_nbr", how="left")

def _add_special_days(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    mmdd = df[date_col].dt.strftime("%m-%d")
    df["is_christmas"] = (mmdd == "12-25").astype(int)
    df["is_newyear"] = (mmdd == "01-01").astype(int)
    df["is_black_friday"] = ((df[date_col].dt.month == 11) & (df[date_col].dt.dayofweek == 4) & (df[date_col].dt.day >= 23) & (df[date_col].dt.day <= 29)).astype(int)
    return df

def _add_group_lags_rolls(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = "sales",
    lags: List[int] = [1, 7, 14, 28],
    roll_windows: List[int] = [7, 30],
) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["date"])
    g = df.groupby(group_cols, sort=False)
    for l in lags:
        df[f"{target_col}_lag_{l}"] = g[target_col].shift(l)
    for w in roll_windows:
        df[f"{target_col}_rollmean_{w}"] = g[target_col].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target_col}_rollstd_{w}"] = g[target_col].shift(1).rolling(w, min_periods=1).std()
    return df

def make_features(
    train_or_test: pd.DataFrame,
    holidays_events: Optional[pd.DataFrame] = None,
    transactions: Optional[pd.DataFrame] = None,
    oil: Optional[pd.DataFrame] = None,
    stores: Optional[pd.DataFrame] = None,
    group_cols: List[str] = ["store_nbr", "family"],
    target_col: str = "sales",
    add_lags: bool = True,
    lags: List[int] = [1, 7, 14, 28],
    roll_windows: List[int] = [7, 30],
    dropna_target: bool = False,
):
    # Поддержка альтернативного порядка аргументов (transactions, oil, holidays, stores)
    # Если второй аргумент похож на transactions, переупорядочим
    def _is_transactions(x: Optional[pd.DataFrame]) -> bool:
        return isinstance(x, pd.DataFrame) and ("transactions" in x.columns)
    def _is_oil(x: Optional[pd.DataFrame]) -> bool:
        return isinstance(x, pd.DataFrame) and ("dcoilwtico" in x.columns or "oil_price" in x.columns)
    def _is_holidays(x: Optional[pd.DataFrame]) -> bool:
        return isinstance(x, pd.DataFrame) and ("locale" in x.columns or "type" in x.columns or "transferred" in x.columns)
    # Если holidays_events фактически transactions, сдвинем порядок: (transactions, oil, holidays)
    if _is_transactions(holidays_events) and _is_oil(transactions) and _is_holidays(oil):
        transactions, oil, holidays_events = holidays_events, transactions, oil

    df = train_or_test.copy()
    df = _ensure_datetime(df, "date")
    df = _add_calendar_features(df, "date")
    df = _add_special_days(df, "date")
    # Важно: сначала добавим метаданные магазинов, чтобы holidays могли учитывать city/state
    df = _merge_stores_meta(df, stores)
    df = _merge_holidays(df, holidays_events)
    df = _merge_transactions(df, transactions)
    df = _merge_oil(df, oil)
    # Дополнительные временные признаки
    df = _add_txn_oil_lags_rolls(df, group_cols, txn_col="transactions", oil_col="oil_price", lags=[7], roll_windows=[7, 30])
    # Признаки по промо-акциям (из train/test: onpromotion)
    df = _add_onpromotion_features(df, group_cols, promo_col="onpromotion", lags=[1, 7, 14, 28], roll_windows=[7, 30])
    df = _add_holiday_distance_features(df, "date")
    if add_lags and (target_col in df.columns):
        df = _add_group_lags_rolls(df, group_cols, target_col, lags, roll_windows)
    y = df[target_col] if target_col in df.columns else None
    if (y is not None) and dropna_target:
        df = df[~y.isna()]
        y = y.loc[df.index]
    return df, y
