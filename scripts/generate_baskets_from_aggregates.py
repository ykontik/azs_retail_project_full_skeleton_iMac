# scripts/generate_baskets_from_aggregates.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data_raw")
OUT = RAW_DIR / "baskets.csv"

def load_data():
    sales = pd.read_csv(RAW_DIR/"train.csv")  # date, store_nbr, family, sales
    sales["date"] = pd.to_datetime(sales["date"])
    tx_path = RAW_DIR/"transactions.csv"
    tx = pd.read_csv(tx_path) if tx_path.exists() else pd.DataFrame(columns=["date","store_nbr","transactions"])
    if not tx.empty:
        tx["date"] = pd.to_datetime(tx["date"])
    return sales, tx

def sample_hours(n):
    # усреднённое распределение для АЗС: утро/день/вечер/ночь
    probs = np.array([0.30, 0.40, 0.20, 0.10])
    bins  = [(5,10),(11,16),(17,22),(23,4)]  # диапазоны часов
    choices = np.random.choice(4, size=n, p=probs)
    hours = []
    for c in choices:
        a,b = bins[c]
        if a <= b:
            hours.append(np.random.randint(a, b+1))
        else:
            # диапазон через полночь
            h = np.random.choice(list(range(a,24))+list(range(0,b+1)))
            hours.append(h)
    return hours

def generate_for_day(group, n_tx, fuel_anchor=False, fuel_names=("FUEL","GASOLINE","DIESEL"), anchor_prob=0.45):
    """
    group: df по одному (store,date) с колонками family,sales
    n_tx: число чеков
    fuel_anchor: если True — часть чеков будет содержать 'топливо' (если есть в продажах)
    """
    # расклад по семьям для сэмплинга
    fam = group["family"].astype(str).values
    qty = group["sales"].clip(lower=0).round().astype(int).values
    total_units = int(qty.sum())
    if total_units == 0 or n_tx <= 0:
        return pd.DataFrame(columns=["date","time","store_nbr","ticket_id","family","qty","price"])

    # распределим размер корзин (сколько позиций в чеке) ~ Poisson
    mean_items = max(1.2, total_units / n_tx)  # среднее по товарам в чеке
    sizes = np.random.poisson(lam=mean_items, size=n_tx)
    # гарантируем минимум 1
    sizes = np.maximum(sizes, 1)
    # подгоним сумму к total_units
    diff = total_units - sizes.sum()
    while diff != 0:
        idx = np.random.randint(0, n_tx)
        if diff > 0:
            sizes[idx] += 1
            diff -= 1
        elif diff < 0 and sizes[idx] > 1:
            sizes[idx] -= 1
            diff += 1
        else:
            # не можем уменьшить — перераспределим иначе
            idxs = np.where(sizes > 1)[0]
            if len(idxs)==0: break
            i = np.random.choice(idxs)
            sizes[i] -= 1
            diff += 1

    # вероятности по семьям пропорциональны дневным продажам
    p = qty / qty.sum()

    # подготовим билдер чеков
    date  = group["date"].iloc[0]
    store = int(group["store_nbr"].iloc[0])
    hours = sample_hours(n_tx)

    fuel_present = [i for i,f in enumerate(fam) if any(t in f.upper() for t in fuel_names)]
    want_anchor  = fuel_anchor and len(fuel_present)>0
    out_rows = []
    ticket_counter = 0

    for k in range(n_tx):
        ticket_counter += 1
        t_id = f"{store}-{pd.to_datetime(date).strftime('%Y%m%d')}-{ticket_counter}"
        # базовая выборка позиций
        m = sizes[k]
        # если нужен якорь топлива — с вероятностью anchor_prob кладём 1 позицию fuel
        items = []
        if want_anchor and np.random.rand() < anchor_prob:
            fi = np.random.choice(fuel_present)
            items.append(fam[fi])

        # добираем остальные позиции по распределению p
        need = m - len(items)
        if need > 0:
            picks = np.random.choice(fam, size=need, p=p, replace=True)
            items.extend(picks.tolist())

        # агрегируем по семье → qty
        if len(items)==0:
            continue
        dfb = pd.Series(items).value_counts().reset_index()
        dfb.columns = ["family","qty"]
        dfb["date"] = date
        dfb["time"] = f"{hours[k]:02d}:{np.random.randint(0,60):02d}:00"
        dfb["store_nbr"] = store
        dfb["ticket_id"] = t_id
        dfb["price"] = np.nan  # нет цен — оставим пустыми
        out_rows.append(dfb[["date","time","store_nbr","ticket_id","family","qty","price"]])

    if not out_rows:
        return pd.DataFrame(columns=["date","time","store_nbr","ticket_id","family","qty","price"])
    return pd.concat(out_rows, ignore_index=True)

def main(min_tx=50, max_tx=1200, fuel_anchor=True):
    sales, tx = load_data()

    # ожидаем, что sales — по дням; если у тебя weekly, предварительно ресемплируй в daily
    base = (sales.groupby(["store_nbr","date","family"], as_index=False)["sales"].sum())

    if not tx.empty:
        base = base.merge(tx.groupby(["store_nbr","date"], as_index=False)["transactions"].sum(),
                          on=["store_nbr","date"], how="left")
    else:
        base["transactions"] = np.nan

    rows = []
    for (store,date), g in base.groupby(["store_nbr","date"]):
        n_tx = g["transactions"].iloc[0]
        # если транзакций нет — оценим через объём продаж
        if (pd.isna(n_tx)) or (n_tx <= 0):
            total_units = int(g["sales"].sum())
            # грубая эвристика: средний чек 1.4 позиции
            n_tx = max(1, int(total_units / 1.4))
        n_tx = int(np.clip(n_tx, min_tx, max_tx))
        day_rows = generate_for_day(g, n_tx, fuel_anchor=fuel_anchor)
        if not day_rows.empty:
            rows.append(day_rows)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date","time","store_nbr","ticket_id","family","qty","price"]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[baskets.csv] saved: {OUT} rows={len(out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_tx", type=int, default=50, help="нижняя отсечка транзакций на день")
    ap.add_argument("--max_tx", type=int, default=1200, help="верхняя отсечка транзакций на день")
    ap.add_argument("--no_fuel_anchor", action="store_true", help="не форсировать присутствие топлива в части чеков")
    args = ap.parse_args()
    main(min_tx=args.min_tx, max_tx=args.max_tx, fuel_anchor=not args.no_fuel_anchor)
