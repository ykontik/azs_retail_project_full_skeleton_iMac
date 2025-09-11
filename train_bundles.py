# train_bundles.py
# Запуск:
#   python train_bundles.py --file data_raw/baskets.csv --level family --top_k 20
#
# Файл baskets.csv может иметь колонки с разными именами:
#  - order:  order_id | ticket_id | receipt_id | basket_id | transaction_id
#  - product: product_id | sku | item_id | upc | family | category
#  - store: store_nbr | store | shop
#  - date: date | datetime
#  - time: time (опц.), hour (опц.)
#  - qty: qty | quantity (опц.)
#  - price: price | amount | net_price (опц.)

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

DEFAULT_OUT = Path("data_dw/bundles.parquet")
DEFAULT_ANCHORS = {"FUEL","GASOLINE","DIESEL","FUEL_OCTANE95","FUEL_OCTANE98"}

def _pick(cols_lower, *cands):
    for c in cands:
        if c in cols_lower:
            return cols_lower[c]
    return None

def _read_baskets(path: Path,
                  order_col: Optional[str],
                  product_col: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    # авто-детект имён
    order = order_col or _pick(cols_lower, "order_id","ticket_id","receipt_id","basket_id","transaction_id")
    product = product_col or _pick(cols_lower, "product_id","sku","item_id","upc","family","category")
    store = _pick(cols_lower, "store_nbr","store","shop")
    date  = _pick(cols_lower, "date","datetime")
    time  = _pick(cols_lower, "time")
    hour  = _pick(cols_lower, "hour")
    qty   = _pick(cols_lower, "qty","quantity")
    price = _pick(cols_lower, "price","amount","net_price")

    # дружелюбная ошибка с подсказками
    if order is None or product is None:
        raise ValueError(
            "Не нашёл колонки заказа/товара.\n"
            f"Колонки в файле: {list(df.columns)}\n"
            "Решения: \n"
            "  1) Переименуй в CSV (order_id, product_id), или\n"
            "  2) Запусти с флагами: --order_col YOUR_TICKET --product_col YOUR_PRODUCT\n"
            "  3) Если товар в 'family', просто укажи --product_col family"
        )

    # нормализуем
    df = df.rename(columns={order: "order_id", product: "product_id"})
    if store: df = df.rename(columns={store: "store_nbr"})
    if date:  df = df.rename(columns={date:  "date"})
    if time:  df = df.rename(columns={time:  "time"})
    if hour:  df = df.rename(columns={hour:  "hour"})
    if qty:   df = df.rename(columns={qty:   "qty"})
    if price: df = df.rename(columns={price: "price"})

    # типы
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "store_nbr" in df.columns:
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").astype("Int64")

    # час: либо из hour, либо из time
    if "hour" not in df.columns:
        if "time" in df.columns:
            try:
                df["hour"] = pd.to_datetime(df["time"], errors="coerce").dt.hour
            except Exception:
                df["hour"] = pd.to_numeric(df["time"], errors="coerce")
        else:
            df["hour"] = np.nan

    # qty по умолчанию 1
    if "qty" not in df.columns:
        df["qty"] = 1

    # продукт → строка
    df["product_id"] = df["product_id"].astype(str)
    return df

def _daypart(hour: float) -> str:
    if pd.isna(hour): return "ALL"
    h = int(hour) % 24
    if 5 <= h < 11:  return "MORNING"
    if 11 <= h < 17: return "DAY"
    if 17 <= h < 23: return "EVENING"
    return "NIGHT"

def _prepare_baskets(df: pd.DataFrame,
                     level: str,
                     min_items_per_order: int = 2) -> pd.DataFrame:
    # level: 'family' или 'sku'; если в файле только family — всё равно ок (product_id уже = family)
    item_col = "product_id" if level in ("sku","family") else "product_id"
    g = df.copy()
    g["daypart"] = g["hour"].apply(_daypart)

    # агрегируем строки одного заказа → уникальные позиции
    # если qty >1, это не влияет на правила 1→1
    keys = ["order_id","daypart"] + (["store_nbr"] if "store_nbr" in g.columns else [])
    bask = (g.groupby(keys)[item_col]
              .apply(lambda x: sorted(set(x.astype(str))))
              .reset_index(name="items"))
    # фильтр по длине корзины
    bask["len"] = bask["items"].str.len()
    bask = bask[bask["len"] >= min_items_per_order].drop(columns=["len"])
    return bask

def _fit_rules_segment(b: pd.DataFrame,
                       min_support: float,
                       min_confidence: float,
                       min_lift: float,
                       anchors: Optional[set],
                       top_k: int) -> pd.DataFrame:
    if len(b) < 50:
        return pd.DataFrame()
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(b["items"])
    names = mlb.classes_
    oh = pd.DataFrame.sparse.from_spmatrix(X, columns=names)

    # частые наборы длины 2
    itemsets = fpgrowth(oh, min_support=min_support, use_colnames=True, max_len=2)
    if itemsets.empty: return pd.DataFrame()

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty: return pd.DataFrame()

    # только 1→1
    rules = rules[(rules["antecedents"].apply(len)==1) & (rules["consequents"].apply(len)==1)].copy()

    # якоря (например, FUEL слева)
    if anchors:
        rules = rules[rules["antecedents"].apply(lambda s: next(iter(s)) in anchors)]

    rules = rules[rules["lift"] >= min_lift]
    if rules.empty: return pd.DataFrame()

    rules["antecedent"] = rules["antecedents"].apply(lambda s: next(iter(s)))
    rules["consequent"] = rules["consequents"].apply(lambda s: next(iter(s)))
    rules["score"] = rules["lift"] * (rules["confidence"] * np.sqrt(rules["support"]))
    cols = ["antecedent","consequent","support","confidence","lift","leverage","conviction","score"]
    rules = rules.sort_values(["score","lift","confidence","support"], ascending=False)[cols]
    if top_k:
        rules = rules.head(top_k)
    return rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="data_raw/baskets.csv")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--level", choices=["family","sku"], default="family")
    ap.add_argument("--min_support", type=float, default=0.005)
    ap.add_argument("--min_confidence", type=float, default=0.15)
    ap.add_argument("--min_lift", type=float, default=1.1)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--only_fuel_anchor", action="store_true")
    ap.add_argument("--order_col", type=str, default=None, help="если в файле другое имя для order_id")
    ap.add_argument("--product_col", type=str, default=None, help="если в файле другое имя для product_id/sku/family")
    args = ap.parse_args()

    df = _read_baskets(Path(args.file), args.order_col, args.product_col)
    bask = _prepare_baskets(df, level=args.level)

    anchors = DEFAULT_ANCHORS if args.only_fuel_anchor else None

    # группировка по сегментам (store_nbr + daypart, если store_nbr нет — только daypart)
    seg_keys = (["store_nbr"] if "store_nbr" in bask.columns else []) + ["daypart"]
    all_rules = []
    seg_groups = list(bask.groupby(seg_keys))
    total = len(seg_groups)
    t0 = time.perf_counter()
    pbar = tqdm(total=total, desc="Bundles training", unit="segment")
    for idx, (seg_vals, seg_df) in enumerate(seg_groups, start=1):
        seg_rules = _fit_rules_segment(seg_df,
                                       min_support=args.min_support,
                                       min_confidence=args.min_confidence,
                                       min_lift=args.min_lift,
                                       anchors=anchors,
                                       top_k=args.top_k)
        if seg_rules.empty:
            pbar.update(1)
            # обновим скорость/ETA
            elapsed = max(time.perf_counter() - t0, 1e-6)
            speed = idx / elapsed
            eta = max(total - idx, 0) / speed if speed > 0 else float('inf')
            pbar.set_postfix_str(f"{speed:.2f} seg/s, ETA {eta:.1f}s")
            continue
        seg_rules = seg_rules.copy()
        if "store_nbr" in seg_keys:
            seg_rules.insert(0, "store_nbr", seg_vals[0] if isinstance(seg_vals, tuple) else seg_vals)
            seg_rules.insert(1, "daypart", seg_vals[1] if isinstance(seg_vals, tuple) else "ALL")
        else:
            seg_rules.insert(0, "daypart", seg_vals if not isinstance(seg_vals, tuple) else seg_vals[0])
        all_rules.append(seg_rules)
        pbar.update(1)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        speed = idx / elapsed
        eta = max(total - idx, 0) / speed if speed > 0 else float('inf')
        pbar.set_postfix_str(f"{speed:.2f} seg/s, ETA {eta:.1f}s")
    pbar.close()

    out = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame(
        columns=["store_nbr","daypart","antecedent","consequent","support","confidence","lift","leverage","conviction","score"]
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[bundles] saved → {out_path}  rows={len(out)}  segments={len(seg_groups)}")

if __name__ == "__main__":
    main()
