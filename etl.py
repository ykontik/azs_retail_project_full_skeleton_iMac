import os
from pathlib import Path

import duckdb as ddb
import pandas as pd
from dotenv import load_dotenv  # загрузка .env

from validation import validate_folder

# Подхватываем переменные окружения из .env
load_dotenv()

WAREHOUSE_PATH = os.getenv("WAREHOUSE_PATH", "data_dw/warehouse.duckdb")
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "data_dw/parquet"))
RAW_DIR = Path(os.getenv("RAW_DIR", "data_raw"))

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
Path(WAREHOUSE_PATH).parent.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, name: str, partition_cols=None):
    """Сохраняет датафрейм в Parquet.

    Если указаны partition_cols — чистим целевую папку и пишем партиционированный набор файлов
    через DuckDB COPY в саму папку (а не в файл внутри неё).
    """
    path = PARQUET_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        # Очищаем каталог, чтобы избежать ошибки DuckDB "Directory ... is not empty"
        import shutil

        for p in path.glob("*"):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception:
                pass
        con = ddb.connect()
        con.register("df", df)
        # Пишем непосредственно в каталог `path`. PARTITION_BY ожидает список идентификаторов колонок без кавычек.
        part_cols_sql = ", ".join(partition_cols)
        con.execute(f"COPY df TO '{path}' (FORMAT PARQUET, PARTITION_BY ({part_cols_sql}))")
        con.close()
    else:
        df.to_parquet(path / f"{name}.parquet", index=False)


def main():
    if not validate_folder(str(RAW_DIR)):
        print("Валидация провалена — поправь файлы в data_raw/")
        return

    train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["date"])
    transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    save_parquet(train, "train", partition_cols=["store_nbr", "family"])
    save_parquet(transactions, "transactions", partition_cols=["store_nbr"])
    save_parquet(oil, "oil")
    save_parquet(holidays, "holidays_events")
    save_parquet(stores, "stores")

    con = ddb.connect(WAREHOUSE_PATH)
    # Используем рекурсивные шаблоны, чтобы находить партиционированные parquet в подпапках
    con.execute(
        "CREATE OR REPLACE VIEW train AS SELECT * FROM read_parquet('data_dw/parquet/train/**/*.parquet');"
    )
    con.execute(
        "CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet('data_dw/parquet/transactions/**/*.parquet');"
    )
    con.execute(
        "CREATE OR REPLACE VIEW oil AS SELECT * FROM read_parquet('data_dw/parquet/oil/**/*.parquet');"
    )
    con.execute(
        "CREATE OR REPLACE VIEW holidays_events AS SELECT * FROM read_parquet('data_dw/parquet/holidays_events/**/*.parquet');"
    )
    con.execute(
        "CREATE OR REPLACE VIEW stores AS SELECT * FROM read_parquet('data_dw/parquet/stores/**/*.parquet');"
    )
    con.close()

    print("ETL готово: Parquet + DuckDB views.")


if __name__ == "__main__":
    main()
