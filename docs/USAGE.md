# Использование и Quickstart

## Быстрый старт
- Подготовка: `make venv && make validate && make etl && make features`
- Обучение: `make train` (или `make demo` для игрушечных данных)
- Запуск: `make api` и `make ui` → открыть `/docs` и Streamlit

## Ручной запуск (без Makefile)
```bash
python validation.py
python etl.py
python features.py
python train_forecast.py --train data_raw/train.csv --transactions data_raw/transactions.csv \
  --oil data_raw/oil.csv --holidays data_raw/holidays_events.csv --stores data_raw/stores.csv
uvicorn service.app:app --reload
streamlit run ui/dashboard.py
```

## Demo
- Полное авто‑демо: `python scripts/demo_presentation.py` или `make demo-full`

