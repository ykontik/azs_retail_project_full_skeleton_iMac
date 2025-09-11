# Архитектура Проекта

Краткое описание конвейера данных и сервисов. Диаграмма отражает ключевые шаги от загрузки сырых данных до инференса через API и UI.

## Конвейер Данных и Моделей

```mermaid
flowchart LR
    subgraph Raw[Сырые данные]
      A[train.csv]
      B[transactions.csv]
      C[oil.csv]
      D[holidays_events.csv]
      E[stores.csv]
    end

    A & B & C & D & E --> ETL[ETL (etl.py)\nCSV -> Parquet + DuckDB]
    ETL --> FE[Фичи (make_features)\nfeatures.py/make_features.py]
    FE -->|per-SKU top-N| LGBM[Обучение per-SKU LGBM\ntrain_forecast.py]
    FE -->|global| CAT[Глобальная CatBoost\ntrain_global_catboost.py]
    FE -->|global| XGB[Глобальная XGBoost\ntrain_global_xgboost.py]

    LGBM --> M[models/*.joblib]
    CAT  --> M
    XGB  --> M

    FE --> MTR[data_dw/metrics_per_sku.csv\nsummary_metrics.txt]

    subgraph Serve[Сервисы]
      API[FastAPI\nservice/app.py]
      UI[Streamlit\nui/dashboard.py]
      [()]
    end

    M & MTR --> API
    API <--> UI
    FE & Train[[Обучающие скрипты]] --> 
```

## Компоненты

- ETL: нормализация CSV, сохранение Parquet и DuckDB (`etl.py`).
- Feature Engineering: календарные признаки, лаги, скользящие статистики, onpromotion/transactions/oil/holidays (`features.py`, `make_features.py`).
- Обучение:
  - per‑SKU LightGBM с holdout‑валидацией + опциональные квантили (`train_forecast.py`).
  - Глобальные CatBoost/XGBoost (`train_global_catboost.py`, `train_global_xgboost.py`).
  - Optuna‑тюнинг и ablation‑эксперименты (`experiments/`).
- Инференс: FastAPI (`service/app.py`) с Swagger, CORS, API‑ключом, Prometheus‑метриками.
- Веб‑интерфейс: Streamlit (`ui/dashboard.py`) — метрики, лидерборд, интерактивный прогноз.
- Трекинг: сводки метрик в `data_dw/summary_metrics.txt`, сравнение моделей в `docs/model_comparison.csv`.

## Наблюдаемость и Качество

- Тесты: unit/integration/smoke, покрытие 70%+ (`tests/`, `pytest.ini`).
- Линт/типы: ruff, black, mypy (Makefile, CI).
- Метрики API: `/metrics-prom` (Prometheus формат).
- Бенчмарк: `scripts/performance_monitor.py` → отчёты в `data_dw`/`artifacts`.

## Деплой

- Локально: `make api`, `make ui` (Swagger: `/docs`).
- Docker Compose: сервисы `api` и `ui`, общий том с моделями и данными.
- Переменные окружения в `.env.example` и `configs/.env` (для Compose).
