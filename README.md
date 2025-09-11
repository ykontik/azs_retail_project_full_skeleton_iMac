
# AZS + Retail — Forecast MVP (TL;DR)

Этот README — краткий. Подробности см. в `docs/` и `PRESENTATION_README.md`.

Быстрый старт:
- make venv && make validate && make etl && make features
- make train (или make demo для игрушечных данных)
- make api и make ui → открой `/docs` и Streamlit

Полезные ссылки:
- Архитектура: docs/ARCHITECTURE.md | Model Card: docs/MODEL_CARD.md
- Презентация/видео: PRESENTATION_README.md | API: docs/API.md
- UI‑страницы: 02__Сравнение_моделей, 05_Error_Leaderboard, 06_Агрегаты_по_времени, 07_Бизнес_метрики



---
Ниже — краткие разделы. Полные версии перенесены в docs/.
- Готовые команды для показа

## Быстрый старт
1) Создай папку `data_raw/` и положи CSV:
   - `train.csv`, `transactions.csv`, `oil.csv`, `holidays_events.csv`, `stores.csv`
2) Валидация:
   ```bash
   python validation.py
   ```
3) ETL:
   ```bash
   python etl.py
   ```
4) Фичи:
   ```bash
   python features.py
   ```
5) Обучение прогноза:
   ```bash
   python train_forecast.py \
     --train data_raw/train.csv \
     --transactions data_raw/transactions.csv \
     --oil data_raw/oil.csv \
     --holidays data_raw/holidays_events.csv \
     --stores data_raw/stores.csv \
     --models_dir models \
     --warehouse_dir data_dw \
     --top_n_sku 50 \
     --top_recent_days 90 \
     --valid_days 28 \
     --cv_folds 3 \
     --cv_step_days 14 \
     --quantiles 0.5 0.9
   ```
6) Запасы:
   ```bash
   python train_stock.py
   
   ```
7) Бандлы (если есть `data_raw/baskets.csv`):
   ```bash
   
   python train_bundles.py
   ```
8) API + UI:
  ```bash
  uvicorn service.app:app --reload
  streamlit run ui/dashboard.py
  ```

## Как проверять критерии
- Постановка задачи и результат: см. `docs/MODEL_CARD.md` (задача, данные, метрики, ограничения) и раздел Business Impact ниже.
- Исследование и подготовка данных: `validation.py`, `etl.py`, `features.py`, визуализации в Streamlit (`ui/dashboard.py`).
- Модели и гиперпараметры: `train_forecast.py` (переключаемые цели LGBM), `train_global_catboost.py`, `train_global_xgboost.py`, `experiments/`.
- Метрики качества: `data_dw/metrics_per_sku.csv`, `data_dw/summary_metrics.txt`, глобальные `data_dw/metrics_global_*.json`.
- Интерпретация: `scripts/shap_report.py` (артефакты в `data_dw/`).
- Вывод в прод: FastAPI API (`service/app.py`) + Swagger `/docs`, Streamlit UI (`ui/dashboard.py`), Docker Compose (`docker-compose.yml`).
- Демонстрация: `make demo` / `make demo-full` поднимут пайплайн «из коробки» на toy‑данных.
- Архитектура: `docs/ARCHITECTURE.md` (Mermaid‑диаграмма конвейера).

## Демо/презентация

### Автоматизированное демо (рекомендуется)
```bash
# Полное демо с автоматическим запуском всех компонентов
python scripts/demo_presentation.py

# Или через Makefile
make demo-full
```

### Ручной запуск
```bash
# 1. Подготовка
make venv
make validate && make etl && make features

# 2. Обучение (быстрое для демо)
make train_global
make train_global_xgb

# 3. Запуск сервисов
make api          # в одном терминале
make ui           # в другом терминале

# 4. Бенчмарк производительности
make benchmark-fast
```

## 🧪 Тестирование и качество

### Запуск тестов
```bash
# Все тесты
make test

# С покрытием
make test-cov

# Быстрые тесты
make test-fast

# Только unit тесты
make test-unit
```

### Проверка качества кода
```bash
# Линтинг
make lint

# Автоисправление
make lint-fix

# Проверка типов
make typecheck

# Полная проверка
make quality
```


## 💰 Business Impact

- Сгенерировать шаблон цен/маржи/стоимости хранения по семьям:
```bash
make prices-template  # создаст configs/prices.csv
```
- Заполнить реальные значения в `configs/prices.csv` (можно добавить колонку `store_nbr` для переопределений).
- Сформировать отчёт по всему датасету (CSV + сводка в консоли):
```bash
make impact LEAD_TIME_DAYS=2 SERVICE_LEVEL=0.95 PRICE_CSV=configs/prices.csv
# Репорт: data_dw/business_impact_report.csv
```
- Для одной пары (быстрый расчёт):
```bash
make biz-metrics STORE=1 FAMILY=AUTOMOTIVE PRICE=3.5 MARGIN_RATE=0.25 HOLDING_COST=0.05 LEAD_TIME_DAYS=2 SERVICE_LEVEL=0.95
```

## 🔥 Smoke Quickstart (30–60 сек)

- Установка окружения:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install -r requirements.txt -r requirements-dev.txt
```
- Запуск smoke‑тестов (генерируют toy‑данные, кладут Dummy‑модель и проверяют API через TestClient):
```bash
pytest -q -m "smoke"
```

## 🖥️ Установка: Windows / Linux

- Windows (PowerShell):
  - Создать venv: `py -m venv .venv; . .venv\Scripts\Activate.ps1`
  - Установить зависимости: `pip install -r requirements.txt -r requirements-dev.txt`
  - Без `make`: запускайте команды напрямую, напр.: `python validation.py`, `python etl.py`, `python features.py`, `python train_forecast.py ...`, `uvicorn service.app:app --reload`, `streamlit run ui/dashboard.py`.
- Linux/macOS:
  - Создать venv: `python3 -m venv .venv && . .venv/bin/activate`
  - Использовать `make` цели из разделов выше или те же команды напрямую.

## 🧹 Repo Cleanup (тяжёлые артефакты)

- Исключить артефакты из git (останутся на диске):
```bash
make clean-repo
# затем закоммитьте изменения при необходимости
# git commit -m "chore: cleanup artifacts"
```
- Полностью удалить локальные артефакты:
```bash
make clean-artifacts
```

## 🧮 Экспорт и квантизация (SOTA-бонус)

Опционально можно экспортировать обученную per‑SKU модель (LightGBM/Sklearn API) в ONNX и сделать dynamic INT8‑квантизацию для ускорения инференса и уменьшения размера:

```bash
# Установить зависимости (локально):
pip install skl2onnx onnx onnxruntime

# Экспорт модели пары store=1, family=AUTOMOTIVE в models/1__AUTOMOTIVE.onnx
python scripts/export_onnx.py --store 1 --family AUTOMOTIVE --quantize
```

Это добавит файлы `.onnx` и `.int8.onnx` рядом с .joblib, а также `.onnx.meta.json` с метаданными (список фич и их число). В презентации можно показать выигрыш в задержке onnxruntime vs joblib (см. мини-бенчмарк в конце скрипта).


## 📚 Документация

- **API документация:** `/docs` (Swagger UI)
- **Техническая документация:** `docs/API.md`
- **Архитектура:** `docs/ARCHITECTURE.md`
- **Model Card:** `docs/MODEL_CARD.md`
- **Презентационный README:** `PRESENTATION_README.md`
- **Слайды (10 мин):** `docs/slides.md` → `make slides-html` или `make slides-pdf`
- **Colab демо:** `notebooks/colab_demo.ipynb` (замените `REPO_URL` и запустите по шагам)

## 🔌 API Эндпоинты

- `GET /health` — статус сервиса
- `GET /live` / `GET /ready` / `GET /version` — пробы и версия
- `GET /models` — список доступных моделей
- `GET /feature_names?store_nbr=1&family=AUTOMOTIVE` — фичи модели
- `GET /metrics` — метрики обучения (таблица per‑SKU)
- `GET /quantiles_available?store_nbr=1&family=AUTOMOTIVE` — доступные квантили
- `POST /predict_demand` — точечный прогноз по паре `(store_nbr, family)`
- `POST /predict_bulk` — пакетный прогноз
- `POST /predict_demand_quantiles?qs=0.5,0.9` — прогноз по квантилям
- `POST /reorder_point` — расчёт Safety Stock и Reorder Point (использует P50/P90 при наличии)
- `GET /metrics-prom` — метрики приложения для Prometheus (latency, счётчики)

Пример `/reorder_point`:
```bash
curl -X POST http://localhost:8000/reorder_point \
  -H 'Content-Type: application/json' \
  -d '{
        "store_nbr":1,
        "family":"AUTOMOTIVE",
        "features":{"year":2017,"month":8,"sales_lag_7":5.0},
        "lead_time_days":2,
        "service_level":0.95
      }'
```

## 🧩 Новые скрипты и фичи

### Признак onpromotion
- В фичах добавлены: `onpromotion`, `onpromotion_lag_{1,7,14,28}`, `onpromotion_rollmean_{7,30}`, `onpromotion_rollstd_{7,30}`.
- Если исходной колонки нет, она создаётся (0.0) — пайплайн не ломается.

### Optuna‑тюнинг (per‑SKU LGBM)
```bash
python experiments/tune_optuna.py \
  --train data_raw/train.csv \
  --transactions data_raw/transactions.csv \
  --oil data_raw/oil.csv \
  --holidays data_raw/holidays_events.csv \
  --stores data_raw/stores.csv \
  --store 1 --family AUTOMOTIVE \
  --valid_days 28 --n_trials 50
# Артефакты: data_dw/best_params_{store}__{family}.json, models/{store}__{family}__optuna.joblib
```

### SHAP‑интерпретация важности
```bash
python scripts/shap_report.py \
  --store 1 --family AUTOMOTIVE \
  --train data_raw/train.csv \
  --transactions data_raw/transactions.csv \
  --oil data_raw/oil.csv \
  --holidays data_raw/holidays_events.csv \
  --stores data_raw/stores.csv
# Артефакты: data_dw/shap_summary_{store}__{family}.png, data_dw/shap_top_{store}__{family}.csv
```

### Бизнес‑метрики (MAPE → деньги, SS/ROP)
```bash
python scripts/business_metrics.py \
  --store 1 --family AUTOMOTIVE \
  --price 3.5 --margin_rate 0.25 --holding_cost 0.05 \
  --lead_time_days 2 --service_level 0.95
```

## 🎯 Что показывать на презентации

### 1. **Swagger UI** (`/docs`)
- Полная документация API
- Интерактивное тестирование эндпоинтов
- Примеры запросов и ответов

### 2. **Streamlit Dashboard**
- Метрики качества по всем SKU
- Сравнение моделей
- Лидерборд ошибок
- Интерактивные прогнозы

### 3. Трекинг результатов
- Сводки метрик и сравнение запусков: data_dw/summary_metrics.txt
- Сравнение моделей: docs/model_comparison.csv
- Интерпретация: SHAP (data_dw/shap_*.png, раздел «SHAP — важность признаков» в UI)

### 4. **Docker контейнеризация**
- `docker compose up --build`
- Показ health checks
- Логи сервисов

### 5. **Тестирование**
- `make test-cov` - покрытие тестами
- `make quality` - проверка качества кода

### 6. **Производительность**
- `make benchmark` - отчет о производительности
- Время инференса < 50ms
- Сравнение размеров моделей

## 🔑 Ключевые преимущества 

✅ **Полный ML пайплайн** - от данных до API  
✅ **Production-ready** - Docker, health checks, мониторинг  
✅ **Качество кода** - тесты 70%+, линтинг, типы  
✅ **Документация** - API docs, техническая документация  
✅ **Инновации** - ONNX/квантизация, бенчмарк, множественные алгоритмы  
✅ **Веб-интерфейс** - Streamlit dashboard  
✅ **Автоматизация** - Makefile, скрипты демо  

## 📱 Новые сервисные эндпоинты/возможности API:
- `GET /live` — liveness-проба
- `GET /ready` — readiness-проба (проверка наличия метрик/моделей)
- `GET /version` — версия приложения и основные пути
- `GET /metrics-prom` — метрики Prometheus (если установлен `prometheus-client`)
- Базовая авторизация по ключу заголовка `X-API-Key` (отключаемая через `DISABLE_AUTH=true`)

Полезные эндпоинты API:
- `GET /health` — статус
- `GET /live` — liveness-проба
- `GET /ready` — readiness-проба
- `GET /version` — версия/пути
- `GET /models` — список моделей
- `GET /metrics` — метрики обучения
- `GET /feature_names?store_nbr=1&family=AUTOMOTIVE` — список фич модели
- `POST /predict_demand` — прогноз по одной паре `(store_nbr, family)`
- `POST /predict_bulk` — прогноз пачкой (список запросов)
- `POST /predict_demand_quantiles?qs=0.5,0.9` — прогноз по квантилям (если обучены модели квантилей)
- `GET /metrics-prom` — Prometheus метрики (text/plain)

UI добавлено:
- Переключение `API URL` (вверху дашборда, раздел «⚙️ Подключение к API»)
- Кнопка «Автозаполнить фичи по последней дате» — считает фичи по данным из `data_raw` и подставляет признаки для выбранной пары с учётом списка фич обученной модели.

Конфигурация
- Копируй `.env.example` в `.env` и при необходимости меняй пути/настройки.
- Для обучения есть `configs/train.yaml`. Параметры из YAML используются, если не заданы явно через CLI.

Makefile
- Удобные команды: `make validate`, `make etl`, `make features`, `make train`, `make api`, `make ui`, `make docker`.
  - `make demo` — сгенерировать игрушечные данные, подготовить фичи и обучить маленький топ-N (для быстрого демо).
9) Docker:
   ```bash
  docker compose up --build
  ```

## 10 минут до демо (короткий сценарий)
- Создать окружение и зависимости: `make venv`
- Проверка и подготовка: `make validate && make etl && make features`
- Обучение (per‑SKU LGBM): `make train`
- Запуск API и UI (в двух терминалах): `make api` и `make ui`
- В UI (Streamlit) откройте «Лидерборд ошибок» и «Сравнение моделей»

Если нет `make` (Windows):
- Создать venv: `python -m venv .venv && .venv\\Scripts\\activate`
- `pip install -r requirements.txt`
- `python validation.py && python etl.py && python features.py`
- `python train_forecast.py --config configs/train.yaml`
- `python -m uvicorn service.app:app --reload`
- `python -m streamlit run ui/dashboard.py`

## Архитектура (данные → прод)
- Raw CSV → `validation.py` → `etl.py` → `make_features.py` → `train_*` → модели в `models/`
- Метрики и отчёты → `data_dw/metrics_per_sku.csv`, `data_dw/summary_metrics.txt`, `data_dw/report.html`
- Прод: FastAPI (`service/app.py`) + Streamlit (`ui/*`), запуск локально или через Docker Compose

## Отчёт по метрикам
- Скрипт `scripts/report.py` собирает HTML‑отчёт: агрегаты MAE/MAPE по магазинам/семействам, топ‑ошибки, сравнение с baseline.
- Запуск: `python scripts/report.py` → `data_dw/report.html`
- В UI добавлена страница «Лидерборд ошибок» (по данным `metrics_per_sku.csv`).

## CI / Качество
- В репозитории есть GitHub Actions workflow (`.github/workflows/ci.yml`): линтеры, типы, тесты.
- Локально: `make dev-setup && make lint && make typecheck && make test`

## Новое в инфраструктуре
- CORS включён в API (см. переменную `CORS_ORIGINS`), UI в docker-compose подключается к `http://api:8000` (см. `configs/.env`).
- Добавлен `configs/.env` для docker-compose с путями `/app/...` и `API_URL=http://api:8000`.
- Добавлен `configs/train.yaml` с типовыми параметрами обучения; `make train` подхватывает его по умолчанию.

**Makefile Команды**
- **validate**: проверяет входные CSV в `data_raw/`.
- **etl**: кладёт parquet и настраивает DuckDB в `data_dw/`.
- **features**: генерирует `features.parquet` и `target.parquet`.
- **train / train_xgb / train_global / train_global_xgb**: обучение моделей.
- **api / ui**: запуск API и Streamlit локально.
- **docker**: сборка/запуск `api` и `ui` через Docker Compose.
- **venv / setup**: создаёт `.venv` и ставит зависимости из `requirements.txt`.
- **dev-setup**: ставит dev‑зависимости (`requirements-dev.txt`) и pre-commit хуки.
- **lint**: ruff + black, **typecheck**: mypy, **test**: pytest, **cov**: pytest с покрытием, **all**: validate→etl→features→train→test.

**Переменные Окружения**
- **RAW_DIR, PARQUET_DIR, WAREHOUSE_DIR, MODELS_DIR**: пути к данным/моделям (локально) — есть шаблон `.env.example`.
- Для Docker Compose используйте `configs/.env` (пути вида `/app/...`, `API_URL=http://api:8000`).
- **CORS_ORIGINS**: список разрешённых источников для UI.
- **MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT**: трекинг экспериментов (опционально).
- **DISABLE_AUTH**: `true|false` — отключить/включить проверку API‑ключа (по умолчанию true).
- **API_KEY**: секретный ключ для доступа к API (используется, если `DISABLE_AUTH=false`).

**Запуск Локально**
- Создать окружение и поставить зависимости: `make venv` (и при необходимости `make dev-setup`).
- Подготовка данных/фич: `make validate && make etl && make features`.
- Обучение: `make train` (использует `configs/train.yaml`, можно переопределять флаги).
- API и UI: `make api` и в другом терминале `make ui`.

**Запуск В Docker**
- Заполнить `configs/.env` (шаблон уже в репозитории).
- Запуск: `make docker` или `docker compose up --build`.

**Линт И Тесты**
- Установка dev‑зависимостей: `make dev-setup`.
- Линт/формат: `make lint`, типы: `make typecheck`.
- Тесты: `make test`, покрытие: `make cov`.

**API Примеры**
- `GET /health`: `curl http://localhost:8000/health`.
- `GET /models`: `curl http://localhost:8000/models`.
- `POST /predict_demand`:
  ```bash
  curl -X POST http://localhost:8000/predict_demand \
    -H 'Content-Type: application/json' \
    -d '{"store_nbr":1, "family":"AUTOMOTIVE", "features": {"year":2017, "month":8}}'
  ```

## Baseline и Ablation
- В `train_forecast.py` добавлены baseline-метрики на валидации: Naive lag-7 и MA(7). В `data_dw/metrics_per_sku.csv` появляются колонки `NAIVE_*` и `MAE_GAIN_vs_*`.
- Для экспериментов ablation (отключение групп признаков) добавлен скрипт:
  ```bash
  python experiments/ablation.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --warehouse_dir data_dw \
    --top_n_sku 10 --top_recent_days 90
  # результат: data_dw/ablation_results.csv
  ```

## Тесты
- Минимальные тесты (API health, make_features):
  ```bash
  pip install pytest
  pytest -q
  ```

## Дополнительно: цели LightGBM и глобальная CatBoost
- Переключение цели LightGBM (MAE/Tweedie/Poisson) в `train_forecast.py`:
  - Флаги:
    - `--lgb_objective l1|tweedie|poisson` (по умолчанию `l1`)
    - `--tweedie_variance_power 1.2` (для tweedie)
  - Пример:
    ```bash
    LGB_OBJECTIVE=tweedie TWEEDIE_POWER=1.2 make train
    # или
    python train_forecast.py --config configs/train.yaml --lgb_objective tweedie --tweedie_variance_power 1.2
    ```
- Глобальная модель CatBoost (одна модель на все пары):
  ```bash
  make train_global
  # или
  python train_global_catboost.py \
    --train data_raw/train.csv \
    --transactions data_raw/transactions.csv \
    --oil data_raw/oil.csv \
    --holidays data_raw/holidays_events.csv \
    --stores data_raw/stores.csv \
    --valid_days 28
  ```
  - Модель сохраняется в `models/global_catboost.cbm`, метрики — `data_dw/metrics_global_catboost.json`.

## XGBoost: обучение и сравнение
- Глобальная XGBoost-модель (одна на все пары):
  ```bash
  make train_global_xgb
  # модель: models/global_xgboost.joblib
  # метрики: data_dw/metrics_global_xgboost.json
  ```
- Per‑SKU XGBoost (аналогично LGBM, на top‑N пар):
  ```bash
  make train_xgb \
    TOP_N_SKU=50 \
    TOP_RECENT_DAYS=90 \
    VALID_DAYS=28
  # модели: models/{store}__{family}__xgb.joblib
  ```
- Обучить одну пару XGBoost (точечно):
  ```bash
  python scripts/train_one_pair_xgb.py --store 2 --family AUTOMOTIVE
  ```
- Сравнение в UI:
  - Страница «Сравнение моделей: per‑SKU LGBM vs Global CatBoost vs ALT LGBM» теперь показывает и линию XGBoost (если обучена глобальная модель) и строит heatmap «XGBoost vs LGBM».
  - Отдельная страница: «XGBoost Compare» (ui/pages/04_XGBoost_Compare.py) — графики и heatmap «XGBoost vs LGBM» с выгрузкой CSV.
## 🛠️ Troubleshooting (установка и запуск)

- LightGBM/CatBoost на Windows: поставьте Microsoft C++ Build Tools; проще использовать `python3.11` и готовые wheels.
- Apple Silicon (M1/M2): используйте системный `python3.11`, свежий `pip`, и по возможности `xgboost==2.1.x` (есть arm64‑wheels). Для LightGBM лучше без GPU.
- Ошибки компиляции: сначала `pip install -U pip setuptools wheel`, затем `pip install -r requirements.txt`.
- Streamlit не видит API: задайте `API_URL` (например, `export API_URL=http://127.0.0.1:8000`) или укажите в верхнем поле дашборда.
- «Мало памяти/долго обучается»: уменьшите `TOP_N_SKU`, `TOP_RECENT_DAYS`, `VALID_DAYS` или запустите `make demo` для ускоренного сценария.
- Docker образ слишком тяжёлый: для чистого API можно вынести UI/SHAP/Optuna в отдельный образ (напишите — покажу пример `Dockerfile` с multi‑stage и split‑requirements).
