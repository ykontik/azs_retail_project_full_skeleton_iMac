# Простые команды для воспроизводимости. Используется .env, если он существует.

# Выбор интерпретатора: предпочитаем локальный venv, если он есть
# Предпочитаем локальный venv, если он есть, без сложных shell-выражений
ifneq ("$(wildcard .venv/bin/python)", "")
PY := .venv/bin/python
else
PY := $(shell command -v python || echo python)
endif

ifneq ("$(wildcard .venv/bin/pip)", "")
PIP := .venv/bin/pip
else
PIP := $(shell command -v pip || echo pip)
endif

SHELL := /bin/bash
.ONESHELL:

.PHONY: help env validate etl features train api ui docker train_global train_global_xgb venv setup train_xgb train_rf train_one_xgb train_lstm \
        lint typecheck test test-cov test-fast test-unit test-integration cov all dev-setup precommit-install \
        benchmark benchmark-fast lint-fix quality demo demo-full \
        optuna shap biz-metrics clean-artifacts clean-repo impact prices-template retrain-all \
        export-onnx biz-report clean-mlflow \
        batch-train-selected train-quantiles-selected export-eda model-summary shap-one

# Параметры
DOCKER_PROJECT_NAME ?= azs-retail

# Показать справку по всем командам
help: ## Показать справку по всем командам
	@echo "🚀 AZS + Retail Project - Доступные команды:"
	@echo ""
	@echo "📊 Основные операции:"
	@echo "  validate     - Валидация входных данных"
	@echo "  etl          - ETL процесс (CSV → Parquet + DuckDB)"
	@echo "  features     - Создание признаков"
	@echo "  train        - Обучение per-SKU моделей LightGBM"
	@echo "  train_global - Обучение глобальной CatBoost модели"
	@echo "  train_global_xgb - Обучение глобальной XGBoost модели"
		@echo "  train_rf       - Обучение per-SKU RandomForest"
		@echo "  train_lstm     - Обучение per-SKU LSTM (baseline)"
	@echo ""
	@echo "🚀 Запуск сервисов:"
	@echo "  api          - Запуск FastAPI сервера"
	@echo "  ui           - Запуск Streamlit UI"
	@echo "  docker       - Запуск через Docker Compose"
	@echo ""
	@echo "🧩 Опциональные артефакты:"
	@echo "  optuna       - Optuna-тюнинг LGBM per-SKU (best params + модель)"
	@echo "  shap         - SHAP отчёт важности признаков для пары"
	@echo "  biz-metrics  - Перевод MAPE в деньги и расчёт SS/ROP"
	@echo "  export-onnx  - Экспорт модели пары в ONNX (+INT8)"
	@echo "  biz-report   - Сводный бизнес-отчёт по всем парам (CSV)"
	@echo "  batch-train-selected - Обучить per-SKU (LGBM+XGB) для первых 20 и последних 10 магазинов"
	@echo "  train-quantiles-selected - Обучить квантильные модели (P50/P90) для тех же пар"
	@echo "  export-eda   - Сохранить EDA-графики (сезонность/промо) в docs/"
	@echo "  model-summary - Сводная таблица сравнения моделей в docs/model_comparison.csv"
	@echo "  shap-one     - SHAP для одной пары (STORE/FAMILY)"
	@echo "     Параметры batch/train-quantiles: STORE_LIST=\"1,2\" FAMILY_LIST=\"BEVERAGES,PRODUCE\" MAX_PAIRS=100 STOP_FILE=STOP_BATCH"
	@echo "  clean-mlflow - Удалить локальные артефакты MLflow (mlflow.db, mlruns/)"
	@echo ""
	@echo "🧪 Тестирование и качество:"
	@echo "  test         - Запуск всех тестов"
	@echo "  test-cov     - Тесты с покрытием кода"
	@echo "  test-fast    - Быстрые тесты (без медленных)"
	@echo "  test-unit    - Только unit тесты"
	@echo "  test-integration - Только integration тесты"
	@echo "  lint         - Проверка и форматирование кода"
	@echo "  lint-fix     - Автоисправление линтера"
	@echo "  typecheck    - Проверка типов (MyPy)"
	@echo "  quality      - Полная проверка качества"
	@echo ""
	@echo "⚡ Производительность:"
	@echo "  benchmark    - Полный бенчмарк моделей"
	@echo "  benchmark-fast - Быстрый бенчмарк"
	@echo ""
	@echo "🎯 Демо и презентация:"
	@echo "  demo         - Быстрое демо с игрушечными данными"
	@echo "  demo-full    - Полное демо для презентации"
	@echo ""
	@echo "🔧 Разработка:"
	@echo "  venv         - Создание виртуального окружения"
	@echo "  dev-setup    - Установка dev-зависимостей"
	@echo "  precommit-install - Установка pre-commit хуков"
	@echo ""
	@echo "📚 Дополнительно:"
	@echo "  pairs        - Показать доступные пары (store_nbr, family)"
	@echo "  all          - validate → etl → features → train → test"
	@echo ""
	@echo "💡 Примеры использования:"
	@echo "  make help              - Эта справка"
	@echo "  make demo-full         - Полное демо для презентации"
	@echo "  make quality           - Проверка качества кода"
	@echo "  make benchmark         - Анализ производительности"
	@echo "  make slides-html       - Сгенерировать HTML-слайды из docs/slides.md (pandoc)"
	@echo "  make slides-pdf        - Сгенерировать PDF-слайды (требуется LaTeX)"

env:
	@[ -f .env ] && echo "Using .env" || echo "No .env found (using defaults)"

validate: env
	$(PY) validation.py

etl: env
	$(PY) etl.py

features: env
	$(PY) features.py

train: env
	$(PY) train_forecast.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --config configs/train.yaml \
	  --lgb_objective $${LGB_OBJECTIVE:-l1} \
	  --tweedie_variance_power $${TWEEDIE_POWER:-1.2}

train_global: env
	$(PY) train_global_catboost.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --valid_days $${VALID_DAYS:-28}

train_global_xgb: env
	$(PY) train_global_xgboost.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --valid_days $${VALID_DAYS:-28}

api: env
	$(PY) -m uvicorn service.app:app --reload

ui: env
	$(PY) -m streamlit run ui/dashboard.py

docker: env
	@if [ ! -f configs/.env ]; then \
	  echo "# auto-generated placeholder" > configs/.env; \
	  echo "INFO: создан пустой configs/.env (заполните при необходимости)"; \
	fi
	COMPOSE_DOCKER_CLI_BUILD=0 DOCKER_BUILDKIT=0 docker compose -p $(DOCKER_PROJECT_NAME) up --build

# Быстрый демо-режим: сгенерировать игрушечные данные и обучить top-N
demo: env
	$(PY) scripts/generate_toy_data.py
	$(PY) validation.py
	$(PY) etl.py
	$(PY) features.py
	$(PY) train_forecast.py \
	  --train data_raw/train.csv \
	  --transactions data_raw/transactions.csv \
	  --oil data_raw/oil.csv \
	  --holidays data_raw/holidays_events.csv \
	  --stores data_raw/stores.csv \
	  --models_dir models \
	  --warehouse_dir data_dw \
	  --top_n_sku 4 \
	  --top_recent_days 60 \
	  --valid_days 14

# Создать локальное окружение и установить зависимости
venv:
	@echo "Creating venv in .venv ..."
	@(python3 -m venv .venv || python -m venv .venv)
	@.venv/bin/python -m pip install -U pip setuptools wheel
	@.venv/bin/pip install -r requirements.txt
	@echo "OK: venv ready. Use: source .venv/bin/activate"

setup: venv

train_xgb: env
	$(PY) train_forecast_xgb.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --top_n_sku $${TOP_N_SKU:-50} \
	  --top_recent_days $${TOP_RECENT_DAYS:-90} \
	  --valid_days $${VALID_DAYS:-28}

train_rf: env
	$(PY) train_random_forest.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --top_n_sku $${TOP_N_SKU:-50} \
	  --top_recent_days $${TOP_RECENT_DAYS:-90} \
	  --valid_days $${VALID_DAYS:-28}

train_lstm: env
	$(PY) experiments/train_lstm.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  --top_n_sku $${TOP_N_SKU:-20} \
	  --top_recent_days $${TOP_RECENT_DAYS:-90} \
	  --valid_days $${VALID_DAYS:-28} \
	  --window $${WINDOW_SIZE:-30} \
	  --epochs $${EPOCHS:-50}

train_one_xgb: env
	@if [ -z "$(STORE)" ] || [ -z "$(FAMILY)" ]; then \
	  echo "ERROR: укажите STORE и FAMILY, например: make train_one_xgb STORE=2 FAMILY=AUTOMOTIVE"; \
	  exit 1; \
	else \
	  $(PY) scripts/train_one_pair_xgb.py \
	    --store $(STORE) \
	    --family "$(FAMILY)" \
	    --valid_days $${VALID_DAYS:-28} \
	    --models_dir $${MODELS_DIR:-models} \
	    --data_dir $${DATA_DIR:-data_raw} ; \
	fi


# Показать примеры доступных пар (store_nbr, family) из train.csv
pairs: env
	@echo "Первые 30 уникальных пар из $${DATA_DIR:-data_raw}/train.csv"; \
	$(PY) - <<-'PY'
		import os, pandas as pd
		p = os.environ.get('DATA_DIR', 'data_raw')
		try:
	    	df = pd.read_csv(f"{p}/train.csv")
	    	pairs = df[['store_nbr','family']].drop_duplicates().head(30)
	    	print(pairs.to_string(index=False))
		except Exception as e:
	    	print(f"Не удалось прочитать {p}/train.csv: {e}")
	PY

# ---- Инфраструктура: линтеры, типы, тесты ----

lint: env ## Проверка кода (без авто-фикс)
	@echo "Running ruff and black checks..."
	$(PY) -m ruff check .
	$(PY) -m black --check .

lint-fix: env ## Автоисправление линтера
	@echo "Running ruff and black with auto-fix..."
	$(PY) -m ruff check . --fix
	$(PY) -m black .

typecheck: env
	@echo "Running mypy..."
	$(PY) -m mypy . --ignore-missing-imports

# Тестирование и качество кода
test: env ## Запуск тестов
	@echo "Running tests (pytest)..."
	$(PY) -m pytest -q

test-cov: env ## Тесты с покрытием
	@echo "Running tests with coverage..."
	$(PY) -m pytest --cov=. --cov-report=html:htmlcov --cov-report=term-missing

test-fast: env ## Быстрые тесты (без медленных)
	@echo "Running fast tests..."
	$(PY) -m pytest -m "not slow" -q

test-unit: env ## Только unit тесты
	@echo "Running unit tests..."
	$(PY) -m pytest -m "unit" -q

test-integration: env ## Только integration тесты
	@echo "Running integration tests..."
	$(PY) -m pytest -m "integration" -q

cov: test-cov ## Алиас для test-cov

# Мониторинг производительности
benchmark: env ## Бенчмарк производительности моделей
	@echo "Running full benchmark..."
	$(PY) scripts/performance_monitor.py --generate-report --generate-plot

benchmark-fast: env ## Быстрый бенчмарк
	@echo "Running fast benchmark..."
	$(PY) scripts/performance_monitor.py --test-size 100

# Дополнительные команды для качества
quality: env ## Полная проверка качества
	@echo "Running full quality check..."
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

# Команды для презентации
demo-full: env ## Полное демо для презентации
	@echo "Running full demo for presentation..."
	$(MAKE) validate
	$(MAKE) etl
	$(MAKE) features
	$(MAKE) train
	$(MAKE) benchmark
	$(MAKE) api
	$(MAKE) ui

precommit-install:
	@echo "Installing pre-commit hooks..."
	@([ -x .venv/bin/pre-commit ] && .venv/bin/pre-commit install) || pre-commit install

dev-setup: venv
	@echo "Installing dev dependencies..."
	@.venv/bin/pip install -r requirements-dev.txt
	@.venv/bin/pre-commit install || true

all: validate etl features train test

# --- Repo cleanup helpers ---
.PHONY: clean-artifacts clean-repo

clean-artifacts: ## Удалить локальные артефакты (без git)
	rm -rf .pytest_cache __pycache__ tests/__pycache__ service/__pycache__
	rm -rf data_raw/* data_dw/* models/*

clean-repo: ## Убрать тяжёлые каталоги из индекса git (файлы останутся локально) — ТРЕБУЕТ CONFIRM=1
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "⚠️  Эта цель изменит индекс git (git rm --cached). Запустите: make clean-repo CONFIRM=1"; \
	  exit 1; \
	fi
	@echo "Cleaning git index from heavy artifacts..."
	- git rm -r --cached .venv .pytest_cache __pycache__ tests/__pycache__ service/__pycache__ .idea mlruns || true
	- git rm -r --cached data_raw/* data_dw/* models/* || true
	- find . -name .DS_Store -print0 | xargs -0 git rm --cached -f || true
	@echo "Done. Commit the changes: git commit -m 'chore: cleanup artifacts'"

# ---- Дополнительные цели: Optuna / SHAP / Бизнес-метрики ----

optuna: env ## Optuna-тюнинг LGBM per-SKU (STORE/FAMILY не обязательны)
	@echo "Running Optuna tuning..."
	$(PY) experiments/tune_optuna.py \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  $$( [ -n "$(STORE)" ] && echo --store $(STORE) ) \
	  $$( [ -n "$(FAMILY)" ] && echo --family "$(FAMILY)" ) \
	  --valid_days $${VALID_DAYS:-28} \
	  --n_trials $${N_TRIALS:-50}

shap: env ## SHAP отчёт важности (требует STORE и FAMILY)
	@if [ -z "$(STORE)" ] || [ -z "$(FAMILY)" ]; then \
	  echo "ERROR: укажите STORE и FAMILY, например: make shap STORE=1 FAMILY=AUTOMOTIVE"; \
	  exit 1; \
	fi
	$(PY) scripts/shap_report.py \
	  --store $(STORE) --family "$(FAMILY)" \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv \
	  $$( [ -n "$(SHAP_SAMPLE)" ] && echo --sample $(SHAP_SAMPLE) )

biz-metrics: env ## MAPE→деньги и расчёт SS/ROP (требует STORE/FAMILY/PRICE)
	@if [ -z "$(STORE)" ] || [ -z "$(FAMILY)" ] || [ -z "$(PRICE)" ]; then \
	  echo "ERROR: укажите STORE, FAMILY и PRICE, например: make biz-metrics STORE=1 FAMILY=AUTOMOTIVE PRICE=3.5"; \
	  exit 1; \
	fi
	$(PY) scripts/business_metrics.py \
	  --store $(STORE) --family "$(FAMILY)" \
	  --price $(PRICE) \
	  --margin_rate $${MARGIN_RATE:-0.25} \
	  --holding_cost $${HOLDING_COST:-0.05} \
	  --lead_time_days $${LEAD_TIME_DAYS:-2} \
	  --service_level $${SERVICE_LEVEL:-0.95}

# Отчёт по экономическому эффекту на всём датасете (по всем парам)
impact: env ## Сформировать business impact report (CSV и сводка в консоли)
	$(PY) scripts/business_impact_report.py \
	  --train $${RAW_TRAIN:-data_raw/train.csv} \
	  --metrics $${METRICS_CSV:-data_dw/metrics_per_sku.csv} \
	  --price_csv $${PRICE_CSV:-configs/prices.csv} \
	  --lead_time_days $${LEAD_TIME_DAYS:-2} \
	  --service_level $${SERVICE_LEVEL:-0.95} \
	  --tail_days $${TAIL_DAYS:-30} \
	  --valid_days $${VALID_DAYS:-28} \
	  --out_csv $${OUT_CSV:-data_dw/business_impact_report.csv}

prices-template: env ## Сгенерировать шаблон цен/маржи/хранения по family в configs/prices.csv
	$(PY) scripts/generate_prices_template.py --train $${RAW_TRAIN:-data_raw/train.csv} --out $${OUT_PRICES:-configs/prices.csv}

# Экспорт модели в ONNX (с опциональной квантизацией)
export-onnx: env ## Экспорт per-SKU модели в ONNX (требует STORE/FAMILY; QUANTIZE=1 для INT8)
	@if [ -z "$(STORE)" ] || [ -z "$(FAMILY)" ]; then \
	  echo "ERROR: укажите STORE и FAMILY, например: make export-onnx STORE=1 FAMILY=AUTOMOTIVE [QUANTIZE=1]"; \
	  exit 1; \
	fi
	$(PY) scripts/export_onnx.py --store $(STORE) --family "$(FAMILY)" $$( [ "$(QUANTIZE)" = "1" ] && echo --quantize )

# Сводный бизнес-отчёт: CSV с оценками влияния (см. scripts/business_impact_report.py)
biz-report: env ## Сводный бизнес-отчёт (использует configs/prices.csv)
	$(PY) scripts/business_impact_report.py \
	  --train $${RAW_TRAIN:-data_raw/train.csv} \
	  --metrics $${METRICS_CSV:-data_dw/metrics_per_sku.csv} \
	  --price_csv $${PRICE_CSV:-configs/prices.csv} \
	  --lead_time_days $${LEAD_TIME_DAYS:-2} \
	  --service_level $${SERVICE_LEVEL:-0.95} \
	  --tail_days $${TAIL_DAYS:-30} \
	  --valid_days $${VALID_DAYS:-28} \
	  --out_csv $${OUT_CSV:-data_dw/business_impact_report.csv}

# ---- Утилиты для подготовки материалов презентации ----

batch-train-selected: env ## Первая 5-ка + последние 5 магазинов: per-SKU LGBM + per-SKU XGB
	$(PY) scripts/batch_train_selected.py

train-quantiles-selected: env ## Первая 5-ка + последние 5 магазинов: обучить квантильные LGBM (P50/P90)
	$(PY) scripts/train_quantiles_selected.py

export-eda: env ## Сохранить EDA-графики в docs/
	$(PY) scripts/export_eda_cli.py

model-summary: env ## Сводная таблица сравнения моделей → docs/model_comparison.csv
	$(PY) scripts/model_summary_cli.py

shap-one: env ## SHAP отчёт важности (требует STORE и FAMILY)
	@if [ -z "$(STORE)" ] || [ -z "$(FAMILY)" ]; then \
	  echo "ERROR: укажите STORE и FAMILY, например: make shap-one STORE=3 FAMILY=BEVERAGES"; \
	  exit 1; \
	fi
	PYTHONPATH=. $(PY) scripts/shap_report.py \
	  --store $(STORE) --family "$(FAMILY)" \
	  --train $${RAW_DIR:-data_raw}/train.csv \
	  --transactions $${RAW_DIR:-data_raw}/transactions.csv \
	  --oil $${RAW_DIR:-data_raw}/oil.csv \
	  --holidays $${RAW_DIR:-data_raw}/holidays_events.csv \
	  --stores $${RAW_DIR:-data_raw}/stores.csv

# Полный переобучение моделей (очистка models + train LGBM/XGB per-SKU + глобальные)
retrain-all: env ## Полное переобучение всех моделей (очищает models/ кроме .gitkeep) — ТРЕБУЕТ CONFIRM=1
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "⚠️  Эта цель удалит содержимое models/. Для продолжения запустите: make retrain-all CONFIRM=1"; \
	  exit 1; \
	fi
	@echo "📦 Резервная копия моделей..."
	@ts=$$(date +%Y%m%d_%H%M%S); mkdir -p backup_models_$${ts}; cp -a models/* backup_models_$${ts} 2>/dev/null || true; echo "backup → backup_models_$${ts}"
	@echo "🧹 Очистка каталога models/ (кроме .gitkeep)"
	@find models -type f ! -name '.gitkeep' -delete || true
	@echo "▶️ Обучение per-SKU LGBM"; $(MAKE) train
	@echo "▶️ Обучение per-SKU XGB (с сохранением features.json)"; $(MAKE) train_xgb TOP_N_SKU=$${TOP_N_SKU:-50} TOP_RECENT_DAYS=$${TOP_RECENT_DAYS:-90} VALID_DAYS=$${VALID_DAYS:-28}
	@echo "▶️ Обучение глобальной CatBoost"; $(MAKE) train_global
	@echo "▶️ Обучение глобальной XGBoost"; $(MAKE) train_global_xgb
	@echo "✅ Переобучение завершено."

.PHONY: retrain-all-safe
retrain-all-safe: env ## Дообучение без очистки models/ (LGBM/XGB per-SKU + глобальные)
	@echo "▶️ Обучение per-SKU LGBM"; $(MAKE) train
	@echo "▶️ Обучение per-SKU XGB"; $(MAKE) train_xgb TOP_N_SKU=$${TOP_N_SKU:-50} TOP_RECENT_DAYS=$${TOP_RECENT_DAYS:-90} VALID_DAYS=$${VALID_DAYS:-28}
	@echo "▶️ Обучение глобальной CatBoost"; $(MAKE) train_global
	@echo "▶️ Обучение глобальной XGBoost"; $(MAKE) train_global_xgb
	@echo "✅ Готово без очистки."

# --- Презентация: сборка слайдов (требуется установленный pandoc) ---
.PHONY: slides-html slides-pdf
slides-html:
	@echo "Building HTML slides (Pandoc + revealjs via CDN)..."
	pandoc -t revealjs -s -V theme=white -V revealjs-url=https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.0.4 \
	  -o docs/slides.html docs/slides.md
	@echo "OK: docs/slides.html"

slides-pdf:
	@echo "Building PDF slides (Pandoc Beamer; requires LaTeX)..."
	@if ! command -v pdflatex >/dev/null 2>&1; then \
	  echo "pdflatex не найден. Установите LaTeX (например, MacTeX/TinyTeX) или используй make slides-html и экспортируй PDF из браузера."; \
	  exit 47; \
	fi
	pandoc -t beamer -V aspectratio=169 -o docs/slides.pdf docs/slides.md
	@echo "OK: docs/slides.pdf"
