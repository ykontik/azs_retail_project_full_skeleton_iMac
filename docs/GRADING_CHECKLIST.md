# Чеклист сдачи (на «отлично»)

Ниже — соответствие требований курса возможной проверке проекта. Колонки: есть/сделано → где в репо → что улучшить (если нужно).

## Конвейер и постановка
- Постановка задачи: есть → `PRESENTATION_CHECKLIST.md`, `docs/MODEL_CARD.md` → сверить KPI/бизнес‑метрики с цифрами из вашего запуска.
- Источники данных: есть → `README.md` (быстрый старт), `PRESENTATION_CHECKLIST.md` → добавить 1–2 иллюстрации EDA (png) в `docs/`.
- EDA/визуализации: базово есть → `notebooks/experiments/demo.ipynb` → экспортировать 2–3 ключевых графика в `docs/` и сослаться в слайдах.

## Подготовка данных
- ETL/валидация: есть → `validation.py`, `etl.py`, `features.py`, `make_features.py` → кратко описать шаги в `docs/ARCHITECTURE.md` (2–3 предложения).
- Аугментации/преобразования: есть → `make_features.py` → добавить аргументы в README к ключевым параметрам.

## Модели и обучение
- Несколько моделей: есть → LightGBM/XGBoost/CatBoost (`train_*`) → в `docs/MODEL_CARD.md` свести результаты сравнения в таблицу.
- Тюнинг гиперпараметров: есть → `experiments/tune_optuna.py` → зафиксировать лучшие параметры/метрики в `data_dw/summary_metrics.txt` и перенести сводку в `MODEL_CARD.md`.
- Временная валидация: есть (holdout + CV‑параметры) → показать в презентации схему окон.
- Интерпретация: есть → `scripts/shap_report.py` → приложить 1 png в `docs/` и ссылку в слайды.

## Метрики и бизнес‑связка
- Технические метрики: есть → `data_dw/metrics_per_sku.csv`, `summary_metrics.txt` → в Streamlit уже отражаются.
- Бизнес‑метрики/перевод в деньги: есть → `scripts/business_metrics.py`, `scripts/business_impact_report.py`, цели `make impact`/`biz-metrics` → вставить 1 слайд с итогами.

## Доставка/прод
- API: есть → FastAPI (`service/app.py`) с кастомным Swagger, health‑пробами и ключом → добавить короткий скрин `/docs` в слайды.
- UI: есть → Streamlit (`ui/dashboard.py`) → показать 2 демо‑сценария.
- Docker: есть → `Dockerfile`, `docker-compose.yml` → добавить шаги запуска в README (Windows/Linux заметки).
- Мониторинг: есть → `/metrics-prom`,  (`Makefile`) → пара скриншотов ( runs, метрики API).

## Плюсы к «отлично» (рекомендации)
- Квантизация/ускорение: есть → `scripts/export_onnx.py` (ONNX + INT8) → добавить цифры ускорения в README.
- CI: есть → `.github/workflows/ci.yml` → добавить бейджи в README (опционально).
- Тесты: есть → `tests/` (unit/integration/smoke) → запустить `make test-cov` и вставить % покрытия в README.
- Колаб‑ноутбук: рекомендуется → добавить `notebooks/colab_quickstart.ipynb` (установка + `make demo`).

## Чистота репозитория
- Исключены тяжёлые артефакты: настроено → `.gitignore` (data_raw, data_dw, models, mlruns, mlflow.db, .venv, backups) → периодически `make clean-repo`.
- IDE/OS‑мусор: настроено → `.idea`, `.DS_Store` → `make clean-repo` удаляет из индекса.
- Нота: в `models/` оставить только `.gitkeep` в git.

## Последняя миля перед сдачей
- Прогнать: `make demo-full` или последовательность из README.
- Сохранить артефакты для презентации: 2–3 графика EDA, SHAP, скрин Swagger/.
- Проверить API с ключом: `X-API-Key` (можно отключить `DISABLE_AUTH=true`).
- Записать 8–10 минут демо‑видео по плану слайдов из `docs/slides.md`.
