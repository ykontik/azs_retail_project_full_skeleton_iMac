# Model Card — Прогноз спроса (Retail)

## Обзор
- Задача: прогноз ежедневных продаж per‑SKU (магазин × семья товаров) для поддержки планирования запасов (SS/ROP).
- Вход: агрегированные продажные ряды, транзакции, цены на нефть, календарные и праздничные признаки, onpromotion.
- Выход: прогноз спроса (точка и квантили), метрики на holdout‑валидации.

## Данные
- Источник: Favorita‑подобный набор (train/transactions/oil/holidays_events/stores).
- Объём для демо: top‑N пар по сумме продаж за последние N дней.
- Подготовка: `etl.py` (Parquet + DuckDB), `make_features.py` (календарь, лаги, скользящие, onpromotion, holidays, oil, transactions).

## Модели
- Per‑SKU LightGBM (цели: MAE/Tweedie/Poisson), опционально квантильные модели (P50/P90).
- Глобальные: CatBoost, XGBoost (для сравнения и быстрых демо).
- Нейросетевой baseline: per‑SKU LSTM (окно 30, 2 слоя, hidden 128) — для демонстрации иерархического ансамбля.
- Хранение: `models/{store}__{family}.joblib` (+ `__qXX.joblib` для квантилей), `models/{store}__{family}__lstm.pt`.

## Тренировка и Валидация
- Holdout‑валидация по последним `valid_days`.
- Бейзлайны: Naive lag‑7, MA(7), отчёт о приросте качества vs baseline.
- Метрики: MAE, MAPE, CV‑MAE/CV‑MAPE (если включён time‑series CV).

## Метрики Качества (пример)
- Средний MAE/ MAPE сохраняются в `data_dw/summary_metrics.txt` и `data_dw/metrics_per_sku.csv`.
- RandomForest baseline → `data_dw/summary_metrics_random_forest.txt`, `data_dw/metrics_random_forest.csv`.
- LSTM baseline → `data_dw/summary_metrics_lstm.txt`, `data_dw/metrics_lstm.csv` (+ `data_dw/lstm_preds/*.csv` для UI).
- Глобальные модели: `data_dw/metrics_global_*.json`.
  
| Модель             | MAE | MAPE | Примечание |
|--------------------|-----|------|------------|
| LGBM per-SKU       | 548.1 | 6.61% | Основной продовый вариант |
| RandomForest per-SKU | 570.7 | 9.18% | Interpretable baseline |
| LSTM per-SKU       | 1542.2 | 1419.94% | Демонстрационный baseline (требует донастройки) |
| CatBoost global    | 56.3 | 33.12% | Единая модель |
| XGBoost global     | 51.0 | 52.14% | Единая модель |

## Ограничения и Допущения
- Качество чувствительно к уровню агрегации и полноте фич (цены, промо, внешние факторы).
- Квантильные модели приближённо оценивают дисперсию спроса; для точных интервалов нужна калибровка.
- LSTM baseline требует дополнительного тюнинга (MAPE >> 100%); используем как витрину возможностей и для дальнейшего ансамблирования.
- Для категориальных фич в per‑SKU LightGBM используется встроенная поддержка с указанием категориальных колонок.

## Использование в Продукте
- API: `POST /predict_demand`, `POST /predict_demand_quantiles`, `POST /reorder_point` (SS/ROP).
- UI: Streamlit‑дашборд для интерактивной демонстрации, лидеров по качеству, ручного инференса.
- Экспорт: `scripts/export_onnx.py` (ONNX + опциональная INT8‑квантизация) для ускорения инференса.

## Мониторинг
- Приложение экспонирует Prometheus‑метрики: `/metrics-prom`.
- План расширения: мониторинг дрейфа данных/ошибки, алерты, отчёт по SLA.

## Воспроизводимость
- Фиксированный `random_state` во всех скриптах обучения.
- Команды в `Makefile`, шаблон `.env.example`, Docker Compose для изолированного запуска.
