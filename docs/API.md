# API Документация — AZS + Retail Forecast

## Обзор

API для прогнозирования спроса в розничной сети. Поддерживает точечные прогнозы, пакетные запросы и квантильные прогнозы. Предоставляет сервисные эндпоинты и метрики для мониторинга.

Базовый URL: `http://localhost:8000`

## Аутентификация

По умолчанию аутентификация отключена. Для включения установите `DISABLE_AUTH=false` и задайте `API_KEY` (передаётся в заголовке `X-API-Key`).

## Эндпоинты

### 1) Health

- GET `/health` → `{ "status": "ok" }`
- GET `/live` → `{ "status": "alive" }`
- GET `/ready` → `{ "ready": true|false, "metrics_csv": bool, "has_models": bool }`
- GET `/version` → `{ "version": "1.0.0", "models_dir": "models", "warehouse_dir": "data_dw", "disable_auth": true }`

### 2) Модели

- GET `/models`
  - Ответ: `{ "count": N, "models": [{"store_nbr": 1, "family": "AUTOMOTIVE", "path": "models/1__AUTOMOTIVE.joblib"}, ...] }`

- GET `/feature_names?store_nbr=1&family=AUTOMOTIVE`
  - Ответ при наличии модели: `{ "store_nbr": 1, "family": "AUTOMOTIVE", "feature_names": ["year", "month", ...] }`
  - 404, если модель не найдена

- GET `/quantiles_available?store_nbr=1&family=AUTOMOTIVE`
  - Ответ: `{ "store_nbr": 1, "family": "AUTOMOTIVE", "quantiles": [0.5, 0.9] }` (если квантильные модели сохранены)

### 3) Прогнозирование

- POST `/predict_demand`
  - Тело: `{"store_nbr": int, "family": str, "features": {"year": 2024, ...}}`
  - Ответ: `{ "store_nbr": 1, "family": "AUTOMOTIVE", "pred_qty": 6.4, "used_features": ["..."] }`

- POST `/predict_bulk`
  - Тело: массив объектов `PredictRequest`, пример: `[{"store_nbr":1, "family":"AUTOMOTIVE", "features":{...}}]`
  - Ответ: `{ "results": [ { "store_nbr": 1, "family": "AUTOMOTIVE", "pred_qty": 6.4, "used_features": ["..."] } ] }`

- POST `/predict_demand_quantiles?qs=0.5,0.9`
  - Тело: `PredictRequest`
  - Ответ: `{ "store_nbr": 1, "family": "AUTOMOTIVE", "quantiles": {"0.5": 6.0, "0.9": 9.0}, "used_features": ["..."] }`

- POST `/reorder_point`
  - Тело: `{ "store_nbr": 1, "family": "AUTOMOTIVE", "features": {...}, "lead_time_days": 2, "service_level": 0.95 }`
  - Ответ: `{ "daily_mean": 10.0, "sigma_daily": 2.5, "lead_time_days": 2, "service_level_z": 1.6449, "safety_stock": 5.2, "reorder_point": 25.2, ... }`

### 4) Метрики

- GET `/metrics`
  - Источник: `data_dw/metrics_per_sku.csv`
  - Ответ при наличии файла: `{ "columns": ["store_nbr", "family", "MAE", "MAPE_%"], "rows": [ { ... } ] }`
  - 404, если файл отсутствует

- GET `/metrics-prom` — Prometheus‑метрики (text/plain)

## Коды ошибок

400 (Bad Request), 401 (Unauthorized), 404 (Not Found), 422 (Validation Error), 500 (Internal Server Error)

## Примеры

Прогноз:
```bash
curl -X POST "http://localhost:8000/predict_demand" \
  -H "Content-Type: application/json" \
  -d '{
    "store_nbr": 1,
    "family": "AUTOMOTIVE",
    "features": {"year": 2024, "month": 1, "day": 15}
  }'
```

Квантили:
```bash
curl -X POST "http://localhost:8000/predict_demand_quantiles?qs=0.5,0.9" \
  -H "Content-Type: application/json" \
  -d '{"store_nbr":1, "family":"AUTOMOTIVE", "features": {"year":2024}}'
```

