# Модели и обучение

## Per‑SKU LightGBM
- Скрипт: `train_forecast.py` (цели: l1/tweedie/poisson; квантильные модели P50/P90 при `--quantiles`)
- Запуск: `make train` (использует `configs/train.yaml` и RAW_DIR)

## Глобальные модели
- CatBoost: `make train_global` → `models/global_catboost.cbm`
- XGBoost: `make train_global_xgb` → `models/global_xgboost.joblib`

## Per‑SKU XGBoost
- `make train_xgb` или `python scripts/train_one_pair_xgb.py --store ... --family ...`

## Экспорт ONNX
- `make export-onnx STORE=1 FAMILY=AUTOMOTIVE [QUANTIZE=1]`

