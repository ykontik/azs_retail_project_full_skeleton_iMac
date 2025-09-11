# Troubleshooting

- Установка зависимостей: сначала `pip install -U pip setuptools wheel`, затем `pip install -r requirements*.txt`.
- Apple Silicon (M1/M2): используйте python3.11 и wheels для xgboost==2.1.x; LightGBM без GPU.
- Streamlit не видит API: укажите `API_URL` в окружении или в поле на странице.
- Долгое обучение/мало памяти: уменьшите `TOP_N_SKU`, `TOP_RECENT_DAYS`, `VALID_DAYS` или используйте `make demo`.
- Docker образ тяжёлый: вынесите UI/SHAP/Optuna в отдельный образ; multi‑stage Dockerfile.

