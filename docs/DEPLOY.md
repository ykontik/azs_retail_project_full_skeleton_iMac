
# Деплой и бенчмарк

## Docker / Compose
- Подготовить `configs/.env`
- Запуск: `make docker` или `docker compose up --build`
- Сервисы: API (8000), UI (8501)

## Бенчмарк
- Полный: `make benchmark`
- Быстрый: `make benchmark-fast`

