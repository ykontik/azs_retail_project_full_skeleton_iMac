FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/usr/local/bin:$PATH" \
    UVICORN_WORKERS=2 \
    PORT=8000

WORKDIR /app

# Системные зависимости (минимум)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости проекта
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Копируем исходники
COPY . .

# Значения по умолчанию для API
ENV DISABLE_AUTH=true \
    MODELS_DIR=/app/models \
    WAREHOUSE_DIR=/app/data_dw

EXPOSE 8000 8501

# По умолчанию — запуск API. Для UI будет переопределено командой в docker-compose
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
