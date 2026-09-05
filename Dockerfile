# ============================================================================
# GEO-VEGA // КОСМОХАКАТОН 2026: МОНИТОРИНГ ВЕГЕТАЦИИ С/Х ПОЛЕЙ
# Контейнеризированная среда для воспроизводимого запуска решения и инференса
# ============================================================================

# Базовый образ Linux с Python 3.10
FROM python:3.10-slim

LABEL maintainer="GEO-VEGA Team // Космохакатон 2026"
LABEL description="Веб-сервис мониторинга вегетационной динамики и детекции аномалий с/х территорий"

# Отключение буферизации вывода для мгновенного отображения логов в консоли
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PRODUCTION=1

# Установка системных зависимостей:
# - libgomp1: необходим для многопоточного OpenMP инференса ансамбля LightGBM/CatBoost
# - curl: используется для healthcheck проверки доступности контейнера
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование спецификации библиотек и установка зависимостей Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копирование исходного кода, весов моделей, конфигураций и скриптов
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY run_server.py predict_submission.py predict.py train.py detect_anomalies.py setup_gee.py ./
COPY submission*.csv pyproject.toml environment.yml ./

# Экспорт сетевого порта веб-интерфейса и REST API
EXPOSE 8000

# Автоматическая проверка состояния сервиса (Healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/batch-status || exit 1

# Запуск веб-сервиса GEO-VEGA в продакшен-режиме (без фонового опросчика файлов reload)
CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000", "--no-reload"]
