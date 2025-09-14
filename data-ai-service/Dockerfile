# --- Этап 1: Базовый образ ---
FROM python:3.9-slim

# --- Этап 2: Установка рабочей директории ---
WORKDIR /app

# --- Этап 3: Установка зависимостей ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Этап 4: Копирование кода, данных и конфигов ---
COPY ./app /app/app
COPY ./data_input /app/data_input
COPY ./config /app/config

# --- Этап 5: Запуск скрипта ---
# Заменяем команду запуска сервера на запуск нашего скрипта
CMD ["python", "-m", "app.main"]