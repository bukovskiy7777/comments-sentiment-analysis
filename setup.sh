#!/bin/bash

# Создание виртуального окружения
python3 -m venv sentiment_venv

# Активация окружения
source sentiment_venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка Airflow
pip install apache-airflow==3.1.5

# Установка библиотек для YouTube API и NLP (Transformers)
pip install google-api-python-client transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu/

# Установка библиотек для ML (Scikit-learn для TF-IDF, MLflow)
pip install scikit-learn mlflow pandas numpy joblib

# Установка FastAPI и веб-сервера
pip install fastapi uvicorn

# Установка драйвера для PostgreSQL
pip install psycopg2-binary
