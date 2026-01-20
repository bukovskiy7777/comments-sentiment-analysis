from airflow import DAG
from pendulum import datetime
from utils.tfidf_train_task import train_sentiment_model

#import os
import yaml


#config_path = os.path.join(r'/home/oleksandr/apps/comments-sentiment-analysis/config/youtube_params.yaml')
CONFIG_PATH = '/home/oleksandr/apps/comments-sentiment-analysis/config/youtube_params.yaml'

def get_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)['topic_1']

with DAG(
    dag_id='tfidf_train_dag',
    start_date=datetime(2026, 1, 2, tz="Europe/Rome"), 
    schedule="30 8 * * *",  # Ежедневно в 08:30 утра
    catchup=True,
    max_active_runs=1,
    tags=['youtube', 'sentiment_analysis'],
) as dag:

    # Получаем параметры
    config = get_config()

    # 7. Обучение TF-IDF модели на базе накопленных данных
    training_step = train_sentiment_model(EXPERIMENT_NAME=config['EXPERIMENT_NAME'])

    # --- УСТАНОВКА ЯВНЫХ ЗАВИСИМОСТЕЙ (Database Constraints) ---
    # Сначала видео должны быть в базе, прежде чем мы запишем комментарии к ним (Foreign Key)
    training_step
    # Сначала комментарии должны быть в базе, прежде чем мы запишем их тональность (Foreign Key)
    # Зависимость: обучаем только после того, как новые данные за сегодня сохранены