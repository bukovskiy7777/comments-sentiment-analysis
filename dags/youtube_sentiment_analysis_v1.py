from airflow import DAG
from pendulum import datetime
from utils.load_comments_tasks import (
    fetch_top_videos_from_youtube, 
    save_videos_to_postgres,
    fetch_comments_for_videos,
    save_comments_to_postgres
)
from utils.sentiment_processing_tasks import (
    analyze_comments_sentiment,
    save_sentiment_to_postgres
)
from utils.train_model_tasks import train_sentiment_model

#import os
import yaml


#config_path = os.path.join(r'/home/oleksandr/apps/comments-sentiment-analysis/config/youtube_params.yaml')
CONFIG_PATH = '/home/oleksandr/apps/comments-sentiment-analysis/config/youtube_params.yaml'

def get_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)['topic_1']

with DAG(
    dag_id='youtube_sentiment_analysis_v1',
    start_date=datetime(2026, 1, 2, tz="Europe/Rome"), 
    schedule="30 8 * * *",  # Ежедневно в 08:30 утра
    catchup=True,
    max_active_runs=1,
    tags=['youtube', 'sentiment_analysis'],
) as dag:

    # Получаем параметры
    config = get_config()

    # 1. Получаем список видео
    video_list = fetch_top_videos_from_youtube(
        YOUTUBE_API_KEY=config['YOUTUBE_API_KEY'],
        SEARCH_QUERY=config['SEARCH_QUERY'],
        VIDEOS_LIMIT=config['VIDEOS_LIMIT'], 
        RELEVANCE_LANGUAGE=config['RELEVANCE_LANGUAGE'],
        REGION_CODE=config['REGION_CODE']
    )
    
    # 2. Сохраняем видео (параллельно запускаем сбор комментариев)
    save_videos_step = save_videos_to_postgres(video_list)
    
    # 3. Получаем комментарии по списку видео
    comments_data = fetch_comments_for_videos(
        YOUTUBE_API_KEY=config['YOUTUBE_API_KEY'],
        videos=video_list,
        COMMENTS_LIMIT=config['COMMENTS_LIMIT']
    )
    
    # 4. Сохраняем комментарии в БД
    save_comments_step = save_comments_to_postgres(comments_data)

    # 5. Анализируем полученные комментарии
    sentiment_data = analyze_comments_sentiment(comments_data)

    # 6. Сохраняем результаты анализа
    save_sentiment_step = save_sentiment_to_postgres(sentiment_data)

    # 7. Обучение TF-IDF модели на базе накопленных данных
    training_step = train_sentiment_model(EXPERIMENT_NAME=config['EXPERIMENT_NAME'])

    # --- УСТАНОВКА ЯВНЫХ ЗАВИСИМОСТЕЙ (Database Constraints) ---
    # Сначала видео должны быть в базе, прежде чем мы запишем комментарии к ним (Foreign Key)
    save_videos_step >> save_comments_step >> save_sentiment_step >> training_step
    # Сначала комментарии должны быть в базе, прежде чем мы запишем их тональность (Foreign Key)
    # Зависимость: обучаем только после того, как новые данные за сегодня сохранены