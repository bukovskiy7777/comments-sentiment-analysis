import logging
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from googleapiclient.discovery import build


@task
def fetch_top_videos_from_youtube(YOUTUBE_API_KEY, SEARCH_QUERY, VIDEOS_LIMIT, RELEVANCE_LANGUAGE, REGION_CODE, ds=None):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    time_start, time_end = f"{ds}T00:00:00Z", f"{ds}T23:59:59Z"

    # Поиск видео
    video_items = []
    # Делаем два запроса для разных категорий длительности
    for duration in ['medium', 'long']:
        request = youtube.search().list(
            q=SEARCH_QUERY, 
            part="snippet", 
            maxResults=VIDEOS_LIMIT,
            order="viewCount", 
            publishedAfter=time_start,
            publishedBefore=time_end, 
            type="video",
            videoDuration=duration,
            relevanceLanguage=RELEVANCE_LANGUAGE, 
            regionCode=REGION_CODE
        )
        response = request.execute()
        video_items.extend(response.get('items', []))

    # Собираем ID каналов
    channel_ids = list(set([item['snippet']['channelId'] for item in video_items]))

    if not channel_ids:
        return []

    # Получаем данные о странах этих каналов # Можно передать до 50 ID за один раз через запятую
    channels_response = youtube.channels().list(
        part='snippet',
        id=','.join(channel_ids)
    ).execute()

    # Создаем словарь {channel_id: country_code}
    channel_countries = {
        item['id']: item.get('snippet', {}).get('country', '').upper()
        for item in channels_response.get('items', [])
    }

    # Фильтруем исходный список видео
    filtered_videos = []
    for video in video_items:
        c_id = video['snippet']['channelId']
        if channel_countries.get(c_id) == REGION_CODE.upper():
            filtered_videos.append({
                'video_id': video['id']['videoId'],
                'title': video['snippet']['title'],
                'published_at': video['snippet']['publishedAt'],
                'channel_id': video['snippet']['channelId'],
                'channel_title': video['snippet']['channelTitle'],
                'search_query': SEARCH_QUERY,
                'processed_date': ds,
                'channel_country': REGION_CODE
            })

    return filtered_videos


@task
def save_videos_to_postgres(videos):
    if not videos:
        logging.info("Видео не найдены.")
        return

    pg_hook = PostgresHook(postgres_conn_id='postgres_ubuntu')
    fields = ['video_id', 'title', 'published_at', 'channel_id', 'channel_title', 'search_query', 'processed_date', 'channel_country']
    rows = [tuple(v[f] for f in fields) for v in videos]
    
    pg_hook.insert_rows(table='youtube_videos', rows=rows, target_fields=fields)
    logging.info(f"Успешно сохранено {len(rows)} видео.")


@task
def fetch_comments_for_videos(YOUTUBE_API_KEY, videos, COMMENTS_LIMIT, ds=None):
    if not videos:
        logging.info("Нет видео для получения комментариев.")
        return []

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    all_comments = []

    for video in videos:
        video_id = video['video_id']
        logging.info(f"Загрузка комментариев для видео: {video_id}")
        
        try:
            # YouTube API отдает максимум 100 за раз. 
            # Для 500 комментариев нам нужно пройтись по страницам (pagination).
            comments_collected = 0
            next_page_token = None
            
            while comments_collected < COMMENTS_LIMIT:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    order="relevance" # Топ-комментарии
                )
                response = request.execute()

                for item in response.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'comment_id': item['id'],
                        'video_id': video_id,
                        'text_display': snippet['textDisplay'],
                        'author_name': snippet['authorDisplayName'],
                        'published_at': snippet['publishedAt'],
                        'like_count': snippet['likeCount'],
                        'processed_date': ds
                    })
                
                comments_collected += len(response.get('items', []))
                next_page_token = response.get('nextPageToken')
                
                if not next_page_token:
                    break

        except Exception as e:
            logging.warning(f"Не удалось получить комментарии для {video_id}: {e}")
            continue

    return all_comments


@task
def save_comments_to_postgres(comments):
    if not comments:
        logging.info("Нет комментариев для записи.")
        return

    pg_hook = PostgresHook(postgres_conn_id='postgres_ubuntu')
    fields = ['comment_id', 'video_id', 'text_display', 'author_name', 'published_at', 'like_count', 'processed_date']
    
    rows = [tuple(c[f] for f in fields) for c in comments]
    
    pg_hook.insert_rows(
        table='youtube_comments', 
        rows=rows, 
        target_fields=fields,
        commit_every=1000 # Оптимизация для больших объемов данных
    )
    logging.info(f"Успешно сохранено {len(rows)} комментариев.")
