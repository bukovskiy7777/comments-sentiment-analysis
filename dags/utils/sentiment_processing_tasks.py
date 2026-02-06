from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from transformers import pipeline

@task
def analyze_comments_sentiment(comments, ds=None, **context):
    # yesterday_ds = context['macros'].ds_add(ds, -1)
    yesterday_ds = ds

    if not comments:
        return []

    # Инициализация модели (при первом запуске она скачается ~1GB)
    # Используем CPU или GPU, если доступно
    device = -1
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, device=device)

    results = []
    # Извлекаем только тексты для массовой обработки (batch processing)
    texts = [c['text_display'][:512] for c in comments] # Ограничение 512 токенов для модели
    
    # Запуск анализа
    predictions = sentiment_task(texts, batch_size=16)

    for i, pred in enumerate(predictions):
        results.append({
            'comment_id': comments[i]['comment_id'],
            'label': pred['label'],
            'score': pred['score'],
            'model_name': model_path,
            'processed_date': yesterday_ds
        })
    
    return results

@task
def save_sentiment_to_postgres(sentiment_results):
    if not sentiment_results:
        return

    pg_hook = PostgresHook(postgres_conn_id='postgres_ubuntu')
    fields = ['comment_id', 'label', 'score', 'model_name', 'processed_date']
    rows = [tuple(r[f] for f in fields) for r in sentiment_results]
    
    pg_hook.insert_rows(
        table='comment_sentiment',
        rows=rows,
        target_fields=fields,
        replace=True, # Если анализ перезапсукается, обновляем данные
        replace_index=['comment_id']
    )