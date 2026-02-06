from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.timetables.interval import CronDataIntervalTimetable
from airflow.timetables.trigger import CronTriggerTimetable
from datetime import datetime

def print_context_vars(**kwargs):
    # Извлекаем переменные из контекста
    data_interval_start = kwargs['data_interval_start']
    data_interval_end = kwargs['data_interval_end']
    logical_date = kwargs['logical_date']
    ds = kwargs['ds']
    
    print("-" * 30)
    print(f"data_interval_start: {data_interval_start}")
    print(f"data_interval_end:   {data_interval_end}")
    print(f"logical_date:        {logical_date}")
    print(f"ds (строка):         {ds}")
    print("-" * 30)

# 1. Классический интервальный подход
with DAG(
    dag_id="example_interval_timetable",
    start_date=datetime(2026, 2, 1),
    # Запуск 2 января в 00:00 будет обрабатывать данные за 1 января
    schedule=CronDataIntervalTimetable("30 8 * * *", timezone="UTC"),
    catchup=True,
    tags=['youtube', 'sentiment_analysis'],
) as dag1:
    task1 = PythonOperator(
        task_id="task",
        python_callable=print_context_vars
    )

# 2. Подход "Триггер" (как обычный Cron)
with DAG(
    dag_id="example_trigger_timetable",
    start_date=datetime(2026, 2, 1),
    # Запуск 2 января в 00:00 будет иметь logical_date = 2 января
    schedule=CronTriggerTimetable("30 8 * * *", timezone="UTC"),
    catchup=True,
    tags=['youtube', 'sentiment_analysis'],
) as dag2:
    task2 = PythonOperator(
        task_id="task",
        python_callable=print_context_vars
    )