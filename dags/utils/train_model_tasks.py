import logging
import os
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, mean_absolute_error

@task
def train_sentiment_model(EXPERIMENT_NAME, ds=None):
    pg_hook = PostgresHook(postgres_conn_id='postgres_ubuntu')
    
    # 1. Загрузка данных (связываем текст комментария с его меткой и скором)
    query = """
        SELECT c.text_display, s.label, s.score 
        FROM youtube_comments c
        JOIN comment_sentiment s ON c.comment_id = s.comment_id
    """
    df = pg_hook.get_pandas_df(query)

    max_features = 5000
    
    if len(df) < 20:  # Минимальный порог данных для обучения
        logging.info("Недостаточно данных для обучения модели.")
        return
    
    # Подготовка данных
    # Для классификации (positive, neutral, negative)
    # Для регрессии (уверенность модели)
    X = df['text_display']
    Y = df[['label', 'score']]

    # Разделение данных: 80% на обучение, 20% на тест
    # Используем random_state для воспроизводимости результатов
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        Y, 
        test_size=0.2, 
        random_state=42,
        stratify=Y['label'] # Чтобы пропорции классов (pos/neg) были одинаковы в обеих частях
    )

    y_train_class = y_train['label']
    y_test_class = y_test['label']

    y_train_reg = y_train['score']   
    y_test_reg = y_test['score']

    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000") # Или путь к локальной папке
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_{ds}"):
        # --- Часть 1: Классификация (Label) ---
        class_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        class_pipeline.fit(X_train, y_train_class)
        y_pred_class = class_pipeline.predict(X_test)
        # Считаем метрики на обеих частях
        acc_test = accuracy_score(y_test_class, y_pred_class)
        acc_train = accuracy_score(y_train_class, class_pipeline.predict(X_train))
        f1 = f1_score(y_test_class, y_pred_class, average='weighted')
        
        # --- Часть 2: Регрессия (Score) ---
        reg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('reg', Ridge())
        ])
        
        reg_pipeline.fit(X_train, y_train_reg)
        y_pred_reg = reg_pipeline.predict(X_test)
        # Считаем метрики на обеих частях
        mse_test = mean_squared_error(y_test_reg, y_pred_reg)
        mse_train = mean_squared_error(y_train_reg, reg_pipeline.predict(X_train))

        mae = mean_absolute_error(y_test_reg, y_pred_reg)

        # Логирование в MLflow
        mlflow.log_param("model_type", "tfidf_logistic_ridge")
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("mse_test", mse_test)
        # Также полезно логировать разницу (Overfitting Ratio)
        mlflow.log_metric("acc_gap", abs(acc_test - acc_train))
        mlflow.log_metric("mse_gap", abs(mse_test - mse_train))

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("mae_score", mae)
        
        # Сохранение моделей
        mlflow.sklearn.log_model(class_pipeline, "classifier_model")
        mlflow.sklearn.log_model(reg_pipeline, "regressor_model")
        
        # Сохраняем классификатор локально для быстрого доступа из FastAPI
        model_path = "/home/oleksandr/apps/comments-sentiment-analysis/models/sentiment_models_bundle.pkl"
        #model_path = "/home/oleksandr/apps/comments-sentiment-analysis/models/sentiment_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_pack = {
            'classifier': class_pipeline,
            'regressor': reg_pipeline,
            'metadata': {
                'trained_at': ds,
                'model_name': 'TF-IDF + Logistic/Ridge'
            }
        }
        #joblib.dump(class_pipeline, model_path)
        joblib.dump(model_pack, model_path)
        
        logging.info(f"Модель обучена. Accuracy: {acc_test:.4f}, MSE: {mse_test:.4f}")