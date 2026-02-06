import logging
import os
import matplotlib.pyplot as plt
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, mean_squared_error, f1_score, 
    mean_absolute_error, ConfusionMatrixDisplay
)

@task
def train_sentiment_model(EXPERIMENT_NAME, ds=None, **context):
    # yesterday_ds = context['macros'].ds_add(ds, -1)
    yesterday_ds = ds
    pg_hook = PostgresHook(postgres_conn_id='postgres_ubuntu')
    
    # 1. Загрузка данных
    query = """
        SELECT c.text_display, s.label, s.score 
        FROM youtube_comments c
        JOIN comment_sentiment s ON c.comment_id = s.comment_id
    """
    df = pg_hook.get_pandas_df(query)

    if len(df) < 50:  # Grid Search требует больше данных для кросс-валидации
        logging.info("Недостаточно данных для запуска Grid Search.")
        return
    
    X = df['text_display']
    Y = df[['label', 'score']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y['label']
    )

    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_grid_search_{yesterday_ds}"):
        
        # --- ШАГ 1: Grid Search для Классификации ---
        # Мы ищем лучшие параметры TF-IDF и Логистической регрессии одновременно
        base_class_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        param_grid = {
            'tfidf__max_features': [2500, 5000, 10000, 15000],
            'tfidf__ngram_range': [(1, 1), (1, 2)], # Униграммы и биграммы
            'tfidf__min_df': [2, 5],
            'clf__C': [0.1, 1.0, 10.0] # Сила регуляризации
        }

        logging.info("Запуск GridSearchCV для классификатора...")
        grid_search = GridSearchCV(
            base_class_pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train['label'])
        
        class_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Метрики классификации
        y_pred_class = class_pipeline.predict(X_test)
        acc_test = accuracy_score(y_test['label'], y_pred_class)
        acc_train = accuracy_score(y_train['label'], class_pipeline.predict(X_train))
        f1 = f1_score(y_test['label'], y_pred_class, average='weighted')

        # --- ШАГ 2: Регрессия с лучшими параметрами TF-IDF ---
        # Используем те же параметры TF-IDF, которые победили в классификации
        reg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                max_features=best_params['tfidf__max_features'],
                ngram_range=best_params['tfidf__ngram_range'],
                min_df=best_params['tfidf__min_df']
            )),
            ('reg', Ridge())
        ])
        
        reg_pipeline.fit(X_train, y_train['score'])
        y_pred_reg = reg_pipeline.predict(X_test)
        
        mse_test = mean_squared_error(y_test['score'], y_pred_reg)
        mse_train = mean_squared_error(y_train['score'], reg_pipeline.predict(X_train))
        mae = mean_absolute_error(y_test['score'], y_pred_reg)

        # --- Логирование в MLflow ---
        mlflow.log_params(best_params)
        mlflow.log_param("grid_search_status", "completed")
        
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("acc_gap", abs(acc_test - acc_train))
        mlflow.log_metric("f1_score", f1)
        
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("mse_test", mse_test)
        mlflow.log_metric("mae_score", mae)

        # Логирование Confusion Matrix как артефакт
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test['label'], y_pred_class, 
            display_labels=class_pipeline.classes_, 
            cmap=plt.cm.Blues, ax=ax
        )
        plt.title(f"Confusion Matrix {yesterday_ds}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Сохранение моделей в MLflow
        mlflow.sklearn.log_model(class_pipeline, "classifier_model")
        mlflow.sklearn.log_model(reg_pipeline, "regressor_model")
        
        # Сохранение локального бандла для FastAPI
        model_path = "/home/oleksandr/apps/comments-sentiment-analysis/models/sentiment_models_bundle.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_pack = {
            'classifier': class_pipeline,
            'regressor': reg_pipeline,
            'metadata': {
                'trained_at': yesterday_ds,
                'model_name': 'TF-IDF GridSearch Optimized',
                'best_params': best_params
            }
        }
        joblib.dump(model_pack, model_path)
        
        logging.info(f"Grid Search завершен. Лучшая Accuracy: {acc_test:.4f}. Params: {best_params}")

        save_model_data(class_pipeline)



def save_model_data(class_pipeline):

    # Загружаем модель (если запускаете отдельно)
    # model_pack = joblib.load("/home/oleksandr/apps/comments-sentiment-analysis/models/sentiment_models_bundle.pkl")
    # class_pipeline = model_pack['classifier']

    # 1. Извлекаем TF-IDF и Классификатор из Pipeline
    tfidf = class_pipeline.named_steps['tfidf']
    clf = class_pipeline.named_steps['clf']

    # 2. Получаем Словарь и Веса IDF
    # Словарь: слово -> индекс
    # IDF: индекс -> вес
    feature_names = tfidf.get_feature_names_out()
    idf_weights = tfidf.idf_

    df_tfidf = pd.DataFrame({
        'word': feature_names,
        'idf_weight': idf_weights
    }).sort_values(by='idf_weight', ascending=False)

    # Сохраняем словарь с IDF
    df_tfidf.to_csv("/home/oleksandr/apps/comments-sentiment-analysis/models/models_data/model_debug_tfidf.csv", index=False)

    # 3. Получаем Коэффициенты Логистической Регрессии
    # Для каждого класса (Positive, Negative, Neutral) модель хранит веса слов
    for i, label in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        df_coefs = pd.DataFrame({
            'word': feature_names,
            'coefficient': coefs
        }).sort_values(by='coefficient', ascending=False)
        
        # Сохраняем веса слов для конкретного класса
        df_coefs.to_csv(f"/home/oleksandr/apps/comments-sentiment-analysis/models/models_data/model_debug_coefs_{label}.csv", index=False)

    # 4. Прочие параметры (C, ngram_range и т.д.)
    params = class_pipeline.get_params()
    with open("/home/oleksandr/apps/comments-sentiment-analysis/models/models_data/model_debug_params.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print("📊 Данные для отладки сохранены в CSV файлы.")
