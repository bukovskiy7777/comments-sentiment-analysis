import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Определение путей
BASE_DIR = "/home/oleksandr/apps/comments-sentiment-analysis"
MODEL_PATH = os.path.join(BASE_DIR, "models/sentiment_models_bundle.pkl")

app = FastAPI(title="YouTube Sentiment Analysis API", version="2026.01")

# Модель входных данных
class CommentRequest(BaseModel):
    text: str

# Модель выходных данных
class PredictionResponse(BaseModel):
    label: str
    score: float
    model_info: str

# Загрузка модели при старте сервера
model_bundle = None

@app.on_event("startup")
def load_model():
    global model_bundle
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}. Сначала запустите обучение в Airflow.")
    model_bundle = joblib.load(MODEL_PATH)
    print("✅ Модели (классификатор и регрессор) успешно загружены.")

@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": model_bundle is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: CommentRequest):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст комментария не может быть пустым")

    # Получение предсказаний
    label = model_bundle['classifier'].predict([text])[0]
    score = model_bundle['regressor'].predict([text])[0]
    metadata = model_bundle.get('metadata', {})

    return PredictionResponse(
        label=label,
        score=round(float(score), 4),
        model_info=f"{metadata.get('model_name', 'TF-IDF')} (trained at {metadata.get('trained_at', 'unknown')})"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)