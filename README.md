# YouTube Comments Sentiment Analysis Pipeline 🚀

An end-to-end data processing pipeline: from automated YouTube comment collection to deploying a trained model via FastAPI.

## 📋 Project Overview

This project is an ETL pipeline and ML service designed to:
1. **Collect** the top 10 or 20 videos daily based on specific keywords using the YouTube Data API v3.
2. **Extract** up to 500 top-level comments for each video.
3. **Analyze** sentiment using the SOTA `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` model (Hugging Face Transformers).
4. **Store** structured data in a PostgreSQL database.
5. **Train** a lightweight model (TF-IDF + Logistic Regression) on accumulated historical data for fast inference.
6. **Track** model metrics, parameters, and versions using MLflow.
7. **Serve** the trained model via a REST API (FastAPI).

---

## 🛠 Tech Stack

*   **Orchestration:** Apache Airflow 3.1.5
*   **Database:** PostgreSQL
*   **NLP & ML:** Hugging Face Transformers, PyTorch (CPU-optimized), Scikit-learn
*   **MLOps:** MLflow
*   **API:** FastAPI, Uvicorn
*   **Environment:** WSL 2 (Ubuntu 22.04)

---

## 📂 Project Structure

```text
comments-sentiment-analysis/
├── airflow/              # Airflow configuration and metadata DB
├── config/
│   └── youtube_params.yaml # API keys and search parameters
├── dags/
│   └── youtube_sentiment_analysis_v1.py # Airflow DAG definition
│   └── utils             # Business logic (API calls, ML, DB operations)
│       └── load_comments_tasks.py
│       └── sentiment_processing_tasks.py
│       └── train_model_tasks.py
├── models/               # Serialized .pkl model files
├── venv/                 # Python virtual environment
├── api.py                # FastAPI server script
├── setup.sh              # Automatic installation script
└── README.md             # Project documentation
```

---

## 🚀 Quick Start
1. Environment Setup
Ensure you have WSL 2 and PostgreSQL installed. Run the installation script:
```text
bash
./setup.sh
```

2. Configuration
Create a `config/youtube_params.yaml` file and add your credentials:
```text
yaml
topic_1:
  YOUTUBE_API_KEY: "YOUR_API_KEY_HERE"
  SEARCH_QUERY: "machine learning"
  VIDEOS_LIMIT: 20
  COMMENTS_LIMIT: 500
  RELEVANCE_LANGUAGE: "en"
  REGION_CODE: "US"
  EXPERIMENT_NAME: "machine learning__sentiment_tfidf"
```

3. Database Initialization
Create the required tables in PostgreSQL (using DBeaver or psql):
- youtube_videos
- youtube_comments
- comment_sentiment

---

## ⚙️ Running the Components
To run the full project, start the following services in separate WSL terminals:
A. Airflow Scheduler & Webserver
```text
bash
export AIRFLOW_HOME=$(pwd)/airflow
source venv/bin/activate
airflow scheduler & airflow webserver --port 8080
```
UI available at: http://localhost:8080

B. MLflow Tracking Server
```text
bash
source venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```
UI available at: http://localhost:5000

C. FastAPI Service
```text
bash
source venv/bin/activate
python3 api.py
```
Swagger Documentation: http://localhost:8000/docs

---

## 📊 Monitoring & Metrics
During the daily TF-IDF model training, the following metrics are logged in MLflow:

**Classification**: Accuracy, F1-Score, Confusion Matrix.

**Regression**: MSE (Train/Test), MAE.

This allows for monitoring the quality of the "fast" TF-IDF model relative to the "heavy" XLM-RoBERTa teacher model.

---

## 📝 License
This project is for educational purposes and social data analysis.

---

Author: Oleksandr

Updated: January 2026
