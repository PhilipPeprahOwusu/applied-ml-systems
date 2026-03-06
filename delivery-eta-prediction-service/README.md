# 🚕 NYC Delivery ETA & Fare Prediction Service

An end-to-end, production-grade Machine Learning system that predicts delivery/trip times and costs across New York City in real-time. Built with a focus on modern MLOps architecture, this system processes millions of NYC TLC taxi records and features a beautiful Next.js + Mapbox UI, automated Airflow orchestration, and a sub-50ms inference API.

## 🌟 Key Features

* **Multi-Output XGBoost Model:** Simultaneously predicts both ETA (minutes) and Cost (USD) with high accuracy (ETA MAE: 3.44 mins, Fare MAE: $2.73).
* **Automated MLOps Pipeline:** Apache Airflow DAG automates the monthly ingestion of new NYC TLC data, recalculates route features, and dynamically updates the feature store.
* **Low-Latency Feature Store:** Feast backed by Redis ensures <50ms p95 latency by serving pre-calculated `route_stats` directly to the FastAPI inference server.
* **Model Registry & Tracking:** MLflow tracks all experiments, metrics, and model artifacts, ensuring full reproducibility and seamless transitions between model versions.
* **Geospatial UI:** A stunning, light-mode Next.js frontend integrated with Mapbox GL JS and Turf.js for real-road routing animations and dynamic point-to-point estimation.
* **Observability:** Built-in Prometheus metrics and a Grafana dashboard for real-time API monitoring.

## 🏗️ Architecture

1. **Ingestion (`pipelines/ingest_data.py`):** Downloads and processes monthly Parquet files (e.g., ~3M trips) to calculate historical averages (Time, Distance, Fare) per unique route.
2. **Feature Store (`feature_repo/`):** Feast materializes these aggregated route features into a Redis online store.
3. **Training (`pipelines/train_model.py`):** Trains a `MultiOutputRegressor` (wrapping XGBoost) on historical data, logging artifacts to MLflow.
4. **Inference API (`api/main.py`):** A FastAPI service that accepts Pickup/Dropoff IDs, fetches live features from Feast/Redis, and returns dual predictions. Includes a Haversine/grid-routing fallback for unknown routes.
5. **Frontend (`frontend/`):** A React/Next.js dashboard that visualizes the predictions on an interactive map.

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose
* Node.js (v18+)

### 1. Start the Infrastructure (Airflow, MLflow, Redis, Grafana)
```bash
docker compose up airflow-init
docker compose up -d
```
* **Airflow UI:** [http://localhost:8080](http://localhost:8080) (admin / admin)
* **MLflow UI:** [http://localhost:5005](http://localhost:5005)
* **Grafana:** [http://localhost:3000](http://localhost:3000) (admin / admin)

### 2. Run the Airflow Pipeline
In the Airflow UI, trigger the `monthly_nyc_eta_pipeline` DAG using **Trigger DAG w/ config** and set the logical date to `2025-01-15` (or a known NYC TLC data release date). This will autonomously download the data, update Redis, and train the model.

### 3. Start the Inference API
```bash
make serve
```
* **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
* **Web UI:** [http://localhost:3000](http://localhost:3000) (Make sure you set your `NEXT_PUBLIC_MAPBOX_TOKEN` in `frontend/.env.local`).

## 📊 Evaluation Metrics (V1 Baseline)
* **ETA MAE:** 3.44 minutes
* **Fare MAE:** $2.73
* **ETA RMSE:** 5.37 minutes
* **Fare RMSE:** $5.81

## 🛠️ Tech Stack
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas
* **MLOps:** Apache Airflow, MLflow, Feast
* **Backend:** FastAPI, Uvicorn
* **Database/Cache:** PostgreSQL, Redis
* **Frontend:** Next.js, React, Tailwind CSS, Mapbox GL JS, Turf.js
* **Observability:** Prometheus, Grafana
* **Deployment:** Docker, Kubernetes (Manifests included for EKS/GKE)