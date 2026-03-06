# NYC Delivery ETA and Fare Prediction Service

A production-grade MLOps system designed to provide real-time estimates for trip duration and total fare across New York City. This project demonstrates a complete machine learning lifecycle, from automated data ingestion and feature engineering to low-latency inference and geospatial visualization.

## Core Features

* **Multi-Output Inference:** Utilizes a custom XGBoost model to simultaneously predict trip duration (minutes) and total cost (USD) based on origin-destination pairs.
* **Automated Data Orchestration:** Employs Apache Airflow to automate monthly data ingestion from the NYC TLC dataset, processing over 3 million records per cycle.
* **Real-Time Feature Serving:** Implements Feast with a Redis backend to provide sub-50ms p95 latency for historical route features.
* **Experiment Tracking:** Uses MLflow for model versioning, artifact storage, and performance monitoring across training runs.
* **Geospatial Interface:** A Next.js frontend integrated with Mapbox GL JS for interactive route selection and road-accurate routing animations.
* **System Observability:** Configured with Prometheus and Grafana to monitor API health and model performance metrics.

## System Architecture

1. **Data Ingestion (pipelines/ingest_data.py):** Processes raw Parquet files from NYC TLC to calculate historical averages for duration, distance, and fare per unique route.
2. **Feature Store (feature_repo/):** Feast materializes aggregated route statistics into a Redis online store for low-latency retrieval.
3. **Model Training (pipelines/train_model.py):** Trains a MultiOutputRegressor using XGBoost, logging model parameters and metrics to the MLflow Registry.
4. **Inference API (api/main.py):** A FastAPI service that integrates with Feast and MLflow to serve real-time predictions. Includes a Haversine-based fallback mechanism for new routes.
5. **Frontend (frontend/):** A React-based dashboard providing a professional interface for trip estimation and mapping.

## Evaluation Metrics (Baseline V1)

Performance metrics calculated on the January 2025 NYC Taxi dataset:

* **Duration Mean Absolute Error (ETA MAE):** 3.44 minutes
* **Fare Mean Absolute Error (Fare MAE):** $2.73
* **Duration Root Mean Squared Error (ETA RMSE):** 5.37 minutes
* **Fare Root Mean Squared Error (Fare RMSE):** $5.81

## Setup and Installation

### Prerequisites
* Docker and Docker Compose
* Node.js (version 18 or higher)

### 1. Infrastructure Deployment
Initialize the Airflow database and start the containerized services (Airflow, MLflow, Redis, Grafana):
```bash
docker compose up airflow-init
docker compose up -d
```

### 2. Automated Pipeline Execution
Access the Airflow UI at http://localhost:8080. Trigger the 'monthly_nyc_eta_pipeline' DAG using the 'Trigger DAG w/ config' option. Set the logical date to '2025-01-15' to simulate a production data run.

### 3. Inference API Startup
Start the FastAPI server:
```bash
make serve
```
Interactive API documentation is available at http://localhost:8000/docs.

### 4. Frontend Startup
Install dependencies and start the Next.js development server:
```bash
cd frontend
npm install
npm run dev
```
The interface is accessible at http://localhost:3000. Ensure the Mapbox API token is configured in 'frontend/.env.local'.

## Technical Stack

* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, GeoPandas
* **MLOps/Orchestration:** Apache Airflow, MLflow, Feast
* **Backend:** FastAPI, Uvicorn, Redis, PostgreSQL
* **Frontend:** Next.js, React, Tailwind CSS, Mapbox GL JS, Turf.js
* **Observability:** Prometheus, Grafana
* **Deployment:** Docker, Kubernetes (Kustomize manifests included)
