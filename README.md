# Applied Machine Learning Systems

A portfolio of production-grade Machine Learning and MLOps projects demonstrating end-to-end system design, model training, feature serving, and scalable deployment.

## Projects

### 1. Delivery ETA & Fare Prediction Service
[View Project](./delivery-eta-prediction-service)

A real-time MLOps architecture designed to estimate trip duration and cost across New York City.
* **Architecture:** Multi-output XGBoost model served via FastAPI.
* **Feature Store:** Feast with Redis for low-latency (<50ms) online feature retrieval.
* **Orchestration:** Apache Airflow automating data ingestion and model retraining.
* **Observability:** MLflow for model tracking; Prometheus and Grafana for system metrics.
* **Frontend:** Next.js and Mapbox GL JS for interactive geospatial visualization.

### 2. Offer Recommendation System
[View Project](./offer-recommendation-system)

An API-driven recommendation engine engineered to deliver context-aware, personalized offers to users.
* **Deployment:** Containerized API designed for robust and scalable cloud deployment.
* **CI/CD:** Automated GitHub Actions workflows for continuous integration and AWS deployment.

## Repository Structure

This repository acts as a monorepo. Each project is contained within its own directory and includes dedicated infrastructure configurations, dependency requirements, and GitHub Actions CI/CD workflows.
