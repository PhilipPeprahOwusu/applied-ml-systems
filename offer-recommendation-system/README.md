# Offer Recommendation System

A production-ready two-stage machine learning pipeline for personalized offer recommendations. The system combines unsupervised clustering for fast candidate retrieval with a supervised ranking model for accurate personalization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Customer ID ──► Redis Cache ──► Stage 1: Retrieval            │
│                   (Features)      (KMeans Clustering)            │
│                                          │                       │
│                                          ▼                       │
│                                   Candidate Offers               │
│                                   (Cluster-based)                │
│                                          │                       │
│                                          ▼                       │
│                                   Stage 2: Ranking               │
│                                   (LightGBM Model)               │
│                                          │                       │
│                                          ▼                       │
│                                   Top-K Offers ──► Response      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Stage Pipeline

**Stage 1 - Retrieval (KMeans Clustering)**
- Segments customers into 10 behavioral clusters based on RFM features
- Pre-computes candidate offers for each cluster during training
- Reduces scoring space from all offers to cluster-relevant subset
- Enables sub-millisecond candidate filtering at inference time

**Stage 2 - Ranking (LightGBM)**
- Gradient boosted decision tree model for offer scoring
- Trained on historical offer interactions with redemption labels
- Scores only cluster-filtered candidates for efficiency
- Returns top-K personalized recommendations

## Features

- **RFM Feature Engineering**: Recency, Frequency, Monetary value metrics
- **Entity Resolution**: Deduplicates customer records using probabilistic matching
- **Real-time Serving**: FastAPI with Redis caching for low-latency inference
- **Experiment Tracking**: MLflow integration for model versioning
- **Automated Retraining**: Airflow DAG for scheduled pipeline execution

## Project Structure

```
offer-recommendation-system/
├── notebooks/
│   ├── generate_data/
│   │   ├── customer_records.ipynb      # Synthetic customer generation
│   │   └── interactions_transaction_data.ipynb  # Transaction & interaction data
│   ├── entity_resolution.ipynb         # Customer deduplication
│   ├── feature_engineering.ipynb       # RFM and behavioral features
│   └── recommendation_model.ipynb      # Model training with MLflow
├── pipelines/
│   ├── generate_data.py                # Data generation script
│   ├── feature_engineering.py          # Feature pipeline
│   └── train_model.py                  # Model training script
├── src/
│   ├── main.py                         # FastAPI application
│   ├── recommender_service.py          # Inference logic
│   ├── load_to_redis.py                # Cache loading utility
│   └── config.py                       # Configuration
├── airflow/
│   └── dags/
│       └── recommender_pipeline.py     # Airflow DAG
├── models/                             # Model artifacts (JSON configs)
└── requirements.txt
```

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.829 |
| Clusters | 10 |

## Installation

```bash
# Clone the repository
git clone https://github.com/PhilipPeprahOwusu/applied-ml-systems.git
cd applied-ml-systems/offer-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Pipeline

Run the notebooks in order or use the pipeline scripts:

```bash
# Generate synthetic data
python pipelines/generate_data.py

# Feature engineering
python pipelines/feature_engineering.py

# Train model
python pipelines/train_model.py
```

### Serving

```bash
# Start Redis (required for caching)
redis-server

# Load customer features to Redis
python -m src.load_to_redis

# Start API server
uvicorn src.main:app --reload
```

### API Endpoints

**Health Check**
```bash
GET /
```

**Get Recommendations**
```bash
GET /recommend/{customer_id}?top_k=3
```

Example response:
```json
{
  "customer_id": "CUST001",
  "cluster": 3,
  "candidates_scored": 12,
  "recommendations": [
    {"offer_id": "OFF_005", "offer_name": "Premium Service Package", "score": 0.89},
    {"offer_id": "OFF_012", "offer_name": "Loyalty Rewards Plus", "score": 0.76},
    {"offer_id": "OFF_008", "offer_name": "Seasonal Discount", "score": 0.71}
  ]
}
```

## Tech Stack

- **ML Framework**: LightGBM, scikit-learn
- **API**: FastAPI, Uvicorn
- **Caching**: Redis
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **Data Processing**: Pandas, NumPy
- **Entity Resolution**: recordlinkage, NetworkX

## License

MIT
