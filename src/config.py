"""Configuration for the recommendation system."""
import os

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Model artifacts
MODEL_PATH = os.path.join(MODELS_DIR, "recommendation_model.lgb")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_cols.json")
CLUSTER_CANDIDATES_PATH = os.path.join(MODELS_DIR, "cluster_candidates.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans.pkl")

# Data files
CUSTOMER_FEATURES_PATH = os.path.join(DATA_DIR, "customer_features_clustered.csv")

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
