import os

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Artifacts
MODEL_PATH = os.path.join(MODELS_DIR, "offer_recommender_v2.txt")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_cols_v2.json")
CLUSTER_CANDIDATES_PATH = os.path.join(MODELS_DIR, "cluster_candidates.json")

# Data
CUSTOMER_FEATURES_PATH = os.path.join(DATA_DIR, "customer_features_clustered.csv")

# Infrastructure
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
