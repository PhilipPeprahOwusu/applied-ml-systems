from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from .recommender_service import OfferRecommender
from .config import (
    MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH,
    REDIS_HOST, REDIS_PORT, REDIS_DB, CUSTOMER_FEATURES_PATH
)
import redis
import json
import os
import pandas as pd


class OfferRecommendation(BaseModel):
    offer_id: str
    offer_name: str
    score: float


class RecommendationResponse(BaseModel):
    customer_id: str
    cluster: Optional[int] = None
    candidates_scored: int
    recommendations: List[OfferRecommendation]


class HealthResponse(BaseModel):
    message: str
    model_loaded: bool
    redis_connected: bool
    data_loaded: bool
    retrieval_enabled: bool


class LoadDataResponse(BaseModel):
    message: str
    customers_loaded: int


# Global state
cache = None
REDIS_CONNECTED = False
DATA_LOADED = False
recommender = None


def connect_cache():
    """Connect to Redis/Valkey cache."""
    global cache, REDIS_CONNECTED
    try:
        use_ssl = REDIS_HOST.endswith('.cache.amazonaws.com')
        cache = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            ssl=use_ssl,
            socket_connect_timeout=5
        )
        cache.ping()
        REDIS_CONNECTED = True
        return True
    except Exception:
        cache = None
        REDIS_CONNECTED = False
        return False


def load_data_to_cache():
    """Load customer features from CSV to cache."""
    global DATA_LOADED
    if not cache:
        raise Exception("Cache not connected")
    if not os.path.exists(CUSTOMER_FEATURES_PATH):
        raise Exception(f"Data file not found: {CUSTOMER_FEATURES_PATH}")

    # Check if data already loaded
    if cache.exists("cust:0003QF4K5R"):
        DATA_LOADED = True
        return -1

    df = pd.read_csv(CUSTOMER_FEATURES_PATH)

    # Load one by one (required for Redis Cluster mode)
    for idx, row in df.iterrows():
        cache.set(f"cust:{row['customer_id']}", json.dumps(row.to_dict()))

    DATA_LOADED = True
    return len(df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    global recommender

    # Load model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_LIST_PATH):
        raise RuntimeError(f"Model files not found at {MODEL_PATH}")
    recommender = OfferRecommender(MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH)

    # Connect to cache (non-blocking, will retry on requests if fails)
    connect_cache()

    yield


app = FastAPI(
    title="Offer Recommender API",
    description="Two-stage recommendation: KMeans retrieval → LightGBM ranking",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    global DATA_LOADED
    if REDIS_CONNECTED and cache and not DATA_LOADED:
        try:
            DATA_LOADED = cache.exists("cust:0003QF4K5R")
        except Exception:
            pass
    return HealthResponse(
        message="Offer Recommender API is running",
        model_loaded=recommender is not None,
        redis_connected=REDIS_CONNECTED,
        data_loaded=DATA_LOADED,
        retrieval_enabled=recommender.cluster_candidates is not None if recommender else False
    )


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(
    customer_id: str,
    top_k: int = Query(default=3, ge=1, le=8, description="Number of offers (1-8)")
):
    """Get top K offer recommendations for a customer."""
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache service unavailable")

    customer_data = cache.get(f"cust:{customer_id}")
    if not customer_data:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    customer_features = json.loads(customer_data)
    recommendations = recommender.predict(customer_features, top_k=top_k)

    cluster = customer_features.get('cluster')
    if recommender.cluster_candidates and cluster is not None:
        candidates_scored = len(recommender.cluster_candidates.get(str(int(cluster)), recommender.offers))
    else:
        candidates_scored = len(recommender.offers)

    return RecommendationResponse(
        customer_id=customer_id,
        cluster=int(cluster) if cluster is not None else None,
        candidates_scored=candidates_scored,
        recommendations=[OfferRecommendation(**rec) for rec in recommendations]
    )


@app.post("/admin/load-data", response_model=LoadDataResponse)
def load_customer_data():
    """Load customer features from CSV to cache."""
    global REDIS_CONNECTED
    if not REDIS_CONNECTED:
        if not connect_cache():
            raise HTTPException(status_code=503, detail="Cannot connect to cache")

    try:
        count = load_data_to_cache()
        if count == -1:
            return LoadDataResponse(message="Data already loaded", customers_loaded=0)
        return LoadDataResponse(message="Customer data loaded successfully", customers_loaded=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
