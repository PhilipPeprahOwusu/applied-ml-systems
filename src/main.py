from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from .recommender_service import OfferRecommender
from .config import (
    MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH,
    REDIS_HOST, REDIS_PORT, REDIS_DB
)
import redis
import json
import os


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
    retrieval_enabled: bool


app = FastAPI(
    title="Offer Recommender API",
    description="Two-stage recommendation: KMeans retrieval → LightGBM ranking",
    version="1.0.0"
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_LIST_PATH):
    raise RuntimeError(f"Model files not found at {MODEL_PATH}")

recommender = OfferRecommender(MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH)

try:
    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    cache.ping()
    REDIS_CONNECTED = True
except redis.ConnectionError:
    cache = None
    REDIS_CONNECTED = False


@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        message="Offer Recommender API is running",
        model_loaded=True,
        redis_connected=REDIS_CONNECTED,
        retrieval_enabled=recommender.cluster_candidates is not None
    )


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(
    customer_id: str,
    top_k: int = Query(default=3, ge=1, le=8, description="Number of offers (1-8)")
):
    """
    Get top K offer recommendations for a customer.

    Two-stage pipeline:
    1. Retrieval: Customer cluster → filter candidate offers
    2. Ranking: LightGBM scores candidates → return top K
    """
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
