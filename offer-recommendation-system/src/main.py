from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import redis
import json
import os
import pandas as pd
import logging

from .recommender_service import OfferRecommender
from .offers import OFFERS
from .config import (
    MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH,
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, CUSTOMER_FEATURES_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    status: str

# Global state
cache = None
REDIS_CONNECTED = False
DATA_LOADED = False
recommender = None

def connect_cache():
    global cache, REDIS_CONNECTED
    try:
        use_ssl = REDIS_HOST.endswith('.cache.amazonaws.com')
        cache = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            ssl=use_ssl,
            socket_connect_timeout=5
        )
        cache.ping()
        REDIS_CONNECTED = True
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        cache = None
        REDIS_CONNECTED = False
        return False

def load_data_to_cache_task():
    global DATA_LOADED
    if not cache:
        return

    try:
        if not os.path.exists(CUSTOMER_FEATURES_PATH):
            logger.error(f"Data file missing: {CUSTOMER_FEATURES_PATH}")
            return

        df = pd.read_csv(CUSTOMER_FEATURES_PATH)
        pipe = cache.pipeline(transaction=False)
        batch_size = 5000
        
        for idx, row in df.iterrows():
            pipe.set(f"cust:{row['customer_id']}", json.dumps(row.to_dict()))
            if (idx + 1) % batch_size == 0:
                pipe.execute()
                pipe = cache.pipeline(transaction=False)

        pipe.execute()
        DATA_LOADED = True
        logger.info(f"Loaded {len(df)} profiles to cache")
    except Exception as e:
        logger.error(f"Cache ingestion error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_LIST_PATH):
        raise RuntimeError(f"Artifacts missing: {MODEL_PATH}")
    
    recommender = OfferRecommender(MODEL_PATH, FEATURE_LIST_PATH, CLUSTER_CANDIDATES_PATH)
    connect_cache()
    yield

app = FastAPI(
    title="Offer Recommender API",
    description="Two-stage recommendation engine",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=HealthResponse)
def health_check():
    global DATA_LOADED
    if REDIS_CONNECTED and cache and not DATA_LOADED:
        try:
            DATA_LOADED = cache.exists("cust:0003QF4K5R")
        except Exception:
            pass
    return HealthResponse(
        message="Service Operational",
        model_loaded=recommender is not None,
        redis_connected=REDIS_CONNECTED,
        data_loaded=DATA_LOADED,
        retrieval_enabled=recommender.cluster_candidates is not None if recommender else False
    )

@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(
    customer_id: str,
    top_k: int = Query(default=3, ge=1, le=8)
):
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache unavailable")

    try:
        customer_data = cache.get(f"cust:{customer_id}")
    except Exception as e:
        raise HTTPException(status_code=503, detail="Cache retrieval error")

    if not customer_data:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_features = json.loads(customer_data)
    recommendations = recommender.predict(customer_features, top_k=top_k)
    cluster = customer_features.get('cluster')
    
    candidates_scored = len(recommender.cluster_candidates.get(str(int(cluster)), OFFERS)) if (recommender.cluster_candidates and cluster is not None) else len(OFFERS)

    return RecommendationResponse(
        customer_id=customer_id,
        cluster=int(cluster) if cluster is not None else None,
        candidates_scored=candidates_scored,
        recommendations=[OfferRecommendation(**rec) for rec in recommendations]
    )

@app.post("/admin/load-data", response_model=LoadDataResponse)
def load_customer_data(background_tasks: BackgroundTasks):
    if not REDIS_CONNECTED and not connect_cache():
        raise HTTPException(status_code=503, detail="Cache connection failed")

    background_tasks.add_task(load_data_to_cache_task)
    return LoadDataResponse(message="Ingestion task initiated", status="processing")
