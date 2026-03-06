from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from feast import FeatureStore
from datetime import datetime
import os
import time
from prometheus_client import make_asgi_app, Counter, Histogram

app = FastAPI(title="NYC Trip Estimation Service")

# --- CORS Configuration for React Frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your exact React domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = FeatureStore(repo_path="feature_repo")

# --- Prometheus Metrics Setup ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
ETA_PREDICTION = Histogram("eta_prediction_minutes", "Distribution of predicted ETAs")
FARE_PREDICTION = Histogram("fare_prediction_dollars", "Distribution of predicted Fares")
FEAST_RETRIEVAL_LATENCY = Histogram("feast_retrieval_latency_seconds", "Latency of fetching features from Feast")

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    if request.url.path == "/predict":
        REQUEST_LATENCY.labels(endpoint="/predict").observe(process_time)
        
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    return response
# --------------------------------

# Load model using MLflow Model Registry
MODEL_NAME = "delivery-eta-version-2"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        model_uri = f"models:/{MODEL_NAME}/{latest_version_info.version}"
        print(f"Loading model from Registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model from registry. Error: {e}")
        model = None


import json

# Load zones for coordinate fallback
with open("frontend/src/app/nyc_zones.json", "r") as f:
    ZONES_DATA = json.load(f)
ZONE_COORDS = {z["id"]: {"lat": z["lat"], "lng": z["lng"]} for z in ZONES_DATA}

import math
def haversine(lat1, lon1, lat2, lon2):
    R = 3959.87433 # Radius of Earth in miles
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class PredictionRequest(BaseModel):
    pickup_location_id: int
    dropoff_location_id: int

@app.post("/predict")
def predict_trip(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or found")

    try:
        now = datetime.now()
        current_hour = now.hour
        current_day_of_week = now.weekday()
        is_weekend = 1 if current_day_of_week >= 5 else 0
        is_rush_hour = 1 if current_hour in [7, 8, 9, 17, 18, 19] else 0

        route_id = f"{request.pickup_location_id}_{request.dropoff_location_id}"

        # 1. Fetch Route Features from Feast
        feast_start = time.time()
        feature_vector = store.get_online_features(
            features=[
                "route_stats:route_avg_duration",
                "route_stats:route_avg_distance",
                "route_stats:route_avg_fare",
            ],
            entity_rows=[{"route_id": route_id}],
        ).to_dict()
        FEAST_RETRIEVAL_LATENCY.observe(time.time() - feast_start)

        # 2. Dynamic Fallback for missing routes
        avg_duration = feature_vector["route_avg_duration"][0]
        avg_distance = feature_vector["route_avg_distance"][0]
        avg_fare = feature_vector["route_avg_fare"][0]

        if avg_duration is None or avg_distance is None or avg_fare is None:
            # We don't have historical data for this route. Calculate dynamically!
            p_coords = ZONE_COORDS.get(request.pickup_location_id)
            d_coords = ZONE_COORDS.get(request.dropoff_location_id)
            
            if p_coords and d_coords:
                # Calculate straight line distance and add 30% for city grid routing
                calculated_dist = haversine(p_coords["lat"], p_coords["lng"], d_coords["lat"], d_coords["lng"]) * 1.3
                avg_distance = max(0.5, calculated_dist) # minimum half mile
                
                # Assume average NYC traffic speed of 10mph (6 mins per mile)
                avg_duration = avg_distance * 6.0
                
                # Standard NYC fare formula: $3 base + $2.50 per mile + $0.50 per minute
                avg_fare = 3.0 + (avg_distance * 2.50) + (avg_duration * 0.50)
            else:
                # Ultimate fallback if we can't even find the coordinates
                avg_duration, avg_distance, avg_fare = 15.0, 3.0, 18.0

        # 3. Prepare input for the model
        input_data = pd.DataFrame({
            "hour": [int(current_hour)],                              
            "day_of_week": [int(current_day_of_week)],    
            "is_weekend": [is_weekend],
            "is_rush_hour": [is_rush_hour],           
            "route_avg_duration": [float(avg_duration)],
            "route_avg_distance": [float(avg_distance)],
            "route_avg_fare": [float(avg_fare)],
        })

        # 4. Predict
        predictions = model.predict(input_data)
        eta_min = predictions[0][0]
        fare_amt = predictions[0][1]

        # Log metrics
        ETA_PREDICTION.observe(float(eta_min))
        FARE_PREDICTION.observe(float(fare_amt))

        return {
            "pickup_id": request.pickup_location_id,
            "dropoff_id": request.dropoff_location_id,
            "estimated_time_minutes": round(float(eta_min), 2),
            "estimated_cost_usd": round(float(fare_amt), 2),
            "historical_route_stats": {
                "avg_duration": round(float(avg_duration), 2),
                "avg_distance_miles": round(float(avg_distance), 2),
                "avg_fare": round(float(avg_fare), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
