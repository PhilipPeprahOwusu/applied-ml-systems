import os
import argparse
import urllib.request
import pandas as pd

REQUIRED_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'trip_distance',
    'PULocationID',
    'DOLocationID',
    'fare_amount',
]

ZONE_URL  = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
TAXI_URL  = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONE_PATH = "data/raw/taxi_zone_lookup.csv"

def _safe_download(url: str, dest: str) -> None:
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"Resource not yet available: {url}")
        raise

def download_nyc_data(month_str: str) -> tuple[str, str]:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(ZONE_PATH):
        _safe_download(ZONE_URL, ZONE_PATH)

    raw_path = f"data/raw/yellow_{month_str}.parquet"
    if not os.path.exists(raw_path):
        _safe_download(TAXI_URL.format(month=month_str), raw_path)

    return raw_path, ZONE_PATH

def process_data(raw_path: str, zone_path: str, month_str: str) -> str:
    print(f"Processing {raw_path} for Route-Based Estimates...")
    df    = pd.read_parquet(raw_path)
    zones = pd.read_csv(zone_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")

    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'fare_amount'])

    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Calculate Time
    df['trip_duration_minutes'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60.0

    # Filter realistic trips
    df = df[
        (df['trip_duration_minutes'].between(1.0, 180.0)) &
        (df['trip_distance'] >= 0.1) &
        (df['fare_amount'] >= 2.0) & (df['fare_amount'] < 500.0)
    ].copy()

    # Create Route ID
    df['route_id'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)

    df['hour']            = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week']     = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour']    = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['event_timestamp'] = df['tpep_pickup_datetime']

    # Keep specific targets for the model
    events_df = df[[
        'route_id', 'PULocationID', 'DOLocationID', 
        'event_timestamp', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'trip_duration_minutes', 'trip_distance', 'fare_amount'
    ]]

    processed_path = f"data/processed/yellow_{month_str}.parquet"
    events_df.to_parquet(processed_path, index=False)

    # Generate Feature Store tables
    print("Generating Route-Level Feature Store tables...")
    os.makedirs("feature_repo/data", exist_ok=True)

    # Historical Route Averages
    route_stats = events_df.groupby('route_id').agg(
        route_avg_duration=('trip_duration_minutes', 'mean'),
        route_avg_distance=('trip_distance', 'mean'),
        route_avg_fare=('fare_amount', 'mean'),
        route_trip_count=('route_id', 'count')
    ).reset_index()
    
    route_stats['event_timestamp'] = pd.to_datetime('2024-01-01 00:00:00', utc=True)
    route_stats.to_parquet("feature_repo/data/route_features.parquet", index=False)

    print(f"Ingestion Summary: Processed {len(events_df):,} trips across {events_df['route_id'].nunique():,} routes.")

    return processed_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default="2025-01", help="YYYY-MM")
    args = parser.parse_args()

    raw_file, zone_file = download_nyc_data(args.month)
    process_data(raw_file, zone_file, args.month)