"""Load customer features to AWS Valkey for API serving."""
import pandas as pd
import redis
import json
import sys

# AWS Valkey configuration
VALKEY_HOST = "offer-recommender-cache-dxl76y.serverless.use1.cache.amazonaws.com"
VALKEY_PORT = 6379

BATCH_SIZE = 500
DATA_PATH = "data/customer_features_clustered.csv"


def load_data():
    """Load customer features from CSV to AWS Valkey."""
    print(f"Connecting to Valkey at {VALKEY_HOST}:{VALKEY_PORT} (SSL enabled)...")

    r = redis.Redis(
        host=VALKEY_HOST,
        port=VALKEY_PORT,
        db=0,
        decode_responses=True,
        ssl=True
    )

    # Test connection
    try:
        r.ping()
        print("Connected to Valkey successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print(f"Loading {len(df):,} customers to Valkey...")
    pipe = r.pipeline()
    loaded = 0

    for idx, row in df.iterrows():
        pipe.set(f"cust:{row['customer_id']}", json.dumps(row.to_dict()))
        loaded += 1

        if loaded % BATCH_SIZE == 0:
            pipe.execute()
            pipe = r.pipeline()
            print(f"  Loaded {loaded:,} customers...")

    # Execute remaining
    pipe.execute()
    print(f"Successfully loaded {loaded:,} customers to Valkey!")

    # Verify a sample
    sample_key = f"cust:{df.iloc[0]['customer_id']}"
    sample = r.get(sample_key)
    if sample:
        print(f"\nVerification - Sample customer: {sample_key}")
        print(f"Data: {sample[:100]}...")


if __name__ == "__main__":
    load_data()
