"""Load customer features to Redis for API serving."""
import pandas as pd
import redis
import json

from .config import CUSTOMER_FEATURES_PATH, REDIS_HOST, REDIS_PORT, REDIS_DB

BATCH_SIZE = 1000


def load_data():
    """Load customer features from CSV to Redis."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)

    pipe = r.pipeline()
    for idx, row in df.iterrows():
        pipe.set(f"cust:{row['customer_id']}", json.dumps(row.to_dict()))

        if idx % BATCH_SIZE == 0 and idx > 0:
            pipe.execute()

    pipe.execute()
    print(f"Loaded {len(df):,} customers to Redis")


if __name__ == "__main__":
    load_data()
