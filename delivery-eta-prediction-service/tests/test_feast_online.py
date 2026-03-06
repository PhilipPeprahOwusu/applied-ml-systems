from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Pick a driver ID that we know exists (from our hash logic)
# Let's just try to fetch features for a few random keys
entity_rows = [
    {"driver_id": "D_00000000"}, # This might not exist, we'll see
    {"PULocationID": 161}        # Midtown Manhattan
]

features = [
    "driver_stats:driver_mean_time",
    "driver_stats:driver_on_time_rate",
    "zone_stats:zone_hour_mean_time"
]

print("🔍 Querying Feast Online Store (Redis)...")
response = store.get_online_features(
    features=features,
    entity_rows=[{"driver_id": "D_0007b542", "PULocationID": 161}] # Using a likely valid hash start
).to_dict()

print("\n--- Online Features ---")
for feature, value in response.items():
    print(f"{feature}: {value}")