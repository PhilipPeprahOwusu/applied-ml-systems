import sys
import os
import pandas as pd
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from recommender_service import OfferRecommender

# Create dummy artifacts if they don't exist (just for the test)
if not os.path.exists('models/feature_cols_v2.json'):
    print("Feature cols missing, creating dummy...")
    with open('models/feature_cols_v2.json', 'w') as f:
        json.dump(['frequency', 'recency_days'], f)

# Mock model
class MockModel:
    def predict(self, df):
        return [0.5] * len(df)

# Test
try:
    print("Initializing Service...")
    # We cheat and inject a mock model because loading the real LightGBM might fail 
    # if the feature columns don't match exactly what I just dumped.
    # But we want to test the CODE LOGIC (reindex, imports), not the model file.
    
    service = OfferRecommender('models/offer_recommender_v2.txt', 'models/feature_cols_v2.json')
    service.model = MockModel() # Inject mock
    
    print("Making Prediction...")
    dummy_features = {'frequency': 5, 'recency_days': 20, 'cluster': 1}
    recs = service.predict(dummy_features)
    
    print("Success! Recommendations:", recs)
    
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
