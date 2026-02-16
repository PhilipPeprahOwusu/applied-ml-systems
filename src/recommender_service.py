import lightgbm as lgb
import pandas as pd
import json
import os
from src.offers import OFFERS, OFFER_TYPES

class OfferRecommender:
    """
    Serves recommendations using a two-stage process:
    1. Retrieval: Filtering candidates based on user cluster (optional)
    2. Ranking: Scoring candidates using a LightGBM model
    """

    def __init__(self, model_path: str, feature_path: str, cluster_candidates_path: str = None):
        self.model = lgb.Booster(model_file=model_path)
        
        with open(feature_path, 'r') as f:
            self.feature_cols = json.load(f)

        self.cluster_candidates = None
        if cluster_candidates_path and os.path.exists(cluster_candidates_path):
            with open(cluster_candidates_path, 'r') as f:
                self.cluster_candidates = json.load(f)

        # Optimize lookup
        self.offers_by_id = {o['offer_id']: o for o in OFFERS}

    def _get_candidates(self, customer_features: dict) -> list:
        """Filter offers by customer cluster if candidate map exists."""
        cluster = customer_features.get('cluster')

        if self.cluster_candidates and cluster is not None:
            cluster_key = str(int(cluster))
            if cluster_key in self.cluster_candidates:
                candidate_ids = self.cluster_candidates[cluster_key]
                # Only return valid offers that exist in our definition
                return [self.offers_by_id[oid] for oid in candidate_ids if oid in self.offers_by_id]

        # Fallback to all offers
        return OFFERS

    def _score_candidates(self, customer_features: dict, candidates: list) -> list:
        """Score candidates using the LightGBM model."""
        if not candidates:
            return []

        # vectorized dataframe creation is faster than list of dicts for larger batches,
        # but for small candidate sets (N<50), list of dicts is fine.
        rows = []
        for offer in candidates:
            row = customer_features.copy()
            row['offer_value'] = offer['value']
            
            # One-hot encode offer type on the fly
            for o_type in OFFER_TYPES:
                row[f'offer_type_{o_type}'] = 1 if offer['type'] == o_type else 0

            row['offer_id'] = offer['offer_id']
            row['offer_name'] = offer['name']
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure all model features exist, filling missing with 0
        df = df.reindex(columns=self.feature_cols, fill_value=0)

        scores = self.model.predict(df)

        results = [
            {
                'offer_id': candidates[i]['offer_id'], 
                'offer_name': candidates[i]['name'], 
                'score': float(scores[i])
            }
            for i in range(len(scores))
        ]
        
        # Sort by score descending
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def predict(self, customer_features: dict, top_k: int = 3) -> list:
        candidates = self._get_candidates(customer_features)
        ranked = self._score_candidates(customer_features, candidates)
        return ranked[:top_k]