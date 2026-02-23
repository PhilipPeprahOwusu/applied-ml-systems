import lightgbm as lgb
import pandas as pd
import json
import os
from .offers import OFFERS, OFFER_TYPES

class OfferRecommender:
    """
    Serves recommendations via a two-stage retrieval-ranking pipeline.
    """

    def __init__(self, model_path: str, feature_path: str, cluster_candidates_path: str = None):
        self.model = lgb.Booster(model_file=model_path)
        with open(feature_path, 'r') as f:
            self.feature_cols = json.load(f)

        self.cluster_candidates = None
        if cluster_candidates_path and os.path.exists(cluster_candidates_path):
            with open(cluster_candidates_path, 'r') as f:
                self.cluster_candidates = json.load(f)

        self.offers_by_id = {o['offer_id']: o for o in OFFERS}

    def _get_candidates(self, customer_features: dict) -> list:
        cluster = customer_features.get('cluster')
        if self.cluster_candidates and cluster is not None:
            cluster_key = str(int(cluster))
            if cluster_key in self.cluster_candidates:
                candidate_ids = self.cluster_candidates[cluster_key]
                return [self.offers_by_id[oid] for oid in candidate_ids if oid in self.offers_by_id]
        return OFFERS

    def _score_candidates(self, customer_features: dict, candidates: list) -> list:
        if not candidates:
            return []

        rows = []
        for offer in candidates:
            row = customer_features.copy()
            row['offer_value'] = offer['value']
            
            for o_type in OFFER_TYPES:
                col = f'offer_{o_type}'
                val = 1 if offer['type'] == o_type else 0
                row[col] = val
                
                # Interaction Features
                if 'frequency' in row:
                    row[f'freq_x_{col}'] = row['frequency'] * val
                if 'recency_days' in row:
                    row[f'recency_x_{col}'] = row['recency_days'] * val
                if 'monetary_avg' in row:
                    row[f'monetary_x_{col}'] = row['monetary_avg'] * val

            row['offer_id'] = offer['offer_id']
            row['offer_name'] = offer['name']
            rows.append(row)

        df = pd.DataFrame(rows).reindex(columns=self.feature_cols, fill_value=0)
        scores = self.model.predict(df)

        results = [
            {
                'offer_id': candidates[i]['offer_id'], 
                'offer_name': candidates[i]['name'], 
                'score': float(scores[i])
            }
            for i in range(len(scores))
        ]
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def predict(self, customer_features: dict, top_k: int = 3) -> list:
        candidates = self._get_candidates(customer_features)
        ranked = self._score_candidates(customer_features, candidates)
        return ranked[:top_k]