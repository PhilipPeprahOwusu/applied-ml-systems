import lightgbm as lgb
import pandas as pd
import json
import os


class OfferRecommender:
    """Two-stage recommender: KMeans retrieval → LightGBM ranking."""

    OFFERS = [
        {'offer_id': 'OFF001', 'name': 'Free Oil Change', 'type': 'Free Service', 'value': 49.99},
        {'offer_id': 'OFF002', 'name': '20% Off Any Service', 'type': 'Discount', 'value': 0.20},
        {'offer_id': 'OFF003', 'name': 'Loyalty Points 2X', 'type': 'Points', 'value': 2.0},
        {'offer_id': 'OFF004', 'name': '$25 Off Next Visit', 'type': 'Credit', 'value': 25.00},
        {'offer_id': 'OFF005', 'name': 'Free Tire Rotation', 'type': 'Free Service', 'value': 29.99},
        {'offer_id': 'OFF006', 'name': 'Winter Package Deal', 'type': 'Bundle', 'value': 50.00},
        {'offer_id': 'OFF007', 'name': 'Refer a Friend $50', 'type': 'Referral', 'value': 50.00},
        {'offer_id': 'OFF008', 'name': 'Birthday Special 30%', 'type': 'Discount', 'value': 0.30},
    ]

    OFFER_TYPES = ['Bundle', 'Credit', 'Discount', 'Free Service', 'Points', 'Referral']

    def __init__(self, model_path: str, feature_path: str, cluster_candidates_path: str = None):
        self.model = lgb.Booster(model_file=model_path)

        with open(feature_path, 'r') as f:
            self.feature_cols = json.load(f)

        self.cluster_candidates = None
        if cluster_candidates_path and os.path.exists(cluster_candidates_path):
            with open(cluster_candidates_path, 'r') as f:
                self.cluster_candidates = json.load(f)

        self.offers = self.OFFERS
        self.offers_by_id = {o['offer_id']: o for o in self.offers}

    def _get_candidates(self, customer_features: dict) -> list:
        """Stage 1: Retrieval - filter offers by customer cluster."""
        cluster = customer_features.get('cluster')

        if self.cluster_candidates and cluster is not None:
            cluster_key = str(int(cluster))
            if cluster_key in self.cluster_candidates:
                candidate_ids = self.cluster_candidates[cluster_key]
                candidates = [self.offers_by_id[oid] for oid in candidate_ids if oid in self.offers_by_id]
                if candidates:
                    return candidates

        return self.offers

    def _score_candidates(self, customer_features: dict, candidates: list) -> list:
        """Stage 2: Ranking - score candidates with LightGBM."""
        rows = []
        for offer in candidates:
            row = customer_features.copy()
            row['offer_value'] = offer['value']

            for o_type in self.OFFER_TYPES:
                row[f'offer_type_{o_type}'] = 1 if offer['type'] == o_type else 0

            row['offer_id'] = offer['offer_id']
            row['offer_name'] = offer['name']
            rows.append(row)

        df = pd.DataFrame(rows)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        scores = self.model.predict(df[self.feature_cols])

        results = [
            {'offer_id': candidates[i]['offer_id'], 'offer_name': candidates[i]['name'], 'score': float(scores[i])}
            for i in range(len(scores))
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def predict(self, customer_features: dict, top_k: int = 3) -> list:
        """
        Generate recommendations using two-stage pipeline.

        Args:
            customer_features: Customer features from Redis
            top_k: Number of recommendations to return

        Returns:
            Top K offers sorted by predicted redemption probability
        """
        candidates = self._get_candidates(customer_features)
        ranked = self._score_candidates(customer_features, candidates)
        return ranked[:top_k]
