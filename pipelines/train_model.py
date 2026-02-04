"""Train recommendation model: KMeans clustering + LightGBM ranking."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pickle
import os
import json

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

CLUSTER_FEATURES = [
    'recency_days', 'frequency', 'monetary_total', 'monetary_avg',
    'unique_service_count', 'unique_service_category',
    'open_rate', 'click_rate', 'redemption_rate'
]


def train_clustering(df: pd.DataFrame, n_clusters: int = 10):
    """Train KMeans clustering for customer segmentation."""
    X = df[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    return df, scaler, kmeans


def prepare_training_data(df_features: pd.DataFrame, df_offers: pd.DataFrame):
    """Prepare data for LightGBM training."""
    df = df_offers.merge(df_features, on='customer_id', how='inner')
    df = pd.get_dummies(df, columns=['offer_type'], prefix='offer_type')

    offer_type_cols = [c for c in df.columns if c.startswith('offer_type_')]
    feature_cols = CLUSTER_FEATURES + ['cluster', 'offer_value'] + offer_type_cols

    exclude = ['redeemed', 'opened', 'clicked', 'customer_id', 'interaction_id', 'offer_id', 'offer_name', 'sent_date']
    feature_cols = [c for c in feature_cols if c in df.columns and c not in exclude]

    X = df[feature_cols].copy()
    y = df['redeemed'].copy()

    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    return X, y, feature_cols


def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    """Train LightGBM ranking model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    train_data = lgb.Dataset(X_train.values.astype(np.float32), label=y_train.values.astype(np.float32))
    test_data = lgb.Dataset(X_test.values.astype(np.float32), label=y_test.values.astype(np.float32), reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    auc = roc_auc_score(y_test, model.predict(X_test.values.astype(np.float32)))
    return model, auc


def generate_cluster_candidates(df_features: pd.DataFrame, df_offers: pd.DataFrame, n_top: int = 5):
    """Generate cluster → top offers mapping for retrieval stage."""
    df = df_offers.merge(df_features[['customer_id', 'cluster']], on='customer_id', how='inner')

    stats = df.groupby(['cluster', 'offer_id']).agg({'redeemed': ['sum', 'count']}).reset_index()
    stats.columns = ['cluster', 'offer_id', 'redemptions', 'total']
    stats['rate'] = stats['redemptions'] / stats['total']

    candidates = {}
    for cluster in stats['cluster'].unique():
        top = stats[stats['cluster'] == cluster].nlargest(n_top, 'rate')['offer_id'].tolist()
        candidates[str(int(cluster))] = top

    return candidates


def main():
    df_features = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
    df_offers = pd.read_csv(os.path.join(DATA_DIR, "offer_interactions.csv"))

    # Stage 1: Clustering
    df_features, scaler, kmeans = train_clustering(df_features)
    df_features.to_csv(os.path.join(DATA_DIR, "customer_features_clustered.csv"), index=False)

    # Stage 2: Train LightGBM
    X, y, feature_cols = prepare_training_data(df_features, df_offers)
    model, auc = train_lightgbm(X, y)

    # Stage 3: Generate retrieval candidates
    cluster_candidates = generate_cluster_candidates(df_features, df_offers)

    # Save artifacts
    model.save_model(os.path.join(MODELS_DIR, "recommendation_model.lgb"))

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, "kmeans.pkl"), 'wb') as f:
        pickle.dump(kmeans, f)

    with open(os.path.join(MODELS_DIR, "feature_cols.json"), 'w') as f:
        json.dump(feature_cols, f)

    with open(os.path.join(MODELS_DIR, "cluster_candidates.json"), 'w') as f:
        json.dump(cluster_candidates, f)

    with open(os.path.join(MODELS_DIR, "metrics.json"), 'w') as f:
        json.dump({'auc': auc, 'best_iteration': model.best_iteration}, f)

    print(f"Training complete. AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
