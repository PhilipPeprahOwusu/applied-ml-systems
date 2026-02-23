import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import os
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")

os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("Offer_Recommendation_Ranking")

def prepare_training_data(df_features, df_offers):
    logger.info("Merging features and constructing interactions...")
    df_train = df_offers.merge(df_features, on='customer_id', how='inner')
    df_train = pd.get_dummies(df_train, columns=['offer_type'], prefix='offer')
    
    offer_type_cols = [c for c in df_train.columns if c.startswith('offer_') and c not in ['offer_id', 'offer_name', 'offer_value']]
    
    for col in offer_type_cols:
        if 'frequency' in df_train.columns:
            df_train[f'freq_x_{col}'] = df_train['frequency'] * df_train[col]
        if 'recency_days' in df_train.columns:
            df_train[f'recency_x_{col}'] = df_train['recency_days'] * df_train[col]
        if 'monetary_avg' in df_train.columns:
            df_train[f'monetary_x_{col}'] = df_train['monetary_avg'] * df_train[col]

    exclude_cols = ['customer_id', 'interaction_id', 'offer_id', 'offer_name', 
                    'sent_date', 'redeemed', 'opened', 'clicked', 
                    'favorite_category', 'favorite_offer_type',
                    'total_redemptions', 'redemption_rate', 'total_clicks', 'click_rate', 'total_opens', 'open_rate']
    
    feature_cols = [c for c in df_train.columns 
                    if c not in exclude_cols 
                    and df_train[c].dtype in ['int64', 'float64', 'int32', 'float32', 'uint8', 'bool']]
    
    return df_train[feature_cols], df_train['redeemed'], feature_cols

def train_model(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,
        'random_state': 42
    }
    
    with mlflow.start_run():
        mlflow.log_params(params)
        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        auc = roc_auc_score(y_test, model.predict(X_test))
        mlflow.log_metric("auc", auc)
        logger.info(f"Model Training Complete. AUC: {auc:.4f}")
        
        model.save_model(os.path.join(MODELS_DIR, "offer_recommender_v2.txt"))
        with open(os.path.join(MODELS_DIR, "feature_cols_v2.json"), 'w') as f:
            json.dump(feature_cols, f)
        mlflow.lightgbm.log_model(model, "model")

def main():
    try:
        df_features = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
        df_offers = pd.read_csv(os.path.join(DATA_DIR, "offer_interactions.csv"))
        X, y, feature_cols = prepare_training_data(df_features, df_offers)
        train_model(X, y, feature_cols)
    except FileNotFoundError as e:
        logger.error(f"Data missing: {e}")

if __name__ == "__main__":
    main()