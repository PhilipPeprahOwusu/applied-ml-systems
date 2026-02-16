# =============================================================================
# TRAINING PIPELINE (LightGBM Ranking)
# =============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")

os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("Offer_Recommendation_Ranking")


def prepare_training_data(df_features, df_offers):
    """
    Merges customer features with offer context and creates interaction features.
    This mirrors the logic in notebooks/recommendation_model.ipynb.
    """
    print("Preparing training data...")
    
    # Merge on customer_id
    df_train = df_offers.merge(df_features, on='customer_id', how='inner')
    
    # One-hot encode offer types
    df_train = pd.get_dummies(df_train, columns=['offer_type'], prefix='offer')
    
    # Identify the new one-hot columns (e.g., 'offer_Discount')
    # Exclude non-feature columns
    offer_type_cols = [c for c in df_train.columns 
                       if c.startswith('offer_') 
                       and c not in ['offer_id', 'offer_name', 'offer_value']]
    
    print(f"  Creating interactions for {len(offer_type_cols)} offer types...")

    # --- CREATE INTERACTION FEATURES ---
    # These features capture the specific affinity (e.g., "Frequent users like Points")
    for col in offer_type_cols:
        # Frequency interactions
        if 'frequency' in df_train.columns:
            df_train[f'freq_x_{col}'] = df_train['frequency'] * df_train[col]
            
        # Recency interactions
        if 'recency_days' in df_train.columns:
            df_train[f'recency_x_{col}'] = df_train['recency_days'] * df_train[col]
            
        # Monetary interactions
        if 'monetary_avg' in df_train.columns:
            df_train[f'monetary_x_{col}'] = df_train['monetary_avg'] * df_train[col]

    # Select final feature set
    # Exclude IDs, dates, target variable, and potential leakage
    exclude_cols = ['customer_id', 'interaction_id', 'offer_id', 'offer_name', 
                    'sent_date', 'redeemed', 'opened', 'clicked', 
                    'favorite_category', 'favorite_offer_type',
                    # Leakage risk
                    'total_redemptions', 'redemption_rate', 'total_clicks', 'click_rate', 'total_opens', 'open_rate']
    
    feature_cols = [c for c in df_train.columns 
                    if c not in exclude_cols 
                    and df_train[c].dtype in ['int64', 'float64', 'int32', 'float32', 'uint8', 'bool']]
    
    X = df_train[feature_cols].copy()
    y = df_train['redeemed'].copy()
    
    print(f"  Final shape: {X.shape}")
    return X, y, feature_cols


def train_model(X, y, feature_cols):
    """Trains the LightGBM model and saves artifacts."""
    print("Training LightGBM...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # LightGBM Dataset
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
        'is_unbalance': True,  # Handle class imbalance
        'random_state': 42
    }
    
    with mlflow.start_run(run_name="pipeline_training"):
        mlflow.log_params(params)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            valid_names=['test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        mlflow.log_metric("auc", auc)
        print(f"\n  Model AUC: {auc:.4f}")
        
        # --- SAVE ARTIFACTS (CRITICAL FOR SERVING) ---
        # 1. Save model as .txt (fast loading for C++ implementation)
        model_path = os.path.join(MODELS_DIR, "offer_recommender_v2.txt")
        model.save_model(model_path)
        
        # 2. Save feature list (so service knows input order)
        feature_path = os.path.join(MODELS_DIR, "feature_cols_v2.json")
        with open(feature_path, 'w') as f:
            json.dump(feature_cols, f)
            
        print(f"  Artifacts saved to {MODELS_DIR}")
        mlflow.lightgbm.log_model(model, "model")


def main():
    print("============================================================")
    print("TRAINING PIPELINE STARTED")
    print("============================================================")
    
    # Load Data
    try:
        df_features = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
        df_offers = pd.read_csv(os.path.join(DATA_DIR, "offer_interactions.csv"))
    except FileNotFoundError:
        print("Error: Input data not found. Run generate_data.py and feature_engineering.py first.")
        return

    # Prepare
    X, y, feature_cols = prepare_training_data(df_features, df_offers)
    
    # Train & Save
    train_model(X, y, feature_cols)
    print("\nPipeline Complete!")


if __name__ == "__main__":
    main()