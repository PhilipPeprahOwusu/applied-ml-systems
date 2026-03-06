import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "route_avg_duration",
    "route_avg_distance",
    "route_avg_fare",
]

TARGETS = ["trip_duration_minutes", "fare_amount"]

def fetch_training_data(data_path: str) -> pd.DataFrame:
    print(f"Fetching historical data from Feast for: {data_path}")
    store = FeatureStore(repo_path="feature_repo")

    # Load processed trips
    entity_df = pd.read_parquet(data_path)
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

    features_to_fetch = [
        "route_stats:route_avg_duration",
        "route_stats:route_avg_distance",
        "route_stats:route_avg_fare",
    ]

    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    ).to_df()

    return training_data

def train_and_log_model(df: pd.DataFrame, run_name: str = "V1_Baseline") -> None:
    print("Starting XGBoost multi-output model training...")
    
    # Drop nulls from point-in-time join
    df = df.dropna()

    if df.empty:
        raise ValueError("Training dataframe is empty after dropna. Feast join found no historical matches.")

    X = df[FEATURES]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    import os
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Delivery_ETA_Prediction")

    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators": 150,
            "max_depth": 7,
            "learning_rate": 0.05,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        # MultiOutputRegressor for Time and Cost
        base_model = xgb.XGBRegressor(**params)
        model = MultiOutputRegressor(base_model)
        
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        
        mae_time = mean_absolute_error(y_test["trip_duration_minutes"], predictions[:, 0])
        rmse_time = np.sqrt(mean_squared_error(y_test["trip_duration_minutes"], predictions[:, 0]))
        
        mae_fare = mean_absolute_error(y_test["fare_amount"], predictions[:, 1])
        rmse_fare = np.sqrt(mean_squared_error(y_test["fare_amount"], predictions[:, 1]))

        print(f"Training Complete: {run_name} | ETA MAE: {mae_time:.2f} mins | Fare MAE: ${mae_fare:.2f}")

        mlflow.log_params(params)
        mlflow.log_metric("eta_mae", mae_time)
        mlflow.log_metric("eta_rmse", rmse_time)
        mlflow.log_metric("fare_mae", mae_fare)
        mlflow.log_metric("fare_rmse", rmse_fare)
        
        mlflow.sklearn.log_model(model, artifact_path="eta_model")
        print("Model successfully logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/yellow_2024-01.parquet", help="Path to processed training data")
    parser.add_argument("--run_name", type=str, default="V1_Baseline", help="MLflow run name")
    args = parser.parse_args()

    training_df = fetch_training_data(args.data_path)
    train_and_log_model(training_df, run_name=args.run_name)