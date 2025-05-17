from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import numpy as np
import joblib
import os

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def retrain_and_compare_model():
    # File paths
    train_path = "/data/train80.csv"
    test_path = "/data/test.csv"
    model_output_path = "/ml_model/best_model.pkl"

    # Tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("xgboost_model_selection_v2")

    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    columns_to_drop = [
        "Legislative District",
        "Vehicle Location",
        "Postal Code",
        "City",
        "2020 Census Tract",
        "County",
        "Electric Utility",
    ]

    # Data cleaning
    df_train = df_train.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

    df_train = df_train[df_train["Electric Range"] > 0]
    df_test = df_test[df_test["Electric Range"] > 0]

    # Split features/target
    X_train = df_train.drop(columns=["Electric Range"]).fillna(
        df_train.mean(numeric_only=True)
    )
    y_train = df_train["Electric Range"]
    X_test = df_test.drop(columns=["Electric Range"]).fillna(
        df_test.mean(numeric_only=True)
    )
    y_test = df_test["Electric Range"]

    # Train new model
    new_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    new_model.fit(X_train, y_train)

    new_predictions = new_model.predict(X_test)

    # Calculate multiple metrics for new model
    new_r2 = r2_score(y_test, new_predictions)
    new_rmse = np.sqrt(mean_squared_error(y_test, new_predictions))
    new_mae = mean_absolute_error(y_test, new_predictions)
    # MAPE can't handle zero values in y_test, so adding small epsilon
    y_test_safe = np.maximum(y_test, 1e-7)
    new_mape = mean_absolute_percentage_error(y_test_safe, new_predictions)

    print(
        f"New model metrics - R²: {new_r2:.4f}, RMSE: {new_rmse:.4f}, MAE: {new_mae:.4f}, MAPE: {new_mape:.4f}"
    )

    # Try to load existing model
    prod_r2 = 0
    prod_rmse = float("inf")  # Initialize to worst case
    prod_mae = float("inf")
    prod_mape = float("inf")

    if os.path.exists(model_output_path):
        try:
            # Load existing model
            existing_model = joblib.load(model_output_path)
            # Evaluate existing model on same test data
            existing_predictions = existing_model.predict(X_test)

            # Calculate multiple metrics for existing model
            prod_r2 = r2_score(y_test, existing_predictions)
            prod_rmse = np.sqrt(mean_squared_error(y_test, existing_predictions))
            prod_mae = mean_absolute_error(y_test, existing_predictions)
            prod_mape = mean_absolute_percentage_error(
                y_test_safe, existing_predictions
            )

            print(
                f"Existing model metrics - R²: {prod_r2:.4f}, RMSE: {prod_rmse:.4f}, MAE: {prod_mae:.4f}, MAPE: {prod_mape:.4f}"
            )
        except Exception as e:
            print(f"Error loading existing model: {e}")
    else:
        print("No existing model found.")

    # MLflow logging
    with mlflow.start_run(run_name="xgboost_auto_retrain") as run:
        # Log all metrics
        mlflow.log_metric("r2_score", new_r2)
        mlflow.log_metric("rmse", new_rmse)
        mlflow.log_metric("mae", new_mae)
        mlflow.log_metric("mape", new_mape)

        mlflow.log_param("model_type", "XGBoostRegressor")
        mlflow.log_artifact(train_path, artifact_path="input_data")
        mlflow.log_artifact(test_path, artifact_path="input_data")

        # Compare models based on primary metric (R²)
        # You can change this comparison logic to use any combination of metrics
        if new_r2 > prod_r2:
            print(f"New model is better: R² {new_r2:.4f} > {prod_r2:.4f}")
            joblib.dump(new_model, model_output_path)
            mlflow.log_artifact(model_output_path, artifact_path="model")
            mlflow.log_param("model_selected", "new_model")

            # Log improvement percentages
            if prod_r2 > 0:  # Avoid division by zero
                mlflow.log_metric(
                    "r2_improvement_percent", 100 * (new_r2 - prod_r2) / abs(prod_r2)
                )
            if prod_rmse < float("inf"):
                mlflow.log_metric(
                    "rmse_improvement_percent", 100 * (prod_rmse - new_rmse) / prod_rmse
                )
        else:
            print(f"Existing model is better: R² {prod_r2:.4f} >= {new_r2:.4f}")
            mlflow.log_param("model_selected", "existing_model")


with DAG(
    dag_id="xgboost_model_retrain_and_selection",
    default_args=default_args,
    start_date=datetime(2025, 5, 16),
    schedule_interval="*/50 * * * *",
    catchup=False,
) as dag:

    retrain_model_task = PythonOperator(
        task_id="retrain_and_compare",
        python_callable=retrain_and_compare_model,
    )
