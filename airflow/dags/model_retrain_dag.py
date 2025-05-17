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
import psycopg2
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv(override=True)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def download_data_from_rds():
    import psycopg2
    import csv

    host = os.environ.get("AWS_RDS")
    port = os.environ.get("PORT")
    database = os.environ.get("DATABASE")
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    ssl_cert_path = "/opt/airflow/dags/global-bundle.pem"

    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
        sslmode="verify-full",
        sslrootcert=ssl_cert_path,
    )

    cur = conn.cursor()

    tablename = "test_ev_data"

    cur.execute(f"SELECT * FROM {tablename}")
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]

    with open(f"../../data/{tablename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

    print(f"Data saved to {tablename}.csv")

    cur.close()
    conn.close()


def retrain_and_compare_model():
    train_path = "/data/test_ev_data.csv"
    test_path = "/data/test.csv"
    model_output_path = "/ml_model/best_model.pkl"

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("xgboost_model_selection_v2")

    print("Loading dataset files...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading data files: {e}")
        print(f"Train path exists: {os.path.exists(train_path)}")
        print(f"Test path exists: {os.path.exists(test_path)}")
        raise

    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    # Normalize column names to lowercase to avoid case sensitivity issues
    df_train.columns = [col.lower() for col in df_train.columns]
    df_test.columns = [col.lower() for col in df_test.columns]

    # Print column names for debugging
    print("Training columns:", df_train.columns.tolist())
    print("Test columns:", df_test.columns.tolist())

    # Determine the target column name - try various common names
    target_candidates = ["electric_range", "electric range", "electricrange"]
    target_column = None

    for candidate in target_candidates:
        if candidate in df_train.columns and candidate in df_test.columns:
            target_column = candidate
            print(f"Found target column: {target_column}")
            break

    if target_column is None:
        raise ValueError(
            f"Target column not found in data. Available columns: {df_train.columns.tolist()}"
        )

    # Check for potential unique identifiers and drop them
    columns_to_drop = [
        "legislative_district",
        "dol_vehicle_id",
        "vehicle_location",
        "postal_code",
        "city",
        "2020_census_tract",
        "county",
        "electric_utility",
        "vin",
        "vehicle_id",
    ]

    # Remove columns that might not exist
    columns_to_drop = [col for col in columns_to_drop if col in df_train.columns]

    df_train = df_train.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

    # Remove rows with zero or negative target values
    print(f"Rows before filtering {target_column}: {len(df_train)}")
    df_train = df_train[df_train[target_column] > 0]
    df_test = df_test[df_test[target_column] > 0]
    print(f"Rows after filtering {target_column}: {len(df_train)}")

    # Print dataset information
    print("\nTraining data info:")
    print(df_train.describe().T)

    # Prepare features and target
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    # Handle categorical features
    print("\nIdentifying categorical columns...")
    categorical_cols = [
        col for col in X_train.columns if X_train[col].dtype == "object"
    ]
    print(f"Categorical columns: {categorical_cols}")

    # Use pandas get_dummies for one-hot encoding (more stable than manual encoding)
    if categorical_cols:
        print("Applying one-hot encoding to categorical columns...")
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

        # Ensure X_test has same columns as X_train
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0

        # Keep only columns that are in the training data
        X_test = X_test[X_train.columns]

    # Handle missing values
    print("\nHandling missing values...")
    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_test.mean(numeric_only=True))

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")

    # Train model with more reasonable parameters
    print("\nTraining new model...")
    new_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    try:
        new_model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        # Print detailed information to help diagnose the issue
        print(f"X_train types: {X_train.dtypes}")
        print(f"y_train type: {type(y_train)}")
        print(f"y_train range: {y_train.min()} to {y_train.max()}")
        raise
    try:
        new_model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        # Print detailed information to help diagnose the issue
        print(f"X_train types: {X_train.dtypes}")
        print(f"y_train type: {type(y_train)}")
        print(f"y_train range: {y_train.min()} to {y_train.max()}")
        raise

    # Make predictions and evaluate new model
    print("\nEvaluating new model...")
    new_predictions = new_model.predict(X_test)

    # Calculate metrics for new model
    new_r2 = r2_score(y_test, new_predictions)
    new_rmse = np.sqrt(mean_squared_error(y_test, new_predictions))
    new_mae = mean_absolute_error(y_test, new_predictions)

    # Handle potential zero values in y_test for MAPE calculation
    y_test_safe = np.maximum(y_test, 1e-7)  # Avoid division by zero
    new_mape = mean_absolute_percentage_error(y_test_safe, new_predictions)

    print(
        f"New model metrics - R²: {new_r2:.4f}, RMSE: {new_rmse:.4f}, MAE: {new_mae:.4f}, MAPE: {new_mape:.4f}"
    )

    # Check for feature importance to identify potential data leakage
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": new_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\nTop 10 feature importance:")
    print(feature_importance.head(10))

    # Try to load existing model for comparison
    prod_r2 = 0  # Default value if no existing model
    prod_rmse = float("inf")
    prod_mae = float("inf")
    prod_mape = float("inf")

    if os.path.exists(model_output_path):
        try:
            print("\nLoading existing model for comparison...")
            existing_model = joblib.load(model_output_path)

            # Evaluate existing model on same test data
            existing_predictions = existing_model.predict(X_test)

            # Calculate metrics for existing model
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
            print(f"Error loading or evaluating existing model: {e}")
    else:
        print("No existing model found for comparison.")

    # Log everything to MLflow
    with mlflow.start_run(
        run_name=f"xgboost_ev_range_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        # Log model parameters
        mlflow.log_params(
            {
                "model_type": "XGBoostRegressor",
                "objective": "reg:squarederror",
                "n_estimators": new_model.n_estimators,
                "learning_rate": new_model.learning_rate,
                "max_depth": new_model.max_depth,
                "subsample": new_model.subsample,
                "colsample_bytree": new_model.colsample_bytree,
                "train_rows": len(X_train),
                "train_cols": len(X_train.columns),
                "test_rows": len(X_test),
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {"r2_score": new_r2, "rmse": new_rmse, "mae": new_mae, "mape": new_mape}
        )

        importance_path = "/tmp/feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.barh(
                feature_importance["Feature"][:10],
                feature_importance["Importance"][:10],
            )
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title("Top 10 Feature Importance")
            plt.tight_layout()

            plot_path = "/tmp/feature_importance.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")

        # Log training & test data sample
        train_sample_path = "/tmp/train_sample.csv"
        test_sample_path = "/tmp/test_sample.csv"

        # Save only a small sample to avoid large files
        df_train.sample(min(100, len(df_train))).to_csv(train_sample_path, index=False)
        df_test.sample(min(100, len(df_test))).to_csv(test_sample_path, index=False)

        mlflow.log_artifact(train_sample_path, artifact_path="data_samples")
        mlflow.log_artifact(test_sample_path, artifact_path="data_samples")

        # Log the model itself
        mlflow.xgboost.log_model(new_model, "model")

        # Compare with existing model and decide whether to save
        if new_r2 > prod_r2:
            print(f"\nNew model is better: R² {new_r2:.4f} > {prod_r2:.4f}")
            print(f"Saving new model to {model_output_path}")

            # Save the model using joblib for production use
            joblib.dump(new_model, model_output_path)

            # Log improvement percentage
            if prod_r2 > 0:  # Avoid division by zero
                mlflow.log_metric(
                    "r2_improvement_percent", 100 * (new_r2 - prod_r2) / abs(prod_r2)
                )

            mlflow.log_param("model_selected", "new_model")
            mlflow.log_param("model_improved", "true")
        else:
            print(
                f"\nExisting model is better or equivalent: R² {prod_r2:.4f} >= {new_r2:.4f}"
            )
            print("Keeping existing model")

            mlflow.log_param("model_selected", "existing_model")
            mlflow.log_param("model_improved", "false")

        # Log actual vs predicted values
        prediction_df = pd.DataFrame({"actual": y_test, "predicted": new_predictions})

        pred_path = "/tmp/predictions.csv"
        prediction_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, artifact_path="predictions")

        # Create scatter plot of actual vs predicted
        try:
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, new_predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            plt.xlabel("Actual Range (miles)")
            plt.ylabel("Predicted Range (miles)")
            plt.title("Actual vs Predicted EV Range")

            scatter_path = "/tmp/actual_vs_predicted.png"
            plt.savefig(scatter_path)
            mlflow.log_artifact(scatter_path, artifact_path="plots")
        except Exception as e:
            print(f"Error creating scatter plot: {e}")

        # Add MLflow run ID to log
        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"MLflow experiment ID: {run.info.experiment_id}")

    return new_r2 > prod_r2


with DAG(
    dag_id="xgboost_model_retrain_and_selection",
    default_args=default_args,
    start_date=datetime(2025, 5, 16),
    schedule_interval="*/50 * * * *",
    catchup=False,
) as dag:
    download_data_task = PythonOperator(
        task_id="download_data_from_rds",
        python_callable=download_data_from_rds,
    )

    retrain_model_task = PythonOperator(
        task_id="retrain_and_compare",
        python_callable=retrain_and_compare_model,
    )

    # Set task dependencies
    download_data_task >> retrain_model_task
