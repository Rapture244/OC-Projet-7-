"""
FAILURE: im trying to load the model right from the mlflow server before doing the rest but i cant seem to make it work so i will stop and move on for now.

- cd to 'api/' and 'python main.py'
- the script starts by running the mlflow server so I can have access to the model and load it
- the API code then starts
- then I can test it using 'streamlit run streamlit_test.py' inside the 'api/'
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
import numpy as np
from loguru import logger
from pathlib import Path
from packages.constants.paths import MODEL_DIR, LOG_DIR, PROCESSED_DATA_DIR
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Type, Any, Dict, Optional, List, Union
import mlflow.pyfunc
from werkzeug.exceptions import BadRequest  # Import BadRequest exception
from mlflow.lightgbm import load_model
from mlflow import MlflowClient
import subprocess
import time

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_BACKEND_STORE_URI = "sqlite:///C:/Users/KDTB0620/Documents/Study/Open Classrooms/Git Repository/projet7/ml_flow/ml_flow.db"
MLFLOW_ARTIFACT_ROOT = "file:///C:/Users/KDTB0620/Documents/Study/Open Classrooms/Git Repository/projet7/ml_flow/artifacts"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model Details
MODEL_NAME = 'LGBMClassifier - business'
MODEL_ALIAS = 'champion'

# Scaler Details
SCALER_DIR = "scalers"
SCALER_FILENAME = "2025-01-14 - RobustScaler.joblib"  # Example filename

# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR / DATASET_NAME)

if not DATASET_PATH.exists():
    logger.error(f"Dataset file not found at {DATASET_PATH}")
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

# Loguru Configuration
LOG_PATH = Path(LOG_DIR / "api")
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")


# ==================================================================================================================== #
#                                             START MLFLOW SERVER PROGRAMMATICALLY                                     #
# ==================================================================================================================== #
def start_mlflow_server():
    """
    Starts the MLflow server in a subprocess and waits for it to be ready.
    """
    logger.info("Starting MLflow Tracking Server...")
    mlflow_process = subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", MLFLOW_BACKEND_STORE_URI,
            "--default-artifact-root", MLFLOW_ARTIFACT_ROOT,
            "--host", "127.0.0.1",
            "--port", "5000"
            ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )

    # Wait for MLflow server to initialize
    for _ in range(10):  # Retry up to 10 times
        try:
            if requests.get(MLFLOW_TRACKING_URI).status_code == 200:
                logger.info("MLflow server is ready.")
                return mlflow_process
        except requests.ConnectionError:
            time.sleep(2)
    raise RuntimeError("Failed to start MLflow server.")


# Start MLflow server
mlflow_process = start_mlflow_server()


# ==================================================================================================================== #
#                                            LOADING RESOURCES FROM MLFLOW                                             #
# ==================================================================================================================== #
# Load Model from MLflow Registry using alias
try:
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = load_model(model_uri)  # Load the model using the alias
    logger.success(f"Model loaded successfully using alias: {model_uri}")

    # Retrieve the run ID associated with the alias
    client = MlflowClient()
    alias_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = alias_version.run_id  # Get the run ID
    logger.info(f"Retrieved run ID for the model: {run_id}")
except Exception as e:
    logger.error(f"Error loading model using alias: {e}")
    raise RuntimeError("An error occurred while loading the model from MLflow. Please check the logs.")

# Load Scaler from MLflow Artifacts
try:
    logger.info(f"Fetching scaler directly for run ID: {run_id}")

    # Construct the full path to the scaler artifact
    scaler_path = f"{SCALER_DIR}/{SCALER_FILENAME}"
    logger.info(f"Expected scaler path: {scaler_path}")

    # Download and load the scaler directly
    scaler_local_path = mlflow.artifacts.download_artifacts(artifact_path=scaler_path, run_id=run_id)
    robust_scaler = joblib.load(scaler_local_path)
    logger.success(f"Scaler loaded successfully from: {scaler_local_path}")
except Exception as e:
    logger.error(f"Error loading scaler from MLflow: {e}")
    raise RuntimeError("An error occurred while loading the scaler from MLflow. Please check the logs.")

# Fetch the custom threshold from the model's run
try:
    custom_threshold = float(mlflow.get_run(run_id).data.params["custom_threshold"])
    logger.info(f"Retrieved custom threshold: {custom_threshold}")
except KeyError:
    logger.error("The parameter 'custom_threshold' was not found in the run's parameters.")
    raise RuntimeError("The 'custom_threshold' parameter is missing from the MLflow run parameters.")
except Exception as e:
    logger.error(f"Error fetching custom threshold: {e}")
    raise RuntimeError("An error occurred while retrieving the custom threshold. Please check the logs.")


# ==================================================================================================================== #
#                                                    LOADING DATASET                                                   #
# ==================================================================================================================== #
try:
    dataset = pd.read_csv(DATASET_PATH)  # Load the CSV into a DataFrame
    logger.success(f"Dataset loaded successfully from {DATASET_PATH}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("An error occurred while loading the dataset. Please check the logs for more details.")

# =================================================== PREPROCESSING ================================================== #
# Apply RobustScaler with PassThrough
try:
    # Separate numeric and passthrough features
    numeric_features = [col for col in dataset.select_dtypes(include=["number"]).columns if col != "SK_ID_CURR"]
    passthrough_features = ["SK_ID_CURR"]  # Explicit passthrough feature

    # Scale only numeric features using the loaded scaler
    numeric_scaled = robust_scaler.transform(dataset[numeric_features])

    # Create a DataFrame for scaled numeric columns
    numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_features, index=dataset.index)

    # Combine scaled numeric features with passthrough columns
    passthrough_data = dataset[passthrough_features]  # Keep passthrough columns untouched
    dataset_scaled = pd.concat([numeric_scaled_df, passthrough_data], axis=1)

    # Ensure columns are in the original order
    dataset_scaled = dataset_scaled[dataset.columns]  # Reorder to match the original structure

    logger.success("Applied RobustScaler to numeric features successfully.")
except Exception as e:
    logger.error(f"Error applying RobustScaler: {e}")
    raise RuntimeError("An error occurred while scaling the dataset. Please check the logs for more details.")


# ==================================================================================================================== #
#                                                          API                                                         #
# ==================================================================================================================== #
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict using the model loaded from MLflow for a given SK_ID_CURR.
    Request Body Example:
    {
        "id": 100001
    }
    """
    try:
        # Parse JSON request
        user_request = request.get_json(force=True)
        user_id = str(user_request.get("id", None))

        if not user_id:
            return jsonify({"error": "Missing or invalid 'id' in request payload"}), 400

        # Validate ID
        if user_id not in dataset_scaled["SK_ID_CURR"].astype(str).values:
            return jsonify({"error": f"ID {user_id} not found in the dataset"}), 404

        # Prepare data for prediction
        user_data = dataset_scaled.loc[dataset_scaled["SK_ID_CURR"] == user_id].drop(columns=["SK_ID_CURR"])
        proba = model.predict_proba(user_data)[0][1]  # Predicted probability
        prediction = int(proba >= custom_threshold)  # Adjust threshold as needed
        status = "Granted" if prediction == 0 else "Denied"

        return jsonify({
            "SK_ID_CURR": user_id,
            "predicted_proba": round(proba, 2),
            "predicted_target": prediction,
            "status": status
            }), 200
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        # Ensure MLflow server is terminated when the Flask app exits
        logger.info("Shutting down MLflow server...")
        mlflow_process.terminate()
        mlflow_process.wait()
        logger.info("MLflow server stopped.")
