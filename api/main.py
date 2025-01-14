from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from packages.constants.paths import MODEL_DIR, LOG_DIR, PROCESSED_DATA_DIR
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Type, Any, Dict, Optional, List, Union
import mlflow.pyfunc
from werkzeug.exceptions import BadRequest  # Import BadRequest exception

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model Details
MODEL_NAME = 'LGBMClassifier - business'
MODEL_VERSION = None  # Use "None" for the latest version or specify an integer version

# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR / DATASET_NAME)

if not DATASET_PATH .exists():
    logger.error(f"Dataset file not found at {DATASET_PATH }")
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH }")

# Loguru Configuration
LOG_PATH = Path(LOG_DIR / "api")
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")



# Scaler
SCALER_NAME = "2025-01-11 - RobustScaler.joblib"
SCALER_PATH = Path(MODEL_DIR / SCALER_NAME)


# ==================================================================================================================== #
#                                            LOADING RESOURCES FROM MLFLOW                                             #
# ==================================================================================================================== #
# Load Model from MLflow Model Registry
try:
    if MODEL_VERSION:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"

    model = mlflow.pyfunc.load_model(model_uri)
    logger.success(f"Model loaded successfully from MLflow Model Registry: {model_uri}")
except Exception as e:
    logger.error(f"Error loading model from MLflow: {e}")
    raise RuntimeError("An error occurred while loading the model from MLflow. Please check the logs.")

# Load Scaler from MLflow Artifacts
try:
    with mlflow.start_run() as run:
        # Get the scaler artifact URI from the model's run
        scaler_path = mlflow.artifacts.download_artifacts(artifact_path="scalers/2025-01-14 - RobustScaler.joblib", run_id=run.info.run_id)

    robust_scaler = joblib.load(scaler_path)
    logger.success(f"Scaler loaded successfully from MLflow artifacts: {scaler_path}")
except Exception as e:
    logger.error(f"Error loading scaler from MLflow: {e}")
    raise RuntimeError("An error occurred while loading the scaler from MLflow. Please check the logs.")


# ==================================================================================================================== #
#                                                    LOADING DATASET                                                   #
# ==================================================================================================================== #
try:
    dataset = pd.read_csv(DATASET_PATH )  # Load the CSV into a DataFrame
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
#                                                      DEBUG CHECK                                                     #
# ==================================================================================================================== #
# Log predicted probabilities for the first 1000 rows
try:
    logger.info("Calculating predicted probabilities for the first 50 rows...")
    # Exclude "SK_ID_CURR" for predictions
    user_data_first_50 = dataset_scaled.drop(columns=["SK_ID_CURR"]).head(50)

    # Predict probabilities for the first 1000 rows
    proba_first_1000 = model.predict_proba(user_data_first_50)[:, 1]

    # Log the SK_ID_CURR with the probabilities
    for idx, proba in zip(dataset_scaled["SK_ID_CURR"].head(1000), proba_first_1000):
        logger.debug(f"SK_ID_CURR: {idx}, Predicted Proba: {round(proba, 2)}")

    logger.info("Logged predicted probabilities for the first 1000 rows successfully.")
except Exception as e:
    logger.error(f"Error logging predicted probabilities for the first 1000 rows: {e}")
    raise RuntimeError("An error occurred while logging predicted probabilities. Please check the logs for details.")



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
        try:
            user_request = request.get_json(force=True)
        except BadRequest:
            return jsonify({"error": "Invalid JSON payload"}), 400

        if "id" not in user_request:
            return jsonify({"error": "Missing 'id' in request payload"}), 400

        # Ensure both user_id and SK_ID_CURR are strings for comparison
        user_id = str(user_request["id"])                           # Convert input to string
        dataset["SK_ID_CURR"] = dataset["SK_ID_CURR"].astype(str)   # Convert column to string

        # Check if ID exists in the dataset
        if user_id not in dataset["SK_ID_CURR"].values:
            return jsonify({"error": f"ID {user_id} not found in the dataset"}), 404

        # Select the row for the given ID, excluding SK_ID_CURR
        user_data = dataset_scaled[dataset_scaled["SK_ID_CURR"] == user_id].drop(columns=["SK_ID_CURR"])

        # Predict using the loaded model
        proba = model.predict(user_data).iloc[0]  # Predict method for MLflow model
        prediction = 1 if proba >= 0.46 else 0  # Apply custom threshold

        # Define status based on prediction
        status = "Granted" if prediction == 0 else "Denied"

        # Prepare the response
        response = {
            "SK_ID_CURR": user_id,
            "predicted_proba": round(proba, 2),
            "predicted_target": int(prediction),
            "status": status
            }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        return jsonify({"error": "An error occurred while processing the prediction request"}), 500


if __name__ == "__main__":
    app.run(debug=True)

