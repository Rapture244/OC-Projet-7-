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
MODEL_ALIAS = 'champion'

# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR / DATASET_NAME)

if not DATASET_PATH .exists():
    logger.error(f"Dataset file not found at {DATASET_PATH }")
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH }")

# Loguru Configuration
LOG_PATH = Path(LOG_DIR / "api")
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")


# ==================================================================================================================== #
#                                            LOADING RESOURCES FROM MLFLOW                                             #
# ==================================================================================================================== #
# Load Model from MLflow Model Registry
try:
    # Use the alias instead of stages
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.success(f"Model loaded successfully from MLflow Model Registry: {model_uri}")

    # Retrieve the run ID associated with the loaded model
    run_id = mlflow.get_run(model.metadata.run_id).info.run_id
    logger.info(f"Retrieved run ID for the model: {run_id}")
except Exception as e:
    logger.error(f"Error loading model from MLflow: {e}")
    raise RuntimeError("An error occurred while loading the model from MLflow. Please check the logs.")


# Load Scaler from MLflow Artifacts
try:
    # Retrieve the run ID from the loaded model
    run_id = model.metadata.run_id
    logger.info(f"Using run ID: {run_id} to fetch artifacts")

    # List all artifacts under the "scalers" directory in the run
    artifact_list = mlflow.artifacts.list_artifacts(run_id=run_id, path="scalers")
    logger.debug(f"Artifacts found in 'scalers': {[artifact.path for artifact in artifact_list]}")

    # Find the scaler artifact with 'RobustScaler.joblib' in its name
    scaler_artifact = next(
        artifact for artifact in artifact_list if "RobustScaler.joblib" in artifact.path
        )
    logger.info(f"Scaler artifact found: {scaler_artifact.path}")

    # Download the scaler
    scaler_path = mlflow.artifacts.download_artifacts(artifact_path=scaler_artifact.path, run_id=run_id)
    robust_scaler = joblib.load(scaler_path)
    logger.success(f"Scaler loaded successfully from: {scaler_path}")
except StopIteration:
    logger.error("No scaler artifact found with 'RobustScaler.joblib' in its name.")
    raise RuntimeError("Could not find the scaler artifact in the MLflow run artifacts.")
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
# Log predicted probabilities for the first 50 rows
try:
    logger.info("Calculating predicted probabilities for the first 50 rows...")
    # Exclude "SK_ID_CURR" for predictions
    user_data_first_50 = dataset_scaled.drop(columns=["SK_ID_CURR"]).head(50)

    # Predict probabilities for the first 50 rows
    proba_first_50 = model.predict_proba(user_data_first_50)[:, 1]

    # Log the SK_ID_CURR with the probabilities
    for idx, proba in zip(dataset_scaled["SK_ID_CURR"].head(50), proba_first_50):
        logger.debug(f"SK_ID_CURR: {idx}, Predicted Proba: {round(proba, 2)}")

    logger.info("Logged predicted probabilities for the first 50 rows successfully.")
except Exception as e:
    logger.error(f"Error logging predicted probabilities for the first 50 rows: {e}")
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
        prediction = 1 if proba >= custom_threshold else 0  # Apply dynamic custom threshold

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

