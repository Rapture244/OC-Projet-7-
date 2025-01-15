"""
This Flask application integrates with MLflow to provide a prediction API for a pre-trained LightGBM model.

Key Features:
1. **MLflow Integration**:
   - Sets up the MLflow tracking URI to fetch registered models and artifacts.
   - Dynamically loads the model, scaler, and custom threshold using MLflow's artifact and registry services.
   - Provides logging of key configuration and loading steps for traceability.

2. **Dataset Handling**:
   - Loads a preprocessed dataset from a specified path.
   - Applies preprocessing using a RobustScaler downloaded as an MLflow artifact.

3. **Prediction API**:
   - `/predict`: A POST endpoint that accepts an `SK_ID_CURR` ID and returns the predicted probability, target, and status.
   - Validates input data and ensures that the requested ID exists in the dataset.
   - Computes predictions using the LightGBM model and compares them against a custom threshold fetched from MLflow.

4. **Error Handling and Logging**:
   - Provides robust error handling for missing files, invalid input data, and API requests.
   - Uses Loguru for detailed logging of processes and exceptions.

5. **Development-Friendly**:
   - Includes endpoints for testing, and logs predicted probabilities for the first 20 rows of the dataset.

Dependencies:
- **Flask**: Framework for the API.
- **MLflow**: For model and artifact management.
- **Loguru**: For structured and detailed logging.
- **LightGBM**: Predictive model.
- **Pandas, NumPy**: For data handling and preprocessing.
- **Joblib**: For deserializing the scaler artifact.

Notes:
- Ensure `MLFLOW_TRACKING_URI` is configured correctly in the constants file.
- The dataset, model, and scaler must exist and be properly configured in the MLflow registry for the API to function.
- Use the `list_all_registered_models_and_versions_with_details` utility to debug model configurations and versions.

"""

# ====================================================== IMPORTS ===================================================== #
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from typing import Any
from packages.constants.paths import LOG_DIR, PROCESSED_DATA_DIR, MODEL_DIR
from werkzeug.exceptions import BadRequest  # Import BadRequest exception
from mlflow.lightgbm import load_model  # Use the LightGBM-specific loader
from mlflow.tracking import MlflowClient
import mlflow
from packages.mflow_utils import *
from packages.constants.paths import MLFLOW_TRACKING_URI
from packages.utils import log_section_header
# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
log_section_header(title = "Configuration")

# Set the Ml Flow tracking uri & check registered models
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.debug(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")
list_all_registered_models_and_versions_with_details()

# Model Details
MODEL_NAME = "LGBMClassifier - business"
MODEL_ALIAS = "champion"  # Specify the alias of the model to use

# Scaler Details
SCALER_NAME = "2025-01-14 - RobustScaler.joblib"
ARTIFACT_SCALER_PATH = Path("scalers") / SCALER_NAME  # Define the relative path within the artifact directory


# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR) / DATASET_NAME

# Loguru Configuration
LOG_PATH = Path(LOG_DIR) / "api"
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")

# Path existence check
paths_to_check = {
    # "Scaler": SCALER_PATH,
    "Dataset": DATASET_PATH,
    }
for name, path in paths_to_check.items():
    if not path.exists():
        logger.error(f"{name} file not found at {path}")
        raise FileNotFoundError(f"{name} file not found at {path}")

# ==================================================================================================================== #
#                                            LOADING RESOURCES USING MLFLOW                                            #
# ==================================================================================================================== #

log_section_header(title = "Loading ressources using MlFlow")

# ================================================== FETCHING RUN ID ================================================= #
# Initialize MLflow Client
client = MlflowClient()

# Fetch the model's run_id using the alias
try:
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = model_version.run_id
    logger.success(f"{MODEL_NAME:<25} | Alias: @{MODEL_ALIAS:<15} | Fetched run_id '{run_id}")
except Exception as e:
    logger.error(f"{MODEL_NAME:<25} | Alias: @{MODEL_ALIAS:<15} | Error fetching run_id: {e}")
    raise RuntimeError("Failed to retrieve run_id for the model. Check if the model and alias exist in the registry.")

# ==================================================== LOAD MODEL ==================================================== #
# Load Model from MLflow using alias
try:
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = load_model(model_uri)  # Use LightGBM-specific loader
    logger.success(f"{MODEL_NAME:<25} | Alias: @{MODEL_ALIAS:<15} | Model loaded successfully using alias: {model_uri}")
except Exception as e:
    logger.error(f"Error loading model from MLflow Registry: {e}")
    raise RuntimeError("An error occurred while loading the model from MLflow Registry. Please check the logs.")

# ==================================================== LOAD SCALER =================================================== #
# Load Scaler dynamically from the same run as the model
try:
    # Convert the artifact path to a string for MLflow
    artifact_path = ARTIFACT_SCALER_PATH.as_posix()

    # Download the artifact locally using the fetched run_id
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

    # Load the scaler using joblib
    robust_scaler = joblib.load(local_path)
    logger.success(f"Scaler loaded successfully from MLflow artifact: {local_path}")
except Exception as e:
    logger.error(f"Error loading scaler from MLflow artifact: {e}")
    raise RuntimeError("An error occurred while loading the scaler from MLflow. Please check the logs.")

# ================================================== LOAD THRESHOLD ================================================== #
# Fetch the custom threshold parameter
try:
    run_data = client.get_run(run_id).data
    # Ensure the custom_threshold parameter exists
    if "custom_threshold" not in run_data.params:
        raise ValueError(f"Parameter 'custom_threshold' not found in MLflow run '{run_id}'")

    # Extract and set the custom threshold
    THRESHOLD = float(run_data.params["custom_threshold"])
    logger.success(f"ML FLOW run id: {run_id:<45} | Fetched custom threshold '{THRESHOLD}'")
except Exception as e:
    logger.error(f"Error fetching custom threshold from MLflow run '{run_id}': {e}")
    raise RuntimeError("Failed to fetch custom threshold from MLflow. Please check the logs.")

# ==================================================================================================================== #
#                                                     LOAD DATASET                                                     #
# ==================================================================================================================== #
log_section_header(title = "Loading Dataset")

try:
    dataset = pd.read_csv(DATASET_PATH)  # Load the CSV into a DataFrame
    logger.success(f"Dataset loaded successfully from {DATASET_PATH}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("An error occurred while loading the dataset. Please check the logs for more details.")

# ==================================================================================================================== #
#                                                     PREPROCESSING                                                    #
# ==================================================================================================================== #
log_section_header(title = "Preprocessing")

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
    passthrough_data = dataset[passthrough_features]        # Keep passthrough columns untouched
    dataset_scaled = pd.concat([numeric_scaled_df, passthrough_data], axis=1)
    dataset_scaled = dataset_scaled[dataset.columns]        # Reorder to match the original structure

    logger.success("Applied RobustScaler to numeric features successfully.")
except Exception as e:
    logger.error(f"Error applying RobustScaler: {e}")
    raise RuntimeError("An error occurred while scaling the dataset. Please check the logs for more details.")

# ==================================================================================================================== #
#                                  LOG PREDICTED PROBABILITIES FOR THE FIRST 20 ROWS                                   #
# ==================================================================================================================== #
log_section_header(title = "Log predicted probabilities for the first 20 rows")

try:
    logger.info("Calculating predicted probabilities for the first 20 rows...")
    # Exclude "SK_ID_CURR" for predictions
    user_data_first_20 = dataset_scaled.drop(columns=["SK_ID_CURR"]).head(20)

    # Predict probabilities for the first 20 rows
    proba_first_20 = model.predict_proba(user_data_first_20)[:, 1]  # LightGBM's predict_proba

    # Log the SK_ID_CURR with the probabilities
    for idx, proba in zip(dataset_scaled["SK_ID_CURR"].head(20), proba_first_20):
        logger.debug(f"SK_ID_CURR: {idx}, Predicted Proba: {round(proba, 2)}")

    logger.success("Logged predicted probabilities for the first 20 rows successfully.")
except Exception as e:
    logger.error(f"Error logging predicted probabilities for the first 20 rows: {e}")
    raise RuntimeError("An error occurred while logging predicted probabilities. Please check the logs for details.")

# ==================================================================================================================== #
#                                                          API                                                         #
# ==================================================================================================================== #
log_section_header(title = "API ")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict using the model loaded from MLflow Registry for a given SK_ID_CURR.
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

        user_id = user_request.get("id", None)

        # Validate input ID
        if user_id is None or not isinstance(user_id, int):
            return jsonify({"error": "Missing or invalid 'id' in request payload"}), 400

        # Check if ID exists in the dataset
        if user_id not in dataset_scaled["SK_ID_CURR"].values:
            return jsonify({"error": f"ID {user_id} not found in the dataset"}), 404

        # Prepare data for prediction
        user_data = dataset_scaled.loc[dataset_scaled["SK_ID_CURR"] == user_id].drop(columns=["SK_ID_CURR"])
        if user_data.empty:
            return jsonify({"error": "No valid data for prediction"}), 400

        # Predict probability and target
        proba = model.predict_proba(user_data)[0][1]  # Get the probability of the positive class
        prediction = int(proba >= THRESHOLD)
        status = "Granted" if prediction == 0 else "Denied"

        # Return response
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
    app.run(debug=True)
