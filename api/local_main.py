"""
This module provides a Flask API for predicting credit status using a pre-trained LightGBM Classifier model.

Key Features:
1. Loads a pre-trained model, scaler, and dataset from specified file paths.
2. Preprocesses the dataset by scaling numeric features using RobustScaler while retaining essential ID columns.
3. Predicts probabilities for the first 20 rows of the dataset and logs the results for debugging.
4. Implements an API endpoint `/predict` that accepts a JSON payload containing an SK_ID_CURR (user ID) and
   returns the predicted probability, target, and credit status (Granted or Denied).
5. Handles exceptions at each stage and provides detailed logging for debugging purposes using Loguru.

Dependencies:
- Flask: API framework.
- Pandas, NumPy: Data manipulation and numerical computations.
- Scikit-learn: Preprocessing and model handling.
- Joblib: For loading pre-trained model and scaler.
- Loguru: Logging.
- werkzeug.exceptions: To handle HTTP exceptions in the API.

Note:
- The module is configured to fail gracefully if any required files (model, scaler, dataset) are missing.
- Ensure that all paths specified in `MODEL_DIR`, `LOG_DIR`, and `PROCESSED_DATA_DIR` are valid before execution.
- Update the `MODEL_NAME`, `SCALER_NAME`, and `DATASET_NAME` as per the required versions of the files.

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
from packages.constants.paths import MODEL_DIR, LOG_DIR, PROCESSED_DATA_DIR
from werkzeug.exceptions import BadRequest  # Import BadRequest exception

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #

# Model Details
MODEL_NAME = "2025-01-14 - LGBMClassifier - business.joblib"
MODEL_PATH = Path(MODEL_DIR) / MODEL_NAME
THRESHOLD = 0.43

# Scaler Details
SCALER_NAME = "2025-01-14 - RobustScaler.joblib"
SCALER_PATH = Path(MODEL_DIR) / SCALER_NAME

# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR) / DATASET_NAME

# Loguru Configuration
LOG_PATH = Path(LOG_DIR) / "api"
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")

# Path existence check
paths_to_check = {
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Dataset": DATASET_PATH,
    }
for name, path in paths_to_check.items():
    if not path.exists():
        logger.error(f"{name} file not found at {path}")
        raise FileNotFoundError(f"{name} file not found at {path}")



# ==================================================================================================================== #
#                                            LOADING RESOURCES LOCALLY                                                 #
# ==================================================================================================================== #

# Load Model
try:
    model = joblib.load(MODEL_PATH)
    logger.success(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("An error occurred while loading the model. Please check the logs.")

# Load Scaler
try:
    robust_scaler = joblib.load(SCALER_PATH)
    logger.success(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    raise RuntimeError("An error occurred while loading the scaler. Please check the logs.")

# Load Dataset
try:
    dataset = pd.read_csv(DATASET_PATH)  # Load the CSV into a DataFrame
    logger.success(f"Dataset loaded successfully from {DATASET_PATH}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("An error occurred while loading the dataset. Please check the logs for more details.")


# ==================================================================================================================== #
#                                                     PREPROCESSING                                                    #
# ==================================================================================================================== #

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
try:
    logger.info("Calculating predicted probabilities for the first 20 rows...")
    # Exclude "SK_ID_CURR" for predictions
    user_data_first_20 = dataset_scaled.drop(columns=["SK_ID_CURR"]).head(20)

    # Predict probabilities for the first 20 rows
    proba_first_20 = model.predict_proba(user_data_first_20)[:, 1]

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
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict using the locally loaded model for a given SK_ID_CURR.
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
        proba = model.predict_proba(user_data)[0][1]
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
