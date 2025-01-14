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

# Names
model_name = "2025-01-11 - LGBMClassifier - business.joblib"
scaler_name = "2025-01-11 - RobustScaler.joblib"
dataset_name = "04_prediction_df.csv"

# Paths
log_path = Path(LOG_DIR / "api")
model_path = Path(MODEL_DIR / model_name)
scaler_path = Path(MODEL_DIR / scaler_name)
dataset_path = Path(PROCESSED_DATA_DIR / dataset_name)

# Loguru Configuration
logger.add(log_path, rotation="1 MB", retention="7 days", level="INFO")


# Checks
if not model_path.exists():
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

if not scaler_path.exists():
    logger.error(f"Scaler file not found at {scaler_path}")
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

if not dataset_path.exists():
    logger.error(f"Dataset file not found at {dataset_path}")
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

# Loading Model
try:
    model = joblib.load(model_path)
    # Additional verification step
    if not model:
        raise ValueError("Failed to load the model. The loaded object is invalid.")
    logger.success("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("An error occurred while loading the model. Please check the logs for more details.")

# Loading Scaler
try:
    robust_scaler = joblib.load(scaler_path)
    if not robust_scaler:
        raise ValueError("Failed to load the scaler. The loaded object is invalid.")
    logger.success("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    raise RuntimeError("An error occurred while loading the scaler. Please check the logs for more details.")


# Loading Dataset
try:
    dataset = pd.read_csv(dataset_path)  # Load the CSV into a DataFrame
    logger.success(f"Dataset loaded successfully from {dataset_path}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("An error occurred while loading the dataset. Please check the logs for more details.")

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



# ========== API ==========
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict using the preloaded model for a given SK_ID_CURR.
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
        user_id = str(user_request["id"])  # Convert input to string
        dataset["SK_ID_CURR"] = dataset["SK_ID_CURR"].astype(str)  # Convert column to string

        # Check if ID exists in the dataset
        if user_id not in dataset["SK_ID_CURR"].values:
            return jsonify({"error": f"ID {user_id} not found in the dataset"}), 404

        # Select the row for the given ID, excluding SK_ID_CURR
        user_data = dataset[dataset["SK_ID_CURR"] == user_id].drop(columns=["SK_ID_CURR"])

        # Predict using the loaded model
        proba = model.predict_proba(user_data)[:, 1][0]  # Get the probability of the positive class
        prediction = 1 if proba >= 0.46 else 0          # Apply the custom threshold

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

