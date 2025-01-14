from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from typing import Any
from packages.constants.paths import MODEL_DIR, LOG_DIR, PROCESSED_DATA_DIR

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #

# Model Details
MODEL_NAME = "2025-01-14 - LGBMClassifier - business.joblib"
MODEL_PATH = Path(MODEL_DIR / MODEL_NAME)
THRESHOLD = 0.43

# Scaler Details
SCALER_NAME = "2025-01-14 - RobustScaler.joblib"
SCALER_PATH = Path(MODEL_DIR / SCALER_NAME)

# Dataset Path
DATASET_NAME = "04_prediction_df.csv"
DATASET_PATH = Path(PROCESSED_DATA_DIR / DATASET_NAME)


# Loguru Configuration
LOG_PATH = Path(LOG_DIR / "api")
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")


# ================================================ PYTEST HERE MAYBE ? =============================================== #
# File details
paths_to_check = {
    "Model": Path(MODEL_DIR / "2025-01-14 - LGBMClassifier - business.joblib"),
    "Scaler": Path(MODEL_DIR / "2025-01-14 - RobustScaler.joblib"),
    "Dataset": Path(PROCESSED_DATA_DIR / "04_prediction_df.csv")
    }

# Path existence check
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
    Endpoint to predict using the locally loaded model for a given SK_ID_CURR.
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
        prediction = int(proba >= 0.5)  # Default threshold of 0.5
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
    app.run(debug=True)
