# ================================================= IMPORTS ================================================= #
# Standard library imports
from pathlib import Path

# Third-party library imports
import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from loguru import logger

# Local application imports
from prod.paths import API_DIR, API_MODELS_DIR

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# Model Details
MODEL_NAME: str = "2025-01-17 - LGBMClassifier - business.joblib"
MODEL_PATH: Path = API_MODELS_DIR / MODEL_NAME

# Scaler Details
SCALER_NAME: str = "2025-01-17 - RobustScaler.joblib"
SCALER_PATH: Path = API_MODELS_DIR / SCALER_NAME

# Pipeline details
PIPELINE_NAME: str = "model_pipeline.joblib"
PIPELINE_PATH: Path = API_MODELS_DIR / PIPELINE_NAME


# ==================================================================================================================== #
#                                                       FUNCTIONS                                                      #
# ==================================================================================================================== #
def create_pipeline():
    """
    Creates a scikit-learn pipeline with RobustScaler and LightGBMClassifier.

    Returns:
        Pipeline: A trained pipeline combining preprocessing and model.
    """
    # Load Model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        logger.success(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("An error occurred while loading the model.")

    # Load Scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

    try:
        robust_scaler = joblib.load(SCALER_PATH)
        logger.success(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        raise RuntimeError("An error occurred while loading the scaler.")

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", robust_scaler),  # Apply RobustScaler
        ("model", model)            # Pre-trained LightGBM model
        ], verbose=True)  # Enable verbosity to check pipeline stages

    return pipeline


def save_pipeline():
    """
    Saves the pipeline to disk.
    """
    pipeline = create_pipeline()
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.success(f"Pipeline saved to {PIPELINE_PATH}")


def load_pipeline():
    """Loads the saved pipeline from disk."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline file not found at {PIPELINE_PATH}")

    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise RuntimeError("Pipeline loading failed, check logs.")



#
# # Imports
# from prod.utils import log_section_header
# from api.utils.database_utils import get_db_connection
# from api.utils.model_pipeline import load_pipeline
# import numpy as np
# import pandas as pd
#
# # Config
# pipeline = load_pipeline()
# THRESHOLD = 0.48
#
# # The function
# def log_predicted_probabilities():
#     """
#     Extracts the first 20 rows from the 'model_input_data' table, applies the prediction pipeline,
#     and logs the predicted probabilities.
#     """
#     log_section_header(title="Log predicted probabilities for the first 20 rows")
#
#     try:
#         logger.info("Fetching first 20 rows from 'model_input_data'...")
#
#         # Connect to the database
#         conn = get_db_connection()
#         cursor = conn.cursor()
#
#         # Fetch column names dynamically
#         cursor.execute("PRAGMA table_info(model_input_data);")
#         columns_info = cursor.fetchall()
#         all_columns = [col[1] for col in columns_info]  # Extract column names
#         feature_columns = [col for col in all_columns if col != "SK_ID_CURR"]  # Remove SK_ID_CURR
#
#         # Fetch the first 20 rows from the database
#         cursor.execute("SELECT * FROM model_input_data LIMIT 20;")
#         rows = cursor.fetchall()
#         conn.close()
#
#         if not rows:
#             logger.warning("No data found in 'model_input_data'.")
#             return
#
#         # Convert rows to Pandas DataFrame
#         df = pd.DataFrame(rows, columns=all_columns)  # Assign column names
#         sk_ids = df["SK_ID_CURR"].tolist()  # Extract client IDs
#         input_data = df.drop(columns=["SK_ID_CURR"], errors="ignore")  # Exclude SK_ID_CURR
#
#         # Ensure float32 dtype (optional but keeps consistency)
#         input_data = input_data.astype(np.float32)
#
#         # Apply prediction pipeline
#         logger.info("Calculating predicted probabilities...")
#         probabilities = pipeline.predict_proba(input_data)[:, 1]  # Probability of class 1
#
#         # Log the SK_ID_CURR with the probabilities
#         for sk_id, proba in zip(sk_ids, probabilities):
#             logger.debug(f"SK_ID_CURR: {sk_id}, Predicted Proba: {round(proba, 2)}")
#
#         logger.success("Logged predicted probabilities for the first 20 rows successfully.")
#
#     except Exception as e:
#         logger.error(f"Error logging predicted probabilities: {e}")
#         raise RuntimeError("An error occurred while logging predicted probabilities. Check logs for details.")


# ==================================================================================================================== #
#                                                     MAIN EXECUTION                                                   #
# ==================================================================================================================== #

def main():
    # create_pipeline()
    # save_pipeline()
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"SCALER_PATH: {SCALER_PATH}")
    logger.info(f"PIPELINE_PATH: {PIPELINE_PATH}")



if __name__ == "__main__":
    main()