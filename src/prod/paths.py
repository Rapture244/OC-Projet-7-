# ====================================================== IMPORTS ===================================================== #
# Standard Library Imports
import os
from pathlib import Path

# Third-Party Library Imports
from loguru import logger


# ==================================================================================================================== #
#                                                    CONSTANT PATHS                                                    #
# ==================================================================================================================== #

# Use environment variables if defined; otherwise, default to project-relative paths.
ROOT_DIR = Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parents[2]))

# Paths derived from ROOT_DIR
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", ROOT_DIR / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "assets" / "models"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))
API_DIR = Path(os.getenv("API_DIR", ROOT_DIR / "api" ))

# MLflow Tracking URI
ML_FLOW_DIR = Path(os.getenv("ML_FLOW_DIR", ROOT_DIR / "ml_flow"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ML_FLOW_DIR / 'ml_flow.db'}")

def main():
    # Logging the paths
    logger.debug(f"ROOT_DIR : {ROOT_DIR}")
    logger.debug(f"RAW_DATA_DIR : {RAW_DATA_DIR}")
    logger.debug(f"PROCESSED_DATA_DIR : {PROCESSED_DATA_DIR}")
    logger.debug(f"ML_FLOW_DIR : {ML_FLOW_DIR}")
    logger.debug(f"MLFLOW_TRACKING_URI : {MLFLOW_TRACKING_URI}")
    logger.debug(f"MODEL_DIR : {MODEL_DIR}")
    logger.debug(f"LOG_DIR : {LOG_DIR}")
    logger.debug(f"API_DIR : {API_DIR}")

if __name__ == "__main__":
    main()
