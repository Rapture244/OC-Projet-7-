"""
This module defines and initializes configuration constants for the project.

Features:
1. **Dynamic Path Configuration**:
   - Paths are derived from environment variables, allowing for customization without modifying the codebase.
   - Default paths are project-relative, ensuring consistent directory structure.

2. **Key Paths**:
   - `ROOT_DIR`: Root directory of the project, defaulting to three levels above the script location.
   - `RAW_DATA_DIR`: Directory for raw datasets.
   - `PROCESSED_DATA_DIR`: Directory for processed datasets.
   - `MODEL_DIR`: Directory to store trained models and artifacts.
   - `LOG_DIR`: Directory for logging output.
   - `ML_FLOW_DIR`: Directory for MLflow tracking and artifacts.
   - `MLFLOW_TRACKING_URI`: URI for MLflow tracking, defaulting to a local SQLite database.

3. **Automatic Directory Creation**:
   - Ensures that all specified directories exist, creating them recursively if necessary.

4. **Logging Configuration**:
   - Logs the configured paths for debugging and validation purposes.

Dependencies:
- **Core Libraries**:
  - `os`, `pathlib`: For handling file system paths and environment variables.

- **Third-Party Libraries**:
  - `loguru`: Enhanced logging capabilities.

Usage:
- Import this module to access pre-configured paths for consistent file management across the project.
- Customize paths by setting corresponding environment variables before running the script.
"""


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
ROOT_DIR = Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parents[3]))

# Paths derived from ROOT_DIR
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", ROOT_DIR / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "assets" / "models"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))

# MLflow Tracking URI
ML_FLOW_DIR = Path(os.getenv("ML_FLOW_DIR", ROOT_DIR / "ml_flow"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ML_FLOW_DIR / 'ml_flow.db'}")


# Automatically create directories if they don't exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ML_FLOW_DIR, MODEL_DIR, LOG_DIR]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def main():
    # Logging the paths
    logger.debug(f"ROOT_DIR : {ROOT_DIR}")
    logger.debug(f"RAW_DATA_DIR : {RAW_DATA_DIR}")
    logger.debug(f"PROCESSED_DATA_DIR : {PROCESSED_DATA_DIR}")
    logger.debug(f"ML_FLOW_DIR : {ML_FLOW_DIR}")
    logger.debug(f"MLFLOW_TRACKING_URI : {MLFLOW_TRACKING_URI}")
    logger.debug(f"MODEL_DIR : {MODEL_DIR}")
    logger.debug(f"LOG_DIR : {LOG_DIR}")

if __name__ == "__main__":
    main()
