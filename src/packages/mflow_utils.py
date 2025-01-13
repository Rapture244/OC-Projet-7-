import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from loguru import logger
import sqlite3
from pathlib import Path
from typing import Optional, List

# ==================================================================================================================== #
#                                                      EXPERIMENTS                                                     #
# ==================================================================================================================== #
def list_all_experiments() -> None:
    """
    Lists all active and deleted MLflow experiments.

    This function uses the `MlflowClient` to retrieve and log details about
    active and deleted experiments separately.

    Logs the following:
    - Active experiments with their ID, name, and lifecycle stage.
    - Deleted experiments with their ID, name, and lifecycle stage.
    """
    client: MlflowClient = MlflowClient()

    try:
        # Log active experiments
        logger.success("=== ACTIVE EXPERIMENTS ===")
        active_experiments: List = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        for exp in active_experiments:
            logger.debug(f"ID: {exp.experiment_id:<5} | Name: {exp.name:<35} | Lifecycle: {exp.lifecycle_stage}")

        # Log deleted experiments
        logger.success("=== DELETED EXPERIMENTS ===")
        deleted_experiments: List = client.search_experiments(view_type=ViewType.DELETED_ONLY)
        for exp in deleted_experiments:
            logger.debug(f"ID: {exp.experiment_id:<5} | Name: {exp.name:<35} | Lifecycle: {exp.lifecycle_stage}")

    except Exception as e:
        # Log any exception that occurs
        logger.error(f"An error occurred while listing experiments: {e}")

# Example: List all experiments
# list_all_experiments()

