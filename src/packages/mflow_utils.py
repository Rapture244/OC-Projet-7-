import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.entities import Experiment
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


def create_experiment(experiment_name: str, artifact_uri: str) -> str:
    """
    Creates a new MLflow experiment or retrieves an existing one.
    If the experiment exists but is in the 'deleted' lifecycle stage, it is restored and cleared of all previous runs.

    Args:
        experiment_name (str): Name of the experiment.
        artifact_uri (str): Location to store artifacts for the experiment.

    Returns:
        str: The ID of the created or existing experiment.
    """
    logger.info("=== CREATING OR RETRIEVING EXPERIMENT ===")
    client: MlflowClient = MlflowClient()

    try:
        # Check if the experiment already exists
        experiment: Optional[Experiment] = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            # Create the experiment if it doesn't exist
            experiment_id: str = client.create_experiment(name=experiment_name, artifact_location=artifact_uri)
            logger.success(f"Created new experiment ---> {experiment_name:<25} | ID: {experiment_id}\n")

        elif experiment.lifecycle_stage == "deleted":
            # Restore the deleted experiment
            experiment_id: str = experiment.experiment_id
            logger.warning(f"Experiment '{experiment_name}' is in 'deleted' state. Restoring it.")
            client.restore_experiment(experiment_id)
            logger.success(f"Restored experiment ---> {experiment_name:<25} | ID: {experiment_id}")

            # Delete all runs to clear the experiment
            logger.debug(f"Deleting all runs associated with experiment '{experiment_name}' to refresh it.")
            runs = client.search_runs(experiment_ids=[experiment_id])
            for run in runs:
                client.delete_run(run.info.run_id)
            logger.success(f"All runs associated with experiment '{experiment_name}' have been deleted\n.")

        else:
            # Retrieve the existing active experiment
            experiment_id: str = experiment.experiment_id
            logger.success(f"Experiment already exists ---> {experiment_name:<25} | ID: {experiment_id}\n")

        return experiment_id

    except Exception as e:
        # Log and re-raise any unexpected exceptions
        logger.error(f"An error occurred while creating or retrieving the experiment: {e}")
        raise
# Example: Create or get an experiment
# experiment_name = "Fine-Tuning Experiment"
# artifact_uri = f"file:///{ML_FLOW_DIR}/artifacts"
# experiment_id = create_experiment(experiment_name, artifact_uri)



