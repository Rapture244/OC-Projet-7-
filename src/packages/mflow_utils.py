# === Logging and Utilities ===
from loguru import logger
from pathlib import Path
import sqlite3

# === Type Hints ===
from typing import List, Optional, Dict

# === MLflow Core ===
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, ViewType

# === MLflow Model Registry ===
from mlflow.entities.model_registry import RegisteredModel, ModelVersion

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



def erase_deleted_experiments(db_path: str) -> None:
    """
    Permanently deletes all deleted MLflow experiments from the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    logger.info("=== ERASING DELETED EXPERIMENTS ===")
    logger.debug(f"Using SQLite database at: {db_path}")

    client: MlflowClient = MlflowClient()
    deleted_experiments: List = client.search_experiments(view_type=ViewType.DELETED_ONLY)
    deleted_experiment_ids: List[int] = [int(exp.experiment_id) for exp in deleted_experiments]

    # Log the number of deleted experiments found
    logger.debug(f"Found {len(deleted_experiment_ids)} deleted experiments to process.")

    # Exit early if no deleted experiments are found
    if not deleted_experiment_ids:
        logger.success("No deleted experiments to hard delete.")
        return

    logger.debug(f"Hard deleting experiments with IDs: {deleted_experiment_ids}")

    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Delete records from the "experiments" table
            logger.debug("Deleting records from the 'experiments' table.")
            query = (
                f"DELETE FROM experiments WHERE experiment_id IN ({', '.join(['?'] * len(deleted_experiment_ids))})"
            )
            cursor.execute(query, deleted_experiment_ids)

            # Delete records from the "runs" table associated with the experiments
            logger.debug("Deleting records from the 'runs' table.")
            query = (
                f"DELETE FROM runs WHERE experiment_id IN ({', '.join(['?'] * len(deleted_experiment_ids))})"
            )
            cursor.execute(query, deleted_experiment_ids)

            # Commit changes
            conn.commit()
            logger.success(f"Successfully hard deleted experiments: {deleted_experiment_ids}")

    except sqlite3.Error as e:
        logger.error(f"Error interacting with SQLite database: {e}")
    finally:
        logger.debug("SQLite connection closed")

# usage example
# sqlite_db_path: str = f"{ML_FLOW_DIR}/ml_flow.db"
# erase_deleted_experiments(sqlite_db_path)



def delete_active_experiment(experiment_identifier: str) -> None:
    """
    Deletes an active MLflow experiment by its name or ID.

    Args:
        experiment_identifier (str): The name or ID of the experiment to delete.
    """
    logger.info("=== DELETING EXPERIMENT ===")
    client: MlflowClient = MlflowClient()

    try:
        # Determine if the identifier is numeric (assume it's an ID)
        if experiment_identifier.isdigit():
            experiment_id: str = experiment_identifier
            experiment: Optional[Experiment] = client.get_experiment(experiment_id)
        else:
            # Assume it's a name and retrieve the experiment by name
            experiment: Optional[Experiment] = client.get_experiment_by_name(experiment_identifier)
            experiment_id: Optional[str] = experiment.experiment_id if experiment else None

        # Validate the experiment exists
        if not experiment:
            logger.error(f"Experiment not found ---> {experiment_identifier:<25}")
            return

        # Ensure the experiment is active before attempting deletion
        if experiment.lifecycle_stage != "active":
            logger.warning(
                f"Experiment is not active ---> {experiment_identifier:<25} | Lifecycle: {experiment.lifecycle_stage}"
                )
            return

        # Delete the experiment
        client.delete_experiment(experiment_id)
        logger.success(f"Deleted experiment ---> {experiment.name:<25} | ID: {experiment_id}")

    except Exception as e:
        logger.error(f"Failed to delete experiment ---> {experiment_identifier:<25} | Error: {e}")

# Usage example
# delete_experiment("Fine-Tuning Models")
# delete_experiment("3")


# ==================================================================================================================== #
#                                                    MODEL REGISTRY                                                    #
# ==================================================================================================================== #
def list_all_registered_models_and_versions() -> None:
    """
    Lists all registered models and all their versions in the MLflow Model Registry.
    """
    logger.info("=== LISTING ALL REGISTERED MODELS AND ALL THEIR VERSIONS ===")
    client: MlflowClient = MlflowClient()
    registered_models = client.search_registered_models()

    if not registered_models:
        logger.success("No registered models found in the MLflow Model Registry\n")
        return

    for model in registered_models:
        logger.success(f"Model Name ---> '{model.name}'")

        # Fetch all versions of the current model
        model_versions = client.search_model_versions(f"name='{model.name}'")

        for version in model_versions:
            logger.debug(
                f" - Version {version.version:<3} | Stage: {version.current_stage:<10} | "
                f"Run ID: {version.run_id:<35} | Description: {version.description or 'None'}"
                )

# Usage Example:
# list_all_registered_models_and_versions()

def list_all_registered_models_and_versions_with_details() -> None:
    """
    Lists all registered models and all their versions in the MLflow Model Registry,
    including aliases and tags.
    """
    logger.info("=== LISTING ALL REGISTERED MODELS, VERSIONS, ALIASES, AND TAGS ===")
    client: MlflowClient = MlflowClient()
    registered_models = client.search_registered_models()

    if not registered_models:
        logger.success("No registered models found in the MLflow Model Registry\n")
        return

    for model in registered_models:
        logger.success(f"Model Name ---> '{model.name}'")

        # Fetch all versions of the current model
        model_versions = client.search_model_versions(f"name='{model.name}'")

        for version in model_versions:
            # Extract aliases from tags
            aliases = [
                key.replace("mlflow.alias.", "")
                for key, value in version.tags.items()
                if key.startswith("mlflow.alias.")
                ]

            # Get tags for the current version
            tags = version.tags

            logger.debug(
                f" - Version {version.version:<3} | Stage: {version.current_stage:<10} | "
                f"Run ID: {version.run_id:<35} | Description: {version.description or 'None'}"
                )
            logger.debug(f"   - Aliases: {aliases or 'None'}")
            logger.debug(f"   - Tags: {tags or 'None'}")

# mlflow_set_alias_and_tags(
#     registered_model_name="LGBMClassifier - business",
#     version=1,
#     alias="candidate",
#     tags={"environment": "staging", "custom_threshold": "0.46", "business_cost_std": "0.71"},
#     description="Initial staging model with tuned hyperparameters."
#     )
#

# ============================================= MLFLOW_SET_ALIAS_AND_TAGS ======================================== #
def mlflow_set_alias_tags_and_description(registered_model_name: str, version: int, alias: str, tags: Dict[str, str], description: Optional[str] = None) -> None:
    """
    Sets an alias, tags, and optionally a description for a specific version of a registered model
    in the MLflow Model Registry.

    Args:
        registered_model_name (str): Name of the registered model.
        version (int): Specific version of the model to target.
        alias (str): Alias to assign to the model version.
        tags (Dict[str, str]): Tags to associate with the model version.
        description (Optional[str]): Description to associate with the model version. Default is None.

    Returns:
        None
    """
    client = MlflowClient()

    try:
        # Log the header
        logger.info(f"{registered_model_name:<15} | Version {version:<10}")

        # Assign alias to the specific version
        client.set_registered_model_alias(name=registered_model_name, alias=alias, version=version)
        logger.success(f"    - Assigned alias @'{alias}'")

        # Add tags to the specific version
        for key, value in tags.items():
            client.set_model_version_tag(registered_model_name, version, key, value)
            logger.success(f"    - Added tag '{key}: {value}'")

        # Add a description if provided
        if description:
            client.update_model_version(
                name=registered_model_name,
                version=version,
                description=description
                )
            logger.success(f"    - Added description: '{description}'")
    except Exception as e:
        logger.error(f"Failed to set alias, tags, or description for model '{registered_model_name}' version '{version}': {e}")
        raise


def hard_delete_registered_model(model_name: str) -> None:
    """
    Hard deletes a registered model along with all its versions.

    Args:
        model_name (str): Name of the registered model to delete.
    """
    logger.info(f"=== DELETING REGISTERED MODEL ===")
    client: MlflowClient = MlflowClient()

    try:
        # Attempt to delete the registered model
        client.delete_registered_model(name=model_name)
        logger.success(f"Successfully hard deleted model ---> {model_name}")
    except Exception as e:
        # Log the error with details
        logger.error(f"Failed to hard delete model ---> {model_name} | Error: {e}")
        raise

# Usage Example:
# hard_delete_registered_model("MyModelName")


def hard_delete_model_version(model_name: str, version: int) -> None:
    """
    Hard deletes a specific version of a registered model.

    Args:
        model_name (str): Name of the registered model.
        version (int): Version number of the model to delete.
    """
    logger.info(f"=== DELETING MODEL VERSION ===")
    client: MlflowClient = MlflowClient()

    try:
        # Attempt to delete the specified model version
        client.delete_model_version(name=model_name, version=version)
        logger.success(f"Successfully deleted version {version} of model ---> {model_name}")
    except Exception as e:
        logger.error(f"Failed to delete version {version} of model ---> {model_name} | Error: {e}")

# Usage Example:
# hard_delete_model_version("MyModelName", version=3)


