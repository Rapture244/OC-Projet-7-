# ---- Imports ---------------------------------------------------------------------------------------------------------
# ---- Standard Library Imports ----
import gc
import warnings
import json
import time
from pathlib import Path
from datetime import datetime
from threading import Thread
from functools import partial, lru_cache
from typing import Tuple, Type, Any, Dict, Optional, List, Union
from itertools import groupby

from typing import Optional

# ---- Third-Party Library Imports ----
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from loguru import logger
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import optuna
import plotly.io as pio
import GPUtil

# ---- Machine Learning and Data Preprocessing ----
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    )
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    make_scorer,
    )
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

# ---- MLflow ----
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# --- Local packages ---
from src.packages.constants.paths import ROOT_DIR


# ==================================================================================================================== #
#                                                 MODEL PIPELINE CLASS                                                 #
# ==================================================================================================================== #
class ModelPipeline(BaseEstimator, ClassifierMixin):
    """
    A pipeline for managing data splitting and handling machine learning model workflows.

    This class provides functionality for preparing datasets, managing results, and optionally
    integrating MLflow tracking for machine learning experiments.

    Attributes:
        random_state (int): Controls the randomness of the train-test split. Defaults to 42.
        test_size (float): Specifies the proportion of the dataset to include in the test split. Defaults to 0.3.
        mlflow_tracking (bool): If True, enables MLflow tracking if MLflow is available.
        study (Optional[object]): Stores the Optuna study object for plotting optimization history and parameter importance.
        results_df (pd.DataFrame): DataFrame to store results of model experiments.
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.3, mlflow_tracking: bool = False) -> None:
        """
        Initializes the ModelPipeline with specified parameters.

        Args:
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            test_size (float, optional): Proportion of dataset for testing. Defaults to 0.3.
            mlflow_tracking (bool, optional): Enables MLflow tracking if True. Defaults to False.
        """
        self.random_state: int = random_state
        self.test_size: float = test_size
        self.mlflow_tracking: bool = mlflow_tracking
        self.study: Optional[object] = None  # Stores the Optuna study object
        self.results_df: pd.DataFrame = pd.DataFrame()  # Stores experiment results

        # Log initialization details
        logger.info(
            "ModelPipeline initialized with:\n"
            f"{'Random State':<20} | {self.random_state}\n"
            f"{'Test Size':<20} | {self.test_size}\n"
            f"{'MLflow Tracking':<20} | {'Enabled' if self.mlflow_tracking else 'Disabled'}"
            )


# ==================================================================================================================== #
#                                                    UTILITY METHODS                                                   #
# ==================================================================================================================== #
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the dataset by sampling a fraction of it and splitting into features and target.

        Args:
            df (pd.DataFrame): The complete dataframe to sample from.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): The feature matrix.
                - y (pd.Series): The target vector.
        """
        # Log the original DataFrame shape
        logger.info(f"Original DataFrame shape: {df.shape}")

        # Split the DataFrame into features (X) and target (y)
        X: pd.DataFrame = df.drop('TARGET', axis=1)  # Features matrix
        y: pd.Series = df['TARGET']  # Target vector

        # Log the shapes of X and y for debugging purposes
        logger.debug(f"X shape: {X.shape}")  # Log feature matrix shape
        logger.debug(f"y shape: {y.shape}")  # Log target vector shape

        return X, y


    def get_results_df(self) -> pd.DataFrame:
        """
        Returns the current state of the results DataFrame.

        Returns:
            pd.DataFrame: The results DataFrame.
        """
        # Return the current results DataFrame
        return self.results_df



    def reset_results_df(self) -> None:
        """
        Resets the results DataFrame to an empty state.
        """
        self.results_df = pd.DataFrame()
        logger.success("Results DataFrame has been reset.")

# ==================================================================================================================== #
#                                                  DATA PREPROCESSING                                                  #
# ==================================================================================================================== #
    def split_data(self, X: pd.DataFrame, y: pd.Series, shuffle: bool = True, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
            stratify (bool, optional): Whether to stratify the split by the target variable. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                A tuple containing:
                - X_train (pd.DataFrame): Training feature matrix.
                - X_test (pd.DataFrame): Testing feature matrix.
                - y_train (pd.Series): Training target vector.
                - y_test (pd.Series): Testing target vector.

        Raises:
            ValueError: If the dataset is too small or stratified splitting is not feasible.
        """
        logger.info("Starting Data Splitting.")

        # Determine stratification parameter
        stratify_param: Optional[pd.Series] = y if stratify and y.value_counts().min() >= 2 else None

        # Validate dataset size and stratification compatibility
        if len(y) < 2:
            logger.error("The dataset is too small to split.")
            raise ValueError("Dataset must contain more than one sample for splitting.")

        # Perform the data split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = self.test_size,
            random_state = self.random_state,
            shuffle = shuffle,
            stratify = stratify_param)

        # == LOGGING ====
        # Map class labels to human-readable names
        class_names: Dict[int, str] = {0: "Class 0 (Repaid)", 1: "Class 1 (Not Repaid)"}

        # Calculate the distribution of target classes in the train and test sets
        train_distribution: Dict[str, str] = {
            class_names[int(k)]: f"{v / y_train.size * 100:.2f}% ({v})"
            for k, v in zip(*np.unique(y_train, return_counts=True))
            }

        test_distribution: Dict[str, str] = {
            class_names[int(k)]: f"{v / y_test.size * 100:.2f}% ({v})"
            for k, v in zip(*np.unique(y_test, return_counts=True))
            }

        # Log the distribution
        logger.debug(f"Class distribution in training set: {train_distribution}")
        logger.debug(f"Class distribution in testing set: {test_distribution}")

        logger.debug(
            "Types:\n"
            f"X_train ---> {type(X_train)}\n"
            f"X_test ---> {type(X_test)}\n"
            f"y_train ---> {type(y_train)}\n"
            f"y_test ---> {type(y_test)}"
            )

        # Successful loging
        logger.success(
            "Data splitting successful:\n"
            f"X_train: {X_train.shape} | X_test: {X_test.shape}\n"
            f"y_train: {y_train.shape} | y_test: {y_test.shape}."
            )

        return X_train, X_test, y_train, y_test


    def split_data_sample(self, X: pd.DataFrame, y: pd.Series, train_sample_size: float, shuffle: bool = True, stratify: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates a smaller stratified sample from the training dataset.

        Args:
            X (pd.DataFrame): Full training feature matrix.
            y (pd.Series): Full training target vector.
            train_sample_size (float): The proportion of the training dataset to sample.
            shuffle (bool, optional): Whether to shuffle the data before sampling. Defaults to True.
            stratify (bool, optional): Whether to stratify the sample by the target variable. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
                - X_train_sample (pd.DataFrame): Subsampled training feature matrix.
                - y_train_sample (pd.Series): Subsampled training target vector.

        Raises:
            ValueError: If the training dataset is too small or stratified sampling is not feasible.
        """
        logger.info("Starting training data subsampling.")

        # Validate sample size
        if not 0 < train_sample_size < 1:
            logger.error("`train_sample_size` must be a float between 0 and 1.")
            raise ValueError("`train_sample_size` must be a float between 0 and 1.")

        # Determine stratification parameter
        stratify_param: Optional[pd.Series] = y if stratify and y.value_counts().min() >= 2 else None

        # Validate dataset size and stratification compatibility
        if len(y) < 2:
            logger.error("The dataset is too small for sampling.")
            raise ValueError("Dataset must contain more than one sample for sampling.")

        if stratify and stratify_param is not None and y.value_counts().min() < 2:
            logger.error("Stratified sampling is not feasible due to insufficient class distribution.")
            raise ValueError("Each class needs at least 2 samples for stratified sampling.")

        # Perform the subsampling
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X,
            y,
            train_size=train_sample_size,
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=stratify_param
            )

        # Log the distribution of target classes in the sampled training set
        class_names: Dict[int, str] = {0: "Class 0 (Repaid)", 1: "Class 1 (Not Repaid)"}
        sample_distribution: Dict[str, str] = {
            class_names[int(k)]: f"{v / y_train_sample.size * 100:.2f}% ({v})"
            for k, v in zip(*np.unique(y_train_sample, return_counts=True))
            }

        logger.debug(f"Class distribution in subsampled training set: {sample_distribution}")

        logger.success(
            "Subsampled training data:\n"
            f"X_train_sample: {str(X_train_sample.shape):<20} | y_train_sample: {str(y_train_sample.shape)}"
            )
        return X_train_sample, y_train_sample


    def apply_preprocessing(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            preprocessing_used: Dict[str, Any],
            save_scaler: bool = False,
            ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[Any]]:
        """
        Applies preprocessing by scaling numeric features using RobustScaler and passing through non-numeric features.
        Handles resampling using SMOTETomek for imbalanced datasets.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_test (pd.DataFrame): Testing feature matrix.
            preprocessing_used (Dict[str, Any]): Dictionary specifying preprocessing settings:
            save_scaler (bool): Whether to include the fitted scaler object in the return value.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[RobustScaler]]:
                - Resampled and scaled training feature matrix.
                - Resampled training target vector.
                - Scaled testing feature matrix.
                - Fitted RobustScaler object, if `save_scaler` is True.

        Raises:
            ValueError: If preprocessing settings are incomplete or unsupported configurations are provided.
        """
        logger.info(f"Starting preprocessing with the following settings: {preprocessing_used}")

        # Extract the nested preprocessing settings
        preprocessing_settings = preprocessing_used.get("preprocessing_used", {})
        if not preprocessing_settings:
            logger.error("Missing 'preprocessing_used' key in preprocessing_used.")
            raise ValueError("Preprocessing settings must be provided under the key 'preprocessing_used'.")

        # Identify numeric features
        numeric_features: List[str] = X_train.select_dtypes(include=["number"]).columns.tolist()

        # Identify passthrough features (non-numeric features)
        passthrough_features = [col for col in X_train.columns if col not in numeric_features]
        logger.debug(f"Passthrough features: {passthrough_features}")

        # Validate preprocessing settings
        scaling_steps = preprocessing_settings.get("scaling", [])
        sampler_choice = preprocessing_settings.get("sampler")
        sampling_strategy = preprocessing_settings.get("sampling_strategy")
        smote_params = preprocessing_settings.get("custom_smote", {})
        tomek_params = preprocessing_settings.get("custom_tomek", {})

        if not all([sampler_choice, sampling_strategy, scaling_steps]):
            logger.error("Incomplete preprocessing settings in preprocessing_used.")
            raise ValueError("Preprocessing settings (sampler, sampling_strategy, scaling) must be provided.")

        # Extract scaler type and parameters
        scaler_type = scaling_steps[0].get("type")
        scaler_params = scaling_steps[0].get("parameters", {})
        if scaler_type != "RobustScaler":
            logger.error(f"Unsupported scaler type: {scaler_type}")
            raise ValueError(f"Only 'RobustScaler' is supported. Got {scaler_type}.")

        # Initialize RobustScaler
        scaler = RobustScaler(**scaler_params)

        # Scale numeric features
        X_train_scaled_numeric = pd.DataFrame(
            scaler.fit_transform(X_train[numeric_features]),
            columns=numeric_features,
            index=X_train.index
            )
        X_test_scaled_numeric = pd.DataFrame(
            scaler.transform(X_test[numeric_features]),
            columns=numeric_features,
            index=X_test.index
            )
        logger.debug("Numeric features scaled successfully.")

        # Concatenate numeric features with passthrough features
        X_train_final = pd.concat([X_train_scaled_numeric, X_train[passthrough_features]], axis=1)
        X_test_final = pd.concat([X_test_scaled_numeric, X_test[passthrough_features]], axis=1)

        # Ensure columns are in the original order
        X_train_final = X_train_final[X_train.columns]
        X_test_final = X_test_final[X_test.columns]
        logger.debug("Columns reordered to match original structure.")

        # Handle sampling
        if sampler_choice == "SMOTETomek":
            sampler = SMOTETomek(
                smote=SMOTE(**smote_params),
                tomek=TomekLinks(**tomek_params),
                sampling_strategy=sampling_strategy
                )
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_final, y_train)
            logger.success("Resampling completed successfully.")
        else:
            logger.error(f"Unsupported sampler choice: {sampler_choice}")
            raise ValueError(f"Sampler {sampler_choice} is not supported.")

        # Return results
        if save_scaler:
            return X_train_resampled, y_train_resampled, X_test_final, scaler
        else:
            return X_train_resampled, y_train_resampled, X_test_final


    def get_preprocessor_and_pipeline(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, classifier: BaseEstimator) -> ImbPipeline:
        """
        Returns a simplified pipeline with RobustScaler and SMOTETomek for preprocessing.

        Args:
            trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.
            X_train (pd.DataFrame): Training feature matrix.
            classifier (BaseEstimator): Classifier to include in the pipeline.

        Returns:
            ImbPipeline: A complete pipeline with preprocessing and the classifier.
        """
        # Suggest sampling strategy for SMOTETomek
        sampler_strategy: float = trial.suggest_categorical("sampling_strategy", [0.15, 0.25, 0.30, 0.35])

        # Define custom SMOTE for oversampling
        custom_smote = SMOTE(
            sampling_strategy=sampler_strategy,  # Oversample minority class to specified ratio
            random_state=self.random_state       # Use self.random_state for reproducibility
            )

        # Define custom TomekLinks for undersampling only the majority class
        custom_tomek = TomekLinks(
            sampling_strategy="majority"         # Only clean links from the majority class
            )

        # Define the oversampler
        oversampler = SMOTETomek(
            smote=custom_smote,
            tomek=custom_tomek,
            random_state=self.random_state
            )

        # Set up the RobustScaler
        robust_scaler = RobustScaler()

        # Identify numeric features
        numeric_features: List[str] = X_train.select_dtypes(include=["number"]).columns.tolist()

        # Define numeric transformation pipeline
        numeric_transformer: Pipeline = Pipeline(
            steps=[("robust_scaler", robust_scaler)]  # Apply RobustScaler
            )

        # Define preprocessor for numeric features
        preprocessor: ColumnTransformer = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features)],
            remainder="passthrough"
            )

        # Assemble the pipeline
        pipeline_steps = [
            ("preprocessor", preprocessor),  # Apply scaling
            ("oversampler", oversampler),    # Perform oversampling and undersampling
            ("classifier", classifier),      # Add the classifier
            ]

        clf: ImbPipeline = ImbPipeline(pipeline_steps)

        # Attach preprocessing metadata
        clf.metadata = {
            "preprocessing_used": {
                "sampler": "SMOTETomek",
                "sampling_strategy": sampler_strategy,
                "scaling": [{"type": "RobustScaler", "parameters": {}}],
                "custom_smote": {"sampling_strategy": sampler_strategy},
                "custom_tomek": {"sampling_strategy": "majority"}
                }
            }

        return clf


# =================================================================================================================== #
#                                                HYPERPARAMETER TUNING                                                #
# =================================================================================================================== #
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, model: str, scorer: str = 'roc_auc', max_trials: int = 20) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Tunes hyperparameters using Optuna for a given model using 'roc_auc' as the scoring metric.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            model (str): The model type to tune. Currently supports 'LogisticRegression', 'RandomForest',
                         'XGBoost', 'LightGBM', and 'DummyClassifier'.
            scorer (str): Scoring metric for evaluation ('roc_auc' or 'business').
            max_trials (int): Maximum number of trials for optimization. Defaults to 30.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                - Best hyperparameters (Dict[str, Any]): A dictionary of the best hyperparameters for the model.
                - Preprocessing details (Dict[str, Any]): A dictionary of preprocessing settings.

        Raises:
            ValueError: If the specified model is unsupported.
        """
        # Supported models and objectives
        supported_models: Dict[str, Any] = {
            'LogisticRegression': self.logistic_regression_objective,
            'RandomForest': self.random_forest_objective,
            'XGBoost': self.xgboost_objective,
            'LightGBM': self.lightgbm_objective,
            'DummyClassifier': self.dummy_classifier_objective,
            }

        # Check for valid model
        if model not in supported_models:
            supported_models_list: str = "\n".join(supported_models)
            logger.warning(f"Unsupported model type: {model}.\nPlease select one of the supported models:\n{supported_models_list}")
            raise ValueError(f"Unsupported model type: {model}. Supported models are:\n{supported_models_list}")

        # Log the start of tuning
        logger.info(f"Starting hyperparameter tuning using Optuna --> {model} & scorer: {scorer}")

        # Define the objective function
        objective_func: Any = partial(
            supported_models[model],
            X_train=X_train,
            y_train=y_train,
            scorer=scorer,  # Pass scorer dynamically
            )

        sampler = TPESampler(
            n_startup_trials=7,     # IMPORTANT: n_startup_trials defines initial random exploration, while max_trials sets the total number of trials including TPE-guided optimization
            n_ei_candidates=24,      # Default EI candidates
            multivariate=False,       # Joint parameter sampling
            group=False,              # Group correlated parameters
            seed=self.random_state                  # Random seed for reproducibility
            )


        pruner = optuna.pruners.NopPruner()       # No pruning to avoid plateau

        # Determine n_jobs based on whether the model is GPU-based
        gpu_models = ['XGBoost', 'LightGBM']
        use_gpu = model in gpu_models

        if use_gpu:
            n_jobs = 1  # Single job for GPU-based models to avoid contention
            logger.info(f"Detected GPU-based model: {model}. Using n_jobs=1 to avoid resource contention.")
        else:
            n_jobs = -1  # Use all available CPU cores for CPU-based models
            logger.info(f"Detected CPU-based model: {model}. Using n_jobs=-1 for parallel optimization.")

        # Define the optimization direction dynamically
        #direction = "maximize" if scorer == "roc_auc" else "maximize"

        # Create the study with the conditional sampler and pruner
        self.study: optuna.Study = optuna.create_study(
            direction= "maximize",  # business_cost -> returns a negative value, in make_scorer, greater_is_better= True which mean we try to minimize the cost so to keep the logic, 'maximize' with optuna
            sampler = sampler,
            pruner = pruner
            )

        # Conditionally run opimization based on if model is GPU based or not
        self.study.optimize(objective_func, n_trials=max_trials, n_jobs =n_jobs)

        # Retrieve metadata from the best trial
        best_trial = self.study.best_trial
        best_metadata: Dict[str, Any] = best_trial.user_attrs["metadata"]
        logger.debug(f"Raw metadata from best trial: {best_metadata}")

        # Extract preprocessing metadata and model parameters separately
        preprocessing_used = best_metadata.get("preprocessing_used", {})
        raw_best_params = best_metadata.get("best_params", {})

        best_params = {"model": model, **raw_best_params}


        # Log best parameters and preprocessing
        logger.success(f"Optuna optimization completed. Best score: {self.study.best_value:.4f}")
        logger.success(f"Best model hyperparameters: {best_params}")
        logger.success(f"Preprocessing steps used: {preprocessing_used}")

        # Log to MLflow if tracking is enabled
        if self.mlflow_tracking:
            # Log model name and scorer
            mlflow.log_param("model", model)
            mlflow.log_param("scorer", scorer)

            # Log best parameters as a single JSON string
            mlflow.log_param("best_parameters", json.dumps(best_params))

            # Log preprocessing details as a single JSON string
            mlflow.log_param("preprocessing_used", json.dumps(preprocessing_used))

            # Log the best score
            mlflow.log_metric("best_score", round(self.study.best_value, 4))


            logger.success(f"MLFLOW --> Model: {model} | Best parameters, preprocessing, and score logged as single entries")

        return best_params, preprocessing_used


    def plot_optimization_history(self, model: str, scorer: str, save_img: bool = False) -> None:
        """
        Plot optimization history using Matplotlib, save the plot locally, and log it to MLflow.

        Args:
            model (str): Name of the model being optimized.
            scorer (str): Scorer used during optimization.
            save_img (bool): Flag to save the image locally and log to MLflow.

        Raises:
            ValueError: If no Optuna study exists in the current instance.
        """
        if self.study is None:
            raise ValueError("No Optuna study found. Please run hyperparameter tuning first.")

        # Extract trial numbers and values
        trial_numbers = [t.number for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # Plot the optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, values, marker="o", linestyle="-", label="Trial Value")
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value")
        plt.title(f"{model} Optimization History (Scorer: {scorer})")
        plt.legend()
        plt.grid()

        if save_img:
            # Define the directory and file path for the image
            image_dir: Path = Path(ROOT_DIR) / "assets" / "plots"
            image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            image_path: Path = image_dir / f"optimization_history_{model}_{scorer}.png"

            try:
                # Save the plot locally
                plt.savefig(image_path, bbox_inches='tight')
                logger.success(f"Optimization history plot saved locally at: {image_path}")

                # Log the plot to MLflow
                if self.mlflow_tracking:
                    mlflow.log_artifact(str(image_path))  # Directly log to artifacts folder
                    logger.info(f"Optimization history plot logged to MLflow for model: {model}")
            except Exception as e:
                logger.error(f"Failed to save or log optimization history plot: {e}")

        plt.show()


    def plot_param_importance(self, model: str, scorer: str, save_img: bool = False) -> None:
        """
        Plot parameter importance from the Optuna study, save the plot locally, and log it to MLflow.

        Args:
            model (str): Name of the model being optimized.
            scorer (str): Scorer used during optimization.
            save_img (bool): Flag to save the image locally and log to MLflow.

        Raises:
            ValueError: If no Optuna study exists in the current instance.
        """
        if self.study is None:
            raise ValueError("No Optuna study found. Please run hyperparameter tuning first.")

        # Get parameter importances
        param_importances = optuna.importance.get_param_importances(self.study)

        if not param_importances:
            raise ValueError("No parameter importances found. Ensure the study contains completed trials.")

        # Log parameter importances using loguru
        logger.info("Logging parameter importances:")
        for param, importance in param_importances.items():
            logger.info(f"Parameter: {param:<20} | Importance: {importance:.4f}")

        # Convert to DataFrame for sorting and visualization
        importance_df = pd.DataFrame({
            "Parameter": list(param_importances.keys()),
            "Importance": list(param_importances.values())
            }).sort_values(by="Importance", ascending=True)  # Sort for horizontal bar plot

        # Plot parameter importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Parameter"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Parameter")
        plt.title(f"{model} Parameter Importance (Scorer: {scorer})")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_img:
            # Define the directory and file path
            image_dir: Path = Path(ROOT_DIR) / "assets" / "plots"
            image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            image_path: Path = image_dir / f"parameter_importance_{model}_{scorer}.png"

            try:
                # Save the plot locally
                plt.savefig(image_path, bbox_inches='tight')
                logger.success(f"Parameter importance plot saved locally at: {image_path}")

                # Log the plot to MLflow
                if self.mlflow_tracking:
                    mlflow.log_artifact(str(image_path))  # Directly log to artifacts folder
                    logger.info(f"Parameter importance plot logged to MLflow for model: {model}")
            except Exception as e:
                logger.error(f"Failed to save or log parameter importance plot: {e}")

        plt.show()


    def logistic_regression_objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scorer: str = 'roc_auc') -> float:
        """
        Defines the optimization objective for LogisticRegression using Optuna.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object for suggesting hyperparameters.
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            scorer (str): Scoring method to evaluate model performance. Either 'roc_auc' or 'business'.

        Returns:
            float: The mean score from cross-validation based on the specified scoring method.
        """
        # Define scoring metric
        if scorer == "roc_auc":
            scoring = "roc_auc"
        elif scorer == 'business':
            scoring = make_scorer(
                ModelPipeline.business_cost,
                greater_is_better=True,
                response_method="predict",
                )
        else:
            raise ValueError("Invalid scorer specified. Choose either 'roc_auc' or 'business'.")

        # Suggest hyperparameters
        solver_penalty_combinations: list[str] = [
            "saga_elasticnet",  # Scalable, supports elasticnet
            "saga_l1",          # Scalable, supports l1
            "saga_l2",          # Scalable, supports l2
            "sag_l2"            # Efficient, supports l2 only
            ]
        solver_penalty = trial.suggest_categorical("solver_penalty", solver_penalty_combinations)
        solver, penalty = solver_penalty.split("_")
        C = trial.suggest_loguniform("C", 0.01, 10)
        max_iter = trial.suggest_categorical("max_iter", [1000, 2000])
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9) if penalty == "elasticnet" else None

        # Model parameters
        model_params = {
            'C': C,
            'solver': solver,
            'penalty': penalty,
            'max_iter': max_iter,
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1,
            }
        if l1_ratio is not None:
            model_params['l1_ratio'] = l1_ratio

        # Create classifier
        classifier = LogisticRegression(**model_params)

        # Create pipeline with preprocessing
        pipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

        # Attach metadata
        pipeline.metadata = {
            "preprocessing_used": pipeline.metadata,  # Preprocessing details
            "best_params": model_params,             # Model parameters
            }

        # Save metadata to the trial
        trial.set_user_attr("metadata", pipeline.metadata)

        # Evaluate the model using cross-validation with pruning on convergence warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1)

        except ConvergenceWarning as e:
            logger.warning(f"Trial {trial.number} pruned due to convergence warning: {e}")
            raise TrialPruned(f"Trial pruned due to convergence warning: {e}")

        # Log the trial score
        mean_score = np.mean(scores)
        logger.info(
            f"Trial {trial.number}: "
            f"Score = {mean_score:.4f}, "
            f"Scorer = {scoring}"
            )

        return mean_score


    def random_forest_objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scorer: str) -> float:
        """
        Define the optimization objective for RandomForestClassifier.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object for suggesting hyperparameters.
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            scorer (str): Scoring method to evaluate the model.

        Returns:
            float: The mean score from cross-validation based on the chosen scorer.
        """

        # Define the scorer
        if scorer == "roc_auc":
            scoring = "roc_auc"
        elif scorer == 'business':
            scoring = make_scorer(
                ModelPipeline.business_cost,  # Using your static method
                greater_is_better=True,       # Keep the raw score positive such that Optuna can maximize
                response_method="predict",    # Ensure binary predictions
                )
        else:
            raise ValueError("Invalid scorer specified. Choose either 'roc_auc' or 'business'.")

        # Hyperparameters to tune
        model_params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),  # Reduce range for faster tuning
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, None]),  # Focus on meaningful depths
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),  # Retain as it's crucial for tree growth
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),  # Focus on commonly used options
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),  # Sampling impacts generalization
            "random_state": self.random_state,  # Consistency in results
            "n_jobs": -1,  # Use all available cores
            }
        # Add class_weight if the scorer is 'business'
        if scorer == 'business':
            model_params['class_weight'] = {0: 1, 1: 10}

        # Model and pipeline
        classifier: RandomForestClassifier = RandomForestClassifier(**model_params)
        pipeline: ImbPipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

        # Attach metadata
        pipeline.metadata = {
            "preprocessing_used": pipeline.metadata,  # Preprocessing details
            "best_params": model_params,              # Model parameters
            }

        # Evaluate the model using cross-validation
        skf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores: np.ndarray = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1)

        # Calculate mean score
        mean_score = np.mean(scores)

        # Log the trial score
        logger.info(
            f"Trial {trial.number}: "
            f"Score = {mean_score:.4f}, "
            f"Scorer = {scoring}"
            )

        # Save metadata for the trial
        trial.set_user_attr("metadata", pipeline.metadata)

        return mean_score


    def xgboost_objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scorer: str) -> float:
        """
        Optimization objective for XGBoost.
        """
        # Define the parameter space for XGBoost
        model_params = {
            'verbosity': 1,  # Controls verbosity of training
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'auc',  # Evaluation metric
            'tree_method': 'hist',  # Use histogram-based optimization
            'device': 'cuda',  # Use GPU for training

            # High-impact parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  # Allow slower learning rates
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),  # Wider range for boosting rounds
            'max_depth': trial.suggest_int('max_depth', 2, 10),  # Include shallower trees
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Wider range for balancing splits
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),  # Increase range for regularization
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Allow slightly smaller subsamples
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Allow smaller feature fractions
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),

            # Fixed Parameters
            'random_state': self.random_state,  # Ensures reproducibility
            }

        if scorer == 'business':  # Custom scoring scenario
            model_params['scale_pos_weight'] = 10  # Adjust for class imbalance

        # Create the XGBoost classifier
        classifier = xgb.XGBClassifier(**model_params)

        # Create the pipeline with preprocessing
        pipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

        # Attach full metadata but ensure separation of preprocessing and model parameters
        pipeline.metadata = {
            "preprocessing_used": pipeline.metadata,  # Preprocessing details
            "best_params": model_params,  # Model parameters
            }

        # Custom scoring function for business cost
        if scorer == "roc_auc":
            scoring = make_scorer(roc_auc_score)
        elif scorer == 'business':
            scoring = make_scorer(
                ModelPipeline.business_cost,  # Using your static method
                greater_is_better=True,                # Keep the raw score positive such that optuna can try to minize it
                response_method="predict",              # Ensure binary predictions
                )
        else:
            raise ValueError("Invalid scorer specified.")

        # Cross-validation with n_jobs=1 to prevent GPU contention
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=skf, n_jobs=1)

        # Log the trial score
        mean_score = np.mean(scores)
        logger.info(
            f"Trial {trial.number}: "
            f"Score = {mean_score:.4f}, "
            f"Scorer = {scoring}"
            )

        # Log GPU state during and after the trial
        gpus = GPUtil.getGPUs()
        if not gpus:
            logger.warning("No GPUs detected after the trial. Ensure GPU is properly configured.")
        else:
            for gpu in gpus:
                logger.info(f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                            f"Load: {gpu.load * 100:.1f}%, "
                            f"Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")

        # Save metadata for the trial
        trial.set_user_attr("metadata", pipeline.metadata)

        # Return mean score
        return mean_score


    def lightgbm_objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scorer: str = 'roc_auc') -> float:
        """
        Define the optimization objective for LightGBM.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object for suggesting hyperparameters.
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            scorer (str): Scoring method to evaluate model performance. Either 'roc_auc' or 'business'.

        Returns:
            float: The mean score from cross-validation based on the chosen scorer.
        """
        # Suggest parameters for LightGBM
        model_params: Dict[str, Any] = {
            'objective': 'binary',
            'metric': 'auc',  # Fix metric to 'auc' as it has low importance for the business metric
            'verbosity': -1,  # Minimal verbosity
            'boosting_type': 'gbdt',

            # High-Impact Parameters (Retained and Refined)
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),

            # Fixed or Less Important Parameters
            'random_state': self.random_state,  # Use consistent random state
            'device': 'gpu',  # GPU for faster computation
            'gpu_platform_id': 0,
            'gpu_device_id': 0
            }

        # Add class_weight if the scorer is 'business'
        if scorer == 'business':
            model_params['class_weight'] = {0: 1, 1: 10}


        # Create the LightGBM classifier
        classifier: lgb.LGBMClassifier = lgb.LGBMClassifier(**model_params)

        # Create the pipeline with preprocessing
        pipeline: ImbPipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

        # Attach full metadata but ensure separation of preprocessing and model parameters
        pipeline.metadata = {
            "preprocessing_used": pipeline.metadata,  # Preprocessing details
            "best_params": model_params,  # Model parameters
            }

        # Custom scoring function for business cost
        if scorer == "roc_auc":
            scoring = "roc_auc"
        elif scorer == 'business':
            scoring = make_scorer(
                ModelPipeline.business_cost,  # Using your static method
                greater_is_better=True,                # Keep the raw score positive such that optuna can try to minize it
                response_method="predict",              # Ensure binary predictions
                )
        else:
            raise ValueError("Invalid scorer specified. Choose either 'roc_auc' or 'business'.")

        # Cross-validation with n_jobs=1 to prevent GPU contention
        skf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores: np.ndarray = cross_val_score(pipeline, X_train, y_train, scoring= scoring, cv=skf, n_jobs=1)

        # Log the trial score
        mean_score = np.mean(scores)
        logger.info(
            f"Trial {trial.number}: "
            f"Score = {mean_score:.4f}, "
            f"Scorer = {scoring}"
            )

        # Log GPU state during and after the trial
        gpus = GPUtil.getGPUs()
        if not gpus:
            logger.warning("No GPUs detected after the trial. Ensure GPU is properly configured.")
        else:
            for gpu in gpus:
                logger.info(f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                            f"Load: {gpu.load * 100:.1f}%, "
                            f"Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")

        # Save metadata for the trial
        trial.set_user_attr("metadata", pipeline.metadata)

        # Return mean score
        return mean_score


    def dummy_classifier_objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scorer: str) -> float:
        """
        Objective function for tuning the DummyClassifier, including preprocessing steps.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object for suggesting parameters.
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            scorer (str): scorer method to evaluate the model.

        Returns:
            float: The mean score from cross-validation, including preprocessing.
        """
        # Define the scorer
        if scorer == "roc_auc":
            scoring = "roc_auc"
        elif scorer == 'business':
            scoring = make_scorer(
                ModelPipeline.business_cost,  # Using your static method
                greater_is_better=True,       # Keep the raw score positive such that Optuna can maximize
                response_method="predict",    # Ensure binary predictions
                )
        else:
            raise ValueError("Invalid scorer specified. Choose either 'roc_auc' or 'business'.")

        # Suggest strategy for the DummyClassifier
        strategy: str = trial.suggest_categorical("strategy", ["stratified"])

        # Model + pipeline
        classifier: DummyClassifier = DummyClassifier(strategy=strategy, random_state=self.random_state)
        pipeline: ImbPipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

        # Attach metadata
        pipeline.metadata = {
            "preprocessing_used": pipeline.metadata,  # Preprocessing details
            "best_params": {"strategy": strategy},    # Model parameters
            }

        # Setup cross-validation
        skf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores: np.ndarray = cross_val_score(pipeline, X_train, y_train, scoring=scorer, cv=skf, n_jobs=-1)

        # Log the trial score
        mean_score = np.mean(scores)
        logger.info(
            f"Trial {trial.number}: "
            f"Score = {mean_score:.4f}, "
            f"Scorer = {scoring}"
            )

        # Save metadata for the trial
        trial.set_user_attr("metadata", pipeline.metadata)

        return mean_score


    def refined_lightgbm_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, max_trials: int = 30) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Performs a refined hyperparameter tuning for LightGBM using Optuna and business score as the scorer.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            max_trials (int): Maximum number of trials for Optuna. Defaults to 50.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                - Best hyperparameters (Dict[str, Any]): A dictionary of the best hyperparameters for LightGBM.
                - Preprocessing details (Dict[str, Any]): A dictionary of preprocessing settings.
        """
        def lightgbm_objective(trial: optuna.trial.Trial) -> float:
            """Objective function for Optuna optimization."""
            # Define a refined hyperparameter space based on the logs
            model_params: Dict[str, Any] = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 0.5),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 120),
                'class_weight': {0: 1, 1: 10},  # Fixed for imbalanced target
                'random_state': self.random_state,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
                }

            # Create the LightGBM classifier
            classifier: lgb.LGBMClassifier = lgb.LGBMClassifier(**model_params)

            # Preprocessing and pipeline creation
            pipeline: ImbPipeline = self.get_preprocessor_and_pipeline(trial, X_train, classifier)

            # Business scoring function
            scoring = make_scorer(
                ModelPipeline.business_cost,
                greater_is_better=True,
                response_method="predict"
                )

            # Cross-validation with n_jobs=1 to prevent GPU contention
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=skf, n_jobs=1)

            # Log the trial score
            mean_score = np.mean(scores)
            logger.info(
                f"Trial {trial.number}: "
                f"Score = {mean_score:.4f}, "
                f"Scorer = {scoring}"
                )

            # Log GPU state during and after the trial
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.warning("No GPUs detected after the trial. Ensure GPU is properly configured.")
            else:
                for gpu in gpus:
                    logger.info(f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                                f"Load: {gpu.load * 100:.1f}%, "
                                f"Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")

            # Save trial metadata
            trial.set_user_attr("metadata", {
                "preprocessing_used": pipeline.metadata,
                "best_params": model_params
                })

            return mean_score

        # Configure Optuna sampler and study
        sampler = TPESampler(
            n_startup_trials=10,  # n_startup_trials defines initial random exploration, while max_trials sets the total number of trials including TPE-guided optimization.
            n_ei_candidates=24,
            seed=self.random_state
            )

        # Assign study to self for future access
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        # Optimize with n_jobs=1 for GPU-based models
        self.study.optimize(lightgbm_objective, n_trials=max_trials, n_jobs=1)

        # Retrieve best parameters and preprocessing metadata
        best_trial = self.study.best_trial
        best_metadata: Dict[str, Any] = best_trial.user_attrs["metadata"]
        best_params = {"model": "LightGBM", **best_metadata.get("best_params", {})}
        preprocessing_used = best_metadata.get("preprocessing_used", {})

        # Logging results
        logger.success(f"Refined LightGBM tuning completed. Best score: {self.study.best_value:.4f}")
        logger.success(f"Best hyperparameters: {best_params}")
        logger.success(f"Preprocessing details: {preprocessing_used}")

        # Log results to MLflow
        if self.mlflow_tracking:
            mlflow.log_param("model", "LightGBM")
            mlflow.log_param("scorer", "business")
            mlflow.log_param("best_parameters", json.dumps(best_params))
            mlflow.log_param("preprocessing_used", json.dumps(preprocessing_used))
            mlflow.log_metric("best_score", round(self.study.best_value, 4))
            logger.success("MLflow logging completed for LightGBM tuning.")

        return best_params, preprocessing_used


