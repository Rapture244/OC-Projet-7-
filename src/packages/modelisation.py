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


# ==================================================================================================================== #
#                                               MODEL SAVING AND LOADING                                               #
# ==================================================================================================================== #
    def save_scaler(self, scaler: Any, dir_path: Path, log_to_mlflow: bool = False) -> None:
        """
        Saves a fitted scaler to the specified directory and optionally logs it to MLflow as an artifact.

        Args:
            scaler (Any): The fitted scaler to save (e.g., RobustScaler).
            dir_path (Path): Directory to save the scaler locally.
            log_to_mlflow (bool): If True, logs the scaler to MLflow as an artifact.

        Raises:
            ValueError: If the scaler is not fitted or dir_path is invalid.
        """
        # Ensure the directory exists
        dir_path.mkdir(parents=True, exist_ok=True)

        # Dynamically generate the scaler name
        scaler_name = f"{type(scaler).__name__}"

        # Create a filename with the current date and scaler name
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{current_date} - {scaler_name}.joblib"
        scaler_file = dir_path / filename

        try:
            # Save locally using joblib
            joblib.dump(scaler, scaler_file)
            logger.success(f"Scaler saved locally at: {scaler_file}")

            # Log to MLflow as an artifact if enabled
            if log_to_mlflow and self.mlflow_tracking:
                try:
                    mlflow.log_artifact(str(scaler_file), artifact_path="scalers")
                    logger.success(f"Scaler logged to MLflow under artifact path 'scalers'")
                except Exception as e:
                    logger.error(f"Failed to log scaler to MLflow: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to save/log scaler: {e}")
            raise

    @staticmethod
    def save_model_locally(model: Any, scorer: str, dir_path: Path) -> str:
        """
        Saves a fitted model locally using joblib.

        Args:
            model (Any): The fitted model to save.
            scorer (str): Scorer used for evaluation.
            dir_path (Path): Directory to save the model locally.

        Returns:
            str: The name of the saved model (model_name).
        """
        dir_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_name = f"{type(model).__name__} - {scorer}"
        filename = f"{current_date}_{model_name}.joblib"
        model_file = dir_path / filename

        try:
            joblib.dump(model, model_file)  # Save model using joblib
            logger.success(f"==== [LOCAL] ====\n---> Model Saved at: {model_file}")
            return model_name  # Return the model name instead of the file path
        except Exception as e:
            logger.error(f"Failed to save model locally: {e}")
            raise


    @staticmethod
    def detect_framework(model: Any) -> str:
        """
        Detects the framework of the given model.

        Args:
            model (Any): The model object to detect.

        Returns:
            str: Framework name ("sklearn", "lightgbm", "xgboost").
        """
        if isinstance(model, (LogisticRegression, RandomForestClassifier, DummyClassifier)):
            return "sklearn"
        elif isinstance(model, lgb.LGBMClassifier):
            return "lightgbm"
        elif isinstance(model, xgb.XGBClassifier):
            return "xgboost"
        else:
            raise ValueError(f"Unsupported model type: {type(model).__name__}")


    def mlflow_log_model(self, model: Any, x_train: pd.DataFrame, registered_model_name: Optional[str] = None) -> None:
        """
        Logs a model as an artifact to the current MLflow run and optionally registers it in the MLflow Model Registry.

        Args:
            model (Any): The model to log.
            x_train (pd.DataFrame): Training data for inferring input-output signatures.
            registered_model_name (Optional[str]): If provided, registers the model in the Model Registry under this name.

        Returns:
            None
        """
        # Detect the framework
        framework = ModelPipeline.detect_framework(model)

        # Infer input-output signature
        signature = None
        try:
            if x_train is not None and hasattr(model, "predict"):
                predictions = model.predict(x_train.sample(min(50, len(x_train)), random_state=self.random_state))
                signature = infer_signature(x_train, predictions)
        except Exception as e:
            logger.warning(f"Could not infer signature: {e}")

        # Log the model
        artifact_path = "model"  # Default artifact path for the model
        try:
            if framework == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    registered_model_name=registered_model_name
                    )
            elif framework == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    registered_model_name=registered_model_name
                    )
            elif framework == "xgboost":
                mlflow.xgboost.log_model(
                    booster=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    registered_model_name=registered_model_name
                    )
            else:
                raise ValueError(f"Unsupported framework: {framework}")

            # Verify registration if a registered model name is provided
            if registered_model_name:
                self._verify_model_registration(registered_model_name)
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

    def _verify_model_registration(self, model_name: str) -> None:
        """
        Verifies if a model is successfully registered in the MLflow Model Registry.

        Args:
            model_name (str): The name of the registered model to verify.

        Returns:
            None

        Raises:
            RuntimeError: If the model is not found in the Model Registry.
        """
        client = MlflowClient()
        try:
            registered_model = client.get_registered_model(name=model_name)
            logger.success(f"Model --> '{model_name}' is successfully registered. Details:\n {registered_model}")
        except Exception as e:
            logger.error(f"Model '{model_name}' could not be found in the Model Registry: {e}")
            raise RuntimeError(f"Model '{model_name}' is not registered. Please check the registration process.") from e


    @staticmethod
    def load_model(model_path: Path) -> Any:
        """
        Loads a saved model from the specified file path.

        Args:
            model_path (Path): Path to the saved model file.

        Returns:
            Any: The loaded model object.
        """
        try:
            # Load the model using joblib
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")

            # Check if the model is LightGBM or XGBoost and re-wrap if needed
            if isinstance(model, lgb.Booster):
                logger.warning("Loaded a raw LightGBM Booster. Ensure the Python wrapper (LGBMClassifier/Regressor) is preserved.")
            elif isinstance(model, xgb.Booster):
                logger.warning("Loaded a raw XGBoost Booster. Ensure the Python wrapper (XGBClassifier/Regressor) is preserved.")

            return model
        except FileNotFoundError:
            logger.error(f"File not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise



# ==================================================================================================================== #
#                                                     ISUALIZATION                                                     #
# ==================================================================================================================== #
    def display_confusion_matrix(self, y_test: pd.Series, y_pred: pd.Series, model_name: str, scorer: str, threshold: float = 0.5, save_img: bool = False) -> None:
        """
        Displays and logs the confusion matrix to MLflow if tracking is enabled, and optionally saves the image locally.

        Args:
            y_test (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values by the model.
            model_name (str): Name of the model to display in the title.
            scorer (str): Scorer used for evaluation ('roc_auc', 'business', etc.).
            threshold (float): Threshold used for predictions. Default is 0.5.
            save_img (bool): Flag to save the image locally and log it to MLflow.

        Returns:
            None
        """
        # Compute the confusion matrix
        cm: np.ndarray = confusion_matrix(y_test, y_pred)
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Class 0 (Repaid)", "Class 1 (Not Repaid)"]
            )

        # Construct the title and filename based on the threshold
        if threshold == 0.5:
            title = f"Confusion Matrix for {model_name} (Scorer: {scorer})"
            filename = f"Confusion_Matrix_{model_name}_{scorer}.png"
        else:
            title = f"Confusion Matrix for {model_name} (Scorer: {scorer}) at Threshold {threshold:.2f}"
            filename = f"Confusion_Matrix_{model_name}_{scorer}_Threshold_{threshold:.2f}.png"

        # Plotting the confusion matrix without default annotations
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format='d')  # Default annotations with integer formatting
        ax.grid(False)  # Disable the grid
        plt.title(title, pad=20)
        plt.tight_layout()  # Adjust layout to make room for the title

        # Explicitly setting axis labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        # Save and log the plot if save_img is True
        if save_img:
            # Define the directory and file path for the image using pathlib
            image_dir: Path = Path(ROOT_DIR) / "assets" / "plots"
            image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            image_path: Path = image_dir / filename

            try:
                # Save the plot as a file
                plt.savefig(image_path, bbox_inches='tight')  # Use bbox_inches to include all elements
                logger.success(f"Confusion matrix image saved successfully at: {image_path}")

                # Log the image to MLflow
                if self.mlflow_tracking:
                    mlflow.log_artifact(str(image_path))  # Directly log to artifacts folder
                    logger.info(f"Confusion matrix image logged to MLflow successfully for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to save or log confusion matrix image: {e}")

        # Show plot
        plt.show()


    def display_cost_matrix(self, y_test: pd.Series, y_pred: pd.Series, model_name: str, scorer: str, threshold: Optional[float] = None, save_img: bool = False) -> None:
        """
        Displays and logs the cost matrix to MLflow if tracking is enabled, and optionally saves the image locally.

        Args:
            y_test (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values by the model.
            model_name (str): Name of the model to display in the title.
            scorer (str): Scorer used for evaluation ('roc_auc', 'business', etc.).
            threshold (Optional[float]): Threshold used for predictions. Default is None.
            save_img (bool): Flag to save the image locally and log it to MLflow.

        Returns:
            None
        """
        # Cost constants
        FN_BASE_COST = 10  # Base cost for FN
        FP_COST = 1        # Cost for FP

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        # Calculate costs for FN and FP, set TN and TP to 0
        cost_matrix = np.array([
            [0, -fp * FP_COST],   # Row for actual negative (TN, FP)
            [-fn * FN_BASE_COST, 0]  # Row for actual positive (FN, TP)
            ])

        # Define labels for the matrix
        labels = ["Class 0 (Repaid)", "Class 1 (Not Repaid)"]

        # Create the plot with constrained layout for better adjustment
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        cax = ax.matshow(cost_matrix, cmap="viridis")
        ax.grid(False)  # Disable grid lines

        # Add color bar with adjusted size
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # Annotate each cell with formatted values and dynamic text color
        def format_value(value):
            return f"{value:,.0f}".replace(",", " ")  # Replace commas with spaces

        # Determine the text color based on the cell's background brightness
        norm = plt.Normalize(vmin=cost_matrix.min(), vmax=cost_matrix.max())
        cmap = plt.cm.viridis

        for (i, j), value in np.ndenumerate(cost_matrix):
            background_color = cmap(norm(value))
            text_color = "white" if sum(background_color[:3]) / 3 < 0.5 else "black"
            ax.text(j, i, format_value(value), va='center', ha='center', color=text_color)

        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Move X-ticks to the bottom
        ax.xaxis.set_ticks_position('bottom')

        # Add labels and title
        if threshold is None or threshold == 0.5:
            title = f"Cost Matrix for {model_name} (Scorer: {scorer})"
            filename = f"Cost_Matrix_{model_name}_{scorer}.png"
        else:
            title = f"Cost Matrix for {model_name} (Scorer: {scorer}) at Threshold = {threshold:.2f}"
            filename = f"Cost_Matrix_{model_name}_{scorer}_Threshold_{threshold:.2f}.png"
        plt.title(title, pad=20)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Save and log the plot if save_img is True
        if save_img:
            # Define the directory and file path for the image
            image_dir: Path = Path(ROOT_DIR) / "assets" / "plots"
            image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            image_path: Path = image_dir / filename

            try:
                # Save the plot as a file
                plt.savefig(image_path, bbox_inches='tight')  # Use bbox_inches to include all elements
                logger.success(f"Cost matrix image saved successfully at: {image_path}")

                # Log the image to MLflow
                if self.mlflow_tracking:
                    mlflow.log_artifact(str(image_path))  # Directly log to artifacts folder
                    logger.info(f"Cost matrix image logged to MLflow successfully for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to save or log cost matrix image: {e}")

        # Show the plot
        plt.show()


    def display_confusion_cost_matrices(self, y_test: pd.Series, y_pred: pd.Series, model_name: str, scorer: str, threshold: float = 0.5, save_img: bool = False) -> None:
        """
        Displays the confusion matrix and cost matrix side by side in a single figure.
        Optionally saves the combined image locally and logs it to MLflow.

        Args:
            y_test (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values by the model.
            model_name (str): Name of the model to display in the title.
            scorer (str): Scorer used for evaluation ('roc_auc', 'business', etc.).
            threshold (float): Threshold used for predictions. Default is 0.5.
            save_img (bool): Flag to save the combined image locally and log it to MLflow.

        Returns:
            None
        """
        # Cost constants
        FN_BASE_COST = 10  # Base cost for FN
        FP_COST = 1        # Cost for FP

        # Compute the confusion matrix
        cm: np.ndarray = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compute the cost matrix
        cost_matrix = np.array([
            [0, -fp * FP_COST],   # Row for actual negative (TN, FP)
            [-fn * FN_BASE_COST, 0]  # Row for actual positive (FN, TP)
            ])

        # Compute the combined cost value
        combined_cost = -fn * FN_BASE_COST + -fp * FP_COST

        # Labels for matrices
        labels = ["Class 0 (Repaid)", "Class 1 (Not Repaid)"]

        # Construct the main title based on the model name, scorer, and threshold
        main_title = f"Confusion-Cost Matrices for {model_name} (Scorer: {scorer})"
        if threshold != 0.5:
            main_title += f" at Threshold {threshold:.2f}"

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(main_title, fontsize=16, weight='bold', y=1.05)  # Add main title with extra padding

        # ---- Plot Confusion Matrix ----
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[0], values_format='d')
        axes[0].set_title("Confusion Matrix")  # Subplot title
        axes[0].grid(False)  # Disable grid lines
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")

        # ---- Plot Cost Matrix ----
        cax = axes[1].matshow(cost_matrix, cmap="viridis")
        fig.colorbar(cax, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Cost Matrix (cost = {combined_cost:,.0f})")  # Subplot title with combined cost
        axes[1].grid(False)  # Disable grid lines
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].set_yticklabels(labels)
        axes[1].xaxis.set_ticks_position('bottom')
        axes[1].set_xlabel("Predicted Label")  # No y-axis label for cost matrix

        # Annotate the cost matrix with values
        def format_value(value):
            return f"{value:,.0f}".replace(",", " ")  # Replace commas with spaces

        norm = plt.Normalize(vmin=cost_matrix.min(), vmax=cost_matrix.max())
        cmap = plt.cm.viridis

        for (i, j), value in np.ndenumerate(cost_matrix):
            background_color = cmap(norm(value))
            text_color = "white" if sum(background_color[:3]) / 3 < 0.5 else "black"
            axes[1].text(j, i, format_value(value), va='center', ha='center', color=text_color)

        # Adjust layout to avoid overlapping elements
        plt.tight_layout()

        # Save and log the plot if save_img is True
        if save_img:
            # Define the directory and file path for the image
            filename = f"Combined_Matrix_{model_name}_{scorer}_Threshold_{threshold:.2f}.png"
            image_dir: Path = Path(ROOT_DIR) / "assets" / "plots"
            image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            image_path: Path = image_dir / filename

            try:
                # Save the plot as a file
                plt.savefig(image_path, bbox_inches='tight')  # Use bbox_inches to include all elements
                logger.success(f"Combined matrix image saved successfully at: {image_path}")

                # Log the image to MLflow
                if self.mlflow_tracking:
                    mlflow.log_artifact(str(image_path))  # Directly log to artifacts folder
                    logger.info(f"Combined matrix image logged to MLflow successfully for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to save or log combined matrix image: {e}")

        # Show the combined plot
        plt.show()


    @staticmethod
    def display_combined_confusion_matrices(directory: Path, model_names: list[str], fraction_prefix: str = "Fraction", save_combined: bool = False):
        """
        Combines and displays confusion matrix images from a directory, grouping by fraction, model name, and scoring method.

        Args:
            directory (Path): Path to the directory containing confusion matrix images.
            model_names (List[str]): List of model names to filter images for grouping.
            fraction_prefix (str): Prefix for fraction grouping in filenames. Defaults to "Fraction".
            save_combined (bool): If True, saves the combined images in the same directory.

        Returns:
            None
        """
        # Ensure the directory exists
        if not directory.exists():
            raise FileNotFoundError(f"The directory {directory} does not exist.")

        # Storing
        scoring_methods: List[str] = ["business", "roc_auc"]
        image_files: List[Path] = []

        # Fetch all PNG files matching any of the model names
        for model_name in model_names:
            image_files.extend(directory.glob(f"*{model_name}*.png"))

        if not image_files:
            logger.warning(f"No confusion matrix images found in {directory} for models: {', '.join(model_names)}.")
            return

        # Parse filenames to extract group keys (Fraction, Model, and Scoring)
        def extract_group_key(file_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            file_stem: str = file_path.stem
            parts: List[str] = file_stem.split(" - ")
            fraction: Optional[str] = next((p for p in parts if fraction_prefix in p), None)
            model: Optional[str] = parts[-1] if parts else None
            scoring: Optional[str] = next((p for p in scoring_methods if p in file_stem), None)
            return fraction, model, scoring


        # Group files by fraction and model
        image_files.sort()  # Sort for consistent grouping
        grouped_files: Dict[Tuple[str, str], Dict[str, Path]] = {}

        for file in image_files:
            fraction, model, scoring = extract_group_key(file)
            if not fraction or not model or not scoring:
                logger.warning(f"Skipping file {file}, missing fraction, model, or scoring key.")
                continue

            key: Tuple[str, str] = (fraction, model)
            grouped_files.setdefault(key, {})[scoring] = file

        # Process each group
        for (fraction, model), scoring_files in grouped_files.items():
            # Ensure both scoring methods are present
            if len(scoring_files) != len(scoring_methods):
                logger.info(f"Skipping group {fraction}-{model}, not all scoring methods found.")
                continue

            # Load images for the two scoring methods
            images: List[Image.Image] = [Image.open(scoring_files[method]) for method in scoring_methods]

            # Create a combined image (horizontal layout)
            widths, heights = zip(*(img.size for img in images))
            combined_width: int = sum(widths)
            max_height: int = max(heights)

            combined_image: Image.Image = Image.new("RGB", (combined_width, max_height))
            x_offset: int = 0

            for img in images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.width

            # Display the combined image
            plt.figure(figsize=(16, 8))
            plt.imshow(combined_image)
            plt.axis("off")
            plt.title(f"Combined Confusion Matrices: {fraction} - {model}")
            plt.show()

            # Optionally save the combined image
            if save_combined:
                combined_filename = f"Combined - {fraction} - {model}.png"
                combined_path = directory / combined_filename
                combined_image.save(combined_path)
                logger.info(f"Combined image saved as {combined_path}")


# ==================================================================================================================== #
#                                                   MODEL EVALUATION                                                   #
# ==================================================================================================================== #
    def instantiate_fit_model(self, X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict[str, Union[str, int, float, Any]]) -> BaseEstimator:
        """
        Instantiates a model using the best parameters and fits it, raising a warning if convergence fails.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            best_params (Dict[str, Union[str, int, float, Any]]): Dictionary of the best hyperparameters for the model.

        Returns:
            BaseEstimator: The instantiated and fitted model.

        Raises:
            ValueError: If the specified model type is unsupported.
        """
        # Extract the model type from the parameters and remove it from the dictionary
        params_copy: Dict[str, Any] = best_params.copy()
        model_type: str = params_copy.pop("model")

        # Dynamically instantiate the appropriate model based on the 'model' type
        try:
            if model_type == "LogisticRegression":
                model = LogisticRegression(**params_copy)
            elif model_type == "RandomForest":
                model = RandomForestClassifier(**params_copy)
            elif model_type == "XGBoost":
                model = xgb.XGBClassifier(**params_copy)
            elif model_type == "LightGBM":
                model = lgb.LGBMClassifier(**params_copy)
            elif model_type == "DummyClassifier":
                strategy: str = best_params.get("strategy", "most_frequent")
                model = DummyClassifier(strategy=strategy)
            else:
                raise ValueError(f"Model type {model_type} is not supported.")
        except Exception as e:
            logger.error(f"Error instantiating model of type {model_type}: {e}")
            return None  # Gracefully handle instantiation errors

        logger.success(f"Model object instantiated:\n {model}")

        # Fit the model and handle convergence warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)  # Treat ConvergenceWarning as an error
                model.fit(X_train, y_train)  # Attempt to fit the model
                logger.success(f"Model successfully fitted:\n {model}")
                return model  # Return the fitted model if successful
        except ConvergenceWarning as e:
            logger.warning(f"{model_type} model failed to converge: {e}")
        except Exception as e:
            logger.error(f"Error fitting {model_type} model: {e}")

        # If the model failed to fit, return None and log the failure
        logger.warning(f"Skipping model with parameters: {best_params}")
        return None


    def model_evaluation(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: Optional[float], scorer: str, save_img: bool = False, log_to_mlflow: bool = False) -> None:
        """
        Evaluates the model on the test data using specified metrics and updates the results DataFrame.

        Args:
            model (BaseEstimator): Trained classification model.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Test target vector.
            threshold (Optional[float]): Custom threshold for classification. Default is None.
            scorer (str): Scoring method used ('roc_auc' or 'business').
            save_img (bool, optional): If True, saves visualizations (confusion matrix, cost matrix) locally. Default is False.
            log_to_mlflow (bool, optional): If True, disables MLflow tracking for this function. Default is False.

        Returns:
            None
        """
        # Dynamically retrieve the model's class name for logging and display
        model_name: str = type(model).__name__

        # Generate probabilities for the positive class
        y_proba: pd.Series = model.predict_proba(X_test)[:, 1]

        # ---- Determine the threshold to use ----
        if threshold is None:
            threshold_to_use = 0.5
            logger.info(f"Evaluating model '{model_name}' with default threshold = {threshold_to_use}.")
        else:
            threshold_to_use = threshold
            logger.info(f"Evaluating model '{model_name}' with custom threshold = {threshold_to_use}.")

        # Generate predictions using the determined threshold
        y_pred: pd.Series = (y_proba >= threshold_to_use).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Confusion matrix breakdown

        # Compute metrics and costs for the determined threshold
        metrics = self._calculate_metrics(y_test, y_pred, y_proba, tn, fp, fn, tp)
        total_cost = metrics["cost"]
        total_profit = metrics["profit"]
        business_cost_std = self.business_cost_std(y_test, y_pred)

        # Log metrics and costs for the determined threshold
        logger.debug(
            f"Standardized Business Cost for threshold ({threshold_to_use}):\n"
            f"TN: {tn}, FP: {fp}\n"
            f"FN: {fn}, TP: {tp}\n"
            f"Profit: {total_profit:<10,} | Cost: {total_cost:<10,} | Standardized Cost: {business_cost_std:.2f}"
            )

        # Store results for the determined threshold
        results = {
            "Model": model_name,
            "Scorer": scorer,
            "Threshold": threshold_to_use,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "FNR": metrics["fnr"],
            "F2 Score": metrics["f2_score"],
            "ROC-AUC": metrics["roc_auc"],
            "FN": fn,
            "FP": fp,
            "Cost": total_cost,
            "Profit": total_profit,
            "Business Cost Std": business_cost_std,
            }

        # Log classification report for the determined threshold
        logger.success(f"Classification Report for '{model_name}' with threshold = {threshold_to_use}:\n"
                       f"{classification_report(y_test, y_pred, target_names=['Class 0 (Repaid)', 'Class 1 (Not Repaid)'])}")

        # Display combined visualizations (confusion matrix and cost matrix) for the determined threshold
        self.display_confusion_cost_matrices(y_test, y_pred, model_name, scorer, threshold=threshold_to_use, save_img=save_img)

        # ---- Update results and log metrics to MLflow (if enabled) ----
        new_results = pd.DataFrame([results])  # Convert results to DataFrame
        self._update_results_dataframe(new_results, model_name)  # Update the results DataFrame

        if self.mlflow_tracking and log_to_mlflow:
            mlflow.log_param("Threshold", results['Threshold'])
            mlflow.log_metric("Precision", round(results["Precision"], 4))
            mlflow.log_metric("Recall", round(results["Recall"], 4))
            mlflow.log_metric("FNR", round(results["FNR"], 4))
            mlflow.log_metric("F2_Score", round(results["F2 Score"], 4))
            mlflow.log_metric("ROC_AUC", round(results["ROC-AUC"], 4))
            mlflow.log_metric("FN", results["FN"])
            mlflow.log_metric("FP", results["FP"])
            mlflow.log_metric("Cost", round(results["Cost"], 2))
            mlflow.log_metric("Profit", round(results["Profit"], 2))
            mlflow.log_metric("Business_Cost_Std", round(results["Business Cost Std"], 4))

        return None


    def _calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series, y_proba: pd.Series, tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on predictions and confusion matrix.

        Args:
            y_test (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values.
            y_proba (pd.Series): Predicted probabilities for the positive class.
            tn (int): True Negatives count.
            fp (int): False Positives count.
            fn (int): False Negatives count.
            tp (int): True Positives count.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        # Constants for cost calculation
        FN_BASE_COST = 10  # Base cost for False Negatives
        FP_BASE_COST = 1        # Cost for False Positives

        # Constant profit & loss for profit calculation
        TN_PROFIT = 6
        FP_LOSS = 4
        FN_LOSS = 40
        TP_PROFIT = 0

        # Standard metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f2_score_val = fbeta_score(y_test, y_pred, beta=2, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Avoid division by zero

        # Calculate cost
        total_cost = -fp * FP_BASE_COST - fn * FN_BASE_COST

        # Calculate profit
        total_profit = TN_PROFIT * tn - FP_LOSS * fp - FN_LOSS * fn + TP_PROFIT * tp

        return {
            "precision": precision,
            "recall": recall,
            "f2_score": f2_score_val,
            "roc_auc": roc_auc,
            "fnr": fnr,
            "cost": total_cost,  # Add the cost as part of metrics
            "profit": total_profit,
            }


    def _update_results_dataframe(self, new_results: pd.DataFrame, model_name: str):
        """
        Update the results DataFrame with new evaluation results.

        Args:
            new_results (pd.DataFrame): DataFrame containing new evaluation results.
            model_name (str): Name of the evaluated model.
        """
        if self.results_df.empty:
            self.results_df = new_results
        else:
            for _, new_row in new_results.iterrows():
                # Check if the same Model, Scorer, and Threshold exist
                mask = (
                        (self.results_df["Model"] == new_row["Model"]) &
                        (self.results_df["Scorer"] == new_row["Scorer"]) &
                        (self.results_df["Threshold"] == new_row["Threshold"])
                )

                if mask.any():
                    # Replace existing rows with matching Model and Scorer
                    self.results_df.loc[mask, :] = new_row.values
                else:
                    # Append new row if no match
                    self.results_df = pd.concat(
                        [self.results_df, pd.DataFrame([new_row])], ignore_index=True
                        )

        # Sort by ROC-AUC
        self.results_df.sort_values("Business Cost Std", ascending=False, inplace=True)
        self.results_df.reset_index(drop=True, inplace=True)


    def threshold_evaluation_cost(self, fitted_model, X_test_processed, y_test):
        """
        Evaluates the impact of thresholds on the model's cost, identifies the best threshold,
        and plots the cost vs. threshold curve along with error breakdown.

        Args:
            fitted_model: Trained classification model.
            X_test_processed: Processed test feature matrix.
            y_test: Test target vector.

        Returns:
            float: The best threshold value.
        """
        logger.info("Starting threshold evaluation.")

        # Generate probabilities for the positive class
        y_proba = fitted_model.predict_proba(X_test_processed)[:, 1]
        thresholds = np.linspace(0, 0.7, 71)  # Thresholds from 0 to 0.7 with 71 points
        costs = []
        errors = []

        def get_FN_FP(y_true, y_pred):
            """Helper function to calculate FP and FN from confusion matrix."""
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()
            return FP, FN

        # Calculate costs and errors for each threshold
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.business_cost(y_test, y_pred)  # Use the static business cost method
            fp, fn = get_FN_FP(y_test, y_pred)
            costs.append(cost)
            errors.append((fp, fn))
            logger.debug(f"Threshold: {threshold:.2f}, Cost: {cost:.2f}, FP: {fp}, FN: {fn}")

        costs = np.array(costs)
        errors = np.array(errors)
        FP = errors[:, 0]
        FN = errors[:, 1]

        # Find the best threshold
        best_index = np.argmax(costs)  # Best threshold corresponds to the highest cost (less negative)
        best_threshold = thresholds[best_index]
        logger.success(f"Best threshold identified: {best_threshold:.2f} with cost: {costs[best_index]:.2f}")

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Errors
        ax[0].plot(thresholds, FP + FN, color="indigo", label="Total Errors")
        ax[0].plot(thresholds, FN, color="royalblue", label="False Negatives")
        ax[0].plot(thresholds, FP, color="orange", label="False Positives")
        ax[0].axvline(x=0.5, linestyle="dashed", color="slategray", label="Default Threshold (0.5)")
        ax[0].axvline(x=best_threshold, linestyle="dashed", color="crimson", label=f"Best Threshold ({best_threshold:.2f})")
        ax[0].set_title("Errors vs. Threshold")
        ax[0].set_xlabel("Threshold")
        ax[0].set_ylabel("Number of Errors")
        ax[0].legend()
        ax[0].grid()

        # Plot Costs
        ax[1].plot(thresholds, costs, color="mediumseagreen", label="Cost")
        ax[1].axvline(x=0.5, linestyle="dashed", color="slategray", label="Default Threshold (0.5)")
        ax[1].axvline(x=best_threshold, linestyle="dashed", color="crimson", label=f"Best Threshold ({best_threshold:.2f})")
        ax[1].set_title("Cost vs. Threshold (Less Negative is Better)")
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Cost (Less Negative is Better)")  # Clarify the optimization goal
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.show()

        return best_threshold


    def threshold_evaluation_profit(self, fitted_model, X_test_processed, y_test):
        """
        Evaluates the impact of thresholds on the model's profit, identifies the best threshold,
        and plots errors and profit vs. threshold.

        Args:
            fitted_model: Trained classification model.
            X_test_processed: Processed test feature matrix.
            y_test: Test target vector.

        Returns:
            float: The best threshold value that maximizes profit.
        """
        logger.info("Starting threshold evaluation.")

        # Generate probabilities for the positive class
        y_proba = fitted_model.predict_proba(X_test_processed)[:, 1]
        thresholds = np.round(np.arange(0, 1.01, 0.01), 2)  # Thresholds from 0.00 to 1.00 with step of 0.01

        def get_FN_FP(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()
            return FP, FN

        def get_profit_custom(y_true, y_pred):
            # Similar to how the cost of errors is calculated, the profit calculation must be refined by business experts.
            # Each client correctly identified as creditworthy and given a loan (TN) generates a profit of 6.
            # Each client incorrectly identified as creditworthy (FP) incurs a cost of 4 (lost revenue, slightly lower  than the TN profit to maintain discrimination).
            # Clients incorrectly granted loans despite being insolvent (FN) incur a significant cost of 40 (10 times the FP cost due to maintaining a relationship and potential recovery efforts).
            # Clients correctly identified as non-creditworthy and denied loans (TP) generate neither profit nor cost (0).
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()
            total_revenue = 6 * TN - 4 * FP - 40 * FN + 0 * TP
            return total_revenue

        def to_labels(prob_1, threshold):
            return (prob_1 >= threshold).astype(int)

        # Evaluate profit and errors for each threshold
        profits = []
        errors = []
        for threshold in thresholds:
            y_pred = to_labels(y_proba, threshold)
            profit = get_profit_custom(y_test, y_pred)
            fp, fn = get_FN_FP(y_test, y_pred)
            profits.append(profit)
            errors.append((fp, fn))
            logger.debug(f"Threshold: {threshold:.2f}, Profit: {profit}, FP: {fp}, FN: {fn}")

        # Find the best threshold
        profits = np.array(profits)
        idx_max = np.argmax(profits)
        best_threshold = thresholds[idx_max]
        logger.success(f"Highest profit: {profits[idx_max]:,.2f} at threshold: {best_threshold:.2f}")

        # Extract errors for plotting
        errors = np.array(errors)
        FP = errors[:, 0]
        FN = errors[:, 1]

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Errors
        ax[0].plot(thresholds, FP + FN, color="indigo", label="Total Errors")
        ax[0].plot(thresholds, FN, color="royalblue", label="False Negatives")
        ax[0].plot(thresholds, FP, color="orange", label="False Positives")
        ax[0].axvline(x=0.5, linestyle="dashed", color="slategray", label="Default Threshold (0.5)")
        ax[0].axvline(x=best_threshold, linestyle="dashed", color="crimson", label=f"Best Threshold ({best_threshold:.2f})")
        ax[0].set_title("Errors vs. Threshold")
        ax[0].set_xlabel("Threshold")
        ax[0].set_ylabel("Number of Errors")
        ax[0].legend()
        ax[0].grid()

        # Plot Profits
        ax[1].plot(thresholds, profits, color="mediumseagreen", label="Profit")
        ax[1].axvline(x=0.5, linestyle="dashed", color="slategray", label="Default Threshold (0.5)")
        ax[1].axvline(x=best_threshold, linestyle="dashed", color="crimson", label=f"Best Threshold ({best_threshold:.2f})")
        ax[1].set_title("Profit vs. Threshold")
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Profit")
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.show()

        return best_threshold


