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


