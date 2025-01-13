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
