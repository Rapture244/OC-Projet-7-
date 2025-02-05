"""
This module provides database interaction utilities and model-related functions for credit scoring.

Key Features:
1. Connects to an SQLite database to fetch client and model input data.
2. Extracts client-specific information, including personal, financial, and credit details.
3. Computes SHAP-based global and local feature importance for explainability.
4. Generates histograms and boxplots for client positioning visualization.
5. Loads machine learning artifacts such as trained models and scalers.

Database & Model Paths:
- `credit_scoring.sqlite`: Stores customer and model input data.
- `RobustScaler.joblib`: Pretrained scaler for feature normalization.
- `shap_explainer.joblib`: SHAP explainer for interpretability.

Core Functions:
- `extract_client_info(client_id)`: Retrieves structured client information.
- `extract_predict_client_info(client_id)`: Fetches preprocessed features for model inference.
- `extract_feat_global_importance()`: Computes global feature importance via SHAP.
- `extract_local_feature_importance(client_id)`: Computes SHAP values for a specific client.
- `extract_client_positioning_plot(client_id)`: Generates a visualization of the client's position in the dataset.
- `extract_feature_positioning_plot(client_id, feature_name)`: Compares a client’s feature value to the overall distribution.

Dependencies:
- SQLite: For database queries.
- Pandas & NumPy: Data manipulation and transformations.
- SHAP: Model explainability framework.
- Matplotlib & Seaborn: Data visualization libraries.

Notes:
- Ensure `DATABASE_DIR`, `API_MODELS_DIR`, and `API_STATIC_DIR` are correctly set.
- Data extraction functions assume consistent schema in `model_input_data`.
- SHAP computations rely on a trained pipeline and saved explainer model.
"""


# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from pathlib import Path
import sqlite3
from typing import Optional, Dict, Any, List
import io
import os

# Third-party library imports
from loguru import logger
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import RobustScaler

# For PostgrSQL ! To be used instead of sqlite3 !
import psycopg2

# Local application imports
from prod.paths import DATABASE_DIR, API_STATIC_DIR, API_MODELS_DIR
from api.utils.model_pipeline import load_pipeline


# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# Model paths
SCALER_PATH: Path = API_MODELS_DIR / "2025-01-17 - RobustScaler.joblib"
SHAP_EXPLAINER_PATH: Path = API_MODELS_DIR / "shap_explainer.joblib"

# LOCAL SQLITE DATABASE
DB_PATH: Path = DATABASE_DIR / "credit_scoring.sqlite"

# HEROKU POSTGRES
# When you deploy your app to Heroku, Heroku automatically provides the correct database URL in the environment based on this !
# DATABASE_URL = os.getenv("DATABASE_URL")  # Fetch from Heroku environment variables


# ==================================================================================================================== #
#                                                       FUNCTIONS                                                      #
# ==================================================================================================================== #

# LOCAL DB CONNECTION
def get_db_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    try:
        conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enables dictionary-like row access
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise


# # REMOTE DB CONNECTION
# def get_db_connection():
#     """Create and return a connection to the PostgreSQL database."""
#     try:
#         conn = psycopg2.connect(DATABASE_URL, sslmode="require")
#         return conn
#     except psycopg2.Error as e:
#         print(f"Database connection failed: {e}")
#         return None


# ================================================ EXTRACT CLIENT INFO =============================================== #
def extract_client_info(client_id: int) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Extracts detailed information about a given client ID by querying the 'customer_data' table.

    Args:
        client_id (int): The client ID to look for.

    Returns:
        Optional[Dict[str, Dict[str, Any]]]: A dictionary containing personal, financial, and credit information
                                             for the client, or None if the client ID is not found.
    """
    try:
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()

        # Query the database for the client data
        cursor.execute("SELECT * FROM customer_data WHERE SK_ID_CURR = ?;", (client_id,))
        row: Optional[sqlite3.Row] = cursor.fetchone()
        conn.close()

        if row is None:
            logger.warning(f"Client ID {client_id} not found in the database.")
            return None

        # Extract data from the row
        gender: str = "Female" if row["CODE_GENDER"] == "F" else "Male"
        age_years: int = round(abs(int(row["DAYS_BIRTH"])) / 365.25)
        own_car: str = "Yes" if row["FLAG_OWN_CAR"] == "Y" else "No"
        own_realty: str = "Yes" if row["FLAG_OWN_REALTY"] == "Y" else "No"

        personal_profile: Dict[str, Any] = {
            "Age (years)": age_years,
            "Gender": gender,
            "Family status": row["NAME_FAMILY_STATUS"],
            "Children": int(row["CNT_CHILDREN"]),
            "Family members": int(row["CNT_FAM_MEMBERS"]),
            }

        financial_profile: Dict[str, Any] = {
            "Income type": row["NAME_INCOME_TYPE"],
            "Employment Sector": row["ORGANIZATION_TYPE"],
            "Income": f"${int(row['AMT_INCOME_TOTAL']):,}",
            "Housing situation": row["NAME_HOUSING_TYPE"],
            "Owns Car": own_car,
            "Owns Real Estate": own_realty,
            }

        credit_profile: Dict[str, Any] = {
            "Contract type": row["NAME_CONTRACT_TYPE"],
            "Credit": f"${int(row['AMT_CREDIT']):,}",
            "Annuity": f"${int(row['AMT_ANNUITY']):,}",
            }

        # Construct the final client information dictionary
        client_info: Dict[str, Dict[str, Any]] = {
            "Personal Profile": personal_profile,
            "Financial Profile": financial_profile,
            "Credit Profile": credit_profile,
            }

        return client_info

    except sqlite3.Error as e:
        logger.error(f"Database query failed for client {client_id}: {e}")
        return None


# ============================================ EXTRACT PREDICT CLIENT INFO =========================================== #
def extract_predict_client_info(client_id: int) -> Optional[pd.DataFrame]:
    """
    Extracts a full row from 'model_input_data' corresponding to the given client ID,
    excluding 'SK_ID_CURR' since it's not needed for predictions.

    Args:
        client_id (int): The client ID to look for.

    Returns:
        Optional[pd.DataFrame]: The extracted row as a Pandas DataFrame with column names, or None if not found.
    """
    try:
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()

        # Fetch column names dynamically
        cursor.execute("PRAGMA table_info(model_input_data);")
        columns_info = cursor.fetchall()
        all_columns = [col[1] for col in columns_info]  # Extract column names
        feature_columns = [col for col in all_columns if col != "SK_ID_CURR"]  # Remove SK_ID_CURR

        # Query the database for the full row
        cursor.execute("SELECT * FROM model_input_data WHERE SK_ID_CURR = ?;", (client_id,))
        row: Optional[sqlite3.Row] = cursor.fetchone()
        conn.close()

        if row is None:
            logger.warning(f"Client ID {client_id} not found in 'model_input_data'.")
            return None

        # Convert row to Pandas DataFrame (excluding SK_ID_CURR)
        row_data = pd.DataFrame([list(row)[1:]], columns=feature_columns, dtype=np.float32)

        logger.success(f"Extracted data for Client ID {client_id} from 'model_input_data'.")
        return row_data

    except sqlite3.Error as e:
        logger.error(f"Database query failed for client {client_id}: {e}")
        return None


# ========================================= EXTRACT GLOBAL FEATURE IMPORTANCE ======================================== #
def extract_feat_global_importance() -> Optional[str]:
    """
    Extracts the entire dataset from 'model_input_data', applies the saved RobustScaler,
    computes SHAP global feature importance if not already computed, and generates a beeswarm plot.

    Returns:
        Optional[str]: Path to the precomputed SHAP plot if successful, else None.
    """
    try:
        OUTPUT_FILE = API_STATIC_DIR / "model_predictors.png"

        # If the file already exists, return its path immediately
        if OUTPUT_FILE.exists():
            logger.info(f"SHAP plot already exists at {OUTPUT_FILE}. Skipping computation.")
            return str(OUTPUT_FILE)

        logger.info("Connecting to database to retrieve full dataset...")
        conn: sqlite3.Connection = get_db_connection()
        query = "SELECT * FROM model_input_data;"
        df: pd.DataFrame = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("No data found in 'model_input_data'.")
            return None

        # Drop 'SK_ID_CURR' column (not a feature)
        features_only: pd.DataFrame = df.drop(columns=["SK_ID_CURR"], errors="ignore")

        # Load and apply RobustScaler
        logger.info("Loading and applying RobustScaler...")
        robust_scaler: RobustScaler = joblib.load(SCALER_PATH)
        scaled_features: pd.DataFrame = pd.DataFrame(
            robust_scaler.transform(features_only),
            columns=features_only.columns,
            index=features_only.index
            )

        # Load precomputed SHAP explainer
        logger.info("Loading SHAP Explainer...")
        explainer: shap.Explainer = joblib.load(SHAP_EXPLAINER_PATH)

        # Compute SHAP values for the dataset
        logger.info("Computing SHAP values for global feature importance...")
        shap_values: shap.Explanation = explainer(scaled_features, check_additivity=False)

        logger.success("SHAP values computed successfully.")

        # Generate a beeswarm plot for the top 15 features
        logger.info("Generating and saving SHAP beeswarm plot...")
        plt.figure(figsize=(13, 9))
        shap.summary_plot(
            shap_values=shap_values,
            features=scaled_features,
            plot_type="violin",
            max_display=15,
            show=False
            )

        # Add title with increased padding
        plt.title("Top 15 Model Predictors", pad=20, fontsize=16)
        plt.tight_layout()

        # Save the plot as a .png file in the specified location
        plt.savefig(OUTPUT_FILE)
        plt.close()  # Close the plot to free memory
        logger.success(f"SHAP beeswarm plot saved successfully at {OUTPUT_FILE}.")

        return str(OUTPUT_FILE)

    except Exception as e:
        logger.error(f"Error computing SHAP global feature importance: {e}")
        return None



# ======================================== LOCAL FEATURE IMPORTANCE TABLE ============================================ #
def extract_local_feature_importance(client_id: int) -> Optional[pd.DataFrame]:
    """
    Computes the local feature importance for a given client using SHAP.
    Retrieves the top 15 most impactful features, their original values, and SHAP values.
    Additionally, fetches feature descriptions from 'model_input_metadata'.

    Args:
        client_id (int): The client ID for which local feature importance is computed.

    Returns:
        Optional[pd.DataFrame]: DataFrame with Features, Value, SHAP Value, and Description.
    """
    try:
        logger.info(f"Fetching data for Client ID {client_id}...")

        # Step 1: Extract client features from the database
        df_client_data: Optional[pd.DataFrame] = extract_predict_client_info(client_id)
        if df_client_data is None:
            logger.warning(f"No data found for Client ID {client_id}.")
            return None

        # Step 2: Load the full pipeline
        pipeline = load_pipeline()

        # Step 3: Extract feature names
        feature_names: List[str] = df_client_data.columns.tolist()

        # Step 4: Apply pipeline transformations (RobustScaler + Model preprocessing)
        transformed_data: np.ndarray = pipeline[:-1].transform(df_client_data)

        # Step 5: Load SHAP explainer
        explainer: shap.Explainer = joblib.load(SHAP_EXPLAINER_PATH)

        # Step 6: Compute SHAP values for the transformed instance
        shap_values: shap.Explanation = explainer(transformed_data, check_additivity=False)

        logger.success(f"SHAP values computed successfully for Client ID {client_id}.")

        # Step 7: Select Top 15 Features by Absolute SHAP Value
        df_feature_importance: pd.DataFrame = pd.DataFrame({
            "Features": feature_names,
            "Value": df_client_data.iloc[0].values,  # Original values before transformation
            "SHAP Value": shap_values.values[0]
            })

        df_feature_importance["abs_shap"] = df_feature_importance["SHAP Value"].abs()
        df_feature_importance = df_feature_importance.sort_values(by="abs_shap", ascending=False).drop(columns=["abs_shap"])
        df_feature_importance = df_feature_importance.head(15)

        logger.info("Retrieving feature descriptions from 'model_input_metadata'...")

        # Step 8: Fetch feature descriptions from 'model_input_metadata'
        conn: sqlite3.Connection = get_db_connection()
        metadata_query: str = "SELECT Feature, Description FROM model_input_metadata;"
        df_metadata: pd.DataFrame = pd.read_sql_query(metadata_query, conn)
        conn.close()

        # Merge descriptions with feature importance table
        df_feature_importance = df_feature_importance.merge(df_metadata, left_on="Features", right_on="Feature", how="left").drop(columns=["Feature"])
        df_feature_importance.rename(columns={"Description": "Description"}, inplace=True)

        logger.success("Generated feature importance table with descriptions.")

        return df_feature_importance

    except Exception as e:
        logger.error(f"Error computing local feature importance for Client ID {client_id}: {e}")
        return None



# ========================================== LOCAL FEATURE IMPORTANCE PLOT =========================================== #
def extract_waterfall_plot(client_id: int) -> Optional[io.BytesIO]:
    """
    Generates a SHAP Waterfall Plot for a given client and returns it as an in-memory image.

    Args:
        client_id (int): The client ID for which the waterfall plot is generated.

    Returns:
        Optional[io.BytesIO]: In-memory image buffer (PNG) containing the SHAP Waterfall Plot.
    """
    try:
        logger.info(f"Fetching transformed data for Client ID {client_id}...")

        # Step 1: Extract client features from the database
        client_data: Optional[pd.DataFrame] = extract_predict_client_info(client_id)
        if client_data is None:
            logger.warning(f"No data found for Client ID {client_id}.")
            return None

        # Step 2: Load the full pipeline
        pipeline = load_pipeline()

        # Step 3: Extract feature names BEFORE transformation
        feature_names: List[str] = client_data.columns.tolist()

        # Step 4: Apply pipeline transformations (RobustScaler)
        transformed_data: np.ndarray = pipeline[:-1].transform(client_data)  # Apply transformations but exclude model

        # Step 5: Load SHAP explainer
        explainer: shap.Explainer = joblib.load(SHAP_EXPLAINER_PATH)

        # Step 6: Compute SHAP values for the transformed instance
        shap_values: shap.Explanation = explainer(transformed_data, check_additivity=False)

        # Set correct feature names in SHAP values
        shap_values.feature_names = feature_names

        logger.success(f"SHAP values computed successfully for Client ID {client_id}.")

        # Step 7: Generate SHAP Waterfall Plot with correct feature names
        plt.figure(figsize=(20, 12))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)  # Show top 15 features
        plt.title("Local Feature Importance", fontsize=18, pad=20)

        # Step 8: Save the plot to an in-memory buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=1.0)  # Ensure proper padding
        plt.close()

        img_buffer.seek(0)  # Reset buffer position

        logger.success("SHAP Waterfall Plot generated successfully as in-memory image.")

        return img_buffer  # Returning the image in-memory

    except Exception as e:
        logger.error(f"Error generating SHAP waterfall plot for Client ID {client_id}: {e}")
        return None


# ============================================= EXTRACT POSITIONING PLOT ============================================= #
def extract_client_positioning_plot(client_id: int) -> Optional[io.BytesIO]:
    """
    Generates a comparative positioning plot for a given client, showing where they stand
    in relation to the overall dataset based on the top 15 SHAP features.

    Args:
        client_id (int): The client ID for which the positioning plot is generated.

    Returns:
        Optional[io.BytesIO]: In-memory image buffer (PNG) containing the comparative plot.
    """
    try:
        logger.info(f"Fetching top 15 features for Client ID {client_id}...")

        # Step 1: Extract client features from the database (ORIGINAL VALUES)
        df_client_data: Optional[pd.DataFrame] = extract_predict_client_info(client_id)
        if df_client_data is None:
            logger.warning(f"No data found for Client ID {client_id}.")
            return None

        # Step 2: Load the full pipeline and extract feature names
        pipeline = load_pipeline()
        feature_names: List[str] = df_client_data.columns.tolist()

        # Step 3: Apply pipeline transformations (RobustScaler) on a copy of client data
        transformed_data: np.ndarray = pipeline[:-1].transform(df_client_data.copy()) # Exclude model

        # Step 4: Load SHAP explainer and compute SHAP values
        explainer: shap.Explainer = joblib.load(SHAP_EXPLAINER_PATH)
        shap_values: shap.Explanation = explainer(transformed_data, check_additivity=False)

        logger.success(f"SHAP values computed successfully for Client ID {client_id}.")

        # Step 5: Retrieve the top 15 most important features (Using ORIGINAL values)
        df_feature_importance: pd.DataFrame = pd.DataFrame({
            "Features": feature_names,
            "Value": df_client_data.iloc[0].values,  # ORIGINAL Values
            "SHAP Value": shap_values.values[0]
            })
        df_feature_importance["abs_shap"] = df_feature_importance["SHAP Value"].abs()
        df_feature_importance = df_feature_importance.sort_values(by="abs_shap", ascending=False).drop(columns=["abs_shap"])
        top_features: List[str] = df_feature_importance["Features"].head(15).tolist()

        logger.info(f"Top 15 features extracted: {top_features}")

        # Step 6: Load only the selected 15 features from the database (ORIGINAL values)
        conn: sqlite3.Connection = get_db_connection()
        feature_columns_str: str = ", ".join([f'"{col}"' for col in top_features])  # Ensure SQL safety
        query: str = f"SELECT {feature_columns_str} FROM model_input_data;"
        df_dataset: pd.DataFrame = pd.read_sql_query(query, conn)  # ORIGINAL Values
        conn.close()

        if df_dataset.empty:
            logger.warning("No data found for selected features in 'model_input_data'.")
            return None

        # Step 7: Prepare to plot histograms and boxplots
        num_features = len(top_features)
        num_columns = 2  # Histogram and boxplot side by side
        num_rows = num_features  # Each feature gets one row

        plt.figure(figsize=(12, 6 * num_rows))

        for i, feature in enumerate(top_features):
            data = df_dataset[feature].dropna()

            if data.empty:
                logger.warning(f"No data available for feature {feature}. Skipping...")
                continue

            client_value = df_client_data[feature].values[0]  # ORIGINAL Value
            mean_value = data.mean()
            median_value = data.median()

            # Histogram
            plt.subplot(num_rows, num_columns, 2 * i + 1)
            sns.histplot(data, kde=False, color="gray", bins=30)
            plt.axvline(client_value, color="gold", linestyle="dashed", linewidth=2, label=f"Client: {client_value:.2f}")
            plt.axvline(mean_value, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_value:.2f}")
            plt.axvline(median_value, color="blue", linestyle="dashed", linewidth=1.5, label=f"Median: {median_value:.2f}")
            plt.title(f"Histogram of {feature}")
            plt.legend()

            # Boxplot
            plt.subplot(num_rows, num_columns, 2 * i + 2)
            sns.boxplot(x=data, color="lightblue")
            plt.axvline(client_value, color="gold", linestyle="dashed", linewidth=2, label=f"Client: {client_value:.2f}")
            plt.title(f"Boxplot of {feature}")
            plt.legend()

        plt.tight_layout(pad=3.0)

        # Step 8: Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)  # Rewind buffer to the start
        plt.close()

        logger.success(f"Generated positioning plot for Client ID {client_id}.")
        return buf

    except Exception as e:
        logger.error(f"Error generating client positioning plot for Client ID {client_id}: {e}")
        return None


# =============================================== EXTRACT FEATURES NAME ============================================== #
def extract_features_name() -> List[str]:
    """
    Extracts all feature names from the 'model_input_data' table, excluding 'SK_ID_CURR'.

    Returns:
        List[str]: A list of feature names (column names) from the table.
    """
    try:
        logger.info("Fetching feature names from 'model_input_data'...")

        # Step 1: Connect to the database
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()

        # Step 2: Fetch column names dynamically
        cursor.execute("PRAGMA table_info(model_input_data);")
        columns_info = cursor.fetchall()
        conn.close()

        if not columns_info:
            logger.warning("No columns found in 'model_input_data'.")
            return []

        # Step 3: Extract column names, excluding 'SK_ID_CURR'
        all_columns = [col[1] for col in columns_info]
        feature_columns = [col for col in all_columns if col != "SK_ID_CURR"]

        logger.success(f"Extracted {len(feature_columns)} features from 'model_input_data'.")
        return feature_columns

    except sqlite3.Error as e:
        logger.error(f"Database query failed while extracting feature names: {e}")
        return []

# ========================================= EXTRACT FEATURE POSITIONING PLOT ========================================= #
def extract_feature_positioning_plot(client_id: int, feature_name: str) -> Optional[io.BytesIO]:
    """
    Generates a comparative positioning plot for a given client based on a single feature.
    The plot includes a histogram and a boxplot showing where the client stands in the dataset.

    Args:
        client_id (int): The client ID for which the positioning plot is generated.
        feature_name (str): The feature to be analyzed.

    Returns:
        Optional[io.BytesIO]: In-memory image buffer (PNG) containing the comparative plot.
    """
    try:
        logger.info(f"Fetching '{feature_name}' for Client ID {client_id}...")

        # Step 1: Extract the client's feature value from the database
        df_client_data: Optional[pd.DataFrame] = extract_predict_client_info(client_id)
        if df_client_data is None or feature_name not in df_client_data.columns:
            logger.warning(f"Feature '{feature_name}' not found for Client ID {client_id}.")
            return None

        client_value = df_client_data[feature_name].values[0]  # ORIGINAL Value

        # Step 2: Load the dataset for this specific feature
        conn: sqlite3.Connection = get_db_connection()
        query: str = f"SELECT \"{feature_name}\" FROM model_input_data;"  # Ensure SQL safety
        df_dataset: pd.DataFrame = pd.read_sql_query(query, conn)  # ORIGINAL Values
        conn.close()

        if df_dataset.empty:
            logger.warning(f"No data found for feature '{feature_name}' in 'model_input_data'.")
            return None

        data = df_dataset[feature_name].dropna()  # Remove NaNs

        if data.empty:
            logger.warning(f"No valid data available for feature '{feature_name}'.")
            return None

        mean_value = data.mean()
        median_value = data.median()

        # Step 3: Generate the plot (Histogram + Boxplot)
        plt.figure(figsize=(12, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data, kde=False, color="gray", bins=30)
        plt.axvline(client_value, color="gold", linestyle="dashed", linewidth=2, label=f"Client: {client_value:.2f}")
        plt.axvline(mean_value, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_value:.2f}")
        plt.axvline(median_value, color="blue", linestyle="dashed", linewidth=1.5, label=f"Median: {median_value:.2f}")
        plt.title(f"Histogram of {feature_name}")
        plt.legend()

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data, color="lightblue")
        plt.axvline(client_value, color="gold", linestyle="dashed", linewidth=2, label=f"Client: {client_value:.2f}")
        plt.title(f"Boxplot of {feature_name}")
        plt.legend()

        plt.tight_layout()

        # Step 4: Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)  # Rewind buffer to the start
        plt.close()

        logger.success(f"Generated feature positioning plot for Client ID {client_id}, Feature: {feature_name}.")
        return buf

    except Exception as e:
        logger.error(f"Error generating feature positioning plot for Client ID {client_id}, Feature: {feature_name}: {e}")
        return None


# ================================================ BIVARIATE ANALYSIS ================================================ #
def bi_variate_analysis(client_id: int, feature_1: str, feature_2: str) -> Optional[io.BytesIO]:
    """
    Generates a scatter plot for two features from the 'model_input_metadata' table,
    highlighting the given client_id's data point in orange.

    Args:
        client_id (int): The ID of the client to highlight.
        feature_1 (str): The first feature to plot.
        feature_2 (str): The second feature to plot.

    Returns:
        io.BytesIO: A BytesIO object containing the generated plot.
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        # Fetch the full dataset for the given features
        query = f"""
        SELECT SK_ID_CURR, "{feature_1}", "{feature_2}" 
        FROM model_input_data;
        """
        df = pd.read_sql_query(query, conn)

        # Fetch the client-specific data
        client_query = f"""
        SELECT "{feature_1}", "{feature_2}"
        FROM model_input_data
        WHERE SK_ID_CURR = ?;
        """
        client_data = pd.read_sql_query(client_query, conn, params=(client_id,))
        conn.close()

        if df.empty:
            logger.warning("No data found for the given features.")
            return None

        if client_data.empty:
            logger.warning(f"Client ID {client_id} not found in dataset.")
            return None

        # Extract client values
        client_x, client_y = client_data.iloc[0][feature_1], client_data.iloc[0][feature_2]

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature_1], y=df[feature_2], alpha=0.5, label="Other Clients")
        plt.scatter(client_x, client_y, color='orange', edgecolors='black', s=100, label=f"Client {client_id}")
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.title(f"Bivariate Analysis: {feature_1} vs {feature_2}")
        plt.legend()

        # Save plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer

    except sqlite3.Error as e:
        logger.error(f"Database query failed: {e}")
        return None




# ==================================================================================================================== #
#                                                     MAIN EXECUTION                                                   #
# ==================================================================================================================== #

def main():
    logger.info(f"DB_PATH path: {DB_PATH}")
    logger.info(f"SCALER_PATH path: {SCALER_PATH}")
    logger.info(f"SHAP_EXPLAINER_PATH path: {SHAP_EXPLAINER_PATH}")

if __name__ == "__main__":
    main()
