"""
This module provides API endpoints for feature importance analysis using SHAP (SHapley Additive exPlanations).

Key Features:
1. Serves precomputed SHAP beeswarm plots to visualize global feature importance.
2. Computes local feature importance for individual clients and returns structured data.
3. Generates SHAP waterfall plots to explain predictions at the instance level.
4. Utilizes `Flask Blueprint` for modular API design.
5. Integrates `Loguru` for structured logging and `Rich` for enhanced console output.

Endpoints:
- `GET /model-predictors`: Returns a SHAP beeswarm plot for global feature importance.
- `POST /local-feature-importance`: Computes and returns local feature importance for a given client.
- `POST /local-waterfall-plot`: Generates a SHAP waterfall plot to explain an individual prediction.

Dependencies:
- Flask: API framework for handling HTTP requests.
- Loguru: Enhanced logging for debugging and error handling.
- Rich: Pretty-printing for console output.
- SHAP: Explainability library for machine learning models.
- Pandas: Data handling and transformation.
- Pathlib: File path management.

Notes:
- Ensure `api.utils.database_utils` contains valid implementations of data extraction methods.
- The API assumes the presence of precomputed SHAP values; update logic if real-time computation is needed.
- The feature importance computations rely on trained models and saved SHAP explainers.
"""


# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Blueprint, request, jsonify, send_file, current_app
from rich.console import Console
from rich.table import Table
from loguru import logger
from pathlib import Path

# Local application imports
from api.utils.database_utils import (
    extract_feat_global_importance,
    extract_local_feature_importance,
    extract_waterfall_plot,
    )

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
console = Console(width=150)

# Create a Blueprint for feature importance-related API
feature_importance_bp = Blueprint("importance", __name__)

# ==================================================================================================================== #
#                                                       ENDPOINTS                                                      #
# ==================================================================================================================== #
@feature_importance_bp.route("/model-predictors", methods=["GET"])
def feature_impact_endpoint():
    """
    API endpoint to serve the precomputed SHAP beeswarm plot.
    If the plot does not exist, it computes it first before serving.
    """
    try:
        shap_plot_path = extract_feat_global_importance()

        if shap_plot_path is None:
            return jsonify({"error": "Failed to compute SHAP feature impact"}), 500

        # Extract only the filename from the full path
        shap_plot_filename = Path(shap_plot_path).name

        # Serve the SHAP plot dynamically from the static directory
        return current_app.send_static_file(shap_plot_filename)  # <-- FIXED

    except Exception as e:
        logger.error(f"Error serving SHAP plot: {e}")
        return jsonify({"error": "Failed to retrieve SHAP feature impact"}), 500


# ====================================== LOCAL FEATURE IMPORTANCE TABLE (JSON) ======================================= #
@feature_importance_bp.route("/local-feature-importance", methods=["POST"])
def local_feature_importance_endpoint():
    """
    API endpoint to compute and return the local feature importance table for a given client.
    """
    try:
        # Parse JSON request
        user_request = request.get_json(force=True)
        client_id = user_request.get("id")

        # Validate client_id
        if client_id is None or not isinstance(client_id, int):
            logger.error("Invalid or missing 'id'")
            return jsonify({"error": "Invalid or missing 'id'"}), 400

        # Compute feature importance table
        feature_importance_df = extract_local_feature_importance(client_id)

        if feature_importance_df is None:
            logger.error(f"Failed to compute local feature importance for Client ID {client_id}")
            return jsonify({"error": f"Failed to compute local feature importance for Client ID {client_id}"}), 500

        # ---- Logging with Loguru ---- #
        logger.success(f"Computed local feature importance for Client ID {client_id}")

        # ---- Pretty-Print with Rich ---- #
        console.rule(f"[bold green]Local Feature Importance for ID {client_id}[/bold green]")
        table = Table(title="Local Feature Importance")

        table.add_column("Feature", justify="left")
        table.add_column("Value", justify="left")
        table.add_column("SHAP Value", justify="left")
        table.add_column("Description", justify="left")

        for _, row in feature_importance_df.iterrows():
            table.add_row(
                str(row["Features"]),
                str(row["Value"]),
                f"{row['SHAP Value']:.4f}",
                str(row["Description"]) if "Description" in row else "N/A"
                )

        console.print(table)
        console.rule("[bold blue]End of Local Feature Importance[/bold blue]")

        # Convert DataFrame to JSON format and return
        return jsonify({"feature_importance": feature_importance_df.to_dict(orient="records")})

    except Exception as e:
        logger.error(f"Error serving local feature importance: {e}")
        console.print_exception()  # Log the stack trace in Rich
        return jsonify({"error": "Failed to retrieve local feature importance"}), 500


# ======================================= LOCAL FEATURE IMPORTANCE PLOT (IMAGE) ====================================== #
@feature_importance_bp.route("/local-waterfall-plot", methods=["POST"])
def local_feature_importance_plot_endpoint():
    """
    API endpoint to generate and return a SHAP waterfall plot for a given client.
    """
    try:
        user_request = request.get_json(force=True)
        client_id = user_request.get("id")

        if client_id is None or not isinstance(client_id, int):
            return jsonify({"error": "Invalid or missing 'id'"}), 400

        # Generate SHAP Waterfall Plot
        img_buffer = extract_waterfall_plot(client_id)

        if img_buffer is None:
            return jsonify({"error": f"Failed to generate SHAP waterfall plot for Client ID {client_id}"}), 500

        return send_file(img_buffer, mimetype="image/png")

    except Exception as e:
        logger.error(f"Error serving SHAP waterfall plot: {e}")
        return jsonify({"error": "Failed to retrieve SHAP feature impact"}), 500

