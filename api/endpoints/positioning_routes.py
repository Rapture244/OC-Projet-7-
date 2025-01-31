# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Blueprint, request, jsonify, send_file
from rich.console import Console
from rich.table import Table
from loguru import logger

# Local application imports
from api.utils.database_utils import (
    extract_client_positioning_plot,
    extract_feature_positioning_plot,
    extract_features_name,
    )

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
console = Console(width=150)

# Create a Blueprint for positioning-related API
positioning_bp = Blueprint("positioning", __name__)

# ==================================================================================================================== #
#                                                       ENDPOINTS                                                      #
# ==================================================================================================================== #

# ============================================== CLIENT POSITIONING PLOT ============================================= #
@positioning_bp.route("/client-positioning-plot", methods=["POST"])
def client_positioning_plot_endpoint():
    """
    API endpoint to generate and return a positioning plot for a given client,
    showing where they stand compared to the dataset based on the top 15 SHAP features.
    """
    try:
        user_request = request.get_json(force=True)
        client_id = user_request.get("id")

        # Validate client_id
        if client_id is None or not isinstance(client_id, int):
            return jsonify({"error": "Invalid or missing 'id'"}), 400

        # Generate Client Positioning Plot
        img_buffer = extract_client_positioning_plot(client_id)

        if img_buffer is None:
            return jsonify({"error": f"Failed to generate client positioning plot for Client ID {client_id}"}), 500

        return send_file(img_buffer, mimetype="image/png")

    except Exception as e:
        logger.error(f"Error serving client positioning plot: {e}")
        return jsonify({"error": "Failed to retrieve client positioning plot"}), 500


# ============================================= FEATURE POSITIONING PLOT ============================================= #
@positioning_bp.route("/feature-positioning-plot", methods=["POST"])
def feature_positioning_plot_endpoint():
    """
    API endpoint to generate and return a feature positioning plot for a given client and feature.
    """
    try:
        user_request = request.get_json(force=True)
        client_id = user_request.get("id")
        feature_name = user_request.get("feature")

        # Validate inputs
        if client_id is None or not isinstance(client_id, int):
            return jsonify({"error": "Invalid or missing 'id'"}), 400
        if not feature_name or not isinstance(feature_name, str):
            return jsonify({"error": "Invalid or missing 'feature'"}), 400

        # Generate Feature Positioning Plot
        img_buffer = extract_feature_positioning_plot(client_id, feature_name)

        if img_buffer is None:
            return jsonify({"error": f"Failed to generate feature positioning plot for Client ID {client_id}"}), 500

        return send_file(img_buffer, mimetype="image/png")

    except Exception as e:
        logger.error(f"Error serving feature positioning plot: {e}")
        return jsonify({"error": "Failed to retrieve feature positioning plot"}), 500


# ================================================= GET FEATURES NAME ================================================ #
@positioning_bp.route("/features-name", methods=["GET"])
def get_features_name_endpoint():
    """
    API endpoint to retrieve all feature names from the 'model_input_data' table, excluding 'SK_ID_CURR'.
    """
    try:
        # Extract feature names from the database
        feature_names = extract_features_name()

        if not feature_names:
            logger.warning("No features found in 'model_input_data'.")
            return jsonify({"error": "No features found"}), 404

        # ---- Logging with Loguru ---- #
        logger.success(f"Successfully retrieved {len(feature_names)} feature names.")

        # ---- Pretty-Print with Rich ---- #
        console.rule("[bold green]Feature Names Retrieved[/bold green]")
        table = Table(title="Feature Names")
        table.add_column("Feature", justify="left")

        for feature in feature_names:
            table.add_row(feature)

        console.print(table)
        console.rule("[bold blue]End of Feature Names[/bold blue]")

        # Return JSON response
        return jsonify({"features": feature_names}), 200

    except Exception as e:
        logger.error(f"Error retrieving feature names: {e}")
        console.print_exception()  # Print stack trace with Rich
        return jsonify({"error": "Internal server error"}), 500
