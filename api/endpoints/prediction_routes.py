# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest
from rich.console import Console
from rich.table import Table
from loguru import logger

# Local application imports
from api.utils.database_utils import extract_predict_client_info
from api.utils.model_pipeline import load_pipeline

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# Console configuration
console = Console(width= 150)

# Load prediction pipeline
pipeline = load_pipeline()
THRESHOLD = 0.48  # Ensure this is consistent across the project

# Create a Blueprint for prediction-related API
predict_bp = Blueprint("predict", __name__)  # "predict" is the namespace

# ==================================================================================================================== #
#                                                       ENDPOINTS                                                      #
# ==================================================================================================================== #
@predict_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Endpoint to predict using the locally loaded pipeline for a given SK_ID_CURR.
    """
    try:
        # Parse JSON request
        user_request = request.get_json(force=True)
        user_id = user_request.get("id", None)

        # Validate user_id
        if user_id is None or not isinstance(user_id, int):
            logger.error("Invalid or missing 'id'")
            return jsonify({"status": "error", "message": "Invalid or missing 'id'"}), 400

        # Extract the client data from the database
        client_data_df = extract_predict_client_info(user_id)

        if client_data_df is None:
            logger.warning(f"Client ID {user_id} not found in 'model_input_data'.")
            return jsonify({"status": "error", "message": f"Client ID {user_id} not found"}), 404

        # Perform prediction using the pipeline
        probability: float = pipeline.predict_proba(client_data_df)[0, 1]
        predicted_target: int = int(probability >= THRESHOLD)
        loan_status: str = "Granted" if predicted_target == 0 else "Rejected"

        # Construct response data
        result = {
            "status": "success",
            "data": {
                "SK_ID_CURR": user_id,
                "predicted_proba": round(probability, 2),
                "predicted_target": predicted_target,
                "status": loan_status,
                "threshold": THRESHOLD
                }
            }

        # ---- Logging with Loguru ---- #
        logger.success(f"Prediction successful for ID {user_id}")
        logger.info(f"Probability: {probability}")
        logger.info(f"Target: {predicted_target}")
        logger.info(f"Loan Status: {loan_status}")
        logger.info(f"Threshold: {THRESHOLD}")

        # ---- Pretty-Print with Rich ---- #
        console.rule(f"[bold green]Prediction for Client ID {user_id}[/bold green]")
        table = Table(title="Prediction Details")

        table.add_column("Attribute", justify="left")
        table.add_column("Value", justify="left")

        for key, value in result["data"].items():
            table.add_row(key, str(value))

        console.print(table)
        console.rule("[bold blue]End of Prediction[/bold blue]")

        return jsonify(result), 200

    except BadRequest:
        logger.error("Invalid JSON payload")
        console.print_exception()  # Print stack trace using Rich
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        console.print_exception()  # Print stack trace using Rich
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500
