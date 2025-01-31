# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Blueprint, request, jsonify
from rich.console import Console
from rich.table import Table
from loguru import logger

# Local application imports
from api.utils.database_utils import extract_client_info


# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# Console configuration
console = Console(width= 150)

# Create a Blueprint for client-related API
client_bp = Blueprint("client", __name__)    # "client" is the namespace


# ==================================================================================================================== #
#                                                       ENDPOINTS                                                      #
# ==================================================================================================================== #
@client_bp.route('/client-info', methods=['POST'])
def get_client_info_endpoint():
    """
    API endpoint to retrieve client information from the database.
    """
    try:
        # Parse JSON request
        user_request = request.get_json(force=True)
        client_id = user_request.get("id")

        # Validate client_id
        if client_id is None or not isinstance(client_id, int):
            logger.error("Invalid or missing 'id'")
            return jsonify({"error": "Invalid or missing 'id'"}), 400

        # Extract client information from the database
        client_info = extract_client_info(client_id)

        if client_info is None:
            logger.error(f"Client ID {client_id} not found in the database.")
            return jsonify({"error": f"Client ID {client_id} not found"}), 404

        # ---- Logging with Loguru ---- #
        logger.success(f"Successfully retrieved information for Client ID {client_id}")

        # ---- Pretty-Print with Rich ---- #
        console.rule(f"[bold green]Client Information for ID {client_id}[/bold green]")
        table = Table(title="Client Details")

        # Add Personal Profile to Rich Table
        table.add_column("Category", justify="left")
        table.add_column("Attribute", justify="left")
        table.add_column("Value", justify="left")

        for key, value in client_info["Personal Profile"].items():
            table.add_row("Personal Profile", key, str(value))

        for key, value in client_info["Financial Profile"].items():
            table.add_row("Financial Profile", key, str(value))

        for key, value in client_info["Credit Profile"].items():
            table.add_row("Credit Profile", key, str(value))

        console.print(table)
        console.rule("[bold blue]End of Client Information[/bold blue]")

        # Return client information as JSON
        return jsonify(client_info), 200

    except Exception as e:
        logger.error(f"Error retrieving client info: {e}")
        console.print_exception()  # Log the stack trace in Rich
        return jsonify({"error": "Internal server error"}), 500
