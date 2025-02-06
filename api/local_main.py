"""
This module initializes and configures the Flask API for credit scoring predictions and explainability.

Key Features:
1. Registers multiple API endpoints using Flask Blueprints.
2. Configures `Loguru` for structured logging with automatic file rotation.
3. Serves static files for API-generated visualizations.
4. Implements modular API design for easy extension and maintenance.

Registered Endpoints:
- `/api/client-info`: Retrieves client details.
- `/api/predict`: Returns loan prediction results.
- `/api/model-predictors`: Provides global feature importance using SHAP.
- `/api/positioning`: Generates client and feature positioning visualizations.

Dependencies:
- Flask: Micro-framework for API routing.
- Loguru: Enhanced logging with file rotation.
- Pathlib: Handles directory and file paths.

Notes:
- Ensure `LOG_DIR` is correctly configured for storing logs.
- All endpoints are prefixed with `/api` for consistency.
- Debug mode is enabled by default; disable in production for security.
"""


# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Flask
from flask_cors import CORS
from loguru import logger
from pathlib import Path

# Local application imports
from api.endpoints.client_routes import client_bp
from api.endpoints.prediction_routes import predict_bp
from api.endpoints.feature_importance_routes import feature_importance_bp
from api.endpoints.positioning_routes import positioning_bp
from prod.paths import LOG_DIR
from prod.utils import log_section_header

# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
# # Loguru Configuration for local testing
# LOG_PATH = LOG_DIR / "api"
# logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")

# ==================================================================================================================== #
#                                                          API                                                         #
# ==================================================================================================================== #
log_section_header(title="API")

app = Flask(__name__, static_folder="static")

# Allow only the Streamlit app's domain
CORS(app, resources={r"/api/*": {"origins": "https://credit-score-attribution-003464da4de3.herokuapp.com"}})


# ==================================================================================================================== #
#                                            REGISTER BLUEPRINTS (ROUTES)                                              #
# ==================================================================================================================== #
app.register_blueprint(client_bp, url_prefix="/api")
app.register_blueprint(predict_bp, url_prefix="/api")
app.register_blueprint(feature_importance_bp, url_prefix="/api")
app.register_blueprint(positioning_bp, url_prefix="/api")



# ==================================================================================================================== #
#                                                       RUN APP                                                        #
# ==================================================================================================================== #

# # For local development
# if __name__ == "__main__":
#     app.run(debug=True)

# For Cloud Deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)