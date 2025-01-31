# ====================================================== IMPORTS ===================================================== #
# Standard library imports
from flask import Flask
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
# Loguru Configuration
LOG_PATH = LOG_DIR / "api"
logger.add(LOG_PATH, rotation="1 MB", retention="7 days", level="INFO")

# ==================================================================================================================== #
#                                                          API                                                         #
# ==================================================================================================================== #
log_section_header(title = "API")

app = Flask(__name__, static_folder = "static")


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
if __name__ == "__main__":
    app.run(debug=True)

