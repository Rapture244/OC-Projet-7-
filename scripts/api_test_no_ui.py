"""
This script demonstrates how to interact with the Credit Score Prediction API hosted on Heroku.

Key Features:
1. Sends a POST request to the API with a user-provided Credit ID (`id`).
2. Logs the response, including predicted probability, target classification, and loan status.
3. Includes error handling for HTTP errors, request exceptions, and invalid API responses.
4. Leverages `loguru` for enhanced logging and debugging.

Functionality:
- Sends the Credit ID as JSON payload to the API endpoint (`/predict`).
- Validates the API's response, ensuring it contains the expected JSON structure.
- Handles various scenarios such as:
  - Successful responses with predictions.
  - HTTP errors and bad status codes.
  - Unexpected or malformed API responses.

Dependencies:
- Requests: For making HTTP POST requests to the API.
- Loguru: For advanced logging and error monitoring.

Notes:
- The API URL (`https://credit-score-attribution-003464da4de3.herokuapp.com/predict`) is hosted on Heroku. Update this if the endpoint changes.
- Ensure the API is live and accessible before running the script.
- The script assumes the API always returns JSON-formatted responses for valid and error scenarios.
"""

# ====================================================== IMPORTS ===================================================== #
from loguru import logger
import requests
import sys


# ==================================================================================================================== #
#                                                       REQUEST NO UI                                                  #
# ==================================================================================================================== #
# Define the API endpoint
url = "https://credit-score-attribution-003464da4de3.herokuapp.com/predict"

# Define the payload
payload = {
    "id": 100001
    }

# Define the headers
headers = {
    "Content-Type": "application/json"
    }

# Make the POST request
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
except requests.exceptions.HTTPError as http_err:
    logger.error(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
    logger.debug(f"Response Content: {response.text}")
    sys.exit(1)
except requests.exceptions.RequestException as err:
    logger.error(f"Other error occurred: {err}")
    sys.exit(1)

# Process the response
try:
    data = response.json()
    logger.success("Prediction Successful")
    logger.info(f"SK_ID_CURR: {data['SK_ID_CURR']}")
    logger.info(f"Predicted Proba: {data['predicted_proba']}")
    logger.info(f"Predicted Target: {data['predicted_target']}")
    logger.info(f"Status: {data['status']}")
except ValueError:
    logger.error("Response content is not valid JSON")
    logger.debug(f"Response Content: {response.text}")
except KeyError as key_err:
    logger.error(f"Missing expected key in the response: {key_err}")
    logger.debug(f"Response JSON: {data}")
except Exception as e:
    logger.error(f"An unexpected error occurred while processing the response: {e}")
    logger.debug(f"Response Content: {response.text}")
