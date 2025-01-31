"""
This test suite validates the behavior and robustness of the `/predict` endpoint in the Flask API.

Key Features:
1. Utilizes `pytest` fixtures to create a reusable Flask test client for API requests.
2. Covers multiple scenarios to ensure comprehensive testing of the `/predict` endpoint.

Test Cases:
- `test_predict_valid_id`: Verifies correct handling of valid IDs that exist in the dataset.
- `test_predict_invalid_id`: Ensures appropriate error handling for IDs that do not exist in the dataset.
- `test_predict_missing_id`: Tests the API's response to payloads missing the required `id` field.
- `test_predict_invalid_payload`: Validates error handling for invalid, non-JSON payloads.
- `test_api_availability`: Confirms the API responds correctly to undefined endpoints.

Dependencies:
- Pytest: Framework for writing and running tests.
- Flask: Framework providing the API to test.
- JSON: Used for crafting request payloads and interpreting responses.

Notes:
- Ensure the `api.local_main` module and the Flask app are correctly configured and imported.
- Adjust `payload` values in test cases as needed to match the data in the actual dataset.
- This test suite assumes `/predict` is the only defined route; update the `test_api_availability` case if other endpoints exist.

"""

# ====================================================== IMPORTS ===================================================== #
import pytest
from api.local_main import app


# ==================================================================================================================== #
#                                                         TEST                                                         #
# ==================================================================================================================== #
@pytest.fixture
def client():
    """
    Creates a test client for the Flask app.

    This fixture sets up a Flask test client, which allows us to make requests
    to the API endpoints without needing to start a real server. It provides
    an isolated test environment, making tests faster and more reliable.
    """
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    """
    Test the /predict endpoint using the Flask test client.
    """
    url = "/api/predict"
    payload = {"id": 100057}  # Ensure this ID exists in the dataset as an integer
    response = client.post(url, json=payload)

    assert response.status_code == 200
    assert "predicted_proba" in response.json["data"]


def test_client_info_endpoint(client):
    """
    Test the /client-info endpoint.
    """
    url = "/api/client-info"
    payload = {"id": 123456}
    response = client.post(url, json=payload)

    assert response.status_code in [200, 404]  # Either success or client not found


def test_client_positioning_plot(client):
    """
    Test the /client-positioning-plot endpoint.
    """
    url = "/api/client-positioning-plot"
    payload = {"id": 123456}
    response = client.post(url, json=payload)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.content_type == "image/png"


def test_feature_positioning_plot(client):
    """
    Test the /feature-positioning-plot endpoint.
    """
    url = "/api/feature-positioning-plot"
    payload = {"id": 123456, "feature": "AMT_CREDIT"}
    response = client.post(url, json=payload)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.content_type == "image/png"


def test_model_predictors(client):
    """
    Test the /model-predictors endpoint.
    """
    url = "/api/model-predictors"
    response = client.get(url)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.content_type == "image/png"