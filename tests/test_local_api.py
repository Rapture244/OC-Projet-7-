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
    """
    with app.test_client() as client:
        yield client

def test_predict_valid_id(client):
    """
    Test case for a valid ID that exists in the dataset.
    """
    payload = {"id": 100001}  # Ensure this ID exists in the dataset as an integer
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert "SK_ID_CURR" in data
    assert data["SK_ID_CURR"] == payload["id"]
    assert "predicted_proba" in data
    assert "predicted_target" in data
    assert "status" in data

def test_predict_invalid_id(client):
    """
    Test case for an invalid ID that does not exist in the dataset.
    """
    payload = {"id": 999999}  # Non-existent ID as an integer
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    data = response.get_json()
    assert data is not None
    assert "error" in data
    assert "not found" in data["error"].lower()

def test_predict_missing_id(client):
    """
    Test case for a missing 'id' field in the payload.
    """
    payload = {}  # Missing 'id'
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data is not None
    assert "error" in data
    assert "missing or invalid 'id'" in data["error"].lower()

def test_predict_invalid_payload(client):
    """
    Test case for an invalid payload that is not JSON.
    """
    response = client.post("/predict", data="not a json payload")
    assert response.status_code == 400
    data = response.get_json()
    assert data is not None
    assert "error" in data
    assert "invalid json payload" in data["error"].lower()

def test_api_availability(client):
    """
    Test case for accessing an undefined endpoint.
    """
    response = client.get("/")  # Assuming no endpoint at "/"
    assert response.status_code == 404
