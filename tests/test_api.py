import pytest
from api.main import app

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
