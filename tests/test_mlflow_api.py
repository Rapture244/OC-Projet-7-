"""
This test suite validates the integration between the Flask API and MLflow, ensuring correct configurations, model loading,
and endpoint functionality.

Key Features:
1. Verifies MLflow configurations and artifacts:
   - Tests the correct setting of the MLflow tracking URI.
   - Confirms that the model and scaler artifacts are successfully loaded from the MLflow registry.
   - Ensures the presence and correctness of custom threshold parameters in the MLflow run.
2. Includes Flask API endpoint tests:
   - Validates the `/predict` endpoint for various scenarios, including valid, invalid, and missing payloads.
   - Tests the API's response to additional unexpected fields in the request payload.
   - Checks for proper handling of undefined endpoints.

Test Cases:
- **MLflow Integration**:
  - `test_mlflow_tracking_uri`: Verifies that the MLflow tracking URI matches the configured value.
  - `test_model_loading`: Confirms the model can be retrieved using the MLflow model registry.
  - `test_scaler_loading`: Validates the scaler artifact can be downloaded from the associated MLflow run.
  - `test_custom_threshold_loading`: Ensures the custom threshold is correctly retrieved from MLflow run parameters.
- **Flask API**:
  - `test_predict_valid_id`: Tests prediction for a valid ID present in the dataset.
  - `test_predict_invalid_id`: Validates error handling for an ID that does not exist in the dataset.
  - `test_predict_missing_id`: Ensures appropriate handling when the `id` field is missing in the payload.
  - `test_predict_invalid_payload`: Tests error handling for invalid, non-JSON payloads.
  - `test_predict_extra_fields`: Verifies response when the payload includes extra fields.
  - `test_api_availability`: Checks the response for undefined API endpoints.

Dependencies:
- `pytest`: Used for writing and running test cases.
- `mlflow`: Interacts with the MLflow model registry and artifact store.
- `Flask`: Framework providing the API under test.
- `src.packages.constants.paths`: Provides the expected MLflow tracking URI configuration.

Notes:
- Ensure the `MLFLOW_TRACKING_URI` is correctly configured and accessible before running tests.
- The MLflow registry must contain the required model, scaler, and parameters for the tests to pass.
- The Flask API must be correctly configured to use the loaded MLflow artifacts and respond to `/predict` requests.
"""


# ====================================================== IMPORTS ===================================================== #
import pytest
import mlflow
from mlflow.tracking import MlflowClient
from api.mlflow_main import app
from src.packages.constants.paths import MLFLOW_TRACKING_URI


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


def test_mlflow_tracking_uri():
    """
    Test that the MLflow tracking URI is set correctly.
    """
    client = MlflowClient()
    tracking_uri = client.tracking_uri

    # Compare with the configured MLFLOW_TRACKING_URI
    assert tracking_uri == MLFLOW_TRACKING_URI, \
        f"Tracking URI mismatch. Expected '{MLFLOW_TRACKING_URI}', got '{tracking_uri}'."


def test_model_loading():
    """
    Test that the model is successfully loaded from MLflow.
    """
    client = MlflowClient()
    model_name = "LGBMClassifier - business"
    model_alias = "champion"

    try:
        model_version = client.get_model_version_by_alias(model_name, model_alias)
        assert model_version is not None, f"Model with alias '{model_alias}' not found in registry."
    except Exception as e:
        pytest.fail(f"Failed to fetch model version: {e}")


def test_scaler_loading():
    """
    Test that the scaler is successfully loaded from MLflow.
    """
    client = MlflowClient()
    model_name = "LGBMClassifier - business"
    model_alias = "champion"
    artifact_path = "scalers/2025-01-14 - RobustScaler.joblib"

    try:
        model_version = client.get_model_version_by_alias(model_name, model_alias)
        run_id = model_version.run_id

        # Validate the scaler artifact
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        assert local_path is not None, "Scaler artifact not found."
    except Exception as e:
        pytest.fail(f"Failed to load scaler artifact: {e}")


def test_custom_threshold_loading():
    """
    Test that the custom threshold is fetched correctly from the MLflow run.
    """
    client = MlflowClient()
    model_name = "LGBMClassifier - business"
    model_alias = "champion"

    try:
        model_version = client.get_model_version_by_alias(model_name, model_alias)
        run_id = model_version.run_id
        run_data = client.get_run(run_id).data

        # Ensure the custom threshold parameter exists
        assert "custom_threshold" in run_data.params, "Custom threshold not found in the MLflow run parameters."
        custom_threshold = float(run_data.params["custom_threshold"])
        assert isinstance(custom_threshold, float), "Custom threshold is not a float."
    except Exception as e:
        pytest.fail(f"Failed to fetch custom threshold: {e}")


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


def test_predict_extra_fields(client):
    """
    Test case for a valid payload with additional unexpected fields.
    """
    payload = {"id": 100001, "extra_field": "unexpected"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert "SK_ID_CURR" in data
    assert data["SK_ID_CURR"] == payload["id"]


def test_api_availability(client):
    """
    Test case for accessing an undefined endpoint.
    """
    response = client.get("/")  # Assuming no endpoint at "/"
    assert response.status_code == 404
