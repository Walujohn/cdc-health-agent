import pytest
from fastapi.testclient import TestClient
from fastapi import status
from fastapi_app import app, API_KEY

client = TestClient(app)

def test_run_guidance_drift():
    # Call the endpoint with correct API key
    response = client.post(
        "/run-guidance-drift",
        headers={"x-api-key": API_KEY}
    )
    assert response.status_code == status.HTTP_200_OK
    json_resp = response.json()
    assert "status" in json_resp
    assert "Drift experiment triggered" in json_resp["status"]

def test_run_guidance_drift_invalid_key():
    # Should reject invalid API key
    response = client.post(
        "/run-guidance-drift",
        headers={"x-api-key": "invalid"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid API Key" in response.text
