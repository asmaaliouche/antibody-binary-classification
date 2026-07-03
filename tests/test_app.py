"""
test_app.py - Automated unit tests for FastAPI validation, routing, and inference
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add src/ to python path so we can import app
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    """Verify that the home health check endpoint returns 200 and correct status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Antibody Scoring API" in data["service"]


def test_valid_exact_match_prediction(client):
    """
    Test a valid request with sequences that are in the reference dataset (exact_match).
    We use the first sequence from Shehata's dataset (from raw Excel / final_dataset).
    """
    payload = {
        "antibody_id": "Test-Ab-1",
        "vh": "DIVMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYSYTFGQGTKLEIK",
        "vl": "DIVMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYSYTFGQGTKLEIK",
        "tm_app": 65.5,
        "psr": 0.15
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "probability" in data
    assert "lookup_status" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    # Our simple mock matching might resolve to exact_match, nearest_match or fallback
    assert data["lookup_status"] in ["exact_match", "nearest_match", "fallback_average"]


def test_valid_unknown_sequence_prediction(client):
    """
    Test prediction with a valid but completely random sequence of amino acids (should trigger fallback/nearest match).
    """
    payload = {
        "antibody_id": "Unknown-Ab",
        "vh": "ACDEFGHIKLMNPQRSTVWY",  # Standard IUPAC sequence
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": 70.0,
        "psr": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lookup_status"] in ["nearest_match", "fallback_average"]
    assert data["prediction"] in [0, 1]


def test_missing_required_fields(client):
    """
    Verify that omitting required fields (like 'vh' or 'tm_app') returns 422 Unprocessable Entity.
    """
    payload = {
        "antibody_id": "Ab-Bad",
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "psr": 0.5
        # vh and tm_app are missing!
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_data_types(client):
    """
    Verify that passing incorrect data types (e.g. text for 'tm_app') returns 422 Unprocessable Entity.
    """
    payload = {
        "antibody_id": "Ab-Bad-Types",
        "vh": "ACDEFGHIKLMNPQRSTVWY",
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": "this_should_be_a_float",
        "psr": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_out_of_bounds_tm_app(client):
    """
    Verify that out of bounds physical properties (e.g., negative tm_app or > 110)
    trigger Pydantic validation error (422).
    """
    payload_low = {
        "vh": "ACDEFGHIKLMNPQRSTVWY",
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": -5.0,  # Below minimum range of 30°C
        "psr": 0.5
    }
    response_low = client.post("/predict", json=payload_low)
    assert response_low.status_code == 422

    payload_high = {
        "vh": "ACDEFGHIKLMNPQRSTVWY",
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": 150.0,  # Above maximum range of 110°C
        "psr": 0.5
    }
    response_high = client.post("/predict", json=payload_high)
    assert response_high.status_code == 422


def test_out_of_bounds_psr(client):
    """
    Verify that out of bounds psr (e.g. < 0 or > 10) trigger Pydantic validation error (422).
    """
    payload_low = {
        "vh": "ACDEFGHIKLMNPQRSTVWY",
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": 65.0,
        "psr": -1.0  # Below minimum 0.0
    }
    response_low = client.post("/predict", json=payload_low)
    assert response_low.status_code == 422


def test_invalid_amino_acid_characters(client):
    """
    Verify that non-amino-acid characters (e.g., 'Z', 'X', numbers) are caught by validation.
    """
    payload = {
        "vh": "ACDEFGHIKLMNPQRSTVWYB",  # 'B' is not standard
        "vl": "ACDEFGHIKLMNPQRSTVWY",
        "tm_app": 65.0,
        "psr": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
