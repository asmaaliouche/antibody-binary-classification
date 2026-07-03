"""
test_pipeline.py - Unit tests for data processing and model pipeline
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

# Add src/ to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from processing import load_and_clean_experimental_data
from modeling import prepare_features_for_embedding


@pytest.fixture
def mock_raw_excel(tmp_path):
    """
    Creates a mock raw Excel file to test load_and_clean_experimental_data
    """
    data = {
        "Clone name": ["antibody_1", "antibody_2", "antibody_3"],
        "B cell subset": ["memory", "naive", "memory"],
        "VH Protein": ["EVQLVESG-G", "DVQLQESG-P", "EVQLVESG-G"],
        "VL Protein": ["DIQMTQSP-S", "EIVLTQSP-G", "DIQMTQSP-S"],
        "HIC retention time (min)": [10.2, 12.5, np.nan],  # antibody_3 has missing critical info
        "TmApp (°C)": [65.0, 72.5, 68.0],
        "PSR Score": [0.12, 0.85, 0.45]
    }
    df = pd.DataFrame(data)
    
    # We append two footer rows mimicking the Shehata footer pattern that gets sliced off (.iloc[:-2])
    footer_data = {
        "Clone name": ["Foot1", "Foot2"],
        "B cell subset": [None, None],
        "VH Protein": [None, None],
        "VL Protein": [None, None],
        "HIC retention time (min)": [None, None],
        "TmApp (°C)": [None, None],
        "PSR Score": [None, None]
    }
    df_footer = pd.DataFrame(footer_data)
    df_full = pd.concat([df, df_footer], ignore_index=True)
    
    excel_path = tmp_path / "mock_antibodies.xlsx"
    df_full.to_excel(excel_path, index=False, engine="openpyxl")
    return str(excel_path)


def test_load_and_clean_experimental_data(mock_raw_excel):
    """
    Verify sequence cleaning (dashes removed), missing value drops, footer slicing,
    and binary classification target engineering.
    """
    df_clean = load_and_clean_experimental_data(mock_raw_excel, hic_threshold=11.5)
    
    # 1. Verify missing HIC drops (antibody_3) and footer slicing
    # Out of 3 real rows, antibody_3 has NaN HIC. The footers are sliced off.
    # Therefore, we should have exactly 2 rows left.
    assert len(df_clean) == 2
    
    # 2. Verify sequence cleaning (dashes removed)
    assert "-" not in df_clean.iloc[0]["vh"]
    assert df_clean.iloc[0]["vh"] == "EVQLVESGG"
    assert df_clean.iloc[0]["vl"] == "DIQMTQSPS"
    
    # 3. Verify Target Engineering (HIC >= 11.5)
    # antibody_1 has HIC = 10.2 (< 11.5) -> low_medium_hic
    # antibody_2 has HIC = 12.5 (>= 11.5) -> high_hic
    assert df_clean.iloc[0]["hic_category"] == "low_medium_hic"
    assert df_clean.iloc[1]["hic_category"] == "high_hic"


def test_prepare_features_for_embedding():
    """
    Verify feature expansion of embedding list columns and biological metadata concatenation.
    """
    # Create mock dataset with embedding list structures
    data = {
        "antibody_id": ["ab1", "ab2"],
        "vh_emb_prott5": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "vl_emb_prott5": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        "tm_app": [65.0, 72.0],
        "psr": [0.1, 0.9]
    }
    df = pd.DataFrame(data)
    
    X = prepare_features_for_embedding(df, "prott5")
    
    # Expected columns: vh_prott5_0, vh_prott5_1, vh_prott5_2, vl_prott5_0, vl_prott5_1, vl_prott5_2, tm_app, psr
    expected_cols = [
        "vh_prott5_0", "vh_prott5_1", "vh_prott5_2",
        "vl_prott5_0", "vl_prott5_1", "vl_prott5_2",
        "tm_app", "psr"
    ]
    
    for col in expected_cols:
        assert col in X.columns
        
    assert X.shape == (2, 8)
    assert X.loc[0, "vh_prott5_0"] == 0.1
    assert X.loc[1, "vl_prott5_2"] == 1.2
    assert X.loc[0, "tm_app"] == 65.0


def test_model_pipeline_inference():
    """
    Load the serialized best_model.joblib and verify that it can make
    a valid prediction with correct types and schemas.
    """
    model_path = os.path.join(os.path.dirname(__file__), "..", "src", "best_model.joblib")
    assert os.path.exists(model_path), "Serialized production model best_model.joblib was not found!"
    
    pipeline = joblib.load(model_path)
    
    # 1. Verify pipeline steps
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    
    # 2. Create a mock single client request vector
    # Best model uses 2050 dimensions: 1024 for VH embedding + 1024 for VL embedding + 2 for physical descriptors (tm_app, psr)
    num_features = 2050
    mock_x = np.random.randn(1, num_features)
    df_mock = pd.DataFrame(mock_x)
    
    # Make prediction
    pred = pipeline.predict(df_mock)
    proba = pipeline.predict_proba(df_mock)
    
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
