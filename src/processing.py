"""
processing.py - Clean Data Preparation for Antibody Classification
"""

import logging
import yaml
import pandas as pd
import numpy as np
import os
from typing import Dict

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_experimental_data(
    raw_gcs_path: str, 
    hic_threshold: float = 11.5
) -> pd.DataFrame:
    """
    Loads raw Excel data from GCS and standardizes sequences and metadata.
    """
    logger.info(f"Loading raw experimental data from: {raw_gcs_path}")
    try:
        # iloc[:-2] handles the specific footer rows in the Shehata dataset
        df = pd.read_excel(raw_gcs_path, engine="openpyxl").iloc[:-2]
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        raise

    cols = {
        "Clone name": "antibody_id",
        "B cell subset": "b_cell_subset",
        "VH Protein": "vh",
        "VL Protein": "vl",
        "HIC retention time (min)": "hic",
        "TmApp (°C)": "tm_app",
        "PSR Score": "psr"
    }
    
    # Filter and rename
    available_cols = [c for c in cols.keys() if c in df.columns]
    df = df[available_cols].rename(columns={k: v for k, v in cols.items() if k in available_cols})
    
    # Sequence standardization: Removes gaps/dashes for PLM tokenizers
    df["vh"] = df["vh"].str.replace("-", "", regex=False)
    df["vl"] = df["vl"].str.replace("-", "", regex=False)
    
    # Numeric conversion for bio-features
    for col in ["hic", "tm_app", "psr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Target Engineering (Binary Classification)
    df["hic_category"] = np.where(df["hic"] >= hic_threshold, "high_hic", "low_medium_hic")
    
    # Drop rows without critical info
    df = df.dropna(subset=["hic", "vh", "vl"])
    
    logger.info(f"Base dataset: {len(df)} antibodies.")
    return df

def merge_all_plm_embeddings(
    main_df: pd.DataFrame, 
    embedding_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Merges multiple PLM embedding files (ESM2, ProtT5, AntiBERTy) into the main dataframe.
    """
    combined_df = main_df.copy()
    for model_name, gcs_path in embedding_map.items():
        logger.info(f"Merging {model_name} embeddings from {gcs_path}")
        try:
            emb_df = pd.read_parquet(gcs_path)
            
            # Identify embedding columns
            vh_col = "vh_embedding" if "vh_embedding" in emb_df.columns else f"vh_emb_{model_name}"
            vl_col = "vl_embedding" if "vl_embedding" in emb_df.columns else f"vl_emb_{model_name}"
            
            emb_df = emb_df.rename(columns={vh_col: f"vh_emb_{model_name}", vl_col: f"vl_emb_{model_name}"})
            cols_to_join = ["antibody_id", f"vh_emb_{model_name}", f"vl_emb_{model_name}"]
            combined_df = combined_df.merge(emb_df[cols_to_join], on="antibody_id", how="left")
        except Exception as e:
            logger.error(f"Error merging {model_name}: {e}")
            
    return combined_df

def run_full_processing_pipeline(config_path: str):
    """
    Orchestrates the full data processing workflow.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load, Clean, and Encode
    df = load_and_clean_experimental_data(
        config['paths']['raw_data'], 
        config['params']['hic_threshold']
    )

    # 2. Merge Embeddings
    df_merged = merge_all_plm_embeddings(df, config['embeddings'])

    # 3. Save to GCS
    output_uri = config['paths']['output_dataset']
    logger.info(f"Saving final dataset to {output_uri}")
    df_merged.to_parquet(output_uri, index=False)

if __name__ == "__main__":
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "path_config.yaml")
    run_full_processing_pipeline(CONFIG_FILE)