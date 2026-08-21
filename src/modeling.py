
"""
modeling.py - Comparative Benchmarking of PLM Embeddings and Model Families
Focus: Binary Classification with feature expansion and benchmarking.

This script implements a complete pipeline for:
1. Loading processed antibody datasets and target categories.
2. Expanding PLM embeddings (ESM2, ProtT5, AntiBERTy) into feature matrices.
3. Conducting Hyperparameter Optimization (HPO) for multiple model families.
4. Comparing the performance of different embedding-model combinations.
"""

import logging
import os
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Acts as a central repository for all available machine learning models
    and their associated hyperparameter search spaces for Optuna.
    """
    _registry: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register(cls, name: str, search_space_fn: Callable):
        """
        Decorator to register a model class with its HPO search space.
        """
        def wrapper(model_class: type):
            cls._registry[name] = {
                "class": model_class, 
                "search_space": search_space_fn
            }
            return model_class
        return wrapper

# -----------------------------------------------------------------------------
# Search Space Definitions
# -----------------------------------------------------------------------------

@ModelRegistry.register("logreg", lambda t: {
    "C": t.suggest_float("C", 1e-3, 10, log=True),
    "max_iter": 1000
})
class RegisteredLR(LogisticRegression): 
    """Logistic Regression with L2 Regularization."""

@ModelRegistry.register("svc", lambda t: {
    "C": t.suggest_float("C", 0.1, 10, log=True),
    "probability": True
})
class RegisteredSVC(SVC): 
    """Support Vector Classifier with RBF kernel."""

@ModelRegistry.register("randomforest", lambda t: {
    "n_estimators": t.suggest_int("n_estimators", 50, 200),
    "max_depth": t.suggest_int("max_depth", 3, 15)
})
class RegisteredRF(RandomForestClassifier): 
    """Random Forest Ensemble Classifier."""

@ModelRegistry.register("xgb", lambda t: {
    "n_estimators": t.suggest_int("n_estimators", 50, 200),
    "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True)
})
class RegisteredXGB(XGBClassifier): 
    """Extreme Gradient Boosting (XGBoost) Classifier."""

@ModelRegistry.register("knn", lambda t: {
    "n_neighbors": t.suggest_int("n_neighbors", 3, 15)
})
class RegisteredKNN(KNeighborsClassifier): 
    """K-Nearest Neighbors Classifier."""

@ModelRegistry.register("ridge", lambda t: {
    "alpha": t.suggest_float("alpha", 0.1, 10.0, log=True)
})
class RegisteredRidge(RidgeClassifier): 
    """Ridge Classifier (Linear model with L2 penalty)."""

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features_for_embedding(df: pd.DataFrame, emb_type: str) -> pd.DataFrame:
    """
    Converts list-like embedding columns into a flattned matrix and 
    merges it with biological metadata (TmApp, PSR, B-Cell Subsets).
    
    Args:
        df: The integrated dataframe.
        emb_type: The model name ('esm2', 'prott5', or 'antiberty').
    """
    vh_col = f"vh_emb_{emb_type}"
    vl_col = f"vl_emb_{emb_type}"
    
    if vh_col not in df.columns or vl_col not in df.columns:
        raise ValueError(f"Embedding columns for {emb_type} not found in dataframe.")
    
    # Expand VH and VL arrays into two matrices
    vh_matrix = np.stack(df[vh_col].values)
    vl_matrix = np.stack(df[vl_col].values)
    
    # Generate meaningful column names for the expanded features
    vh_names = [f"vh_{emb_type}_{i}" for i in range(vh_matrix.shape[1])]
    vl_names = [f"vl_{emb_type}_{i}" for i in range(vl_matrix.shape[1])]
    
    # Horizontally stack VH and VL embeddings
    X_emb = pd.DataFrame(
        np.hstack([vh_matrix, vl_matrix]), 
        columns=vh_names + vl_names, 
        index=df.index
    )
    
    # Merge with biological descriptors (TmApp, PSR)
    bio_prefixes = ["tm_app", "psr"]
    for col in df.columns:
        if any(col.startswith(p) for p in bio_prefixes):
            X_emb[col] = df[col]
            
    return X_emb

def build_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    """
    Encapsulates preprocessing and model training into a scikit-learn Pipeline.
    Ensures that scaling and imputation are done separately for each fold (preventing data leakage).
    """
    model_info = ModelRegistry._registry[model_name]
    
    # If params is a simple dict (like study.best_params), wrap it to get full params
    if not isinstance(params, optuna.trial.BaseTrial):
        full_params = model_info["search_space"](optuna.trial.FixedTrial(params))
    else:
        full_params = model_info["search_space"](params)

    model_instance = model_info["class"](**full_params)
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Fills missing TmApp/PSR with dataset median
        ('scaler', StandardScaler()),                 # Standardizes features (0 mean, 1 variance)
        ('model', model_instance)                     # The classifier itself
    ])

# ══════════════════════════════════════════════════════════════════════════════
# HPO & BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def run_hpo(model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 20) -> optuna.Study:
    """
    Executes Bayesian hyperparameter optimization using the Optuna framework.
    Goal: Maximize AUPRC to ensure high precision on the risk class (High HIC).
    """
    model_info = ModelRegistry._registry[model_name]
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters from the model's search space
        params = model_info["search_space"](trial)
        
        # Build and evaluate the pipeline using cross-validation
        pipeline = build_pipeline(model_name, params)
        scores = cross_validate(
            pipeline, X, y, 
            cv=cv_strategy, 
            scoring='average_precision', 
            n_jobs=-1
        )
        return np.mean(scores['test_score'])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

def run_benchmarking(data_path: str):
    """
    Orchestrates the comparisons between ESM2, ProtT5, and AntiBERTy.
    Saves a summary CSV for analysis.
    """
    df = pd.read_parquet(data_path)
    
    # Encode target to binary (Class 1 = High Risk)
    y = (df["hic_category"] == "high_hic").astype(int)
    
    benchmark_results = []
    
    for emb_name in ["esm2", "prott5", "antiberty"]:
        logger.info(f"--- BENCHMARKING EMBEDDING: {emb_name.upper()} ---")
        
        # Prepare the specific feature set for this embedding
        X = prepare_features_for_embedding(df, emb_name)
        
        # Split into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Test each model family defined in the Registry
        for model_key in ModelRegistry._registry:
            logger.info(f"Searching best parameters for {model_key} using {emb_name}...")
            study = run_hpo(model_key, X_train, y_train, n_trials=20)
            
            # Re-train using only the best parameters found
            final_pipeline = build_pipeline(model_key, study.best_params)
            final_pipeline.fit(X_train, y_train)
            
            # Predict probabilities (with fallback for non-probabilistic models)
            if hasattr(final_pipeline.named_steps['model'], "predict_proba"):
                y_probs = final_pipeline.predict_proba(X_test)[:, 1]
            else:
                # Ridge fallback: convert decision function to quasi-probability
                d_func = final_pipeline.decision_function(X_test)
                y_probs = 1 / (1 + np.exp(-d_func))
            
            # Record performance metrics
            roc_auc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
            
            benchmark_results.append({
                "Embedding": emb_name,
                "Model": model_key,
                "ROC_AUC": roc_auc,
                "AUPRC": pr_auc,
                "Best_Params": str(study.best_params)
            })
            
    # Save the consolidated table
    summary_df = pd.DataFrame(benchmark_results)
    summary_df.to_csv("benchmark_results.csv", index=False)
    logger.info("Success: Comparative benchmarking results saved to 'benchmark_results.csv'")
    return summary_df

if __name__ == "__main__":
    # Point to the master integrated dataset
    INPUT_URI = os.path.join(os.path.dirname(__file__), "..", "data", "final_dataset.parquet")
    if os.path.exists(INPUT_URI):
        run_benchmarking(INPUT_URI)
    else:
        logger.error(f"Input data not found at: {INPUT_URI}")