"""
train_best_model.py - Train and serialize the winning ProtT5 + Logistic Regression model
"""

import logging
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from modeling import prepare_features_for_embedding, build_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    # 1. Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "final_dataset.parquet")
    model_save_path = os.path.join(base_dir, "best_model.joblib")

    logger.info(f"Loading final dataset from {data_path}")
    df = pd.read_parquet(data_path)

    # 2. Encode target to binary (Class 1 = High Risk / high_hic)
    y = (df["hic_category"] == "high_hic").astype(int)

    # 3. Prepare features using ProtT5 embedding
    logger.info("Preparing features for ProtT5 embedding...")
    X = prepare_features_for_embedding(df, "prott5")

    # 4. Stratified Train-Test Split (80/20) for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    logger.info(f"High-HIC prevalence - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # 5. Build pipeline with winning Logistic Regression model
    # Winner hyperparameter space suggested standard or optimal parameters.
    # We will use Logistic Regression with class_weight='balanced'
    best_params = {
        "C": 0.1,  # typical regularized logistic regression strength
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }
    
    # We use build_pipeline from modeling.py to maintain complete architectural consistency
    # Note: modeling.py's build_pipeline uses RegisteredLR which inherits from LogisticRegression.
    # We can pass these parameters to maintain our processing pipeline structure (SimpleImputer, StandardScaler, Model)
    logger.info("Building model pipeline...")
    pipeline = build_pipeline("logreg", best_params)

    # Apply class_weight balanced to the RegisteredLR instance inside the pipeline
    pipeline.named_steps['model'].set_params(class_weight='balanced', random_state=42)

    # 6. Fit and evaluate on validation split
    logger.info("Fitting model on training split...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_prob)
    auprc_score = average_precision_score(y_test, y_prob)

    logger.info("\n--- VALIDATION METRICS ---")
    logger.info(f"ROC-AUC: {auc_score:.4f}")
    logger.info(f"AUPRC (Average Precision): {auprc_score:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # 7. Train on the full dataset for maximum production accuracy
    logger.info("Re-training model pipeline on full dataset for production...")
    full_pipeline = build_pipeline("logreg", best_params)
    full_pipeline.named_steps['model'].set_params(class_weight='balanced', random_state=42)
    full_pipeline.fit(X, y)

    # 8. Serialize and save the model
    logger.info(f"Saving serialized model pipeline to {model_save_path}")
    joblib.dump(full_pipeline, model_save_path)
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
