#!/usr/bin/env python3
"""
SageMaker training script for customer churn prediction model.

This script can run locally or on SageMaker training jobs.
"""
import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path):
    """Load training data from CSV files."""
    logger.info(f"Loading data from {data_path}")

    # SageMaker passes data directory
    if os.path.isdir(data_path):
        files = list(Path(data_path).glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files found in {data_path}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} samples")
    return df


def preprocess_features(df):
    """Preprocess features for training."""
    # Separate features and target
    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' not found in data")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Handle categorical variables (if any)
    categorical_cols = X.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Fill missing values
    X = X.fillna(X.mean())

    logger.info(f"Preprocessed features shape: {X.shape}")
    return X, y


def train_model(X_train, y_train, hyperparameters):
    """Train Random Forest model with specified hyperparameters."""
    logger.info("Training Random Forest model...")
    logger.info(f"Hyperparameters: {hyperparameters}")

    model = RandomForestClassifier(
        n_estimators=hyperparameters.get("n_estimators", 100),
        max_depth=hyperparameters.get("max_depth", 10),
        min_samples_split=hyperparameters.get("min_samples_split", 5),
        min_samples_leaf=hyperparameters.get("min_samples_leaf", 2),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": conf_matrix.tolist(),
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

    return metrics


def save_model(model, model_dir, feature_names, metrics):
    """Save trained model and metadata."""
    logger.info(f"Saving model to {model_dir}")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save feature names
    feature_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_path, "w") as f:
        json.dump({"features": feature_names}, f)

    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = os.path.join(model_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    logger.info("Model saved successfully")
    logger.info(f"\nTop 5 Important Features:\n{importance_df.head()}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data paths (SageMaker specific)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "./data/validation"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    logger.info("Starting training job...")
    logger.info(f"Arguments: {vars(args)}")

    # Load data
    train_df = load_data(args.train)
    val_df = load_data(args.validation)

    # Preprocess
    X_train, y_train = preprocess_features(train_df)
    X_val, y_val = preprocess_features(val_df)

    # Ensure same features in validation set
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for col in missing_cols:
        X_val[col] = 0
    X_val = X_val[X_train.columns]

    # Train model
    hyperparameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
    }

    model = train_model(X_train, y_train, hyperparameters)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Save model
    save_model(model, args.model_dir, list(X_train.columns), metrics)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
