#!/usr/bin/env python3
"""
SageMaker inference handler for the churn prediction model.

This script defines how the model is loaded and how predictions are made.
"""
import json
import logging
import os
import pickle
from io import StringIO

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load the model for inference.

    Args:
        model_dir: The directory where model files are stored

    Returns:
        Tuple of (model, feature_names)
    """
    logger.info(f"Loading model from {model_dir}")

    # Load the model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load feature names
    feature_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_path, "r") as f:
        feature_data = json.load(f)
        feature_names = feature_data["features"]

    logger.info(f"Model loaded successfully with {len(feature_names)} features")

    return {"model": model, "feature_names": feature_names}


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the input data.

    Args:
        request_body: The request payload
        request_content_type: The content type of the request

    Returns:
        pandas DataFrame ready for prediction
    """
    logger.info(f"Processing input with content type: {request_content_type}")

    if request_content_type == "application/json":
        # Parse JSON input
        data = json.loads(request_body)

        # Handle both single instance and batch
        if isinstance(data, dict):
            if "instances" in data:
                # Batch format: {"instances": [{...}, {...}]}
                df = pd.DataFrame(data["instances"])
            else:
                # Single instance: {...}
                df = pd.DataFrame([data])
        elif isinstance(data, list):
            # List of instances: [{...}, {...}]
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")

    elif request_content_type == "text/csv":
        # Parse CSV input
        df = pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    logger.info(f"Processed {len(df)} instances for prediction")
    return df


def predict_fn(input_data, model_dict):
    """
    Make predictions on the input data.

    Args:
        input_data: Preprocessed input DataFrame
        model_dict: Dictionary containing model and feature names

    Returns:
        Dictionary with predictions and probabilities
    """
    model = model_dict["model"]
    feature_names = model_dict["feature_names"]

    logger.info(f"Making predictions on {len(input_data)} instances")

    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_data.columns)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Filling with zeros.")
        for feature in missing_features:
            input_data[feature] = 0

    # Select and order features
    X = input_data[feature_names]

    # Handle missing values
    X = X.fillna(X.mean())

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Feature importance for this prediction (optional)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(5).to_dict("records")

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "feature_importance": feature_importance,
    }


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.

    Args:
        prediction: The prediction results
        response_content_type: The desired response content type

    Returns:
        Serialized prediction response
    """
    logger.info(f"Formatting output as {response_content_type}")

    if response_content_type == "application/json":
        return json.dumps(prediction)
    elif response_content_type == "text/csv":
        # Return simple CSV format with predictions
        df = pd.DataFrame({
            "prediction": prediction["predictions"],
            "probability_class_0": [p[0] for p in prediction["probabilities"]],
            "probability_class_1": [p[1] for p in prediction["probabilities"]],
        })
        return df.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")


# Optional: Health check endpoint
def ping():
    """
    Health check endpoint for SageMaker.

    Returns:
        200 if the model is loaded and ready
    """
    logger.info("Health check requested")
    return 200
