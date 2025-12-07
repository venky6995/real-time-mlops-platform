import os
import logging

import mlflow
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
import numpy as np

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow-server:5000"
MODEL_NAME = os.getenv("MODEL_NAME", "telco_churn_model")
MLFLOW_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"


def _make_dummy_model():
    dummy = DummyClassifier(strategy="prior", random_state=42)
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    dummy.fit(X, y)
    return dummy, "dummy-local-model"


def load_production_model():
    if MLFLOW_OFFLINE:
        logger.warning("MLFLOW_OFFLINE=true, using dummy local model.")
        return _make_dummy_model()

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        try:
            prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        except TypeError:
            prod_versions = []

        if prod_versions:
            version = prod_versions[0].version
        else:
            all_versions = client.get_latest_versions(MODEL_NAME)
            if not all_versions:
                raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'.")
            version = all_versions[0].version

        model_uri = f"models:/{MODEL_NAME}/{version}"
        logger.info("Loading model from MLflow: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri

    except (MlflowException, Exception) as e:
        logger.warning(
            "Failed to load model '%s' from MLflow (tracking_uri=%s). Falling back to dummy. Error: %s",
            MODEL_NAME,
            MLFLOW_TRACKING_URI,
            e,
        )
        return _make_dummy_model()
