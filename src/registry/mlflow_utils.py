import os
import logging

import mlflow
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
import numpy as np

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # may be None
MODEL_NAME = os.getenv("MODEL_NAME", "telco_churn_model")
MLFLOW_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"


def _make_dummy_model():
    """
    Fallback model for local dev/tests or when MLflow is unavailable.
    Binary classifier with two classes [0, 1] so predict_proba has 2 columns.
    """
    dummy = DummyClassifier(strategy="prior", random_state=42)
    X = np.array([[0], [1]])
    y = np.array([0, 1])  # both classes present
    dummy.fit(X, y)
    return dummy, "dummy-local-model"


def load_production_model():
    """
    Loads the latest production model from MLflow.
    - If MLFLOW_OFFLINE=true => dummy model
    - If MLflow is unreachable or no model found => dummy model
    """
    # 1) Offline mode (used in CI/tests or if you explicitly set it)
    if MLFLOW_OFFLINE:
        logger.warning("MLFLOW_OFFLINE=true, using dummy local model.")
        return _make_dummy_model()

    # 2) Online MLflow mode
    tracking_uri = MLFLOW_TRACKING_URI or "http://mlflow-server:5000"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Try Production stage first
        try:
            prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        except TypeError:
            # Some MLflow versions handle this slightly differently
            prod_versions = []

        if prod_versions:
            version = prod_versions[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"
        else:
            # Fallback: any latest version
            all_versions = client.get_latest_versions(MODEL_NAME)
            if not all_versions:
                raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'.")
            version = all_versions[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"

        logger.info("Loading model from MLflow: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri

    except (MlflowException, Exception) as e:
        # Very important: DO NOT crash the app, fall back to dummy
        logger.warning(
            "Failed to load model '%s' from MLflow (tracking_uri=%s). "
            "Falling back to dummy model. Error: %s",
            MODEL_NAME,
            tracking_uri,
            e,
        )
        return _make_dummy_model()
