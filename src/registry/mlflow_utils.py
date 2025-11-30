import os
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
import numpy as np

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "telco_churn_model")


def _make_dummy_model():
    """
    Fallback model for local dev/tests when MLflow is not available.
    Always predicts 'no churn' (0).
    """
    dummy = DummyClassifier(strategy="most_frequent")
    # train on fake data just so predict_proba works
    X = np.array([[0], [1]])
    y = np.array([0, 0])
    dummy.fit(X, y)
    return dummy, "dummy-local-model"


def load_production_model():
    """
    Loads the latest production model from MLflow.
    If MLFLOW_OFFLINE=true, returns a dummy model for local testing.
    """
    # 1) Offline mode for local tests
    if os.getenv("MLFLOW_OFFLINE", "false").lower() == "true":
        return _make_dummy_model()

    # 2) Online MLflow mode
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        # Try to load Production stage model
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            # No production model yet, fall back to latest version if exists
            all_versions = client.get_latest_versions(MODEL_NAME)
            if not all_versions:
                raise RuntimeError(
                    f"No registered versions found for model '{MODEL_NAME}'."
                )
            model_uri = f"models:/{MODEL_NAME}/{all_versions[0].version}"
        else:
            model_uri = f"models:/{MODEL_NAME}/{prod_versions[0].version}"

        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri

    except MlflowException as e:
        # For dev, you can either raise or fallback
        raise RuntimeError(
            f"Failed to load model '{MODEL_NAME}' from MLflow: {e}"
        ) from e
