import os
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier
import numpy as np

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # may be None
MODEL_NAME = os.getenv("MODEL_NAME", "telco_churn_model")
MLFLOW_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"


def _make_dummy_model():
    """
    Fallback model for local dev/tests when MLflow is not available.
    Binary classifier with two classes [0, 1] so predict_proba has 2 columns.
    """
    dummy = DummyClassifier(strategy="prior", random_state=42)
    X = np.array([[0], [1]])
    y = np.array([0, 1])  # has both classes 0 and 1
    dummy.fit(X, y)
    return dummy, "dummy-local-model"


def load_production_model():
    """
    Loads the latest production model from MLflow.
    If MLFLOW_OFFLINE=true, returns a dummy model for local testing.
    """
    # 1) Offline mode for tests / local dev
    if MLFLOW_OFFLINE:
        return _make_dummy_model()

    # 2) Online MLflow mode
    tracking_uri = MLFLOW_TRACKING_URI or "http://mlflow-server:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
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
        raise RuntimeError(
            f"Failed to load model '{MODEL_NAME}' from MLflow: {e}"
        ) from e
