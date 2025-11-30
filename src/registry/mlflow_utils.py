import logging

from sklearn.dummy import DummyClassifier
import numpy as np

logger = logging.getLogger(__name__)


def _make_dummy_model():
    """
    Simple dummy binary classifier for dev/demo.
    Always returns a fixed probability distribution over classes [0, 1].
    """
    dummy = DummyClassifier(strategy="prior", random_state=42)
    X = np.array([[0], [1]])
    y = np.array([0, 1])  # both classes
    dummy.fit(X, y)
    return dummy, "dummy-local-model"


def load_production_model():
    """
    TEMPORARY IMPLEMENTATION:
    - Do NOT call MLflow at all.
    - Always return the dummy model.

    This guarantees FastAPI will start even if MLflow is misconfigured.
    Later you can reintroduce MLflow integration once everything is stable.
    """
    logger.warning("MLflow disabled in this build. Using dummy local model only.")
    return _make_dummy_model()
