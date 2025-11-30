import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data_processing.preprocess import load_data, preprocess

MLFLOW_EXPERIMENT_NAME = "telco-churn-realtime"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # may be None
MLFLOW_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"


def main():
    # 1) Decide tracking backend
    if MLFLOW_OFFLINE:
        # CI / local offline mode: use local SQLite file in project dir
        tracking_uri = "sqlite:///mlflow_ci.db"
    else:
        # Real MLflow server or default local mlruns
        tracking_uri = MLFLOW_TRACKING_URI or "http://mlflow-server:5000"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 2) Load and preprocess data
    df = load_data("data/sample_telco.csv")
    X_train, X_val, y_train, y_val = preprocess(df)

    # 3) Train and log
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=50,  # smaller for faster CI
            max_depth=8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 8)
        mlflow.log_metric("roc_auc", roc_auc)

        # 4) ONLY log the model to MLflow when NOT in offline/CI mode
        if not MLFLOW_OFFLINE:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="telco_churn_model",
            )

        print(f"Validation ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
