import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data_processing.preprocess import load_data, preprocess

MLFLOW_EXPERIMENT_NAME = "telco-churn-realtime"

# Do NOT default to mlflow-server here
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # None by default
MLFLOW_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"


def main():
    # 1) Decide where to log
    if MLFLOW_OFFLINE:
        # local file-based store for tests / local dev
        mlflow.set_tracking_uri("file:./mlruns")
    elif MLFLOW_TRACKING_URI:
        # real MLflow server (e.g. on your VM or inside docker)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # else: mlflow default (./mlruns)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_data("data/sample_telco.csv")
    X_train, X_val, y_train, y_val = preprocess(df)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=50,      # keep smaller for CI speed
            max_depth=8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 8)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="telco_churn_model",
        )

        print(f"Validation ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
