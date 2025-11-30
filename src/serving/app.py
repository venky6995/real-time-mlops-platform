from fastapi import FastAPI
import pandas as pd
from src.serving.schemas import ChurnRequest, ChurnResponse
from src.registry.mlflow_utils import load_production_model
from src.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

app = FastAPI(title="Telco Churn Real-Time API")

model, model_uri = load_production_model()


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    with REQUEST_LATENCY.time():
        REQUEST_COUNT.inc()

        # pydantic v2: model_dump(); v1: dict()
        try:
            payload = req.model_dump()
        except AttributeError:
            payload = req.dict()

        data = pd.DataFrame([payload])

        proba_vector = model.predict_proba(data)[0]
        # If binary classifier with classes [0, 1], proba for churn=1 is column 1
        if proba_vector.shape[0] == 1:
            churn_proba = float(proba_vector[0])
        else:
            churn_proba = float(proba_vector[1])

        return ChurnResponse(
            churn_probability=churn_proba,
            model_version=model_uri,
        )
