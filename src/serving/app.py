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

        data = pd.DataFrame([req.dict()])
        proba = model.predict_proba(data)[:, 1][0]
        return ChurnResponse(churn_probability=float(proba), model_version=model_uri)


from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
