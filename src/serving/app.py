from fastapi import FastAPI
import pandas as pd
from prometheus_client import make_asgi_app, Counter, Histogram


from src.serving.schemas import ChurnRequest, ChurnResponse
from src.registry.mlflow_utils import load_production_model
from src.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

app = FastAPI(title="Telco Churn Real-Time API")

# create your metrics (names used in your repo may vary)
REQUEST_COUNT = Counter('churn_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('churn_request_latency_seconds', 'Request latency')


# mount the prometheus ASGI app at /metrics
app.mount("/metrics", make_asgi_app())

# Will now always return something, never crash
model, model_uri = load_production_model()


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    with REQUEST_LATENCY.time():
        REQUEST_COUNT.inc()

        try:
            payload = req.model_dump()
        except AttributeError:
            payload = req.dict()

        data = pd.DataFrame([payload])
        proba_vector = model.predict_proba(data)[0]

        if len(proba_vector) == 1:
            churn_proba = float(proba_vector[0])
        else:
            churn_proba = float(proba_vector[1])

        return ChurnResponse(
            churn_probability=churn_proba,
            model_version=model_uri,
        )
