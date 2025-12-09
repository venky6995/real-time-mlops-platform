from fastapi import FastAPI
import pandas as pd

# Use a dedicated CollectorRegistry to avoid duplicate registration during tests
from prometheus_client import CollectorRegistry, make_asgi_app, Counter, Histogram

from src.serving.schemas import ChurnRequest, ChurnResponse
from src.registry.mlflow_utils import load_production_model

app = FastAPI(title="Telco Churn Real-Time API")

# Create a dedicated registry for this FastAPI app to prevent duplicate timeseries
PROM_REGISTRY = CollectorRegistry()

# Create metrics attached to the dedicated registry (prevents duplicate registration issues)
REQUEST_COUNT = Counter(
    'churn_requests_total',
    'Total prediction requests',
    registry=PROM_REGISTRY
)

REQUEST_LATENCY = Histogram(
    'churn_request_latency_seconds',
    'Request latency',
    registry=PROM_REGISTRY
)

# Mount the Prometheus ASGI app using our registry on /metrics
app.mount("/metrics", make_asgi_app(registry=PROM_REGISTRY))

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
