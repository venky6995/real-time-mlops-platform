from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter("churn_requests_total", "Total churn prediction requests")
REQUEST_LATENCY = Histogram("churn_request_latency_seconds", "Latency of predictions")
