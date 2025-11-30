# docker/Dockerfile.app
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI="http://mlflow-server:5000"
ENV MODEL_NAME="telco_churn_model"

EXPOSE 8000

# Default command = run API (you can override for training jobs)
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
