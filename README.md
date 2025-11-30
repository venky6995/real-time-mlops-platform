# Real-Time MLOps Platform with CI/CD and Model Monitoring

_M.Tech Project – Venkatesh Doddi (M24DE3084)_  
_Supervisor: Dr. Dip Sankar Banerjee, IIT Jodhpur_

This project implements a **Real-Time MLOps Platform** for the **Telco Customer Churn** problem.  
It covers the full lifecycle: **data processing, model training, CI/CD, real-time inference,
continuous monitoring, and retraining triggers**. :contentReference[oaicite:2]{index=2}

---

## 1. Objectives

- Automate **model training, versioning, and deployment** using CI/CD.
- Expose the churn model via a **FastAPI** endpoint for real-time predictions.
- Monitor **data drift, prediction drift, latency, and accuracy**.
- Trigger **retraining** when performance degrades.
- Deploy everything using containerised, cloud-native technologies on **Google Cloud Platform (GCP)**. :contentReference[oaicite:3]{index=3}

---

## 2. System Architecture

High-level flow (matches project proposal):

1. Source code is hosted on **GitHub**.
2. **GitHub Actions** runs automated tests and builds Docker images.
3. Images are pushed to **Artifact Registry** (GCP).
4. A **Compute Engine VM** pulls and runs containers via **Docker Compose**:
   - MLflow + Postgres (tracking / registry)
   - FastAPI model serving API
   - Prometheus + Grafana (metrics dashboards)
   - Evidently job (drift monitoring)
5. Model artifacts and data are stored in **Google Cloud Storage (GCS)**.
6. Drift & performance issues can trigger **retraining**. :contentReference[oaicite:4]{index=4}

---

## 3. Modules Description

This repository is structured according to the modules in the proposal: :contentReference[oaicite:5]{index=5}

```text
real-time-mlops-platform/
├── data/
│   └── sample_telco.csv          # sample of Telco churn dataset
├── src/
│   ├── data_processing/
│   │   └── preprocess.py         # Data Processing
│   ├── training/
│   │   └── train.py              # Model Training + MLflow logging
│   ├── registry/
│   │   └── mlflow_utils.py       # Model Registry helper
│   ├── serving/
│   │   ├── app.py                # FastAPI API Serving
│   │   └── schemas.py            # Request/response schemas
│   ├── monitoring/
│   │   ├── metrics.py            # Prometheus metrics
│   │   ├── drift_job.py          # Evidently drift report
│   │   └── retrain_trigger.py    # Auto-Retraining trigger
│   └── config/
│       └── config.yaml           # thresholds, paths (optional)
├── docker/
│   ├── Dockerfile.app
│   ├── Dockerfile.mlflow
│   ├── docker-compose.yml
│   └── prometheus.yml
├── .github/
│   └── workflows/
│       └── ci_cd.yml             # CI/CD Pipeline (GitHub Actions)
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_app.py
├── requirements.txt
└── README.md
