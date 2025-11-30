import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from google.cloud import storage

REF_DATA_PATH = "gs://telco-churn-data-m24de3084/reference/reference.csv"
PROD_DATA_PATH = "gs://telco-churn-data-m24de3084/production/latest_window.csv"
REPORT_OUTPUT = "reports/drift_report.html"

def load_from_gcs(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        _, bucket_name, *blob_parts = path.replace("gs://", "").split("/")
        blob_name = "/".join(blob_parts)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        s = blob.download_as_text()
        from io import StringIO
        return pd.read_csv(StringIO(s))
    else:
        return pd.read_csv(path)

def main():
    ref = load_from_gcs(REF_DATA_PATH)
    prod = load_from_gcs(PROD_DATA_PATH)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod)
    report.save_html(REPORT_OUTPUT)

if __name__ == "__main__":
    main()
