import json
from pathlib import Path
import subprocess

THRESHOLD = 0.3  # example drift share threshold

def should_retrain(report_path: str) -> bool:
    # Evidently also generates JSON if you ask it; here assume you parse metrics
    data = json.loads(Path(report_path).read_text())
    drift_share = data["metrics"][0]["result"]["drift_share"]
    return drift_share > THRESHOLD

def main():
    if should_retrain("reports/drift_report.json"):
        # Trigger training pipeline
        subprocess.run(["python", "src/training/train.py"], check=True)

if __name__ == "__main__":
    main()
