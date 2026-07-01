import mlflow
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORT_PATH = PROJECT_ROOT / "docs" / "model_metadata_track.json"

# Setup MLflow
mlflow.set_tracking_uri(f"sqlite:///{str(PROJECT_ROOT / 'mlflow.db')}")
EXPERIMENT_NAME = "RecoMart_Experiments"

def export_metadata():
    print(f"Exporting metadata for experiment: {EXPERIMENT_NAME} from {mlflow.get_tracking_uri()}")
    
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print("Experiment not found.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("No runs found.")
        return

    # Get latest successful run
    latest_run = runs[runs['status'] == 'FINISHED'].sort_values("start_time", ascending=False).iloc[0]
    
    # Convert to dictionary (handle NaN)
    metadata = {
        "export_date": datetime.now().isoformat(),
        "run_id": latest_run.run_id,
        "experiment_id": latest_run.experiment_id,
        "status": latest_run.status,
        "artifact_uri": latest_run.artifact_uri,
        "start_time": str(latest_run.start_time),
        "end_time": str(latest_run.end_time),
        "metrics": {k.replace("metrics.", ""): v for k, v in latest_run.items() if k.startswith("metrics.") and pd.notna(v)},
        "params": {k.replace("params.", ""): v for k, v in latest_run.items() if k.startswith("params.") and pd.notna(v)},
        "tags": {k.replace("tags.", ""): v for k, v in latest_run.items() if k.startswith("tags.")}
    }

    # Write to JSON
    # Write to JSON
    
    # Clean up numpy types for JSON serialization
    def default_converter(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

    with open(EXPORT_PATH, "w") as f:
        json.dump(metadata, f, indent=4, default=default_converter)
        
    print(f"Metadata exported successfully to: {EXPORT_PATH}")

if __name__ == "__main__":
    export_metadata()
