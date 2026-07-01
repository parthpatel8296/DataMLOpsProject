import sys
import logging
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
mlflow.set_tracking_uri(f"sqlite:///{str(PROJECT_ROOT / 'mlflow.db')}")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    client = MlflowClient()
    experiment_name = "RecoMart_Experiments"
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        logger.error(f"Experiment '{experiment_name}' not found.")
        sys.exit(1)

    logger.info(f"Searching for the best Content-Based model based on Hit Rate...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.content_hit_rate_50 > 0",
        order_by=["metrics.content_hit_rate_50 DESC"],
        max_results=1
    )

    if not runs:
        logger.error("No runs found with content_hit_rate_50 metric.")
        sys.exit(1)

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    hit_rate = best_run.data.metrics.get("content_hit_rate_50", 0)

    logger.info(f"Selected Best Run ID: {best_run_id} with Hit Rate@50: {hit_rate:.4f}")

    # Register the model
    model_name = "RecoMart_Model"
    model_uri = f"runs:/{best_run_id}/cb_model"

    logger.info(f"Registering model in MLflow Model Registry as '{model_name}'...")
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Successfully registered model '{model_name}' version {model_version.version}.")
        
        # We can also attempt to transition to Production if the registry supports it locally
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            logger.info(f"Transitioned '{model_name}' version {model_version.version} to Production.")
        except Exception as e:
            logger.warning(f"Could not transition stage to Production (might not be supported on this backend): {e}")

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
