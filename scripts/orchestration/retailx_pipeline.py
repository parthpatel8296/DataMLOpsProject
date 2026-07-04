import os
import sys
import subprocess
import logging
from prefect import flow, task, get_run_logger
from datetime import datetime, timedelta

# --- Setup Logging (File + Stream) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline_orchestration.log")

# Create File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
file_handler.setLevel(logging.INFO)

# Attach to Root Logger and Prefect Loggers
logging.getLogger().addHandler(file_handler)
logging.getLogger("prefect").addHandler(file_handler)
logging.getLogger("prefect.flow_runs").addHandler(file_handler)
logging.getLogger("prefect.task_runs").addHandler(file_handler)

# Define common task configuration
TASK_RETRIES = 3
TASK_RETRY_DELAY = 60 # base seconds
TASK_RETRY_JITTER = 0.1 # 10% jitter

def run_script(script_path, args=None, cwd=None, description="script"):
    """
    Helper to run a python script as a subprocess.
    """
    prefect_logger = get_run_logger()
    
    msg_start = f"Starting {description}: {script_path}"
    prefect_logger.info(msg_start)
    logging.info(msg_start)
    
    python_exe = sys.executable
    cmd = [python_exe, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        msg_out = f"{description} Output:\n{result.stdout}"
        prefect_logger.info(msg_out)
        logging.info(msg_out)
        
        if result.stderr:
             msg_err = f"{description} Stderr:\n{result.stderr}"
             prefect_logger.warning(msg_err)
             logging.warning(msg_err)
             
        msg_end = f"{description} completed successfully."
        prefect_logger.info(msg_end)
        logging.info(msg_end)
        
    except subprocess.CalledProcessError as e:
        prefect_logger.error(f"{description} failed with return code {e.returncode}")
        logging.error(f"{description} failed with return code {e.returncode}")
        prefect_logger.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stdout:\n{e.stdout}")
        prefect_logger.error(f"Stderr:\n{e.stderr}")
        logging.error(f"Stderr:\n{e.stderr}")
        raise RuntimeError(f"{description} failed.") from e

@task(name="Ingest Interactions", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def ingest_interactions_task(target_date):
    script = os.path.join(PROJECT_ROOT, "scripts", "Ingestion", "userinteraction_batch_ingestion.py")
    run_script(script, args=["--date", target_date], description="Ingest Interactions")

@task(name="Ingest Profiles", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def ingest_profiles_task(target_date):
    script = os.path.join(PROJECT_ROOT, "scripts", "Ingestion", "user_profiles_batch_ingestion.py")
    run_script(script, args=["--date", target_date], description="Ingest Profiles")

@task(name="Ingest Products", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def ingest_products_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "Ingestion", "ingest_product.py")
    run_script(script, description="Ingest Products")

@task(name="Validate Data", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def validate_data_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "validation", "validation_script.py")
    run_script(script, description="Data Validation")

@task(name="Prepare Data", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def prepare_data_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "curation", "curate_data.py")
    run_script(script, description="Data Curation")

@task(name="Transform Features", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def transform_features_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "feature_eng", "main.py")
    # For orchestration, we usually run in full mode or incremental depending on policy
    # Here we default to incremental logic as per original
    run_script(script, args=["--incremental"], description="Feature Engineering")

@task(name="Train Model", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def train_model_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "training", "train_eval.py")
    run_script(script, description="Model Training")

def notify_on_failure(flow, flow_run, state):
    msg = f"CRITICAL: Pipeline Flow '{flow.name}' FAILED in state {state}"
    print(msg)
    logging.error(msg)

def get_latest_source_date():
    """Finds the latest date available in the source interactions directory."""
    source_dir = os.path.join(PROJECT_ROOT, "data", "source", "user_interactions")
    if not os.path.exists(source_dir):
        return None
    files = [f for f in os.listdir(source_dir) if f.endswith("_user_interactions.csv")]
    if not files:
        return None
    # Extract dates and find max
    dates = [f.split('_')[0] for f in files]
    return max(dates)

from typing import Optional

@flow(name="RetailX Pipeline", log_prints=True, on_failure=[notify_on_failure])
def retailx_pipeline(override_date: Optional[str] = None):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting RetailX Pipeline Execution")

    # 1. Ingest Data
    if override_date:
        target_date = override_date
    else:
        target_date = get_latest_source_date()
        if not target_date:
            # Fallback to today if no files found (will likely fail but better than erroring here)
            target_date = datetime.now().strftime("%Y%m%d")
        
    prefect_logger.info(f"Using Target Ingestion Date: {target_date}")
    ingest_interactions_task(target_date)
    ingest_profiles_task(target_date)
    ingest_products_task() 
    
    # 2. Validate & Prepare
    validate_data_task()
    prepare_data_task()
    
    # 3. Transform
    transform_features_task()
    
    # 4. Train
    train_model_task()

    prefect_logger.info("RetailX Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run or Serve the RetailX Pipeline")
    parser.add_argument("--serve", action="store_true", help="Serve the flow with a daily schedule")
    parser.add_argument("--date", type=str, help="Target date for ingestion (YYYYMMDD)")
    args = parser.parse_args()
    
    if args.serve:
        retailx_pipeline.serve(
            name="retailx-daily-deployment",
            cron="30 2 * * *", # Run daily at 02:30 AM
            tags=["retailx", "daily"]
        )
    else:
        retailx_pipeline(override_date=args.date)
