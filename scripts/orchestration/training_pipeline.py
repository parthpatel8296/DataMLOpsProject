import os
import sys
import subprocess
import logging
from prefect import flow, task, get_run_logger

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

@task(name="Train Model", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY, retry_jitter_factor=TASK_RETRY_JITTER)
def train_model_task():
    script = os.path.join(PROJECT_ROOT, "scripts", "training", "train_eval.py")
    run_script(script, description="Model Training")

def notify_on_failure(flow, flow_run, state):
    msg = f"CRITICAL: Training Pipeline Flow '{flow.name}' FAILED in state {state}"
    print(msg)
    logging.error(msg)

@flow(name="RetailX Training Pipeline", log_prints=True, on_failure=[notify_on_failure])
def retailx_training_pipeline():
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting RetailX Training Pipeline Execution")

    # Train
    train_model_task()

    prefect_logger.info("RetailX Training Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run or Serve the RetailX Training Pipeline")
    parser.add_argument("--serve", action="store_true", help="Serve the flow with a daily schedule")
    args = parser.parse_args()
    
    if args.serve:
        retailx_training_pipeline.serve(
            name="retailx-training-deployment",
            cron="30 2 * * *", # Run daily at 02:30 AM
            tags=["retailx", "mlops"]
        )
    else:
        retailx_training_pipeline()
