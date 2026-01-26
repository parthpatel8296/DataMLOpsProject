import requests
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import argparse
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
Path(PROJECT_ROOT / "data/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=PROJECT_ROOT / "data/logs/ingestion.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

API_URL = "http://127.0.0.1:8000/products"
MAX_RETRIES = 3

def ingest_products():
    """
    Hits our local FastAPI to pull the latest products and saves them as a raw JSON partition.
    Includes built-in retries because network/startup timing can be tricky.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # We timeout after 5s to avoid hanging the whole pipeline indefinitely
            response = requests.get(API_URL, timeout=5)
            response.raise_for_status()

            data = response.json()
            date_str = datetime.now().strftime("%Y%m%d")
            
            # Save to a Hive-style partition folder so validation/curation can find it easily
            output_dir = PROJECT_ROOT / f"data/raw/api/ingestion_date={date_str}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            with open(output_dir / f"products_{date_str}.json", "w") as f:
                json.dump(data, f, indent=2)

            logging.info(
                f"API ingestion successful | records={len(data)} | path={output_dir}"
            )
            return

        except Exception as e:
            logging.warning(f"API ingestion attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                logging.error("All API ingestion attempts failed. Check if FastAPI is running.")
                raise e
            time.sleep(2) # Short backoff before trying again

   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Product API Ingestion")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode (loop)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for continuous mode")
    args = parser.parse_args()

    if args.continuous:
        print(f"Starting continuous ingestion mode (Interval: {args.interval}s)...")
        while True:
            ingest_products()
            print(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        ingest_products()
