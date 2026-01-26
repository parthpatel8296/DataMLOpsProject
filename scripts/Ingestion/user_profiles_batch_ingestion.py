import sys
import socketserver
import os
from datetime import datetime
import logging
from pathlib import Path
import argparse
import time

# Patch for PySpark on Windows (Python 3.13+)
if not hasattr(socketserver, "UnixStreamServer"):
    class UnixStreamServer(socketserver.TCPServer):
        pass
    socketserver.UnixStreamServer = UnixStreamServer

from pyspark.sql import SparkSession

# Explicitly set HADOOP_HOME for Windows compatibility
os.environ['HADOOP_HOME'] = "C:\\hadoop"
# Ensure hadoop/bin is in PATH
if r"C:\hadoop\bin" not in os.environ['PATH']:
    os.environ['PATH'] += r";C:\hadoop\bin"



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print("=== Spark batch ingestion script started ===")

# Ensure log directory exists
Path(PROJECT_ROOT / "data/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=PROJECT_ROOT / "data/logs/ingestion.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def run_batch_ingestion(ingestion_date_str=None):
    if not ingestion_date_str:
        ingestion_date_str = datetime.now().strftime("%Y%m%d")
        
    source_path = PROJECT_ROOT / "data" / "source" / "user_profiles" / f"{ingestion_date_str}_user_profiles.csv"
    
    # Fail fast if source doesn't exist
    if not os.path.exists(source_path):
        msg = f"Source file not found for date {ingestion_date_str}: {source_path}"
        logging.error(msg)
        print(msg)
        raise FileNotFoundError(msg)

    try:
        print("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("UserProfilesBatchIngestion") \
            .getOrCreate()

        print(f"Reading source CSV from: {source_path}")

        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(str(source_path))

        print(f"CSV read successful. Row count: {df.count()}")
        # Partition by ingestion_date
        output_path = PROJECT_ROOT / "data/raw/batch/user_profiles" / f"ingestion_date={ingestion_date_str}"
        print(f"Writing to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True) # Ensure directory exists
        df.write.mode("overwrite").parquet(str(output_path))

        logging.info(
            f"User Profiles Batch ingestion successful | rows={df.count()} | path={output_path}"
        )

        print("User Profiles Batch ingestion completed successfully.")

    except Exception as e:
        print("ERROR occurred:", e)
        logging.error(f"User Profiles Batch ingestion failed: {e}")
        raise e

    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run User Profiles Batch Ingestion")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode (loop)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for continuous mode")
    parser.add_argument("--date", type=str, help="Target ingestion date (YYYYMMDD)")
    args = parser.parse_args()
    
    if args.continuous:
        print(f"Starting continuous ingestion mode (Interval: {args.interval}s)...")
        while True:
            run_batch_ingestion(args.date)
            print(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        run_batch_ingestion(args.date)
