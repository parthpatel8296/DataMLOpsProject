import pandas as pd
import sys
from pathlib import Path
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from datetime import datetime
import great_expectations as ge
import glob
import os

# We need to grab the root project path so we can locate our data directories.
# This assumes the script is running from scripts/validation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/raw"

# This is where we'll dump all the reports. Pandas and GE will yell if the folder doesn't exist, so create it.
REPORT_DIR = PROJECT_ROOT / "data/quality_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)



# ====================Data Loading Utils=================================
def get_latest_partition(entity_path):
    """Finds the latest ingestion_date partition for a given entity."""
    partitions = glob.glob(str(entity_path / "ingestion_date=*"))
    if not partitions:
        msg = f"No partitions found for entity at: {entity_path}"
        print(f"Error: {msg}")
        raise FileNotFoundError(msg)
    
    try:
        # Extract dates and find max (ingestion_date=YYYYMMDD)
        latest = max(partitions, key=lambda p: datetime.strptime(p.split('=')[-1], '%Y%m%d'))
        return Path(latest)
    except ValueError as e:
        print(f"Error parsing dates in {entity_path}: {e}")
        raise

def read_latest_parquet(base_path):
    """
    Reads parquet files from the LATEST directory partition.
    """
    if not base_path.exists():
        msg = f"Base path not found for parquet reading: {base_path}"
        print(f"Error: {msg}")
        raise FileNotFoundError(msg)
    
    latest_path = get_latest_partition(base_path)
    print(f"Loading latest parquet data from: {latest_path}")
    df = pd.read_parquet(latest_path)
    
    # Standardize types immediately to match system requirements
    # Specifically catching durations and IDs which might be float/int32 in raw
    cols_to_fix = {
        "user_id": "int64",
        "product_id": "int64",
        "category_id": "int64",
        "subcategory_id": "int64",
        "age": "int64",
        "duration_seconds": "int64",
        "timestamp": "int64"
    }
    for col, dtype in cols_to_fix.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
            
    return df

def read_latest_json(base_path):
    """Reads JSON files from the LATEST directory partition."""
    if not base_path.exists():
        msg = f"Base path not found for JSON reading: {base_path}"
        print(f"Error: {msg}")
        raise FileNotFoundError(msg)
    
    latest_path = get_latest_partition(base_path)
    print(f"Loading latest JSON data from: {latest_path}")
    
    dfs = []
    # Pattern: products_*.json
    for file_path in latest_path.glob("products_*.json"):
        if file_path.is_file():
            try:
                data = pd.read_json(file_path)
                if not data.empty:
                    if "products" in data.columns:
                         dfs.append(pd.json_normalize(data["products"].tolist()))
                    else:
                         dfs.append(data)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Cast IDs to int64 for consistency
    cols_to_cast = ["product_id", "category_id", "subcategory_id", "stock_quantity"]
    for col in cols_to_cast:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            
    return df

# ======================create GE validator===============================
def get_validator(df, suite_name):
    # This initializes the Great Expectations context.
    # We're wrapping our raw dataframe in a GE Validator so we can use powerful
    # built-in checks like .expect_column_values_to_not_be_null().
    ge.get_context() 
    
    engine = PandasExecutionEngine()
    suite = ExpectationSuite(name=suite_name)
    
    # A 'Batch' is just the data plus its metadata.
    batch = Batch(
        data=df,
        batch_definition=None
    )
    
    validator = Validator(
        execution_engine=engine,
        batches=[batch],
        expectation_suite=suite
    )
    return validator

# =====================USER INTERACTIONS VALIDATION=====================
def validate_user_interactions():
    path = DATA_DIR / "batch/user_interactions"
    df = read_latest_parquet(path)
    if df.empty: return {"dataset": "user_interactions", "status": "No data found"}

    validator = get_validator(df, "user_interactions_suite")


    # Validations
    # 1. Essential columns cannot be missing (Null checks)
    validator.expect_column_values_to_not_be_null("user_id")
    validator.expect_column_values_to_not_be_null("product_id")
    validator.expect_column_values_to_not_be_null("timestamp")
    
    # 2. Event type must be one of the known valid actions
    validator.expect_column_values_to_be_in_set("event_type", ["view", "click", "add_to_cart", "purchase"])
    
    # 3. Time spent on page can't be negative, that defies physics
    validator.expect_column_values_to_be_between("duration_seconds", min_value=0)
    
    # 4. Strict Type Checks - prevents downstream string/int mixups
    validator.expect_column_values_to_be_of_type("user_id", "int64")
    validator.expect_column_values_to_be_of_type("product_id", "int64")
    validator.expect_column_values_to_be_of_type("timestamp", "int64")
    validator.expect_column_values_to_be_of_type("duration_seconds", "int64")
    
    result = validator.validate().to_json_dict()

    ret = {
        "dataset": "user_interactions",
        "total_records": len(df),
        "stats_date_min": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
        "stats_date_max": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A",
        "stats_unique_users": df["user_id"].nunique() if "user_id" in df.columns else 0,
        "stats_unique_products": df["product_id"].nunique() if "product_id" in df.columns else 0,
        "stats_event_distribution": df["event_type"].value_counts().to_dict() if "event_type" in df.columns else {},
        "duplicates": df.duplicated().sum(),
        "missing_user_ids": df["user_id"].isna().sum(),
        "invalid_events": (~df["event_type"].isin(["view", "click", "add_to_cart", "purchase"])).sum(),
        "invalid_duration_seconds": (~df["duration_seconds"].between(0, 3600)).sum(),
        "invalid_user_ids": (~df["user_id"].astype(int).between(0, 2**31 - 1)).sum(),
        "invalid_product_ids": (~df["product_id"].astype(int).between(0, 2**31 - 1)).sum(),
        "invalid_timestamps": (~pd.to_datetime(df["timestamp"], errors="coerce").notna()).sum(),
        "expectations_passed": sum(r["success"] for r in result["results"]),
        "expectations_failed": sum(not r["success"] for r in result["results"])
    }

    # This list extracts the human-readable failure reason.
    # GE returns a complex object, so we look for 'expectation_type' or 'type' to know WHICH rule broke.
    ret["failed_expectations"] = [
        f"{r['expectation_config'].get('expectation_type') or r['expectation_config'].get('type') or 'Unknown'} failed on {r['expectation_config'].get('kwargs', {}).get('column', 'dataset')}"
        for r in result["results"] if not r["success"]
    ]
    
    return ret

# =============== USER PROFILES VALIDATION=================
def validate_user_profiles():
    path = DATA_DIR / "batch/user_profiles"
    df = read_latest_parquet(path)
    if df.empty: return {"dataset": "user_profiles", "status": "No data found"}

    validator = get_validator(df, "user_profiles_suite")

    # Validations
    validator.expect_column_values_to_be_unique("user_id")
    validator.expect_column_values_to_be_in_set("tier", ["basic", "premium"])
    # City/Country Checks
    validator.expect_column_values_to_not_be_null("location_city")
    validator.expect_column_values_to_not_be_null("location_country")
    validator.expect_column_values_to_not_be_null("age")
    validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)
    validator.expect_column_values_to_match_regex("location_country", r"^[A-Za-z]+$")
    validator.expect_column_values_to_match_regex("location_city", r"^[A-Za-z]+$")
    validator.expect_column_values_to_be_of_type("user_id", "int64")
    validator.expect_column_values_to_be_of_type("age", "int64")
    validator.expect_column_values_to_be_of_type("tier", "object")
    validator.expect_column_values_to_be_of_type("location_country", "object")
    validator.expect_column_values_to_be_of_type("location_city", "object")

    result = validator.validate().to_json_dict()

    return {
        "dataset": "user_profiles",
        "total_records": len(df),
        "stats_age_mean": round(df["age"].mean(), 2) if "age" in df.columns else 0,
        "stats_unique_cities": df["location_city"].nunique() if "location_city" in df.columns else 0,
        "stats_unique_countries": df["location_country"].nunique() if "location_country" in df.columns else 0,
        "stats_tier_counts": df["tier"].value_counts().to_dict() if "tier" in df.columns else {},
        "duplicate_users": df["user_id"].duplicated().sum(),
        "missing_location": df[["location_city", "location_country"]].isna().any(axis=1).sum(),
        "invalid_tier": (~df["tier"].isin(["basic", "premium"])).sum(),
        "invalid_age": (~df["age"].between(18, 120)).sum(),
        "invalid_age_type": (~df["age"].astype(int).between(18, 120)).sum(),
        "invalid_country_type": (~df["location_country"].astype(str).str.strip().str.match(r"^[A-Za-z]+$",case = False, na=False)).sum(),
        "invalid_city_type": (~df["location_city"].astype(str).str.strip().str.match(r"^[A-Za-z]+$",case = False, na=False)).sum(),
        "failed_expectations": [
            f"{r['expectation_config'].get('expectation_type') or r['expectation_config'].get('type') or 'Unknown'} failed on {r['expectation_config'].get('kwargs', {}).get('column', 'dataset')}: {r.get('result', {}).get('unexpected_count', 'N/A')} failures"
            for r in result["results"] if not r["success"]
        ],
        "expectations_passed": sum(r["success"] for r in result["results"]),
        "expectations_failed": sum(not r["success"] for r in result["results"])
    }

# ======================PRODUCT CATALOG VALIDATION===============================
def validate_product_catalog():
    
    path = DATA_DIR / "api"
    df = read_latest_json(path)
    if df.empty: return {"dataset": "product_catalog", "status": "No data found"}

    validator = get_validator(df, "product_catalog_suite")

    # Validations
    validator.expect_column_values_to_be_unique("product_id")
    validator.expect_column_values_to_be_between("rating_avg", min_value=1.0, max_value=5.0)
    validator.expect_column_values_to_be_between("price", min_value=0.01)
    validator.expect_column_values_to_be_between("stock_quantity", min_value=0)
    # Category and Subcategory ID checks
    validator.expect_column_values_to_not_be_null("category_id")
    validator.expect_column_values_to_not_be_null("subcategory_id")
    validator.expect_column_values_to_not_be_null("product_id")
    validator.expect_column_values_to_be_of_type("product_id", "int64")
    validator.expect_column_values_to_be_of_type("category_id", "int64")
    validator.expect_column_values_to_be_of_type("subcategory_id", "int64")
    validator.expect_column_values_to_be_of_type("rating_avg", "float64")
    validator.expect_column_values_to_be_of_type("price", "float64")
    validator.expect_column_values_to_be_of_type("stock_quantity", "int64")

    result = validator.validate().to_json_dict()

    return {
        "dataset": "product_catalog",
        "total_records": len(df),
        "stats_avg_price": round(df["price"].mean(), 2) if "price" in df.columns else 0,
        "stats_max_price": df["price"].max() if "price" in df.columns else 0,
        "stats_total_stock": df["stock_quantity"].sum() if "stock_quantity" in df.columns else 0,
        "stats_category_counts": df["category"].value_counts().to_dict() if "category" in df.columns else {},
        "invalid_ratings": ((df["rating_avg"] < 1) | (df["rating_avg"] > 5)).sum(),
        "missing_category_ids": df["category_id"].isna().sum(),
        "missing_subcategory_ids": df["subcategory_id"].isna().sum(),
        "invalid_prices": (df["price"] <= 0).sum(),
        "invalid_category_ids": (df["category_id"] < 0).sum(),
        "invalid_subcategory_ids": (df["subcategory_id"] < 0).sum(),
        "invalid_product_ids": (df["product_id"] < 0).sum(),
        "invalid_stock_quantity": (df["stock_quantity"] < 0).sum(),
        "failed_expectations": [
            f"{r['expectation_config'].get('expectation_type') or r['expectation_config'].get('type') or 'Unknown'} failed on {r['expectation_config'].get('kwargs', {}).get('column', 'dataset')}"
            for r in result["results"] if not r["success"]
        ],
        "expectations_passed": sum(r["success"] for r in result["results"]),
        "expectations_failed": sum(not r["success"] for r in result["results"])
    }

def generate_report():
    print("Starting Data Profiling & Validation...")
    
    reports = [
        validate_user_interactions(),
        validate_user_profiles(),
        validate_product_catalog()
    ]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"data_quality_report_{timestamp_str}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RetailX Data Quality Audit\n")
        f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        
        for r in reports:
            f.write(f"\nDATASET: {r['dataset'].upper()}\n")
            f.write("-" * 50 + "\n")
            if "status" in r:
                f.write(f"STATUS: {r['status']}\n")
            else:
                for key, value in r.items():
                    if key != "dataset":
                        readable_key = key.replace("_", " ").title()
                        f.write(f"{readable_key}: {value}\n")
    
    print(f"Detailed Validation Report saved to: {report_path}")

    # --- FAIL FAST ---
    # We want to stop the whole pipeline immediately if the data looks sketchy.
    # This prevents 'garbage-in, garbage-out' models from being trained.
    any_failed = False
    for r in reports:
        if r.get("expectations_failed", 0) > 0:
            print(f"CRITICAL: Dataset '{r['dataset']}' failed {r['expectations_failed']} checks.")
            any_failed = True
        if r.get("status") == "No data found":
            print(f"CRITICAL: Dataset '{r['dataset']}' was empty or missing.")
            any_failed = True
            
    if any_failed:
        # Raising an error here stops the Prefect orchestrator
        raise RuntimeError("Data Validation Failed. See reports for details.")
    else:
        print("Data Validation Passed Successfully.")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)