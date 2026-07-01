import sys
import logging
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Import Schema & MetaStore
from scripts.feature_eng.config import (
    USER_FEATURES_COLS,
    PRODUCT_FEATURES_COLS,
    INTERACTION_FEATURES_COLS,
    FEATURE_DB_PATH,
    USER_REGISTRY_METADATA,
    PRODUCT_REGISTRY_METADATA,
    INTERACTION_REGISTRY_METADATA
)
from scripts.feature_eng.transformers import (
    UserFeatureGenerator,
    ProductFeatureGenerator,
    InteractionFeatureGenerator
)
from scripts.feature_eng.metastore import MetaStore

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Loader ---
def load_clean_data(date_str: str = None) -> pd.DataFrame:
    """Loads curated dataset."""
    if date_str:
        partition_path = DATA_DIR / f"curated/event_date={date_str}/curated-dataset.parquet"
        input_path = partition_path
    else:
        curated_dir = DATA_DIR / "curated"
        partitions = list(curated_dir.glob("event_date=*"))
        if not partitions: raise FileNotFoundError("No curated partitions found")
        latest = max(partitions, key=lambda p: datetime.strptime(p.name.split('=')[-1], '%Y%m%d'))
        input_path = latest / "curated-dataset.parquet"
    
    return pd.read_parquet(input_path), str(input_path)

def load_existing_features(metastore, table_name):
    """Loads existing features from SQL."""
    try:
        conn = metastore._get_connection()
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- Alignment Helper ---
DEFAULT_VALUES = {
    'interaction_count': 0, 'activity_frequency': 0.0, 'purchase_ratio': 0.0,
    'category_confidence': 0.0, 'age': -1, 'price': 0.0, 'rating_avg': 0.0,
    'rating_count': 0, 'unique_users': 0, 'popularity_score': 0.0,
    'conversion_rate': 0.0, 'diversity_score': 0.0, 'implicit_score': 0.0,
    'hour_sin': 0.0, 'hour_cos': 0.0, 'day_sin': 0.0, 'day_cos': 0.0,
    'time_since_last_seconds': 0, 'duration_scaled': 0.0, 'price_scaled': 0.0,
    'rating_count_log': 0.0, 'stock_quantity_scaled': 0.0, 'age_scaled': 0.0,
    'location_city': 'Unknown', 'location_country': 'Unknown', 'tier': 'Unknown',
    'preferred_category': 'Unknown', 'category': 'Unknown', 'subcategory': 'Unknown',
    'brand': 'Unknown', 'category_id': 0, 'subcategory_id': 0
}

def fill_and_align(df, target_cols, name="table"):
    """Ensures dataframe matches the expected schema."""
    missing = set(target_cols) - set(df.columns)
    if missing:
        logger.info(f"[{name}] Adding missing columns: {missing}")
        for c in missing:
            df[c] = DEFAULT_VALUES.get(c, 0)
    
    # 1. Select only specific target columns (removes _purchases etc)
    df = df[target_cols]
    
    # 2. Final NaN Cleanup
    df = df.fillna(value=DEFAULT_VALUES)
    return df

# --- Main Execution Flow ---
def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--incremental", action="store_true", help="Run in incremental update mode")
    parser.add_argument("--date", type=str, help="Target date for incremental load")
    args = parser.parse_args()

    logger.info(f"=== Starting Feature Pipeline (Incremental={args.incremental}) ===")
    
    # Initialize MetaStore with explicit schema path for initialization
    schema_file = Path(__file__).resolve().parent / "feature_engineering_schema.sql"
    ms = MetaStore(PROJECT_ROOT / FEATURE_DB_PATH, schema_path=schema_file)

    # 1. Register Features (One-time or idempotent)
    ms.register_feature_view(**USER_REGISTRY_METADATA)
    ms.register_feature_view(**PRODUCT_REGISTRY_METADATA)
    ms.register_feature_view(**INTERACTION_REGISTRY_METADATA)

    # 2. Load Data
    df, source_path = load_clean_data(args.date if args.incremental else None)

    # 3. Transform Features
    user_transformer = UserFeatureGenerator()
    prod_transformer = ProductFeatureGenerator()
    inter_transformer = InteractionFeatureGenerator()

    if args.incremental:
        dim_user = user_transformer.update(load_existing_features(ms, "dim_user_features"), df)
        dim_prod = prod_transformer.update(load_existing_features(ms, "dim_product_features"), df)
        dim_inter = inter_transformer.update(load_existing_features(ms, "fact_user_item_features"), df)
    else:
        dim_user = user_transformer.transform(df)
        dim_prod = prod_transformer.transform(df)
        dim_inter = inter_transformer.transform(df)
    
    # 3.5 Align Columns
    dim_user = fill_and_align(dim_user, USER_FEATURES_COLS, "User Features")
    dim_prod = fill_and_align(dim_prod, PRODUCT_FEATURES_COLS, "Product Features")
    dim_inter = fill_and_align(dim_inter, INTERACTION_FEATURES_COLS, "Interaction Features")
    
    # 4. Save to SQL
    # We always replace because the transformers (transform/update) return the consolidated "latest" state.
    # Appending would cause duplication as the update logic already includes history.
    ms.save_features("dim_user_features", dim_user, mode="replace")
    ms.save_features("dim_product_features", dim_prod, mode="replace")
    ms.save_features("fact_user_item_features", dim_inter, mode="replace")

    # 5. Log Versions
    user_ver = ms.get_next_version("user")
    prod_ver = ms.get_next_version("product")
    inter_ver = ms.get_next_version("interaction")

    ms.log_version("user", "dim_user_features", user_ver, "sql_pipeline", data_source=source_path)
    ms.log_version("product", "dim_product_features", prod_ver, "sql_pipeline", data_source=source_path)
    ms.log_version("interaction", "fact_user_item_features", inter_ver, "sql_pipeline", data_source=source_path)

    logger.info(f"Updated SQL Feature Store: User v{user_ver}, Product v{prod_ver}, Interaction v{inter_ver}")
    logger.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
