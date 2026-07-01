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
FEAST_DATA_DIR = PROJECT_ROOT / "feast_repo" / "data"
FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)

from scripts.feature_eng.config import (
    USER_FEATURES_COLS,
    PRODUCT_FEATURES_COLS,
    INTERACTION_FEATURES_COLS
)
from scripts.feature_eng.transformers import (
    UserFeatureGenerator,
    ProductFeatureGenerator,
    InteractionFeatureGenerator
)

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

def load_existing_features(file_name):
    """Loads existing features from Parquet."""
    try:
        path = FEAST_DATA_DIR / file_name
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not load {file_name}: {e}")
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
    
    # 1. Select only specific target columns
    df = df[target_cols].copy()
    
    # 2. Final NaN Cleanup
    df = df.fillna(value=DEFAULT_VALUES)
    return df

# --- Main Execution Flow ---
def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for Feast")
    parser.add_argument("--incremental", action="store_true", help="Run in incremental update mode")
    parser.add_argument("--date", type=str, help="Target date for incremental load")
    args = parser.parse_args()

    logger.info(f"=== Starting Feature Pipeline for Feast (Incremental={args.incremental}) ===")
    
    # 1. Load Data
    df, source_path = load_clean_data(args.date if args.incremental else None)

    # 2. Transform Features
    user_transformer = UserFeatureGenerator()
    prod_transformer = ProductFeatureGenerator()
    inter_transformer = InteractionFeatureGenerator()

    if args.incremental:
        dim_user = user_transformer.update(load_existing_features("user_features.parquet"), df)
        dim_prod = prod_transformer.update(load_existing_features("product_features.parquet"), df)
        dim_inter = inter_transformer.update(load_existing_features("interaction_features.parquet"), df)
    else:
        dim_user = user_transformer.transform(df)
        dim_prod = prod_transformer.transform(df)
        dim_inter = inter_transformer.transform(df)
    
    # 3. Align Columns
    dim_user = fill_and_align(dim_user, USER_FEATURES_COLS, "User Features")
    dim_prod = fill_and_align(dim_prod, PRODUCT_FEATURES_COLS, "Product Features")
    dim_inter = fill_and_align(dim_inter, INTERACTION_FEATURES_COLS, "Interaction Features")
    
    # 4. Add Feast specific timestamps
    # Feast requires an event_timestamp for point-in-time joins
    # We set features to an old date, and interactions to now, so they always match
    past_date = pd.Timestamp('2000-01-01', tz='UTC')
    now_date = pd.Timestamp.now(tz="UTC")
    
    dim_user['event_timestamp'] = past_date
    dim_prod['event_timestamp'] = past_date
    dim_inter['event_timestamp'] = now_date
    
    # Clean types for parquet
    dim_user['user_id'] = dim_user['user_id'].astype(str)
    dim_prod['product_id'] = dim_prod['product_id'].astype(str)
    dim_inter['user_id'] = dim_inter['user_id'].astype(str)
    dim_inter['product_id'] = dim_inter['product_id'].astype(str)
    
    # Give interactions a strictly unique timestamp to avoid Dask join bugs
    dim_inter['event_timestamp'] = dim_inter['event_timestamp'] + pd.to_timedelta(dim_inter.index, unit='ms')

    # 5. Save to Parquet for Feast Offline Store
    dim_user.to_parquet(FEAST_DATA_DIR / "user_features.parquet", index=False)
    dim_prod.to_parquet(FEAST_DATA_DIR / "product_features.parquet", index=False)
    dim_inter.to_parquet(FEAST_DATA_DIR / "interaction_features.parquet", index=False)
    
    logger.info("=== Feast Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
