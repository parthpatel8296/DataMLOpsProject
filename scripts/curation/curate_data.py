
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import logging
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"

# Map our datasets to their raw landing zones
ENTITIES = {
    "user_interactions": "batch/user_interactions",
    "user_profiles": "batch/user_profiles",
    "product_catalog": "api"
}

def get_latest_partition(entity_path):
    """Finds the latest ingestion_date partition for a given entity."""
    partitions = glob.glob(str(entity_path / "ingestion_date=*"))
    if not partitions:
        msg = f"No partitions found for entity at: {entity_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    # Extract dates and find max
    # Format expected: ingestion_date=YYYYMMDD
    try:
        latest = max(partitions, key=lambda p: datetime.strptime(p.split('=')[-1], '%Y%m%d'))
        return Path(latest)
    except ValueError as e:
        logger.error(f"Error parsing dates in {entity_path}: {e}")
        return None

def load_data(partition_path, entity_name):
    """Loads data from the specified partition."""
    logger.info(f"Loading data for {entity_name} from {partition_path}")
    
    if entity_name == "product_catalog":
        # Handle JSON files for catalog
        # Explicitly load only from the identified latest partition folder
        dfs = []
        # partition_path is already the full path to the specific ingestion_date folder
        for file_path in partition_path.glob("*.json"):
            try:
                data = pd.read_json(file_path)
                if "products" in data.columns:
                     # Flatten nested 'products' list if present (replicating notebook logic)
                    dfs.append(pd.json_normalize(data["products"]))
                else:
                    dfs.append(data)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    
    else:
        # Batch data comes in as Parquet, but we keep CSV as a fallback just in case.
        # Parquet is preferred because it's way faster for Spark and Pandas to read.
        try:
            return pd.read_parquet(partition_path)
        except Exception:
             # If someone dropped a CSV in there manually, let's try to handle it.
            csv_files = list(partition_path.glob("*.csv"))
            if csv_files:
                return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            
            msg = f"No data files found in partition: {partition_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

def process_interactions(df):
    if df.empty: return df
    
    logger.info("Processing Interactions...")
    
    # Clean
    df = df.drop_duplicates()
    df = df.dropna(subset=["user_id", "product_id"])
    df["duration_seconds"] = df["duration_seconds"].fillna(0)
    
    # Cast
    df["user_id"] = pd.to_numeric(df["user_id"], errors='coerce').fillna(0).astype('int64')
    df["product_id"] = pd.to_numeric(df["product_id"], errors='coerce').fillna(0).astype('int64')
    
    # Encode Event Type
    if "event_type" in df.columns:
        event_map = {"view": 1, "click": 2, "add_to_cart": 3, "purchase": 4}
        df["event_strength"] = df["event_type"].map(event_map).fillna(0).astype(int)
    
    # Timestamp Engineering
    if "timestamp" in df.columns:
        # Infer datetime format.
        df["dt"] = pd.to_datetime(df["timestamp"], unit='s') 
        df["hour_of_day"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        
        # Normalize
        df["hour_norm"] = df["hour_of_day"] / 23.0
        df["day_norm"] = df["day_of_week"] / 6.0
        
        # Drop temp dt col if not needed, or keep for partitioning
        # df.drop(columns=['dt'], inplace=True) 

    # Scale Duration
    scaler = MinMaxScaler()
    if "duration_seconds" in df.columns:
        df["duration_scaled"] = scaler.fit_transform(df[["duration_seconds"]])
        
    return df

def process_profiles(df):
    """
    Cleans up user profiles. 
    We fill missing ages with the median to avoid losing records.
    """
    if df.empty: return df
    
    logger.info("Processing Profiles...")
    
    # Validation
    df = df.drop_duplicates()
    df = df.dropna(subset=["user_id"])
    df["user_id"] = pd.to_numeric(df["user_id"], errors='coerce').fillna(0).astype('int64')
    
    # Convert 'tier' into binary flags (one-hot) for the model
    if "tier" in df.columns:
        tier_dummies = pd.get_dummies(df["tier"], prefix="tier", dtype=int)
        df = pd.concat([df, tier_dummies], axis=1)
    
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    
    # Scale age between 0 and 1 so it doesn't overpower other features
    scaler = MinMaxScaler()
    if "age" in df.columns:
        df["age_scaled"] = scaler.fit_transform(df[["age"]])
        
    return df

def process_catalog(df):
    if df.empty: return df
    
    logger.info("Processing Catalog...")
    
    # Clean
    df = df.drop_duplicates()
    df = df.dropna(subset=["product_id"])
    df["product_id"] = pd.to_numeric(df["product_id"], errors='coerce').fillna(0).astype('int64')
    if "category_id" in df.columns:
        df["category_id"] = pd.to_numeric(df["category_id"], errors='coerce').fillna(0).astype('int64')
    if "subcategory_id" in df.columns:
        df["subcategory_id"] = pd.to_numeric(df["subcategory_id"], errors='coerce').fillna(0).astype('int64')
        
    # Encode Category
    if "category" in df.columns:
        cat_dummies = pd.get_dummies(df["category"], prefix="cat", dtype=int)
        df = pd.concat([df, cat_dummies], axis=1)
    
    # Encode Subcategory
    if "subcategory" in df.columns:
        subcat_dummies = pd.get_dummies(df["subcategory"], prefix="subcat", dtype=int)
        df = pd.concat([df, subcat_dummies], axis=1)
        
    # Scale Price & Stock
    scaler = MinMaxScaler()
    if "price" in df.columns:
        median_price = df["price"].median()
        df["price"] = df["price"].fillna(median_price)
        df["price_scaled"] = scaler.fit_transform(df[["price"]])
        
    if "stock_quantity" in df.columns:
        df["stock_quantity"] = df["stock_quantity"].fillna(0)
        df["stock_quantity_scaled"] = scaler.fit_transform(df[["stock_quantity"]])
        
    # Log Transform Rating
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].fillna(0)
        df["rating_count_log"] = np.log1p(df["rating_count"])
        
    return df

def main():
    # Ensure curated dir exists
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store processed dataframes
    processed_dfs = {}
    ingestion_dates = {}

    # 1. Load and Process each entity
    for entity_name, rel_path in ENTITIES.items():
        raw_entity_path = RAW_DIR / rel_path
        
        # Find Latest Partition
        latest_partition = get_latest_partition(raw_entity_path)
        if not latest_partition:
            logger.warning(f"No partitions found for {entity_name} in {raw_entity_path}")
            processed_dfs[entity_name] = pd.DataFrame()
            continue
            
        ingestion_date_str = latest_partition.name.split('=')[-1]
        ingestion_dates[entity_name] = ingestion_date_str
        logger.info(f"Latest partition for {entity_name}: {ingestion_date_str}")
        
        # Load
        df = load_data(latest_partition, entity_name)
        
        if df.empty:
            logger.warning(f"Dataframe empty for {entity_name}. Using empty DF.")
            processed_dfs[entity_name] = pd.DataFrame()
            continue
            
        # Process
        if entity_name == "user_interactions":
            df_curated = process_interactions(df)
        elif entity_name == "user_profiles":
            df_curated = process_profiles(df)
        elif entity_name == "product_catalog":
            df_curated = process_catalog(df)
        else:
            df_curated = df
            
        processed_dfs[entity_name] = df_curated

    # 2. Validation & Merging
    df_interactions = processed_dfs.get("user_interactions", pd.DataFrame())
    df_profiles = processed_dfs.get("user_profiles", pd.DataFrame())
    df_catalog = processed_dfs.get("product_catalog", pd.DataFrame())
    
    if df_interactions.empty:
        msg = "Interactions data is missing or empty. Cannot proceed with curation."
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Merging datasets...")
    
    # Merge Interactions + Profiles
    if not df_profiles.empty:
        # Check for user_id columns
        if "user_id" in df_interactions.columns and "user_id" in df_profiles.columns:
            logger.info(f"Merging profiles ({df_profiles.shape}) into interactions ({df_interactions.shape})...")
            merged_df = pd.merge(df_interactions, df_profiles, on="user_id", how="left", suffixes=("", "_profile"))
        else:
            logger.warning("user_id column missing. Skipping profile merge.")
            merged_df = df_interactions
    else:
        logger.warning("Profiles dataframe is empty. Skipping profile merge.")
        merged_df = df_interactions

    # Merge + Catalog (Final Join)
    if not df_catalog.empty:
        if "product_id" in merged_df.columns and "product_id" in df_catalog.columns:
            logger.info(f"Merging catalog ({df_catalog.shape}) into current df ({merged_df.shape})...")
            merged_df = pd.merge(merged_df, df_catalog, on="product_id", how="left", suffixes=("", "_catalog"))
        else:
            logger.warning("product_id column missing. Skipping catalog merge.")
    else:
        # The catalog is mandatory for recommendations!
        msg = "Catalog dataframe is empty. Cannot proceed as catalog is mandatory."
        logger.error(msg)
        raise RuntimeError(msg)

    # 3. Save to Silver Layer
    # We use 'event_date' partitioning to keep the curated data organized by time
    event_date = ingestion_dates.get("user_interactions", datetime.now().strftime("%Y%m%d"))
    
    target_partition = CURATED_DIR / f"event_date={event_date}"
    target_partition.mkdir(parents=True, exist_ok=True)
    
    output_file = target_partition / "curated-dataset.parquet"
    
    logger.info(f"Saving merged curated data to {output_file}. Shape: {merged_df.shape}")
    merged_df.to_parquet(output_file, index=False)

    # 4. Save metadata
    metadata = {
        "event_date": event_date,
        "created_at": datetime.now().isoformat(),
        "sources": {
            entity: {
                "ingestion_date": date,
                "input_file": str(get_latest_partition(RAW_DIR / ENTITIES[entity])) if get_latest_partition(RAW_DIR / ENTITIES[entity]) else None
            }
            for entity, date in ingestion_dates.items()
        },
        "transformations": [
            "process_interactions",
            "process_profiles" if "user_profiles" in processed_dfs else None,
            "process_catalog" if "product_catalog" in processed_dfs else None,
            "merge_datasets"
        ],
        "row_count": len(merged_df),
        "columns": list(merged_df.columns)
    }
    
    metadata_file = target_partition / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Saved lineage metadata to {metadata_file}")


    logger.info("Curation workflow completed.")

if __name__ == "__main__":
    main()
