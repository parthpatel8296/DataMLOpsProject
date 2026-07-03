import pandas as pd
from pathlib import Path
import sys
import logging
from feast import FeatureStore

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize Feast Store
store = FeatureStore(repo_path=str(PROJECT_ROOT / "feast_repo"))

def demo_offline_retrieval():
    print("\n--- 1. Point-in-Time Correctness (Offline Retrieval via Feast) ---")
    
    try:
        interaction_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features" / "interaction_features.parquet")
        entity_df = interaction_df.head(2).copy()
        
        # Feast expects entities as strings per our schema
        entity_df['user_id'] = entity_df['user_id'].astype(str)
        entity_df['product_id'] = entity_df['product_id'].astype(str)
        
        print("Entity Dataframe (Trigger Events):")
        print(entity_df[['user_id', 'product_id', 'event_timestamp']])
        
        features = [
            "user_features:age", "user_features:location_city",
            "product_features:category", "product_features:price"
        ]
        
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=features
        ).to_df()
        
        print("\nJoined Training Data (Point-in-Time Correct via Feast):")
        print(training_df)
    except Exception as e:
        print(f"Error during offline retrieval: {e}")

def demo_online_retrieval():
    print("\n--- 2. Online Retrieval (Real-time Inference via Feast) ---")
    
    # Ensure entities match Feast types
    entity_rows = [
        {"user_id": "1", "product_id": "1081"}
    ]
    
    features = [
        "user_features:age", "user_features:location_city",
        "product_features:category", "product_features:price"
    ]
    
    try:
        online_features = store.get_online_features(
            features=features,
            entity_rows=entity_rows
        ).to_dict()
        
        print("Feature Vector for User '1' & Product '1081':")
        for k, v in online_features.items():
            print(f"  {k}: {v[0]}")
    except Exception as e:
        print(f"Error during online retrieval (Run 'feast materialize-incremental' in feast_repo): {e}")

if __name__ == "__main__":
    demo_offline_retrieval()
    demo_online_retrieval()
