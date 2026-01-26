import pandas as pd
from pathlib import Path
import sys
import logging

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.feature_eng.config import FEATURE_DB_PATH
from scripts.feature_eng.metastore import MetaStore

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize MetaStore
ms = MetaStore(PROJECT_ROOT / FEATURE_DB_PATH)

def demo_offline_retrieval():
    print("\n--- 1. Point-in-Time Correctness (Offline Retrieval) ---")
    
    # robustly load a sample from source SQL table
    conn = ms._get_connection()
    try:
        source_df = pd.read_sql("SELECT * FROM dim_user_features LIMIT 2", conn)
        if source_df.empty:
            print("No data found in dim_user_features. Please run main.py first.")
            return
            
        # Construct entity_df from actual data
        entity_df = pd.DataFrame({
            'user_id': source_df['user_id'].values,
            'product_id': [1081, 1001], # Sample product IDs
            'event_timestamp': [pd.Timestamp.now()] * 2
        })
        
        print("Entity Dataframe (Trigger Events):")
        print(entity_df)
        
        # Retrieval via MetaStore
        training_df = ms.get_historical_features(entity_df, ["user_features", "product_features"])
        
        print("\nJoined Training Data (Point-in-Time Correct):")
        print(training_df.head())
    finally:
        conn.close()

def demo_online_retrieval():
    print("\n--- 2. Online Retrieval (Real-time Inference) ---")
    
    # User 1 and Product 1081
    entity_rows = [
        {"user_id": 1, "product_id": 1081}
    ]
    
    features = ms.get_online_features(entity_rows, ["user_features", "product_features"])
    
    print("Feature Vector for User 1 & Product 1081:")
    for k, v in features[0].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    demo_offline_retrieval()
    demo_online_retrieval()
