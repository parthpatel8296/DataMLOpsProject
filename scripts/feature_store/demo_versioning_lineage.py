import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import logging

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.feature_eng.config import FEATURE_DB_PATH
from scripts.feature_eng.metastore import MetaStore

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Pandas Display Settings ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Initialize MetaStore
ms = MetaStore(PROJECT_ROOT / FEATURE_DB_PATH)

def demo_lineage():
    print("==================================================")
    print("DEMO: FEATURE LINEAGE")
    print("==================================================")
    
    # In our Custom Registry, lineage is transparent:
    # 1. We can see WHERE the feature came from
    # 2. We can see WHAT its schema is
    
    print("Fetching Feature Lineage from SQL Registry...")
    conn = ms._get_connection()
    try:
        registry_df = pd.read_sql("SELECT * FROM feature_registry", conn)
        metadata_df = pd.read_sql("SELECT * FROM feature_metadata", conn)
        
        for _, view in registry_df.iterrows():
            print(f"\nFeature View: {view['feature_view_name']}")
            print(f"  - Entity: {view['entity_name']}")
            print(f"  - Owner:  {view['owner']}")
            print(f"  - Description: {view['description']}")
            
            # Fetch latest version for this entity
            entity = view['entity_name']
            # If entity is composite (e.g. 'user,product'), we pick the primary one or first one for versioning
            primary_entity = entity.split(',')[0]
            version_df = pd.read_sql(f"SELECT MAX(version_number) as current_v FROM feature_versions WHERE feature_entity = '{primary_entity}'", conn)
            current_v = version_df['current_v'].iloc[0] if not version_df.empty else "N/A"
            print(f"  - Latest Version: v{current_v if current_v else 0}")
            
            # Show columns for this view (Deduplicated with set)
            cols = sorted(list(set(metadata_df[metadata_df['feature_view_name'] == view['feature_view_name']]['column_name'].tolist())))
            print(f"  - Columns ({len(cols)}): {', '.join(cols)}")
            
    finally:
        conn.close()

def demo_time_travel():
    print("\n==================================================")
    print("DEMO: VERSIONING & TIME TRAVEL")
    print("==================================================")
    
    print("Fetching valid user from database...")
    conn = ms._get_connection()
    try:
        user_df = pd.read_sql("SELECT user_id FROM dim_user_features LIMIT 1", conn)
        if user_df.empty:
            print("No users found. Run main_sql.py first.")
            return
        
        target_user = user_df['user_id'].iloc[0]
        print(f"Selected User for Demo: {target_user}")
        
        # 1. Show Latest Version (v3)
        print(f"\n[Scenario A] Requesting Latest Features (Current State)...")
        version_df_latest = pd.read_sql("SELECT version_number FROM feature_versions WHERE feature_entity = 'user' ORDER BY version_id DESC LIMIT 1", conn)
        latest_v = version_df_latest['version_number'].iloc[0] if not version_df_latest.empty else 0
        
        # Test point-in-time request dataframe
        entity_df = pd.DataFrame({
            'user_id': [target_user],
            'product_id': [1081],
            'event_timestamp': [pd.Timestamp("2021-01-01")]
        })

        print(f"--- [Retrieving data tagged as v{latest_v}] ---")
        result_latest = ms.get_historical_features(entity_df, ["user_features"], versions={'user': latest_v, 'product': 1})
        print(result_latest)

        # 2. Show Specific Past Version (v1)
        print(f"\n[Scenario B] Requesting Version 1 Specifically (Time Travel Registry)...")
        print(f"--- [Retrieving data tagged as v1] ---")
        result_v1 = ms.get_historical_features(entity_df, ["user_features"], versions={'user': 1, 'product': 1})
        print(result_v1)
        
        print("\n(Note: Values are the same as tables are currently updated in-place, but metadata provenance is tracked separately.)")
        
    finally:
        conn.close()

if __name__ == "__main__":
    demo_lineage()
    demo_time_travel()
