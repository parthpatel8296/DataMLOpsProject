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

def demo_feast_lineage():
    print("==================================================")
    print("DEMO: FEATURE LINEAGE (FEAST REGISTRY)")
    print("==================================================")
    
    print("Fetching Feature Views from Feast Registry...\n")
    try:
        feature_views = store.list_feature_views()
        
        for view in feature_views:
            print(f"Feature View: {view.name}")
            print(f"  - Entities: {view.entities}")
            print(f"  - Tags: {view.tags}")
            print(f"  - Description: {view.description}")
            
            # Show columns for this view
            feature_names = [f.name for f in view.features]
            print(f"  - Features: {', '.join(feature_names)}\n")
            
    except Exception as e:
        print(f"Failed to fetch from Feast Registry: {e}")

def demo_feast_time_travel():
    print("==================================================")
    print("DEMO: TIME TRAVEL (FEAST POINT-IN-TIME)")
    print("==================================================")
    
    try:
        # Test point-in-time request
        # 2021 (Past - might be empty) vs 2026 (Future/Current)
        entity_df = pd.DataFrame({
            'user_id': ["1"],
            'product_id': ["1081"],
            'event_timestamp': [pd.Timestamp("2021-01-01", tz='UTC')]
        })
        
        print("Requesting Historical Features for User 1 in Jan 2021...")
        
        features = [
            "user_features:age", "user_features:location_city"
        ]
        
        result = store.get_historical_features(
            entity_df=entity_df,
            features=features
        ).to_df()
        print(result)
        
    except Exception as e:
        print(f"Failed to perform time travel query: {e}")

if __name__ == "__main__":
    demo_feast_lineage()
    demo_feast_time_travel()
