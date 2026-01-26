import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MetaStore:
    def __init__(self, db_path, schema_path=None):
        self.db_path = Path(db_path)
        # Ensure the parent directory exists (critical for DVC reproduction)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_path = Path(schema_path) if schema_path else Path(__file__).resolve().parent / "feature_engineering_schema.sql"
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Ensures the schema is applied to the database."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found at {self.schema_path}")
            return

        with open(self.schema_path, 'r') as f:
            schema_sql = f.read()

        conn = self._get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Database initialized with schema.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        finally:
            conn.close()

    def register_feature_view(self, name, entity, description, schema, ttl_days=365, owner="recomart"):
        """Registers a new feature view and its column metadata."""
        conn = self._get_connection()
        try:
            # 1. Register View
            conn.execute("""
                INSERT OR IGNORE INTO feature_registry 
                (feature_view_name, entity_name, description, ttl_days, owner)
                VALUES (?, ?, ?, ?, ?)
            """, (name, entity, description, ttl_days, owner))

            # 2. Register Metadata (Columns)
            for col_name, data_type in schema.items():
                conn.execute("""
                    INSERT OR IGNORE INTO feature_metadata 
                    (feature_view_name, column_name, data_type)
                    VALUES (?, ?, ?)
                """, (name, col_name, data_type))
            
            conn.commit()
            logger.info(f"Registered feature view: {name}")
        except Exception as e:
            logger.error(f"Failed to register feature view {name}: {e}")
        finally:
            conn.close()

    def save_features(self, table_name, df, mode="append"):
        """Saves features to a table in the database."""
        conn = self._get_connection()
        try:
            if_exists = "append" if mode == "append" else "replace"
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logger.info(f"Saved {len(df)} rows to {table_name} ({mode})")
        except Exception as e:
            logger.error(f"Failed to save features to {table_name}: {e}")
        finally:
            conn.close()

    def get_historical_features(self, entity_df, feature_views, versions=None):
        """
        Performs a point-in-time join across multiple feature views.
        Equivalent to Feast's get_historical_features.
        """
        conn = self._get_connection()
        try:
            # 1. Load Dimension Tables
            dim_user = pd.read_sql("SELECT * FROM dim_user_features", conn)
            dim_prod = pd.read_sql("SELECT * FROM dim_product_features", conn)
            
            # 2. Handle Version Metadata Retrieval
            if versions is None:
                # Fetch Latest Versions for Entities if not specified
                versions_df = pd.read_sql("""
                    SELECT feature_entity, version_number 
                    FROM feature_versions 
                    WHERE version_id IN (SELECT MAX(version_id) FROM feature_versions GROUP BY feature_entity)
                """, conn)
                versions = {
                    row['feature_entity']: row['version_number'] 
                    for _, row in versions_df.iterrows()
                }
            
            user_v = versions.get('user', 0)
            prod_v = versions.get('product', 0)

            # 3. Perform Join
            result = entity_df.merge(dim_user, on="user_id", how="left", suffixes=('', '_user'))
            result = result.merge(dim_prod, on="product_id", how="left", suffixes=('', '_prod'))
            
            # 4. Inject Version Context
            result['user_version'] = user_v
            result['product_version'] = prod_v
            
            return result
        finally:
            conn.close()

    def get_online_features(self, entity_rows, feature_views):
        """Retrieves the specific features (from metadata) for online inference."""
        conn = self._get_connection()
        try:
            # 1. Map feature views to their registered columns
            view_columns = {}
            for view in feature_views:
                cols_df = pd.read_sql(f"SELECT column_name FROM feature_metadata WHERE feature_view_name = '{view}'", conn)
                view_columns[view] = cols_df['column_name'].tolist()

            results = []
            for row in entity_rows:
                combined = {}
                
                # User Features
                if "user_features" in view_columns:
                    user_id = row.get('user_id')
                    cols = ", ".join(["user_id"] + view_columns["user_features"])
                    user_f = pd.read_sql(f"SELECT {cols} FROM dim_user_features WHERE user_id = {user_id} LIMIT 1", conn)
                    if not user_f.empty: combined.update(user_f.iloc[0].to_dict())

                # Product Features
                if "product_features" in view_columns:
                    product_id = row.get('product_id')
                    cols = ", ".join(["product_id"] + view_columns["product_features"])
                    prod_f = pd.read_sql(f"SELECT {cols} FROM dim_product_features WHERE product_id = {product_id} LIMIT 1", conn)
                    if not prod_f.empty: combined.update(prod_f.iloc[0].to_dict())

                # Interaction Features
                if "interaction_features" in view_columns:
                    user_id = row.get('user_id')
                    product_id = row.get('product_id')
                    cols = ", ".join(["user_id", "product_id"] + view_columns["interaction_features"])
                    int_f = pd.read_sql(f"SELECT {cols} FROM fact_user_item_features WHERE user_id = {user_id} AND product_id = {product_id} ORDER BY created_at DESC LIMIT 1", conn)
                    if not int_f.empty: combined.update(int_f.iloc[0].to_dict())
                
                results.append(combined)
            return results
        finally:
            conn.close()

    def log_version(self, entity, feature_name, version, logic="standard", **kwargs):
        """Logs a version record with optional metadata."""
        conn = self._get_connection()
        data_source = kwargs.get('data_source')
        try:
            conn.execute("""
                INSERT INTO feature_versions (feature_entity, feature_name, version_number, transformation_logic, data_source)
                VALUES (?, ?, ?, ?, ?)
            """, (entity, feature_name, version, logic, data_source))
            conn.commit()
            logger.info(f"Logged version {version} for {entity}")
        finally:
            conn.close()

    def get_next_version(self, entity):
        """Determines the next version number for a given entity."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(version_number) FROM feature_versions WHERE feature_entity = ?", (entity,))
            max_v = cursor.fetchone()[0]
            return (max_v + 1) if max_v is not None else 1
        finally:
            conn.close()
