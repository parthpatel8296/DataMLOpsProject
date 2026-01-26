# Feature Store Schema Definitions

# Database Path
FEATURE_DB_PATH = "data/processed/features/recomart_features.db"

# User Features Schema
USER_FEATURES_COLS = [
    "user_id", "age", "age_scaled", "location_city", "location_country", "tier",
    "interaction_count", "activity_frequency", "purchase_ratio",
    "first_interaction", "last_interaction", "preferred_category",
    "category_confidence", "created_at", "updated_at"
]

# Product Features Schema
PRODUCT_FEATURES_COLS = [
    "product_id", "category_id", "category", "subcategory_id", "subcategory",
    "brand", "price", "price_scaled", "rating_avg", "rating_count", "rating_count_log",
    "stock_quantity_scaled", "interaction_count", "unique_users", "popularity_score",
    "conversion_rate", "diversity_score", "created_at", "updated_at"
]

# Interaction Features Schema
INTERACTION_FEATURES_COLS = [
    "user_id", "product_id", "interaction_count", "implicit_score",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "duration_scaled",
    "time_since_last_seconds", "interaction_date", "created_at"
]

# Feature Version Metadata Schema
FEATURE_VERSIONS_COLS = [
    "version_id", "feature_entity", "feature_name", "version_number",
    "data_source", "transformation_logic", "quality_score", "is_active",
    "created_at"
]

# Registration Metadata (for Custom MetaStore)
USER_REGISTRY_METADATA = {
    "name": "user_features",
    "entity": "user",
    "description": "Customer behavioral and demographic features",
    "schema": {
        "age_scaled": "Float32",
        "location_city": "String",
        "location_country": "String",
        "tier": "String",
        "interaction_count": "Int64",
        "activity_frequency": "Float32",
        "purchase_ratio": "Float32",
        "preferred_category": "String",
        "category_confidence": "Float32"
    }
}

PRODUCT_REGISTRY_METADATA = {
    "name": "product_features",
    "entity": "product",
    "description": "Product catalog and performance features",
    "schema": {
        "category_id": "Int64",
        "category": "String",
        "subcategory_id": "Int64",
        "subcategory": "String",
        "brand": "String",
        "price_scaled": "Float32",
        "rating_avg": "Float32",
        "rating_count_log": "Float32",
        "stock_quantity_scaled": "Float32",
        "interaction_count": "Int64",
        "popularity_score": "Float32",
        "conversion_rate": "Float32",
        "diversity_score": "Float32"
    }
}

INTERACTION_REGISTRY_METADATA = {
    "name": "interaction_features",
    "entity": "user,product",
    "description": "Contextual interaction features between users and products",
    "schema": {
        "interaction_count": "Int64",
        "implicit_score": "Float32",
        "time_since_last_seconds": "Int64",
        "duration_scaled": "Float32",
        "hour_sin": "Float32",
        "hour_cos": "Float32",
        "day_sin": "Float32",
        "day_cos": "Float32"
    }
}
