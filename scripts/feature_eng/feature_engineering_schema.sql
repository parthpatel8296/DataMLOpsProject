-- RecoMart Feature Engineering Schema (DDL)
-- This file defines the tables for storing model features.

-- 1. Product Dimensions Table
CREATE TABLE IF NOT EXISTS dim_product_features (
    product_id BIGINT PRIMARY KEY,
    category_id BIGINT,
    category VARCHAR(255),
    subcategory_id BIGINT,
    subcategory VARCHAR(255),
    brand VARCHAR(255),
    price DOUBLE PRECISION,
    price_scaled DOUBLE PRECISION,
    rating_avg DOUBLE PRECISION,
    rating_count BIGINT,
    rating_count_log DOUBLE PRECISION,
    stock_quantity_scaled DOUBLE PRECISION,
    interaction_count BIGINT,
    unique_users BIGINT,
    popularity_score DOUBLE PRECISION,
    conversion_rate DOUBLE PRECISION,
    diversity_score DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. User Dimensions Table
CREATE TABLE IF NOT EXISTS dim_user_features (
    user_id BIGINT PRIMARY KEY,
    age INTEGER,
    age_scaled DOUBLE PRECISION,
    location_city VARCHAR(255),
    location_country VARCHAR(255),
    tier VARCHAR(50),
    interaction_count BIGINT,
    activity_frequency DOUBLE PRECISION,
    purchase_ratio DOUBLE PRECISION,
    first_interaction TIMESTAMP,
    last_interaction TIMESTAMP,
    preferred_category VARCHAR(255),
    category_confidence DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. User-Item Interaction (Fact) Table
-- Note: composite primary key includes created_at to support historical tracking/lineage
CREATE TABLE IF NOT EXISTS fact_user_item_features (
    user_id BIGINT,
    product_id BIGINT,
    interaction_count BIGINT,
    implicit_score BIGINT,
    hour_sin DOUBLE PRECISION,
    hour_cos DOUBLE PRECISION,
    day_sin DOUBLE PRECISION,
    day_cos DOUBLE PRECISION,
    duration_scaled DOUBLE PRECISION,
    time_since_last_seconds BIGINT,
    interaction_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, product_id, created_at),
    FOREIGN KEY (user_id) REFERENCES dim_user_features(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_product_features(product_id)
);

-- 4. Feature Version Metadata Table
-- Tracks metadata and versioning for generated features.
CREATE TABLE IF NOT EXISTS feature_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_entity VARCHAR(50) NOT NULL, -- 'user', 'product', or 'interaction'
    feature_name VARCHAR(100) NOT NULL,
    version_number INTEGER NOT NULL,
    data_source VARCHAR(255),
    transformation_logic VARCHAR(255),
    quality_score DOUBLE PRECISION DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Feature Registry (MetaStore)
-- Stores high-level definitions of feature sets (equivalent to Feast FeatureViews)
CREATE TABLE IF NOT EXISTS feature_registry (
    registry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_view_name VARCHAR(100) UNIQUE NOT NULL,
    entity_name VARCHAR(50) NOT NULL,
    description TEXT,
    ttl_days INTEGER DEFAULT 365,
    owner VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Feature Metadata
-- Stores individual column definitions within a feature set (equivalent to Feast Fields)
CREATE TABLE IF NOT EXISTS feature_metadata (
    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_view_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    description TEXT,
    FOREIGN KEY (feature_view_name) REFERENCES feature_registry(feature_view_name),
    UNIQUE(feature_view_name, column_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_product_category ON dim_product_features(category);
CREATE INDEX IF NOT EXISTS idx_user_tier ON dim_user_features(tier);
CREATE INDEX IF NOT EXISTS idx_interaction_lookup ON fact_user_item_features(user_id, product_id);
CREATE INDEX IF NOT EXISTS idx_version_entity ON feature_versions(feature_entity, version_number);
