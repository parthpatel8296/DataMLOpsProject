from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define data sources
# Note: Feast requires an `event_timestamp` column in the data source.
# Our data engineering pipeline will output parquet files with this column.
user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at"
)

product_source = FileSource(
    path="data/product_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at"
)

interaction_source = FileSource(
    path="data/interaction_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at"
)

# Define entities
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="Customer entity",
)

product = Entity(
    name="product",
    join_keys=["product_id"],
    description="Product entity",
)

# Define Feature Views
user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=36500), # 100 years to prevent expiration
    source=user_source,
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="age_scaled", dtype=Float32),
        Field(name="location_city", dtype=String),
        Field(name="location_country", dtype=String),
        Field(name="tier", dtype=String),
        Field(name="interaction_count", dtype=Int64),
        Field(name="activity_frequency", dtype=Float32),
        Field(name="purchase_ratio", dtype=Float32),
        Field(name="preferred_category", dtype=String),
        Field(name="category_confidence", dtype=Float32)
    ]
)

product_features_view = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=36500),
    source=product_source,
    schema=[
        Field(name="category_id", dtype=Int64),
        Field(name="category", dtype=String),
        Field(name="subcategory_id", dtype=Int64),
        Field(name="subcategory", dtype=String),
        Field(name="brand", dtype=String),
        Field(name="price", dtype=Float32),
        Field(name="price_scaled", dtype=Float32),
        Field(name="rating_avg", dtype=Float32),
        Field(name="rating_count", dtype=Int64),
        Field(name="rating_count_log", dtype=Float32),
        Field(name="stock_quantity_scaled", dtype=Float32),
        Field(name="interaction_count", dtype=Int64),
        Field(name="unique_users", dtype=Int64),
        Field(name="popularity_score", dtype=Float32),
        Field(name="conversion_rate", dtype=Float32),
        Field(name="diversity_score", dtype=Float32)
    ]
)

interaction_features_view = FeatureView(
    name="interaction_features",
    entities=[user, product],
    ttl=timedelta(days=36500),
    source=interaction_source,
    schema=[
        Field(name="interaction_count", dtype=Int64),
        Field(name="implicit_score", dtype=Float32),
        Field(name="time_since_last_seconds", dtype=Int64),
        Field(name="duration_scaled", dtype=Float32),
        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),
        Field(name="day_sin", dtype=Float32),
        Field(name="day_cos", dtype=Float32)
    ]
)
