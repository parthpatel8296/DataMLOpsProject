import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from datetime import datetime

class TransformerMixin:
    """
    Base mixin to provide a standard fit/transform interface.
    """
    def fit(self, X, y=None):
        """Mock fit method (stateless)."""
        return self

    def transform(self, X):
        """Base transform method."""
        return X

class UserFeatureGenerator(TransformerMixin):
    """
    Generates user-level features.
    
    Features created:
    - interaction_count: Total actions by user.
    - activity_frequency: Actions per day.
    - purchase_ratio: Proportion of actions that are purchases.
    - preferred_category: Category with most interactions.
    - category_confidence: Ratio of interactions in top category.
    - tier: User loyalty tier (reconstructed or passed through).
    - location/age: Profile attributes.
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        
        # Mandatory Column Check
        mandatory = ["user_id", "timestamp"]
        missing = [c for c in mandatory if c not in df.columns]
        if missing:
            raise ValueError(f"UserFeatureGenerator missing mandatory columns: {missing}")

        print("   [Transformer] Generating User Features...")
        
        # 0. Preprocessing: Ensure timestamps
        if 'dt' not in df.columns:
             df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Group by user to calculate aggregations
        user_grp = df.groupby('user_id')
        
        # --- 1. Interaction Metrics ---
        # Total interactions count
        interaction_count = user_grp.size().rename('interaction_count')
        
        # Activity Frequency (Interactions / Lifespan in days)
        # This tells us if they are a 'daily power user' or just a 'one-hit wonder'.
        date_min = user_grp['dt'].min()
        date_max = user_grp['dt'].max()
        # Add epsilon/replace 0 to handle users who only visit once (avoid division by zero)
        date_range_days = (date_max - date_min).dt.total_seconds() / (24 * 3600)
        date_range_days = date_range_days.replace(0, 1) 
        activity_frequency = interaction_count / date_range_days
        activity_frequency.name = 'activity_frequency'
        
        # --- 2. Purchase Analytics ---
        # Identify purchase events based on available columns
        if 'event_type' in df.columns:
            purchase_mask = df['event_type'] == 'purchase'
        elif 'event_strength' in df.columns:
            # Fallback: Task 5 defined event_strength 4 as purchase
            purchase_mask = df['event_strength'] == 4
        else:
            purchase_mask = pd.Series(False, index=df.index)
            
        purchase_count = df[purchase_mask].groupby('user_id').size()
        # Realign to ensure all users are present (fill non-purchasers with 0)
        purchase_count = purchase_count.reindex(user_grp.groups.keys(), fill_value=0)
        purchase_ratio = purchase_count / interaction_count
        purchase_ratio.name = 'purchase_ratio'
        
        # --- 3. Category Preferences ---
        # Identify columns starting with 'cat_' (one-hot encoded)
        cat_cols = [c for c in df.columns if c.startswith('cat_')]
        
        if 'category' in df.columns:
            # If category is raw string, aggregation is straightforward
            preferred_cat = df.groupby('user_id')['category'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            )
            cat_confidence = df.groupby('user_id')['category'].apply(
                lambda x: x.value_counts(normalize=True).iloc[0] if not x.empty else 0.0
            )
        elif cat_cols:
            # Reconstruction from One-Hot Encoding
            # Sum interactions per category per user
            cat_sums = df.groupby('user_id')[cat_cols].sum()
            
            # Find column with max value for each user
            preferred_cat_col = cat_sums.idxmax(axis=1)
            preferred_cat = preferred_cat_col.str.replace('cat_', '')
            
            # Confidence = (Interactions in Top Category) / (Total Interactions)
            max_val = cat_sums.max(axis=1)
            cat_confidence = max_val / interaction_count
        else:
            # Fallback if no category info exists
            preferred_cat = pd.Series('Unknown', index=user_grp.groups.keys())
            cat_confidence = pd.Series(0.0, index=user_grp.groups.keys())
            
        preferred_cat.name = 'preferred_category'
        cat_confidence.name = 'category_confidence'

        # --- 4. Profile Attributes ---
        # Timestamps
        first_interaction = date_min.rename('first_interaction')
        last_interaction = date_max.rename('last_interaction')
        
        # Reconstruct or Passthrough Static Attributes
        tier_cols = [c for c in df.columns if c.startswith('tier_')]
        
        # Compile features into DataFrame
        dim_user = pd.DataFrame({
            'interaction_count': interaction_count,
            'activity_frequency': activity_frequency,
            'purchase_ratio': purchase_ratio,
            'preferred_category': preferred_cat,
            'category_confidence': cat_confidence,
            'first_interaction': first_interaction,
            'last_interaction': last_interaction
        })
        
        # Add profile columns (age, location) directly
        for c in ['age', 'age_scaled', 'location_city', 'location_country']:
            if c in df.columns:
                dim_user[c] = user_grp[c].first()
                
        # Handle 'Tier' (reconstruction or direct)
        if 'tier' in df.columns:
            dim_user['tier'] = user_grp['tier'].first()
        elif tier_cols:
            # Determine active tier bit
            tier_agg = df.groupby('user_id')[tier_cols].max() 
            dim_user['tier'] = tier_agg.idxmax(axis=1).str.replace('tier_', '')
        
        # Metadata
        dim_user['created_at'] = pd.Timestamp.now(tz='UTC')
        dim_user['updated_at'] = pd.Timestamp.now(tz='UTC')
        
        # --- Final NaN Cleanup ---
        # Fill defaults for user features
        fill_values = {
            'interaction_count': 0,
            'activity_frequency': 0.0,
            'purchase_ratio': 0.0,
            'preferred_category': 'Unknown',
            'category_confidence': 0.0,
            'age': -1, # -1 indicates unknown age
            'location_city': 'Unknown', 
            'location_country': 'Unknown',
            'tier': 'Unknown'
        }
        dim_user = dim_user.fillna(value=fill_values)
        
        return dim_user.reset_index()

    def update(self, prev_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Incrementally updates user features by merging previous state with daily data.
        """
        print("   [Transformer] Updating User Features...")
        new_df = self.transform(daily_df)
        
        if prev_df.empty:
            return new_df

        # 1. Prepare for weighted averaging
        # Back-calculate raw purchase counts (as ratio * total)
        prev_df = prev_df.copy()
        new_df = new_df.copy()
        
        # Ensure we don't have NaNs in key metrics
        for df in [prev_df, new_df]:
            df['purchase_ratio'] = df['purchase_ratio'].fillna(0)
            df['interaction_count'] = df['interaction_count'].fillna(0)
            
        prev_df['_purchases'] = prev_df['purchase_ratio'] * prev_df['interaction_count']
        new_df['_purchases'] = new_df['purchase_ratio'] * new_df['interaction_count']
        
        # 2. Concatenate old and new
        combined = pd.concat([prev_df, new_df], ignore_index=True)
        
        # 2a. Ensure consistent datetime types (SQLite loads strings, transform produces Timestamps)
        for c in ['first_interaction', 'last_interaction', 'created_at']:
            if c in combined.columns:
                combined[c] = pd.to_datetime(combined[c], format='ISO8601', utc=True)
        
        # 3. Aggregation Rules
        # - Sum counts
        # - Min/Max timestamps
        # - Take latest profile info (assume daily is newer)
        agg_rules = {
            'interaction_count': 'sum',
            '_purchases': 'sum',
            'first_interaction': 'min',
            'last_interaction': 'max',
            # Profile attributes - take last (most recent)
            'age': 'last',
            'location_city': 'last', 
            'location_country': 'last',
            'tier': 'last',
            # Categorical heuristics (simplification for incremental)
            'preferred_category': 'last', 
            'category_confidence': 'last',
            'created_at': 'first' # Keep original creation date
        }
        
        # Only aggregate columns that exist
        valid_agg_rules = {k: v for k, v in agg_rules.items() if k in combined.columns}
        
        # GroupBy User
        grouped = combined.groupby('user_id', as_index=False).agg(valid_agg_rules)
        
        # 4. Recompute Derived Metrics
        # Purchase Ratio
        grouped['purchase_ratio'] = grouped['_purchases'] / grouped['interaction_count']
        grouped['purchase_ratio'] = grouped['purchase_ratio'].fillna(0)
        
        # Activity Frequency
        if 'first_interaction' in grouped.columns and 'last_interaction' in grouped.columns:
            date_range = (grouped['last_interaction'] - grouped['first_interaction']).dt.total_seconds() / (24 * 3600)
            date_range = date_range.replace(0, 1)
            grouped['activity_frequency'] = grouped['interaction_count'] / date_range
        
        # Cleanup
        grouped['updated_at'] = pd.Timestamp.now(tz='UTC')
        
        # --- Final NaN Cleanup ---
        fill_values = {
            'interaction_count': 0,
            'activity_frequency': 0.0,
            'purchase_ratio': 0.0,
            'preferred_category': 'Unknown',
            'category_confidence': 0.0,
            'age': -1,
            'location_city': 'Unknown', 
            'location_country': 'Unknown',
            'tier': 'Unknown'
        }
        grouped = grouped.fillna(value=fill_values)
        
        return grouped

class ProductFeatureGenerator(TransformerMixin):
    """
    Generates product-level features.
    
    Features created:
    - popularity_score: Normalized interaction count.
    - conversion_rate: Purchases / Views.
    - diversity_score: Unique users / Total interactions.
    - price, rating_avg: Catalog attributes.
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()

        # Mandatory Column Check
        mandatory = ["product_id"]
        missing = [c for c in mandatory if c not in df.columns]
        if missing:
            raise ValueError(f"ProductFeatureGenerator missing mandatory columns: {missing}")

        print("   [Transformer] Generating Product Features...")
        prod_grp = df.groupby('product_id')
        
        # 1. Basic Metrics
        interaction_count = prod_grp.size().rename('interaction_count')
        unique_users = prod_grp['user_id'].nunique().rename('unique_users')
        
        # 2. Popularity Score (MinMax Scaled Count)
        mx = interaction_count.max()
        popularity_score = interaction_count / mx if mx > 0 else 0
        popularity_score.name = 'popularity_score'
        
        # 3. Conversion Rate
        # Define purchase logic
        if 'event_type' in df.columns:
            purchase_mask = df['event_type'] == 'purchase'
        elif 'event_strength' in df.columns:
            purchase_mask = df['event_strength'] == 4
        else:
            purchase_mask = pd.Series(False, index=df.index)
            
        purchases = df[purchase_mask].groupby('product_id').size()
        purchases = purchases.reindex(prod_grp.groups.keys(), fill_value=0)
        conversion_rate = purchases / interaction_count
        conversion_rate.name = 'conversion_rate'
        
        # 4. Diversity Score (Audience Breadth)
        diversity_score = unique_users / interaction_count
        diversity_score.name = 'diversity_score'
        
        # 5. Compile DataFrame
        dim_prod = pd.DataFrame({
            'interaction_count': interaction_count,
            'unique_users': unique_users,
            'popularity_score': popularity_score,
            'conversion_rate': conversion_rate,
            'diversity_score': diversity_score
        })
        
        # 6. Catalog Attributes Passthrough
        # We try to grab these columns if they exist
        cols_to_grab = [
            'category_id', 'category', 'subcategory_id', 'subcategory', 
            'brand', 'price', 'price_scaled', 'rating_avg', 'rating_count',
            'rating_count_log', 'stock_quantity_scaled'
        ]
        
        for c in cols_to_grab:
            if c in df.columns:
                dim_prod[c] = prod_grp[c].first()
        
        # Reconstruct category from one-hot if 'category' column is missing
        cat_cols = [c for c in df.columns if c.startswith('cat_')]
        if 'category' not in dim_prod.columns and cat_cols:
             # Take max across cat columns (assuming 1 per product)
             cat_agg = df.groupby('product_id')[cat_cols].max()
             dim_prod['category'] = cat_agg.idxmax(axis=1).str.replace('cat_', '')
             
        # Reconstruct subcategory from one-hot if 'subcategory' column is missing
        subcat_cols = [c for c in df.columns if c.startswith('subcat_')]
        if 'subcategory' not in dim_prod.columns and subcat_cols:
             subcat_agg = df.groupby('product_id')[subcat_cols].max()
             dim_prod['subcategory'] = subcat_agg.idxmax(axis=1).str.replace('subcat_', '')
             
        dim_prod['created_at'] = pd.Timestamp.now(tz='UTC')
        dim_prod['updated_at'] = pd.Timestamp.now(tz='UTC')
        
        # --- Final NaN Cleanup ---
        fill_values = {
            'interaction_count': 0,
            'unique_users': 0,
            'popularity_score': 0.0,
            'conversion_rate': 0.0,
            'diversity_score': 0.0,
            'category': 'Unknown',
            'subcategory': 'Unknown',
            'brand': 'Unknown',
            'price': 0.0,
            'rating_avg': 0.0,
            'rating_count': 0
        }
        dim_prod = dim_prod.fillna(value=fill_values)
        
        return dim_prod.reset_index()

    def update(self, prev_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Incrementally updates product features.
        """
        print("   [Transformer] Updating Product Features...")
        new_df = self.transform(daily_df)
        
        if prev_df.empty:
            return new_df
            
        prev_df = prev_df.copy()
        new_df = new_df.copy()
        
        # Back-calculate raw counts for weighted averages
        # conversion_rate = purchases / interactions
        prev_df['_purchases'] = prev_df['conversion_rate'] * prev_df['interaction_count']
        new_df['_purchases'] = new_df['conversion_rate'] * new_df['interaction_count']
        
        combined = pd.concat([prev_df, new_df], ignore_index=True)
        
        # Ensure consistent datetime types
        if 'created_at' in combined.columns:
            combined['created_at'] = pd.to_datetime(combined['created_at'], format='ISO8601', utc=True)
        
        agg_rules = {
            'interaction_count': 'sum',
            'unique_users': 'sum', # Approximation: Upper bound sum (users distinct across days)
            '_purchases': 'sum',
            # Metadata
            'category_id': 'last',
            'category': 'last',
            'subcategory_id': 'last',
            'subcategory': 'last',
            'brand': 'last',
            'price': 'last',
            'rating_avg': 'last',
            'rating_count': 'last',
            'created_at': 'first'
        }
        
        valid_agg_rules = {k: v for k, v in agg_rules.items() if k in combined.columns}
        
        grouped = combined.groupby('product_id', as_index=False).agg(valid_agg_rules)
        
        # Recompute Derived
        # Conversion Rate
        if 'interaction_count' in grouped.columns and '_purchases' in grouped.columns:
             grouped['conversion_rate'] = grouped['_purchases'] / grouped['interaction_count']
             
        # Popularity Score (Re-normalize globally)
        if 'interaction_count' in grouped.columns:
            mx = grouped['interaction_count'].max()
            grouped['popularity_score'] = grouped['interaction_count'] / mx if mx > 0 else 0
            
        # Diversity Score
        if 'unique_users' in grouped.columns:
            grouped['diversity_score'] = grouped['unique_users'] / grouped['interaction_count']
            
        grouped['updated_at'] = pd.Timestamp.now(tz='UTC')
        
        # --- Final NaN Cleanup ---
        fill_values = {
            'interaction_count': 0,
            'unique_users': 0,
            'popularity_score': 0.0,
            'conversion_rate': 0.0,
            'diversity_score': 0.0,
            'category': 'Unknown',
            'subcategory': 'Unknown',
            'brand': 'Unknown',
            'price': 0.0,
            'rating_avg': 0.0,
            'rating_count': 0
        }
        grouped = grouped.fillna(value=fill_values)
        
        return grouped

class InteractionFeatureGenerator(TransformerMixin):
    """
    Generates interaction-level features (User-Item context).
    
    Features created:
    - implicit_score: Aggregated event strength.
    - Temporal Cyclical Features: hour_sin, hour_cos, day_sin, day_cos.
    - Lag Features: time_since_last_seconds (per user-item pair).
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()

        # Mandatory Column Check
        mandatory = ["user_id", "product_id", "timestamp"]
        missing = [c for c in mandatory if c not in df.columns]
        if missing:
            raise ValueError(f"InteractionFeatureGenerator missing mandatory columns: {missing}")

        print("   [Transformer] Generating Interaction Features...")
        
        if 'dt' not in df.columns:
             df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
             
        # Extract date for aggregation grain
        df['interaction_date'] = df['dt'].dt.date
        
        # Group by composite key (User + Product + Date)
        grp = df.groupby(['user_id', 'product_id', 'interaction_date'])
        
        # 1. Base Aggregations
        interaction_count = grp.size().rename('interaction_count')
        
        # Implicit Score: Sum of event strengths (e.g., view(1) + click(2) = 3)
        implicit_score = grp['event_strength'].sum().rename('implicit_score') \
            if 'event_strength' in df.columns \
            else interaction_count
        
        # Average timestamp for temporal feature calculation
        avg_ts = grp['dt'].mean()
        
        # Pull duration_scaled if exists (usually first since it's interaction level)
        duration_scaled = grp['duration_scaled'].mean() if 'duration_scaled' in df.columns else 0
        
        fact_df = pd.DataFrame({
            'interaction_count': interaction_count,
            'implicit_score': implicit_score,
            'avg_ts': avg_ts,
            'duration_scaled': duration_scaled
        }).reset_index()
        
        # 2. Temporal Features (Cyclical Encoding)
        # Models often struggle with raw numbers like '23' vs '0'.
        # By converting to Sin/Cos, we show the model that 11 PM and 1 AM are actually 'close' in distance.
        hour = fact_df['avg_ts'].dt.hour
        day = fact_df['avg_ts'].dt.dayofweek
        
        fact_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        fact_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        fact_df['day_sin'] = np.sin(2 * np.pi * day / 7)
        fact_df['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        # 3. Lag Features
        # Calculate time difference between consecutive interactions for a user-item pair
        fact_df.sort_values(['user_id', 'product_id', 'avg_ts'], inplace=True)
        fact_df['prev_ts'] = fact_df.groupby(['user_id', 'product_id'])['avg_ts'].shift(1)
        
        fact_df['time_since_last_seconds'] = (
            (fact_df['avg_ts'] - fact_df['prev_ts'])
            .dt.total_seconds()
            .fillna(0)
            .astype(int)
        )
        
        # Cleanup
        fact_df['created_at'] = pd.Timestamp.now(tz='UTC')
        fact_df.drop(columns=['avg_ts', 'prev_ts'], inplace=True)
        
        return fact_df.fillna(0) # Interaction features are all numeric (counts/scores), 0 is safe default

    def update(self, prev_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends new interactions to the existing history.
        """
        print("   [Transformer] Updating Interaction Features (Append)...")
        new_df = self.transform(daily_df)
        
        if prev_df.empty:
            return new_df
            
        # Fact table update is just an append (log-based)
        combined = pd.concat([prev_df, new_df], ignore_index=True)
        
        # Ensure consistent types for dedup
        if 'interaction_date' in combined.columns:
             combined['interaction_date'] = pd.to_datetime(combined['interaction_date']).dt.date
        
        # Dedup based on primary key (User, Product, Date)
        if 'interaction_date' in combined.columns:
             combined.drop_duplicates(subset=['user_id', 'product_id', 'interaction_date'], keep='last', inplace=True)
             
        return combined.fillna(0)
