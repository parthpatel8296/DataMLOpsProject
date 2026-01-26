import argparse
import logging
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed/training_output"
# Configure MLflow
mlflow.set_tracking_uri(f"file:///{str(PROJECT_ROOT / 'mlruns')}")
mlflow.set_experiment("RecoMart_Experiments")

# Import MetaStore
from scripts.feature_eng.config import FEATURE_DB_PATH
from scripts.feature_eng.metastore import MetaStore

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class CollaborativeFilteringSVD:
    def __init__(self, n_components=50, n_iter=5):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.user_factors = None
        self.item_factors = None

    def train(self, df):
        logger.info(f"Training SVD (components={self.n_components})...")
        df = df.copy()
        df['user_id'] = df['user_id'].astype("category")
        df['product_id'] = df['product_id'].astype("category")

        self.user_map = dict(enumerate(df['user_id'].cat.categories))
        self.item_map = dict(enumerate(df['product_id'].cat.categories))
        
        user_indices = df['user_id'].cat.codes
        item_indices = df['product_id'].cat.codes
        scores = df['implicit_score'].astype(float)

        self.sparse_matrix = sparse.csr_matrix((scores, (user_indices, item_indices)))
        self.model = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter, random_state=42)
        self.model.fit(self.sparse_matrix)
        
        self.user_factors = self.model.transform(self.sparse_matrix)
        self.item_factors = self.model.components_.T
        
        
        expl_var = self.model.explained_variance_ratio_.sum()
        mlflow.log_metric("svd_explained_variance", expl_var)
        mlflow.log_param("svd_components", self.n_components)
        mlflow.log_param("svd_n_iter", self.n_iter)
        logger.info(f"SVD Training Completed. Var: {expl_var:.4f}")

    def predict(self, user_id, product_id):
        reverse_user_map = {v: k for k, v in self.user_map.items()}
        reverse_item_map = {v: k for k, v in self.item_map.items()}
        u_idx = reverse_user_map.get(user_id)
        i_idx = reverse_item_map.get(product_id)
        if u_idx is None or i_idx is None: return 0.0
        return np.dot(self.user_factors[u_idx], self.item_factors[i_idx])

    def evaluate_rmse(self, test_df):
        logger.info("Evaluating SVD (RMSE)...")
        y_true, y_pred = [], []
        for _, row in test_df.iterrows():
            y_true.append(row['implicit_score'])
            y_pred.append(self.predict(row['user_id'], row['product_id']))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"SVD Test RMSE: {rmse:.4f}")
        mlflow.log_metric("svd_test_rmse", rmse)
        return rmse

    def recommend(self, user_id, n=10):
        reverse_user_map = {v: k for k, v in self.user_map.items()}
        user_idx = reverse_user_map.get(user_id)
        if user_idx is None: return []
        user_vec = self.user_factors[user_idx].reshape(1, -1)
        pred_scores = np.dot(user_vec, self.item_factors.T).flatten()
        top_indices = pred_scores.argsort()[::-1][:n]
        return [(self.item_map[idx], pred_scores[idx]) for idx in top_indices]

class ContentBasedFiltering:
    def __init__(self):
        self.product_vectors = None
        self.product_ids = None
        self.user_profiles = {}
        self.preprocessor = None

    def fit(self, product_df, interaction_df):
        logger.info("Training Content-Based Model...")
        categorical_features = ['category', 'brand', 'subcategory']
        numeric_features = ['price_scaled', 'rating_avg', 'popularity_score', 'diversity_score', 'rating_count_log', 'stock_quantity_scaled']
        
        for c in categorical_features: product_df[c] = product_df[c].astype(str)
        for c in numeric_features: product_df[c] = pd.to_numeric(product_df[c], errors='coerce').fillna(0)

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        self.product_vectors = self.preprocessor.fit_transform(product_df)
        self.product_ids = product_df['product_id'].values
        prod_id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        
        for user_id, group in interaction_df[interaction_df['product_id'].isin(self.product_ids)].groupby('user_id'):
            user_vec = np.zeros(self.product_vectors.shape[1])
            total_weight = 0
            for _, row in group.iterrows():
                pid, weight = row['product_id'], row['implicit_score']
                if pid in prod_id_to_idx:
                    p_vec = self.product_vectors[prod_id_to_idx[pid]]
                    if hasattr(p_vec, "toarray"): p_vec = p_vec.toarray().flatten()
                    user_vec += p_vec * weight
                    total_weight += weight
            if total_weight > 0: self.user_profiles[user_id] = user_vec / total_weight

    def evaluate_metrics(self, test_df, n=10):
        logger.info(f"Evaluating Content-Based (Precision, Recall, HitRate @ {n})...")
        test_user_items = test_df.groupby('user_id')['product_id'].apply(set).to_dict()
        precisions, recalls, hits, evaluated_users = [], [], 0, 0
        
        for user_id, actual_items in test_user_items.items():
            if user_id not in self.user_profiles: continue
            evaluated_users += 1
            rec_items = set([r[0] for r in self.recommend(user_id, n=n)])
            n_hits = len(rec_items.intersection(actual_items))
            precisions.append(n_hits / n)
            recalls.append(n_hits / len(actual_items))
            if n_hits > 0: hits += 1
            
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        hit_rate = hits / evaluated_users if evaluated_users > 0 else 0.0
        
        mlflow.log_metric(f"content_precision_{n}", avg_precision)
        mlflow.log_metric(f"content_recall_{n}", avg_recall)
        mlflow.log_metric(f"content_hit_rate_{n}", hit_rate)
        mlflow.log_param("eval_k", n)
        
        logger.info(f"CB Hit Rate@{n}: {hit_rate:.4f}")
        return avg_precision, avg_recall, hit_rate

    def recommend(self, user_id, n=10):
        if user_id not in self.user_profiles: return []
        user_vec = self.user_profiles[user_id].reshape(1, -1)
        sim_scores = cosine_similarity(user_vec, self.product_vectors).flatten()
        top_indices = sim_scores.argsort()[::-1][:n]
        return [(self.product_ids[idx], sim_scores[idx]) for idx in top_indices]

# --- Workflow Logic ---

def main():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model", type=str, default="all", choices=["svd", "content", "all"])
    args = parser.parse_args()

    logger.info("Initializing Custom MetaStore...")
    ms = MetaStore(PROJECT_ROOT / FEATURE_DB_PATH)

    try:
        conn = ms._get_connection()
        dim_prod = pd.read_sql("SELECT * FROM dim_product_features", conn)
        entity_df = pd.read_sql("SELECT user_id, product_id, implicit_score, created_at as event_timestamp FROM fact_user_item_features", conn)
        conn.close()

        logger.info(f"Retrieving historical features for {len(entity_df)} interactions...")
        training_df = ms.get_historical_features(entity_df, ["user_features", "product_features", "interaction_features"])
        training_df['implicit_score'] = pd.to_numeric(training_df['implicit_score'], errors='coerce').fillna(0)

        training_df = training_df.sort_values(['user_id', 'event_timestamp']).reset_index(drop=True)
        last_indices = training_df.groupby('user_id').tail(1).index
        test_df = training_df.loc[last_indices].copy()
        train_df = training_df.drop(last_indices).copy()
        
        logger.info(f"Data Split: Train={len(train_df)}, Test={len(test_df)}")

    except Exception as e:
        logger.error(f"Failed to load data from MetaStore: {e}")
        sys.exit(1)

    # Set dedicated MLflow experiment
    mlflow.set_experiment("RecoMart_Experiments")

    with mlflow.start_run(run_name=f"Standalone_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_param("engine", "standalone_metastore")
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("split_strategy", "leave-one-out_time_based")

        if args.model in ["svd", "all"]:
            svd = CollaborativeFilteringSVD()
            svd.train(train_df)
            svd.evaluate_rmse(test_df)
            
            # Sample from TRAIN set to ensure we have history to recommend for
            sample_users = train_df['user_id'].unique()[:20]
            with open(OUTPUT_DIR / "svd_top_n_sample.csv", "w") as f:
                f.write("user_id,product_id,score\n")
                for uid in sample_users:
                    recs = svd.recommend(uid, n=50)
                    for pid, score in recs:
                        f.write(f"{uid},{pid},{score}\n")
            
            mlflow.log_artifact(str(OUTPUT_DIR / "svd_top_n_sample.csv"))
            mlflow.sklearn.log_model(svd, "svd_model_standalone")

        if args.model in ["content", "all"]:
            cb_model = ContentBasedFiltering()
            cb_model.fit(dim_prod, train_df)
            cb_model.evaluate_metrics(test_df, n=50)
            
            # Sample from TRAIN set to ensure we have profile built
            sample_users = train_df['user_id'].unique()[:5]
            avg_sim_scores = []
            
            with open(OUTPUT_DIR / "content_recs_sample.csv", "w") as f:
                f.write("user_id,product_id,similarity_score\n")
                for uid in sample_users:
                    recs = cb_model.recommend(uid, n=50)
                    if recs:
                        avg_sim_scores.append(np.mean([r[1] for r in recs]))
                    for pid, score in recs:
                        f.write(f"{uid},{pid},{score}\n")
                        
            avg_conf = np.mean(avg_sim_scores) if avg_sim_scores else 0
            mlflow.log_metric("content_avg_confidence", avg_conf)
            mlflow.log_artifact(str(OUTPUT_DIR / "content_recs_sample.csv"))
            mlflow.sklearn.log_model(cb_model.preprocessor, "cb_model")

    logger.info("Model Training Completed.")

if __name__ == "__main__":
    main()
