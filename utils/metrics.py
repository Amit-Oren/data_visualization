import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score


CATEGORICAL_COLS = [
    "gender", "domain", "current_emotion", "intensity",
    "expressiveness", "self_disclosure_level", "politeness_style", "assertiveness",
]
NUMERICAL_COLS = ["age", "num_turns"]


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Encode and scale features for K-means clustering."""
    encoders = {col: LabelEncoder() for col in CATEGORICAL_COLS}

    encoded = df[CATEGORICAL_COLS].copy()
    for col, enc in encoders.items():
        encoded[col] = enc.fit_transform(df[col])

    numerical = df[NUMERICAL_COLS].copy()

    features = pd.concat([numerical, encoded], axis=1)

    scaler = StandardScaler()
    return scaler.fit_transform(features)


def run_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> dict:
    """
    Run K-means and return cluster labels + evaluation metrics.

    Returns dict with:
        labels       - cluster assignment per row
        inertia      - within-cluster sum of squares
        silhouette   - silhouette score (-1 to 1, higher is better)
        davies_bouldin - Davies-Bouldin index (lower is better)
        pca_coords   - 2D PCA coordinates for plotting
        cluster_sizes - count of rows per cluster
    """
    X = prepare_features(df)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
    tsne_coords = tsne.fit_transform(X)

    return {
        "labels": labels,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "pca_coords": pca_coords,
        "tsne_coords": tsne_coords,
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
    }


def run_tsne(df: pd.DataFrame, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Run t-SNE with a custom perplexity (for interactive use)."""
    X = prepare_features(df)
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=1000)
    return tsne.fit_transform(X)


def elbow_data(df: pd.DataFrame, k_range: range) -> pd.DataFrame:
    """Compute inertia and silhouette for a range of k values (for elbow plot)."""
    X = prepare_features(df)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(rows)


def run_kmeans_on_embeddings(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> dict:
    """
    Run K-means on pre-computed embeddings (pca50_* or emb_* columns).
    Returns labels + quality metrics.
    """
    emb_cols = [c for c in df.columns if c.startswith("pca50_")] or \
               [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=300)
    labels = model.fit_predict(X)

    return {
        "labels": labels,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
    }


def elbow_data_embeddings(df: pd.DataFrame, k_range: range) -> pd.DataFrame:
    """Elbow data using sentence embeddings."""
    emb_cols = [c for c in df.columns if c.startswith("pca50_")] or \
               [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(rows)
