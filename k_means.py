"""
Movie clustering with KMeans (IMDb + engagement scores)
------------------------------------------------------
This script:
1. Loads the movie dataset from a local CSV file
2. Performs basic EDA on numeric features
3. Scales selected features
4. Uses the elbow method to inspect a good value for k
5. Trains a final KMeans model (k = 6 by default)
6. Adds cluster labels to the dataframe
7. Visualizes clusters in 3D using Plotly
8. Projects data to 2D with PCA and plots clusters (matplotlib)
"""

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA   # <--- NEW

import plotly.express as px


def load_data():
    """Load the movie CSV using a relative path."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "updated_movie_team.csv"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Data loaded. Shape:", df.shape)
    return df


def basic_eda(df: pd.DataFrame):
    """Print basic info and numeric summary statistics."""
    print("\n=== DataFrame info ===")
    print(df.info())

    print("\n=== Numeric summary (describe) ===")
    print(df.describe().T)

    print("\n=== Missing values per column ===")
    print(df.isna().sum().sort_values(ascending=False))


def scale_features(df: pd.DataFrame):
    """
    Select numeric features for clustering and scale them.
    """
    feature_cols = [
        "movie_engagement_score",
        "actor_actress_engagement_score",
        "writer_engagement_score",
        "director_engagement_score",
    ]

    # Keep only the columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print("\nUsing the following features for clustering:")
    print(feature_cols)

    X = df[feature_cols].copy()

    # simple missing-value handling: fill NaNs with column means
    X = X.fillna(X.mean(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    print("Scaled feature matrix shape:", scaled_df.shape)
    return scaled_df, feature_cols


def elbow_method(X_scaled: np.ndarray, max_k: int = 15):
    """Compute and plot the elbow curve for KMeans, also save as PNG."""
    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    print("\n=== Inertia values for each k ===")
    for k, inertia in zip(k_values, inertias):
        print(f"k = {k}: inertia = {inertia:.2f}")

    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow curve for KMeans on movie data")
    plt.tight_layout()

    # ---- PNG olarak kaydet ----
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = plots_dir / "elbow_curve_movies.png"
    plt.savefig(elbow_path, dpi=300)
    print(f"Elbow grafiği PNG olarak kaydedildi: {elbow_path}")

    plt.show()



def train_final_kmeans(X_scaled: np.ndarray, n_clusters: int = 6):
    """Train the final KMeans model and return labels + fitted model."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    print(f"\nTrained final KMeans model with k = {n_clusters}")
    return labels, kmeans


def inspect_cluster_sizes(labels: np.ndarray):
    """Print how many samples fall into each cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Cluster size distribution ===")
    for lab, cnt in zip(unique, counts):
        print(f"Cluster {lab}: {cnt} movies")

def summarize_clusters(df: pd.DataFrame):
    """
    Her küme için:
      - film sayısı ve yüzdesi
      - ortalama rating
      - ortalama oy sayısı
      - ortalama süre (dakika)
      - en yaygın tür (genres string modu)
      - 5 örnek film
    bilgilerini yazdırır.
    """
    if "cluster_kmeans" not in df.columns:
        raise ValueError("Önce df['cluster_kmeans'] kolonunun oluşmuş olması gerekiyor.")

    n_total = len(df)
    clusters = sorted(df["cluster_kmeans"].unique())

    print("\n=== Küme özetleri ===")

    for c in clusters:
        c_df = df[df["cluster_kmeans"] == c]

        n_cluster = len(c_df)
        perc = 100.0 * n_cluster / n_total if n_total > 0 else 0.0

        # güvenli kolon erişimi

        avg_engagement_score = c_df["movie_engagement_score"].mean() if "movie_engagement_score" in c_df.columns else np.nan
        avg_runtime = c_df["runtimeMinutes"].mean() if "runtimeMinutes" in c_df.columns else np.nan

        # en yaygın tür (genres string modu)
        if "genres" in c_df.columns and not c_df["genres"].dropna().empty:
            top_genre = c_df["genres"].mode().iloc[0]
        else:
            top_genre = "bilgi yok"

        # örnek filmler
        if "primaryTitle" in c_df.columns:
            sample_titles = c_df["primaryTitle"].dropna().head(5).tolist()
        else:
            sample_titles = []

        print(f"\nKüme {c}:")
        print(f"\tFilm sayısı: {n_cluster:,} ({perc:.1f}%)")

        if not np.isnan(avg_engagement_score):
            print(f"\tOrtalama engagement score: {avg_engagement_score:.2f}")
        else:
            print("\tOrtalama rating: bilgi yok")



        if not np.isnan(avg_runtime):
            print(f"\tOrtalama süre: {avg_runtime:.0f} dakika")
        else:
            print("\tOrtalama süre: bilgi yok")

        print(f"\tEn yaygın tür: {top_genre}")

        if sample_titles:
            print(f"\tÖrnek filmler: {', '.join(sample_titles)}")
        else:
            print("\tÖrnek filmler: bilgi yok")


def plot_3d_clusters(scaled_df: pd.DataFrame, labels: np.ndarray):
    """
    Plot a 3D scatter plot using three selected features.
    Make sure these columns exist in `scaled_df`.
    """
    features_3d = [
        "movie_engagement_score",
        "actor_actress_engagement_score",
        "director_engagement_score",
    ]
    features_3d = [f for f in features_3d if f in scaled_df.columns]

    if len(features_3d) < 3:
        print("\nNot enough features available for 3D plot. Skipping 3D.")
        return

    fig = px.scatter_3d(
        scaled_df,
        x=features_3d[0],
        y=features_3d[1],
        z=features_3d[2],
        color=labels.astype(str),
        title="KMeans clusters in movie engagement feature space (3D)",
        width=700,
        height=700
    )
    fig.show()

def plot_pca_2d(scaled_df: pd.DataFrame, labels: np.ndarray):
    """
    Reduce scaled features to 2D with PCA and plot clusters.
    Uses a custom bright color palette for up to 6 clusters.
    Saves the figure as PNG.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_df.values)

    pc_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pc_df["cluster"] = labels

    explained_var = pca.explained_variance_ratio_.sum()

    # canlı renkler
    base_palette = [
        "red",         # kırmızı
        "navy",        # sarı
        "limegreen",   # yeşil
        "dodgerblue",  # mavi
        "magenta",     # mor/pembe
        "orange"       # turuncu
    ]

    n_clusters = len(np.unique(labels))
    palette = (base_palette * ((n_clusters // len(base_palette)) + 1))[:n_clusters]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pc_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=palette,
        s=20,
        alpha=0.8
    )
    plt.title(f"PCA (2D) of movie features – total explained variance: {explained_var:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # ---- PNG olarak kaydet ----
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pca_path = plots_dir / "pca_clusters_movies.png"
    plt.savefig(pca_path, dpi=300)
    print(f"PCA 2D grafiği PNG olarak kaydedildi: {pca_path}")

    plt.show()




def main():
    # 1. Load data
    df = load_data()

    # 2. Basic EDA
    basic_eda(df)

    # 3. Scale features for clustering
    scaled_df, feature_cols = scale_features(df)

    # 4. Elbow method (optional – visual inspection)
    elbow_method(scaled_df.values, max_k=15)

    # 5. Train final KMeans model (set k based on elbow, here k=6)
    labels, kmeans_model = train_final_kmeans(scaled_df.values, n_clusters=6)

    # 6. Attach labels back to original dataframe
    df["cluster_kmeans"] = labels

    # 7. Inspect cluster sizes
    inspect_cluster_sizes(labels)

    # 🔥 7.5. Her küme için detaylı özet
    summarize_clusters(df)

    # 8. 3D visualization (using scaled features)
    plot_3d_clusters(scaled_df, labels)

    # 9. 2D PCA visualization
    plot_pca_2d(scaled_df, labels)

    # 10. Save the clustered data to a new CSV (film + küme bilgisi birlikte)
    output_path = Path(__file__).resolve().parent / "data" / "updated_movie_team_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nClustered data saved to: {output_path}")



if __name__ == "__main__":
    main()
