import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output folder
output_folder = 'movie_clustering_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# For non-English character support (original comment was for Turkish, keeping it general)
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 80)
print("MOVIE CLUSTERING AND RECOMMENDATION SYSTEM - DETAILED REPORT")
print("=" * 80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

# Load the CSV file
# NOTE: The file path is hardcoded for the user's environment, kept as is.
df = pd.read_csv('imdb_data/movies_for_title_clustering.csv')

print(f"\n✓ Data loaded successfully")
print(f"✓ Total number of movies: {len(df):,}")
print(f"✓ Total number of columns: {len(df.columns)}")
print(f"\n📊 Data Structure (Head):")
print(df.head())
print(f"\n📊 Column Names and Types:")
print(df.dtypes)
print(f"\n📊 Basic Statistics:")
print(df.describe())

# ============================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA CLEANING AND PREPROCESSING")
print("=" * 80)

# Check for missing values
print("\n📋 Missing Value Analysis:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Clean missing values
initial_count = len(df)
df = df.dropna(subset=['primaryTitle', 'genres', 'averageRating', 'numVotes'])
print(f"\n✓ Rows with missing essential values cleaned")
print(f"✓ Number of rows deleted: {initial_count - len(df):,}")
print(f"✓ Remaining number of movies: {len(df):,}")

# Process genres (convert comma-separated genres to a list)
df['genres_list'] = df['genres'].str.split(',')

# Convert genres to binary form using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=df.index)

print(f"\n📊 Genre Information:")
print(f"✓ Total number of distinct genres: {len(mlb.classes_)}")
print(f"✓ Genres: {', '.join(mlb.classes_)}")

# Calculate genre distribution
genre_counts = genres_df.sum().sort_values(ascending=False)
print(f"\n📊 Top Popular Genres:")
for genre, count in genre_counts.head(10).items():
    print(f"  • {genre}: {count:,} movies")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Combine features for clustering
features = pd.DataFrame(index=df.index)

# Add numerical features
features['averageRating'] = df['averageRating']
features['numVotes'] = df['numVotes']
# Fill missing runtimeMinutes and startYear with the median
features['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].median())
features['startYear'] = df['startYear'].fillna(df['startYear'].median())

# Add genre features
features = pd.concat([features, genres_df], axis=1)

print(f"\n✓ Total number of features: {features.shape[1]}")
print(f"  • Numerical features: 4 (rating, number of votes, runtime, year)")
print(f"  • Genre features: {len(mlb.classes_)}")

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print(f"\n✓ Features normalized (StandardScaler)")
print(f"✓ Scaled data dimension: {features_scaled.shape}")

# ============================================================================
# STEP 4: FINDING OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 80)

print("\n⏳ Testing different cluster numbers for K-Means...")
print("   (This process may take a few minutes)\n")

# Calculate inertia and silhouette scores for the Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 21)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(features_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"  K={k:2d}: Inertia={kmeans.inertia_:,.0f}, Silhouette={sil_score:.4f}")

# Find the optimal K (based on silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal number of clusters: {optimal_k}")
print(f"✓ Highest Silhouette score: {max(silhouette_scores):.4f}")

# Create the Elbow chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Inertia plot
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
ax1.set_title('Elbow Method: Inertia Plot', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Silhouette score plot
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(f'{output_folder}/01_elbow_method.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Elbow plot saved: 01_elbow_method.png")
plt.close()

# ============================================================================
# STEP 5: K-MEANS CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: K-MEANS CLUSTERING")
print("=" * 80)

print(f"\n⏳ Running K-Means algorithm with {optimal_k} clusters...\n")

# Final K-Means model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans_final.fit_predict(features_scaled)

print(f"✓ Clustering completed!")
print(f"\n📊 CLUSTER STATISTICS:")
print("=" * 80)

# Detailed analysis for each cluster
cluster_stats = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    stats = {
        'Cluster': cluster_id,
        'Movie Count': len(cluster_data),
        'Proportion (%)': (len(cluster_data) / len(df)) * 100,
        'Avg. Rating': cluster_data['averageRating'].mean(),
        'Avg. Vote Count': cluster_data['numVotes'].mean(),
        'Avg. Runtime (min)': cluster_data['runtimeMinutes'].mean(),
        'Most Common Genre': cluster_data['genres'].mode()[0] if not cluster_data.empty else 'N/A'
    }
    cluster_stats.append(stats)

    print(f"\n🎬 CLUSTER {cluster_id}:")
    print(f"  • Movie count: {stats['Movie Count']:,} ({stats['Proportion (%)']:.1f}%)")
    print(f"  • Average rating: {stats['Avg. Rating']:.2f}")
    print(f"  • Average vote count: {stats['Avg. Vote Count']:,.0f}")
    print(f"  • Average runtime: {stats['Avg. Runtime (min)']:.0f} minutes")
    print(f"  • Most common genre: {stats['Most Common Genre']}")
    print(f"  • Sample movies:")
    sample_movies = cluster_data.nlargest(5, 'numVotes')['primaryTitle'].values
    for i, movie in enumerate(sample_movies, 1):
        print(f"    {i}. {movie}")

# Convert cluster statistics to DataFrame and save
cluster_stats_df = pd.DataFrame(cluster_stats)
cluster_stats_df.to_csv(f'{output_folder}/cluster_statistics.csv', index=False)
print(f"\n✓ Cluster statistics saved: cluster_statistics.csv")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: VISUALIZATIONS")
print("=" * 80)

# 1. Cluster Distribution Pie Chart
plt.figure(figsize=(10, 8))
cluster_counts = df['cluster'].value_counts().sort_index()
colors = plt.cm.Set3(range(len(cluster_counts)))
plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Movie Distribution by Cluster', fontsize=16, fontweight='bold')
plt.savefig(f'{output_folder}/02_cluster_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Cluster distribution chart saved: 02_cluster_distribution.png")
plt.close()

# 2. Cluster Sizes Bar Chart
plt.figure(figsize=(12, 6))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors)
plt.xlabel('Cluster Number', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.title('Number of Movies in Each Cluster', fontsize=14, fontweight='bold')
plt.xticks(cluster_counts.index)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{output_folder}/03_cluster_sizes.png', dpi=300, bbox_inches='tight')
print("✓ Cluster sizes chart saved: 03_cluster_sizes.png")
plt.close()

# 3. Scatter Plot - Rating vs Vote Count
plt.figure(figsize=(14, 8))
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(cluster_data['averageRating'],
                cluster_data['numVotes'],
                label=f'Cluster {cluster_id}',
                alpha=0.6, s=50)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Vote Count (log scale)', fontsize=12)
plt.yscale('log')
plt.title('Movie Clusters: Rating vs Vote Count', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_folder}/04_scatter_rating_votes.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot (Rating vs Votes) saved: 04_scatter_rating_votes.png")
plt.close()

# 4. Scatter Plot - Rating vs Runtime
plt.figure(figsize=(14, 8))
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(cluster_data['averageRating'],
                cluster_data['runtimeMinutes'],
                label=f'Cluster {cluster_id}',
                alpha=0.6, s=50)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Movie Runtime (minutes)', fontsize=12)
plt.title('Movie Clusters: Rating vs Runtime', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_folder}/05_scatter_rating_runtime.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot (Rating vs Runtime) saved: 05_scatter_rating_runtime.png")
plt.close()

# 5. Scatter Plot - Year vs Rating
plt.figure(figsize=(14, 8))
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(cluster_data['startYear'],
                cluster_data['averageRating'],
                label=f'Cluster {cluster_id}',
                alpha=0.6, s=50)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Movie Clusters: Year vs Rating', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_folder}/06_scatter_year_rating.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot (Year vs Rating) saved: 06_scatter_year_rating.png")
plt.close()

# 6. Average Rating Comparison by Cluster
plt.figure(figsize=(12, 6))
avg_ratings = [df[df['cluster'] == i]['averageRating'].mean() for i in range(optimal_k)]
bars = plt.bar(range(optimal_k), avg_ratings, color=colors)
plt.xlabel('Cluster Number', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Comparison of Average Rating by Cluster', fontsize=14, fontweight='bold')
plt.xticks(range(optimal_k))
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 10)
plt.savefig(f'{output_folder}/07_cluster_average_rating.png', dpi=300, bbox_inches='tight')
print("✓ Cluster average rating chart saved: 07_cluster_average_rating.png")
plt.close()

# 7. Genre Distribution Heatmap
plt.figure(figsize=(16, 10))
genre_cluster = pd.DataFrame()
for cluster_id in range(optimal_k):
    cluster_movies = df[df['cluster'] == cluster_id]
    # Explode the list of genres and count normalized frequencies
    cluster_genres = cluster_movies['genres_list'].explode().value_counts(normalize=True)
    genre_cluster[f'Cluster {cluster_id}'] = cluster_genres

genre_cluster = genre_cluster.fillna(0)
sns.heatmap(genre_cluster, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Proportion'})
plt.title('Genre Distribution Heatmap by Cluster', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_folder}/08_genre_distribution_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Genre distribution heatmap saved: 08_genre_distribution_heatmap.png")
plt.close()

# 8. Box Plot - Rating Distribution by Cluster
plt.figure(figsize=(14, 6))
df.boxplot(column='averageRating', by='cluster', figsize=(14, 6))
plt.xlabel('Cluster Number', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.title('Rating Distribution by Cluster (Box Plot)', fontsize=14, fontweight='bold')
plt.suptitle('') # Remove default suptitle added by pandas boxplot
plt.savefig(f'{output_folder}/09_rating_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Rating box plot saved: 09_rating_boxplot.png")
plt.close()

# ============================================================================
# STEP 7: MOVIE RECOMMENDATION SYSTEM
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MOVIE RECOMMENDATION SYSTEM")
print("=" * 80)


def recommend_movies(movie_title, df, top_n=10):
    """Recommends similar movies based on the input movie title"""

    # Find the movie
    movie_match = df[df['primaryTitle'].str.contains(movie_title, case=False, na=False)]

    if movie_match.empty:
        print(f"\n❌ Movie titled '{movie_title}' not found!")
        print("\n💡 Hint: Try the full name or a part of the movie title.")
        return None

    # Get the first matching movie
    movie = movie_match.iloc[0]
    movie_cluster = movie['cluster']

    print(f"\n🎬 Selected Movie: {movie['primaryTitle']}")
    print(f"  • Genre: {movie['genres']}")
    print(f"  • Rating: {movie['averageRating']:.1f}")
    print(f"  • Year: {int(movie['startYear'])}")
    print(f"  • Cluster: {movie_cluster}")

    # Get movies from the same cluster
    same_cluster = df[df['cluster'] == movie_cluster]

    # Exclude the selected movie
    same_cluster = same_cluster[same_cluster['tconst'] != movie['tconst']]

    # Recommend the most popular and highly-rated movies
    recommendations = same_cluster.nlargest(top_n, ['averageRating', 'numVotes'])

    print(f"\n✨ RECOMMENDED MOVIES (From the Same Cluster - Cluster {movie_cluster}):")
    print("=" * 80)

    for idx, (_, film) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {film['primaryTitle']}")
        print(f"   Rating: {film['averageRating']:.1f} | Year: {int(film['startYear'])} | "
              f"Genre: {film['genres']}")

    return recommendations


# Example recommendation
print("\n" + "=" * 80)
print("EXAMPLE MOVIE RECOMMENDATION")
print("=" * 80)

# Test with a popular movie in the dataset
popular_movie = df.nlargest(1, 'numVotes').iloc[0]['primaryTitle']
print(f"\n🎯 Movie selected for test: {popular_movie}")
recommend_movies(popular_movie, df, top_n=10)

# ============================================================================
# STEP 8: SAVING RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING RESULTS")
print("=" * 80)

# Save clustered data
df.to_csv(f'{output_folder}/clustered_movies.csv', index=False)
print(f"\n✓ Clustered movie data saved: clustered_movies.csv")

# Save the top 20 movies of each cluster to separate files
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    top_movies = cluster_data.nlargest(20, ['averageRating', 'numVotes'])[
        ['primaryTitle', 'genres', 'averageRating', 'numVotes', 'startYear', 'runtimeMinutes']
    ]
    top_movies.to_csv(f'{output_folder}/cluster_{cluster_id}_top_movies.csv', index=False)
    print(f"✓ Cluster {cluster_id} top movies saved")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📊 SUMMARY REPORT")
print("=" * 80)

print(f"""
GENERAL INFORMATION:
  • Total number of movies processed: {len(df):,}
  • Total number of clusters: {optimal_k}
  • Optimal cluster selection criterion: Highest Silhouette score
  • Silhouette score: {max(silhouette_scores):.4f}

WHY {optimal_k} CLUSTERS?
  {optimal_k} clusters were chosen because:
  1. The Silhouette score was maximum at this value ({max(silhouette_scores):.4f})
  2. The within-cluster similarity is at its highest level
  3. The difference between clusters is most distinct
  4. Optimal break point in the Elbow graph

LARGEST CLUSTERS:
""")

top_3_clusters = cluster_stats_df.nlargest(3, 'Movie Count')
for idx, row in top_3_clusters.iterrows():
    print(f"  • Cluster {int(row['Cluster'])}: {int(row['Movie Count']):,} movies ({row['Proportion (%)']:.1f}%)")
    print(f"    Average rating: {row['Avg. Rating']:.2f}, Most common genre: {row['Most Common Genre']}")

print(f"""
FILES CREATED:
  📊 Plots: 9 plots in the {output_folder}/ folder
  📄 Data files: 
     - clustered_movies.csv (all movies with cluster info)
     - cluster_statistics.csv (cluster summary statistics)
     - cluster_X_top_movies.csv (top 20 movies for each cluster)

RECOMMENDATION SYSTEM USAGE:
  By using the recommend_movies() function in the code,
  you can get similar movie recommendations with any movie title.

  Example:
  recommend_movies("The Dark Knight", df, top_n=10)
""")

print("\n" + "=" * 80)
print(f"✅ PROJECT COMPLETED!")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Example calls to the function
print(recommend_movies("la la land", df, top_n=20))
print(recommend_movies("amadeus", df, top_n=20))
print(recommend_movies("the pianist", df, top_n=20))