"""
REGRESSION MODEL WITH NAMES
---------------------------

What does this script do?

1) Reads two CSV files you exported from BigQuery:

   a) movies_with_creators_regression.csv
      Each row is a movie.
      EXAMPLE COLUMNS (must have at least these):
        - tconst
        - primaryTitle
        - isAdult
        - startYear
        - runtimeMinutes
        - genres
        - averageRating        (target)
        - numVotes
        - num_directors
        - num_writers
        - num_actors
        - num_principals
        - directors_str        (comma-separated names: "Christopher Nolan, Another Director")
        - writers_str          (comma-separated names)
        - actors_str           (comma-separated names; perhaps just the top 3 actors)

   b) person_stats.csv
      Table of historical performance based on person.
      It roughly comes from something like this on the BigQuery side:

        person_name_norm (LOWER(TRIM(primaryName)))
        category          (director / writer / actor / actress / etc.)
        film_count
        avg_rating        (average IMDb score of films the person starred in/directed)

2) Creates the following dictionaries from person_stats.csv:
      - person_stats[(name, role)] = {avg_rating, film_count}
      - role_defaults[role] = average rating of people in that role
      - global_default = overall average rating of all people

3) For each movie in movies_with_creators_regression.csv:
      - Converts the directors_str, writers_str, actors_str columns into a list of names
      - Calculates the following columns for the movie:
            directors_avg_prev_rating
            writers_avg_prev_rating
            actors_avg_prev_rating

4) Adds these to the regression features and builds a model with scikit-learn:
      - Numeric: runtimeMinutes, isAdult, startYear,
                 num_directors, num_writers, num_actors, num_principals,
                 directors_avg_prev_rating, writers_avg_prev_rating, actors_avg_prev_rating
      - Categorical: main_genre (first genre from the genres column)

5) Performs train/test split, model training, metric calculation, and saves PNG plots.

6) With the predict_new_movie_with_names() function:
      - You can get a rating prediction for a future or hypothetical movie
        by providing its runtime, year, genre, num_* features + a list of names.

NOTE: We previously discussed the exact SQL on the BigQuery side;
     here, we are only providing the Python pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =====================================================================
# 0) FILE PATHS - UPDATE THIS ACCORDING TO YOUR COMPUTER
# =====================================================================

MOVIES_CSV = r"C:\Users\meaki\PycharmProjects\PythonProject\imdb_data\movies_with_creators_regression.csv"
PERSON_STATS_CSV = r"C:\Users\meaki\PycharmProjects\PythonProject\imdb_data\person_stats.csv"
PLOTS_DIR = "plots_names_model"


# =====================================================================
# 1) HELPER FUNCTIONS
# =====================================================================

def add_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes only the first genre from the genres column and writes it to the main_genre column.
    e.g.: "Drama,Romance" -> "Drama"
    """
    df = df.copy()
    df["genres"] = df["genres"].fillna("Unknown")
    df["main_genre"] = df["genres"].apply(
        lambda x: str(x).split(",")[0] if pd.notnull(x) else "Unknown"
    )
    return df


def parse_name_list(s: str):
    """
    Converts a comma-separated name string into a list.
    e.g.: "Christopher Nolan, Another Person" -> ["Christopher Nolan", "Another Person"]
    Returns [] if empty or NaN.
    """
    if not isinstance(s, str):
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


# global dictionaries (to be filled when person_stats is loaded)
person_stats = {}
role_defaults = {}
global_default = None


def load_person_stats(person_stats_csv: str):
    """
    Reads person_stats.csv and populates the global variables:
      - person_stats[(name_norm, role)] = {avg_rating, film_count}
      - role_defaults[role] = avg_rating mean
      - global_default = overall avg_rating
    """
    global person_stats, role_defaults, global_default

    print(f"\n### Loading person_stats: {person_stats_csv}")
    ps_df = pd.read_csv(person_stats_csv)

    # Expected columns:
    expected_cols = {"person_name_norm", "category", "film_count", "avg_rating"}
    missing = expected_cols - set(ps_df.columns)
    if missing:
        raise ValueError(f"Missing column(s) in person_stats.csv: {missing}")

    # Populate dictionary
    person_stats = {}
    for _, row in ps_df.iterrows():
        key = (str(row["person_name_norm"]).strip().lower(), str(row["category"]))
        person_stats[key] = {
            "avg_rating": float(row["avg_rating"]),
            "film_count": int(row["film_count"]),
        }

    # Role-based averages
    role_defaults = (
        ps_df.groupby("category")["avg_rating"]
        .mean()
        .to_dict()
    )

    global_default = float(ps_df["avg_rating"].mean())

    print(f"Total person records: {len(ps_df)}")
    print(f"Number of distinct roles: {len(role_defaults)}")
    print(f"Global default rating: {global_default:.3f}")


def get_person_avg(name: str, role: str):
    """
    Returns the historical average rating for a single person.
    name: string (e.g.: "Christopher Nolan")
    role: 'director' / 'writer' / 'actor' / 'actress' / ...
    """
    if not isinstance(name, str) or not name.strip():
        return None

    name_norm = name.strip().lower()
    role = str(role)

    key = (name_norm, role)
    if key in person_stats:
        return person_stats[key]["avg_rating"]

    # A little hack to soften the confusion between actor/actress:
    if role == "actor":
        alt_key = (name_norm, "actress")
        if alt_key in person_stats:
            return person_stats[alt_key]["avg_rating"]
    if role == "actress":
        alt_key = (name_norm, "actor")
        if alt_key in person_stats:
            return person_stats[alt_key]["avg_rating"]

    # If nothing is found, fall back to the role average,
    # and if that doesn't exist, fall back to the global average
    if role in role_defaults:
        return float(role_defaults[role])
    return global_default


def avg_or_default(names, role: str):
    """
    Calculates the average rating for multiple names in the same role.
    E.g.: If there are two directors in the movie, it takes the average of their ratings.
    """
    vals = [get_person_avg(n, role) for n in names if n]
    vals = [v for v in vals if v is not None]
    if not vals:
        return global_default
    return float(np.mean(vals))


def build_creator_features_for_row(row: pd.Series):
    """
    Calculates the creator-based features for a single movie row.
    Uses the directors_str, writers_str, actors_str columns.
    """
    directors = parse_name_list(row.get("directors_str", None))
    writers = parse_name_list(row.get("writers_str", None))
    actors = parse_name_list(row.get("actors_str", None))

    directors_avg_prev_rating = avg_or_default(directors, "director")
    writers_avg_prev_rating = avg_or_default(writers, "writer")
    actors_avg_prev_rating = avg_or_default(actors, "actor")

    return pd.Series(
        {
            "directors_avg_prev_rating": directors_avg_prev_rating,
            "writers_avg_prev_rating": writers_avg_prev_rating,
            "actors_avg_prev_rating": actors_avg_prev_rating,
        }
    )


# =====================================================================
# 2) DATA LOAD & PREPROCESS
# =====================================================================

def load_and_preprocess_movies(movies_csv: str) -> pd.DataFrame:
    print(f"\n### Loading movie data: {movies_csv}")
    df = pd.read_csv(movies_csv)
    print(f"Total number of rows: {len(df)}")
    print("First 5 rows:")
    print(df.head())

    expected_cols = {
        "runtimeMinutes",
        "isAdult",
        "startYear",
        "genres",
        "num_directors",
        "num_writers",
        "num_actors",
        "num_principals",
        "averageRating",
        "directors_str",
        "writers_str",
        "actors_str",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s) in movies_with_creators_regression.csv: {missing}")

    # Ensure target is not null
    df = df.dropna(subset=["averageRating"])
    # Drop illogical runtimes
    df = df[df["runtimeMinutes"] > 0]
    # Ensure startYear is not null
    df = df.dropna(subset=["startYear"])

    print(f"Cleaned df shape: {df.shape}")

    # Add main_genre
    df = add_main_genre(df)
    print("\nmain_genre examples and top 10 most frequent genres:")
    print(df["main_genre"].value_counts().head(10))

    # Calculate creator-based features
    print("\n### Calculating creator-based features...")
    creator_feats = df.apply(build_creator_features_for_row, axis=1)
    df = pd.concat([df, creator_feats], axis=1)

    print("Creator features added. Example row:")
    print(df[
        [
            "runtimeMinutes",
            "isAdult",
            "startYear",
            "main_genre",
            "num_directors",
            "num_writers",
            "num_actors",
            "num_principals",
            "directors_avg_prev_rating",
            "writers_avg_prev_rating",
            "actors_avg_prev_rating",
            "averageRating",
        ]
    ].head(3))

    return df


def split_features_target(df: pd.DataFrame):
    print("\n### Splitting Features / Target")
    target_col = "averageRating"
    feature_cols = [
        "runtimeMinutes",
        "isAdult",
        "startYear",
        "num_directors",
        "num_writers",
        "num_actors",
        "num_principals",
        "directors_avg_prev_rating",
        "writers_avg_prev_rating",
        "actors_avg_prev_rating",
        "main_genre",
    ]

    X = df[feature_cols]
    y = df[target_col]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, feature_cols, target_col


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    print("\n### Train/Test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


# =====================================================================
# 3) MODEL PIPELINE
# =====================================================================

def build_model_pipeline():
    print("\n### Preprocessing + Model pipeline being set up")

    numeric_features = [
        "runtimeMinutes",
        "isAdult",
        "startYear",
        "num_directors",
        "num_writers",
        "num_actors",
        "num_principals",
        "directors_avg_prev_rating",
        "writers_avg_prev_rating",
        "actors_avg_prev_rating",
    ]
    categorical_features = ["main_genre"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Model used: RandomForestRegressor")
    return clf


def train_model(clf, X_train, y_train):
    print("\n### Model training starting")
    clf.fit(X_train, y_train)
    print("Model training completed.")
    return clf


def evaluate_model(clf, X_test, y_test):
    print("\n### Model evaluation")
    y_pred = clf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return y_pred, {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


def plot_results(y_test, y_pred, output_dir=PLOTS_DIR):
    print("\n### Plots being drawn and saved as PNG")

    os.makedirs(output_dir, exist_ok=True)

    # Scatter: actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Perfect prediction (y = x)",
    )
    plt.xlabel("Actual IMDb rating")
    plt.ylabel("Predicted IMDb rating")
    plt.title("Actual vs Predicted Rating (Named Model)")
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, "actual_vs_predicted_names_model.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved: {scatter_path}")
    plt.show()

    # Residual histogram
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual (Error) Distribution (Named Model)")
    plt.tight_layout()

    residuals_path = os.path.join(output_dir, "residuals_histogram_names_model.png")
    plt.savefig(residuals_path, dpi=300, bbox_inches="tight")
    print(f"Residual histogram saved: {residuals_path}")
    plt.show()


# =====================================================================
# 4) PREDICTION WITH NAMES FOR NEW MOVIE
# =====================================================================

def build_creator_features_for_new_movie(movie_dict: dict):
    """
    movie_dict must contain:
      - "directors": ["Christopher Nolan", ...]
      - "writers": ["Greta Gerwig", ...]
      - "actors": ["Timothée Chalamet", "Florence Pugh", ...]

    This function only calculates and returns the creator-based numeric columns.
    """
    directors = movie_dict.get("directors", []) or []
    writers = movie_dict.get("writers", []) or []
    actors = movie_dict.get("actors", []) or []

    directors_avg_prev_rating = avg_or_default(directors, "director")
    writers_avg_prev_rating = avg_or_default(writers, "writer")
    actors_avg_prev_rating = avg_or_default(actors, "actor")

    return {
        "directors_avg_prev_rating": directors_avg_prev_rating,
        "writers_avg_prev_rating": writers_avg_prev_rating,
        "actors_avg_prev_rating": actors_avg_prev_rating,
    }


def prepare_single_movie_row(movie_dict: dict) -> pd.DataFrame:
    """
    Example input:

    example_movie = {
        "runtimeMinutes": 120,
        "isAdult": 0,
        "startYear": 2026,
        "genres": "Drama,Romance",
        "num_directors": 1,
        "num_writers": 1,
        "num_actors": 3,
        "num_principals": 5,
        "directors": ["Christopher Nolan"],
        "writers": ["Greta Gerwig"],
        "actors": ["Timothée Chalamet", "Florence Pugh"],
    }

    This function returns a single-row DataFrame containing all columns
    expected by the model.
    """
    # Movie core features
    base_cols = [
        "runtimeMinutes",
        "isAdult",
        "startYear",
        "genres",
        "num_directors",
        "num_writers",
        "num_actors",
        "num_principals",
    ]
    data = {col: movie_dict[col] for col in base_cols}

    df = pd.DataFrame([data])
    df = add_main_genre(df)

    # Creator-based features
    creator_feats = build_creator_features_for_new_movie(movie_dict)
    for k, v in creator_feats.items():
        df[k] = v

    # Feature columns expected by the model:
    # runtimeMinutes, isAdult, startYear,
    # num_directors, num_writers, num_actors, num_principals,
    # directors_avg_prev_rating, writers_avg_prev_rating, actors_avg_prev_rating,
    # main_genre
    return df[
        [
            "runtimeMinutes",
            "isAdult",
            "startYear",
            "num_directors",
            "num_writers",
            "num_actors",
            "num_principals",
            "directors_avg_prev_rating",
            "writers_avg_prev_rating",
            "actors_avg_prev_rating",
            "main_genre",
        ]
    ]


def predict_new_movie_with_names(clf, movie_dict: dict) -> float:
    """
    Prediction function with names.

    See prepare_single_movie_row docstring for a movie_dict example.
    """
    X_new = prepare_single_movie_row(movie_dict)
    y_pred = clf.predict(X_new)[0]
    return float(y_pred)


# =====================================================================
# 5) MAIN
# =====================================================================

def main():
    # 1) Load person_stats
    load_person_stats(PERSON_STATS_CSV)

    # 2) Load movie data + add main_genre + creator features
    df_movies = load_and_preprocess_movies(MOVIES_CSV)

    # 3) Split features / target
    X, y, feature_cols, target_col = split_features_target(df_movies)

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 5) Model pipeline
    clf = build_model_pipeline()

    # 6) Train
    clf = train_model(clf, X_train, y_train)

    # 7) Evaluate
    y_pred, metrics = evaluate_model(clf, X_test, y_test)
    plot_results(y_test, y_pred, output_dir=PLOTS_DIR)

    print("\nModel evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 8) Example: prediction with names for a future movie
    print("\n### Example: Prediction for a future movie (with names)")

    example_movie = {
        "runtimeMinutes": 120,
        "isAdult": 0,
        "startYear": 2026,
        "genres": "Drama,Romance",
        "num_directors": 1,
        "num_writers": 1,
        "num_actors": 3,
        "num_principals": 5,
        "directors": ["Christopher Nolan"],
        "writers": ["Greta Gerwig"],
        "actors": ["Timothée Chalamet", "Florence Pugh"],
    }

    example_rating = predict_new_movie_with_names(clf, example_movie)
    print(f"Predicted IMDb rating for example movie: {example_rating:.2f}")


if __name__ == "__main__":
    main()