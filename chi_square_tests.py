"""
GLOBAL CHI-SQUARE ANALYSES
----------------------------
This script tests the following 2 hypotheses using quality_popularity_classification.csv:

1) quality_class x main_genre
2) quality_class x era (year groups)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATH SETTINGS
# ---------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\meaki\PycharmProjects\PythonProject")
DATA_DIR = BASE_DIR / "imdb_data"
OUTPUT_DIR = BASE_DIR / "global_chi_analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "quality_popularity_classification.csv"

# ---------------------------------------------------------------------
# VISUAL SETTINGS
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 10

print("=" * 100)
print(" GLOBAL CHI-SQUARE ANALYSES (QUALITY_POPULARITY) ")
print("=" * 100)


# ---------------------------------------------------------------------
# 1. CORE FUNCTIONS
# ---------------------------------------------------------------------

def run_chi_for_columns(df, col1, nice1, prefix, min_samples=100, top_n_plot=15):
    """
    Performs Chi-Square test for independence between a given column (col1) and 'quality_class'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col1 (str): The column to test against 'quality_class' (e.g., 'main_genre').
        nice1 (str): Human-readable name for the column (for print/plot).
        prefix (str): Prefix for saving plot files.
        min_samples (int): Minimum sample size for filtering categories.
        top_n_plot (int): Number of top categories to show in the plot.
    """
    print(f"\n--- Analysis: {nice1} vs Quality Class ---")

    # 1. Filter out rare categories
    counts = df[col1].value_counts()
    frequent_categories = counts[counts >= min_samples].index
    df_filtered = df[df[col1].isin(frequent_categories)].copy()

    print(f"Total rows before filtering: {len(df):,}")
    print(f"Total rows after filtering (Min {min_samples}): {len(df_filtered):,}")
    print(f"Total categories before filtering: {len(counts)}")
    print(f"Total categories after filtering: {len(frequent_categories)}")

    if len(frequent_categories) < 2 or len(df_filtered['quality_class'].unique()) < 2:
        print("❌ Insufficient data to perform Chi-Square test (too few groups/categories). Skipping.")
        return

    # 2. Build the Contingency Table
    contingency_table = pd.crosstab(
        df_filtered[col1],
        df_filtered['quality_class'],
        normalize='index'  # Normalize by row to see the quality class distribution within each genre/era
    ) * 100

    # 3. Chi-Square Test for Independence
    # H0: The distribution of 'quality_class' is independent of 'col1' (no relationship).
    # Ha: There is a dependency (relationship).

    # We use the unnormalized table for the test itself:
    contingency_test_raw = pd.crosstab(
        df_filtered[col1],
        df_filtered['quality_class']
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency_test_raw)

    print("\n📊 Chi-Square Test Results:")
    print(f"  Chi2 Statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.10f}")
    print(f"  Degrees of Freedom (dof): {dof}")

    alpha = 0.05
    if p_value < alpha:
        print(
            f"  Conclusion: P-value < {alpha}, Null Hypothesis rejected. There is a statistically significant relationship between {nice1} and film quality class.")
    else:
        print(
            f"  Conclusion: P-value >= {alpha}, Null Hypothesis not rejected. No statistically significant relationship found.")

    # 4. Visualization (Stacked Bar Plot)

    # Select the top N categories based on frequency
    top_categories = counts.head(top_n_plot).index
    plot_data = contingency_table.loc[top_categories]

    # Ensure columns are sorted (e.g., from lowest quality to highest)
    quality_order = sorted(plot_data.columns)
    plot_data = plot_data[quality_order]

    ax = plot_data.plot(kind='barh', stacked=True, figsize=(14, 10))

    plt.title(f'Quality Class Distribution by Top {top_n_plot} {nice1}s', fontsize=16, fontweight='bold')
    plt.xlabel('Proportion (%)', fontsize=12)
    plt.ylabel(nice1, fontsize=12)
    plt.legend(title='Quality Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_path = OUTPUT_DIR / f"{prefix}_quality_stacked_bar.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Stacked bar chart saved: {plot_path}")

    # 5. Top categories for 'Masterpiece' (e.g., 'A1')
    if 'A1' in plot_data.columns:
        print(f"\n✨ Top 5 {nice1}s by A1 (Masterpiece) Ratio:")
        top_a1 = plot_data['A1'].sort_values(ascending=False).head(5)
        for cat, ratio in top_a1.items():
            print(f"  • {cat}: {ratio:.2f}% (N={counts[cat]:,})")


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

print("\n--- 1. Data Loading and Preparation ---")

# Load the classification data
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Data loaded successfully. Total films: {len(df):,}")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {CSV_PATH}. Exiting.")
    exit()

# Drop rows where 'quality_class' is null (essential for the analysis)
df = df.dropna(subset=["quality_class"])
print(f"✓ Null 'quality_class' rows removed. Remaining films: {len(df):,}")

# Clean 'main_genre' (take only the first genre and handle nulls)
# Note: Assuming 'main_genre' is already calculated or is the first genre in the 'genres' column.
# If 'main_genre' doesn't exist, we create it from 'genres'.
if "main_genre" not in df.columns and "genres" in df.columns:
    df["main_genre"] = df["genres"].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else np.nan)
elif "main_genre" not in df.columns:
    print("⚠ 'main_genre' column not found. Skipping genre analysis.")


# Create the 'era' column (grouping by year)
def get_era(year):
    """Categorizes the film year into an era."""
    y = int(year)
    if y < 1920:
        return "Pre-1920"
    elif y < 1950:
        return "1920-1949"
    elif y < 1980:
        return "1950-1979"
    elif y < 2000:
        return "1980-1999"
    elif y < 2010:
        return "2000-2009"
    elif y < 2020:
        return "2010-2019"
    else:
        return "2020+"


if "startYear" in df.columns:
    # Ensure startYear is numeric and handle potential missing/invalid values
    df["startYear"] = pd.to_numeric(df["startYear"], errors='coerce').dropna()
    df["era"] = df["startYear"].apply(get_era)
else:
    df["era"] = np.nan
    print("⚠ 'startYear' column not found. Skipping era analysis.")

# Save the prepared data (optional: use for Looker or other tools)
prep_csv = OUTPUT_DIR / "quality_prepared_for_chi_genre_era.csv"
df.to_csv(prep_csv, index=False, encoding="utf-8")
print(f"✓ Prepared data saved: {prep_csv}")

# ---------------------------------------------------------------------
# 2. ANALYSES
# ---------------------------------------------------------------------

# 2.1 quality_class x main_genre
if df["main_genre"].notna().any():
    run_chi_for_columns(
        df=df,
        col1="main_genre",
        nice1="Main Genre",
        prefix="genre_quality",
        min_samples=500,  # Increased sample size for genre stability
        top_n_plot=15
    )
else:
    print("\n--- Skipping Main Genre Analysis (No data) ---")

# 2.2 quality_class x era
if df["era"].notna().any():
    run_chi_for_columns(
        df=df,
        col1="era",
        nice1="Era (Year Group)",
        prefix="era_quality",
        min_samples=100,
        top_n_plot=10
    )
else:
    print("\n--- Skipping Era Analysis (No data) ---")

print("\n" + "=" * 100)
print("✅ GLOBAL CHI-SQUARE ANALYSES COMPLETED!")
print("=" * 100)