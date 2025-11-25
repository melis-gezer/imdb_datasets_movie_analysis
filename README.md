# IMDb Dataset Movie Analysis

A comprehensive data analysis project using IMDb's non-commercial datasets for movie quality classification, rating prediction, and recommendation system development.

## Project Overview

This project analyzes 950K+ movies and TV series from IMDb datasets to:
- Classify content quality (Hidden Gems, Masterpieces, Overrated)
- Predict movie ratings using machine learning
- Build a movie recommendation system using K-Means clustering

## Repository Structure

```
imdb_datasets_movie_analysis/
├── imdb_data/              # Processed CSV tables
├── chi_square_tests.py     # Statistical correlation tests
├── clustering.py           # K-Means clustering for recommendations
├── regression.py           # Rating prediction models
└── README.md
```

## Dataset

**IMDb Tables Used:**
- name.basics - Actor/director information
- title.basics - Movie/series base information
- title.ratings - Ratings and vote counts
- title.crew - Director and writer information
- title.principals - Principal cast members
- title.episode - Episode information

**Additional Data:** TMDB dataset from Kaggle for budget and revenue information.

## Data Cleaning

- Filtered future year content (11.64% removed)
- Removed non-movie/series content types (12.64% removed)
- Cleaned columns with 50%+ missing data
- Split data into `movie`, `series`, and `episode` tables
- Applied filters: minimum 10 votes for quality classification, minimum 1000 votes for regression models

## Technologies Used

- Data Processing: Python, BigQuery, SQL
- ML Libraries: scikit-learn, pandas, numpy
- Storage: Google Cloud Storage
- Algorithm: RandomForestRegressor for predictions, K-Means for clustering

## Data Source

IMDb Non-Commercial Datasets: https://www.imdb.com/interfaces/
Kaggle TMDB Dataset: https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m

## License

This project uses IMDb's non-commercial datasets and is intended for educational purposes only.
