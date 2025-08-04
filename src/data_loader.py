import pandas as pd
import os

def load_cleaned_data(file_path="data/processed/cleaned_reviews.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def save_cleaned_data(df, file_path="data/processed/cleaned_reviews.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def load_raw_data(file_path="data/Reviews.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def save_results(df, file_path="data/processed/results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
