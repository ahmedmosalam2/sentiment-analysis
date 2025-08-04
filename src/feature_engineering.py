import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def load_cleaned_data(file_path):
    
    df = pd.read_csv(file_path)
    return df


def convert_score_to_label(score):
    if score in [1, 2]:
        return 0
    elif score in [4, 5]:
        return 1
    else:
        return None


def prepare_labels(df):
    df = df[df['Score'] != 3].copy()  
    df.loc[:, 'label'] = df['Score'].apply(convert_score_to_label)
    df = df.dropna(subset=['cleaned_text'])  
    return df



def vectorize_text(df, max_features=5000):

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['cleaned_text'])
    return X, vectorizer


def save_features(X, y, vectorizer, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(X, os.path.join(output_dir, "X_tfidf.pkl"))
    joblib.dump(y, os.path.join(output_dir, "y.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    print("Features and vectorizer saved successfully.")


if __name__ == "__main__":
    input_file = "/mnt/d/Sentiment-Analysis/data/processed/cleaned_reviews.csv"
    output_dir = "/mnt/d/Sentiment-Analysis/models"

    df = load_cleaned_data(input_file)
    df = prepare_labels(df)
    X, vectorizer = vectorize_text(df)
    y = df['label']

    save_features(X, y, vectorizer, output_dir)
