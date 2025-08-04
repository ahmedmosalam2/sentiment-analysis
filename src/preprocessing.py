import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_and_save(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'Text' not in df.columns:
        raise KeyError("The expected column 'Text' was not found in the dataset.")
    df['cleaned_text'] = df['Text'].astype(str).apply(clean_text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "/mnt/d/Sentiment-Analysis/data/Reviews.csv"
    output_file = "/mnt/d/Sentiment-Analysis/data/processed/cleaned_reviews.csv"
    preprocess_and_save(input_file, output_file)
