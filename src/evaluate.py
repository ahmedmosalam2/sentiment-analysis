import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os


def load_features(model_dir):
    X = joblib.load(os.path.join(model_dir, "X_tfidf.pkl"))
    y = joblib.load(os.path.join(model_dir, "y.pkl"))
    model = joblib.load(os.path.join(model_dir, "sentiment_model.pkl"))
    return X, y, model


def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    print("Classification Report:\n")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    model_dir = "models"
    X, y, model = load_features(model_dir)
    evaluate_model(X, y, model)
    print("\n✅ Model evaluation completed successfully.")  
    