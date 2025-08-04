
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_sentiment(text):

    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    X = vectorizer.transform([text])
    
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X).max()

    label = "Positive ✅" if prediction == 1 else "Negative ❌"
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    text = input("Enter your product review:\n")
    predict_sentiment(text)
