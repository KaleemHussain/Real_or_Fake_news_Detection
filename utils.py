import joblib

def predict_news(news_text):
    model = joblib.load("models/news_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    vector = vectorizer.transform([news_text])
    prediction = model.predict(vector)
    return "REAL" if prediction[0] == 1 else "FAKE"
