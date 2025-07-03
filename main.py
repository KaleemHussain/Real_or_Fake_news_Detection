import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load both datasets
real_df = pd.read_csv("data/real_news.csv")
fake_df = pd.read_csv("data/fake_news.csv")

# Add labels
real_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake

# Keep only necessary columns
real_df = real_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]

# Combine both
df = pd.concat([real_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/news_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
