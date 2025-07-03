# Fake News Detection

GitHub: https://github.com/KaleemHussain/Real_or_Fake_news_Detection/tree/main

This project detects whether a news article is real or fake using NLP and machine learning.

## Features
- TF-IDF text vectorization
- Logistic Regression classifier
- Accuracy and performance report
- Save & load models using joblib

## How to Run


✅ README.md (Full File Content)
# 📰 Fake News Detection using Machine Learning

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to detect whether a news article is **REAL** or **FAKE**. It is based on logistic regression and TF-IDF vectorization and can be trained using two separate CSV datasets: one for real news and one for fake news.

---

## 📁 Project Structure
```bash
FakeNewsDetection/
│
├── data/
│ ├── real_news.csv # Dataset of real news (must contain 'text' column)
│ └── fake_news.csv # Dataset of fake news (must contain 'text' column)
│
├── models/
│ ├── news_model.pkl # Trained ML model
│ └── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│
├── main.py # Script to train the model
├── test_news.py # Script to test new news articles
├── utils.py # Utility function to load model and predict
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation



---

## 🚀 How to Run the Project

### 🔹 Step 1: Clone or Download the Project

```bash
 manually download and unzip the folder.

🔹 Step 2: Install Required Libraries
Use pip to install the required Python libraries.


pip install -r requirements.txt
🔹 Step 3: Prepare Your Data
Place your two CSV files inside the data/ folder:

real_news.csv should contain real news articles with a column named text

fake_news.csv should contain fake news articles with a column named text

🔹 Step 4: Train the Model
Run the training script to combine the datasets, vectorize the text, and train a logistic regression model:

python main.py
This will:

Combine real and fake news into one dataset

Shuffle and split the data

Train a model

Save the trained model and vectorizer in the models/ folder




🔹 Step 5: Test the Model with a New Article
Open the test_news.py file and edit the news_text variable with any article you'd like to test.


from utils import predict_news

news_text = "The government has launched a new policy to improve the economy."
result = predict_news(news_text)
print("Prediction:", result)
Run the script:

python test_news.py
✅ Sample Output:

Prediction: REAL OR FAKE

```bash 
👨‍💻 Author
Made by Kaleem_Hussain
