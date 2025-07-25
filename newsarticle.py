# Import necessary libraries
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load the dataset (adjust file name if needed)
df = pd.read_csv("E:\Fake.csv")  # Dataset from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# Drop duplicates and NaNs
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Keep only necessary columns
df = df[['title', 'text']]
df['label'] = 'FAKE'  # Assign label to fake dataset

# Repeat for real dataset
real_df = pd.read_csv("E:\True.csv")
real_df.drop_duplicates(inplace=True)
real_df.dropna(inplace=True)
real_df = real_df[['title', 'text']]
real_df['label'] = 'REAL'

# Combine both datasets
data = pd.concat([df, real_df], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Combine title and text for better context
data['content'] = data['title'] + " " + data['text']

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
data['clean_content'] = data['content'].apply(clean_text)

# Split features and labels
X = data['clean_content']
y = data['label']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "E:\ fake_news_model.pkl")
joblib.dump(vectorizer, "E:\ vectorizer.pkl")
