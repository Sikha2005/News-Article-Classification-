import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Paste a news article below to check if it's FAKE or REAL.")

user_input = st.text_area("Enter News Article")

if st.button("Classify"):
    cleaned = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned])
    prediction = model.predict(vector_input)[0]
    st.subheader(f"🧠 Prediction: {prediction}")
