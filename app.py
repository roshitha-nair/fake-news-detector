import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit UI
st.title("📰 Fake News Detector")

article = st.text_area("Enter the news article content below:")

if st.button("Check if it's Real or Fake"):
    if article.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(article)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        st.success(f"This news is predicted to be: **{result}**")
