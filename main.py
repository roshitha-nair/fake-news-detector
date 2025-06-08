import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load the dataset
df = pd.read_csv('news.csv')

#Preprocess Text Column
df['clean_text'] = df['text'].apply(clean_text)


# Print dataset shape and first few rows
print("Dataset Shape:", df.shape)
print(df.head())

# Get labels
labels = df['label']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df['clean_text'], labels, test_size=0.2, random_state=7)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)

# Fit on train, transform both train and test
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on test data
y_pred = pac.predict(tfidf_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")

# Print confusion matrix and classification report
conf_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("\nConfusion Matrix:\n", conf_mat)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'], cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Fake News Detection Confusion Matrix')
plt.show()

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Test on Custom Input
def test_article(article_text):
    vector = tfidf_vectorizer.transform([article_text])
    prediction = pac.predict(vector)
    print("\nPrediction:", prediction[0])

# Example test
test_article("NASA confirms water on the moon, astronauts may soon live there!")

# Save the trained PassiveAggressiveClassifier model
joblib.dump(pac, "model.pkl")

# Save the fitted TfidfVectorizer
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")