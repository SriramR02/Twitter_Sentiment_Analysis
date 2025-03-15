

from fastapi import FastAPI
from pydantic import BaseModel
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from collections.abc import Iterable  # Corrected import
import joblib

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]  # Remove stopwords and non-alphabetic
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return " ".join(lemmas)

# Load the trained model and vectorizer
with open('./models/trained_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

with open('./models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

# Define the request body schema
class Tweet(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

@app.post("/analyze")
async def analyze_sentiment(tweet: Tweet):
    # Preprocess the input text
    processed_text = preprocess_text(tweet.text)
    
    # Vectorize the input text
    input_vectorized = vectorizer.transform([processed_text])
    
    # Predict the sentiment
    prediction = model.predict(input_vectorized)
    
    # Return the result
    if prediction == 1:
        return {"sentiment": "Positive"}
    else:
        return {"sentiment": "Negative"}

# Run the app with: uvicorn main:app --reload