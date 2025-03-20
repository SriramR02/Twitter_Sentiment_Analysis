import streamlit as st
import pandas as pd
import joblib
from preprocess_text import preprocess_text

# Set page configuration
st.set_page_config(page_title='Twitter Sentiment Analysis', layout='wide')

# Load the preprocessed data
@st.cache_data
def load_data():
    return pd.read_csv('processed_twitter_data.csv')

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('trained_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Load data and model
twitter_data = load_data()
model, vectorizer = load_model()

# Streamlit app
# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox("Main Menu", ["Home", "Analyze Tweet", "About"])

# Home tab
if selected == "Home":
    st.title('Welcome to Twitter Sentiment Analysis')
    st.write('This app predicts the sentiment of a given tweet using a machine learning model.')

# Analyze Tweet tab
if selected == "Analyze Tweet":
    st.title('Analyze Tweet Sentiment')
    user_input = st.text_area('Enter a tweet:', '')

    if st.button('Analyze'):
        if user_input:
            # Preprocess the input text
            processed_text = preprocess_text(user_input)
            # Vectorize the processed text
            vectorized_text = vectorizer.transform([processed_text])
            # Predict the sentiment
            prediction = model.predict(vectorized_text)
            sentiment = 'Positive' if prediction[0] == 4 else 'Negative'
            st.write(f'Sentiment: {sentiment}')
        else:
            st.write('Please enter a tweet to analyze.')

# About tab
if selected == "About":
    st.title('About')
    st.write('This app was developed to demonstrate sentiment analysis on tweets using a machine learning model. It uses logistic regression to classify tweets as positive or negative.')