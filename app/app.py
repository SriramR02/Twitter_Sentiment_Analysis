import streamlit as st
import pickle
from modules.preprocess_text import preprocess_text
from collections.abc import Iterable

# Load the trained model and vectorizer
model = pickle.load(open('../models/trained_model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

# Set up the main layout
st.set_page_config(page_title='Twitter Sentiment Analysis', layout='wide')

# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox("Main Menu", ["Home", "Analyze Tweet", "About"])

# Home tab
if selected == "Home":
    st.title('Welcome to Twitter Sentiment Analysis')
    st.write('This app predicts the sentiment of a given tweet using a machine learning model.')
    #st.image('https://example.com/your_image.png', use_column_width=True)  # Add a relevant image

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
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
            st.write(f'Sentiment: {sentiment}')
        else:
            st.write('Please enter a tweet to analyze.')

# About tab
if selected == "About":
    st.title('About')
    st.write('This app was developed to demonstrate sentiment analysis on tweets using a machine learning model. It uses logistic regression to classify tweets as positive or negative.')

if __name__ == '__main__':
    st.run()