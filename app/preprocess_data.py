import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess the input text by removing URLs, mentions, hashtags, punctuation,
    converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
    
    Args:
    text (str): The input text to preprocess.
    
    Returns:
    str: The preprocessed text.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove RT
    text = re.sub(r'RT[\s]+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    # Lemmatize tokens
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return " ".join(lemmas)

# Load the original dataset
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])

# Preprocess the text data
twitter_data['processed_text'] = twitter_data['text'].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
twitter_data.to_csv('processed_twitter_data.csv', index=False)