import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from modules.setup_kaggle import setup_kaggle
from modules.download_dataset import download_dataset
from modules.load_data import load_data
from modules.preprocess_text import preprocess_text
from modules.vectorize_data import vectorize_data
from modules.train_model import train_model
from modules.evaluate import evaluate_model
from modules.visulaize import plot_results
from modules.save_model import save_model
from sklearn.model_selection import train_test_split
import pickle

def main():
    setup_kaggle()
    download_dataset()
    
    twitter_data = load_data()
    
    twitter_data['processed_text'] = twitter_data['text'].apply(preprocess_text)

    X = twitter_data['processed_text'].values
    Y = twitter_data['target'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    X_train, X_test, vectorizer = vectorize_data(X_train, X_test)

    # Save the vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    model = train_model(X_train, Y_train)
    
    X_test_prediction = evaluate_model(model, X_train, Y_train, X_test, Y_test)
    
    plot_results(Y_test, X_test_prediction)
    
    save_model(model)

if __name__ == "__main__":
    main()