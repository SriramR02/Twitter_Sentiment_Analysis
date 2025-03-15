from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, vectorizer