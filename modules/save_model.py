import pickle

def save_model(model, filename='models/trained_model.pkl'):
    pickle.dump(model, open(filename, 'wb'))
    