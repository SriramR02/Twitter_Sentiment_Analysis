import pandas as pd

def load_data():
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=column_names)
    twitter_data.replace({'target': {4: 1}}, inplace=True)
    return twitter_data