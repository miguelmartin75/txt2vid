import pickle

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
