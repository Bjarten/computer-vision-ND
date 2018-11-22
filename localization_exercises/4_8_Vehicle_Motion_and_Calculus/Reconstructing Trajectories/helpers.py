import pickle

def process_data(filename):
    with open(filename, 'rb') as f:
        data_list = pickle.load(f)
    return list(data_list)