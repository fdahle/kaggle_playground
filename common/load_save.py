import pandas as pd

#load data and return as a pd dataframe
def load_data(data_path):

    #differentiate between the different datatypes
    if (data_path.endswith(".csv")):
        data = pd.read_csv(data_path)


    return data

def save_model():
    pass

def load_model():
    pass

def save_results():
    pass
