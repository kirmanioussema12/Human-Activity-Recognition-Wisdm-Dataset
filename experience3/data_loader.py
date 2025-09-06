import pandas as pd
def load_dataset(file_path):
    data=pd.read_csv(file_path)
    return data