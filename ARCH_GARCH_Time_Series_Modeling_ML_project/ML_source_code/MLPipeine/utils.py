import pandas as pd

# Function to read the data file 
def read_data(file_path, **kwargs):
    raw_data = pd.read_excel(file_path  ,**kwargs)
    return raw_data