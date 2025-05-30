"""module"""

import pandas as pd

def load_epex_data(filepath):
    """Function loading a csv file into a DataFrame"""
    return pd.read_csv(filepath)

# End-of-file (EOF)