"""module"""

import pandas as pd

def load_epex_data(filepath):
    """Function loading a csv file into a DataFrame"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Function cleaning the data to make them exploitable"""
    df.drop(['Area', 'Sequences', 'Intraday Period (CET/CEST)', 'Intraday Price (EUR/MWh)'], axis=1)
    return df
# End-of-file (EOF)