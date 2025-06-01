"""module"""

import pandas as pd

def load_epex_data(filepath):
    """Function loading a csv file into a DataFrame"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Function cleaning the data to make them exploitable"""
    df = df.drop(['Area', 'Sequence', 'Intraday Period (CET/CEST)', 'Intraday Price (EUR/MWh)'], axis=1)
    df['MTU (CET/CEST)'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]
    df['MTU (CET/CEST)'] = pd.to_datetime(df['MTU (CET/CEST)'], format="%d/%m/%Y %H:%M:%S")
    df["Date"] = df['MTU (CET/CEST)'].dt.date
    df["Hour"] = df['MTU (CET/CEST)'].dt.hour
    df["Weekday"] = df['MTU (CET/CEST)'].dt.day_name()
    df["Week"] = df['MTU (CET/CEST)'].dt.week()
    df["Weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    return df
# End-of-file (EOF)