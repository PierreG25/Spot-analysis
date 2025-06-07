"""module"""

import pandas as pd
import os

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
    df["Week"] = df['MTU (CET/CEST)'].dt.isocalendar().week
    df["Month"] = df['MTU (CET/CEST)'].dt.month
    df["Weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    return df
# End-of-file (EOF)

def exploitable_data(folderpath):
    dataframes = []
    for file in os.listdir(folderpath):
        if file.endswith('.csv'):
            filepath = os.path.join(folderpath, file)
            df = pd.read_csv(filepath, parse_dates=['Date'])  # adjust column name if needed
            
            year = df['Date'].dt.year.min()  # get the year from the data itself
            dataframes.append((year, df))

    # Sort by year (oldest to newest)
    dataframes.sort(key=lambda x: x[0])

    # Extract only the DataFrames, now in the right order
    df_concat = pd.concat([df for _, df in dataframes], ignore_index=True)
    return df_concat