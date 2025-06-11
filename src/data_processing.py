"""module"""

import pandas as pd
import re
import numpy as np
from typing import List, Dict
import os

def load_epex_data(filepath):
    """Function loading a csv file into a DataFrame"""
    return pd.read_csv(filepath)

########################### Load csv files into DataFrames #################################

def group_files_by_type(folder_path: str, types: List[str], extension: str = "") -> Dict[str, List[str]]:
    grouped_files = {t: [] for t in types}
    ext_pattern = re.escape(extension) if extension else ''

    for f in os.listdir(folder_path):
        for t in types:
            pattern = re.compile(rf'^\d{{4}}_{t}{ext_pattern}$')
            if pattern.match(f):
                grouped_files[t].append(os.path.join(folder_path, f))  # full path

    return grouped_files

def load_dataframes(grouped_files: Dict[str, List[str]]) -> Dict[str, List[pd.DataFrame]]:
    dataframes = {}
    for dtype, files in grouped_files.items():
        dfs = [pd.read_csv(f, na_values=['N/A', 'n/a', 'NA', '-', '']) for f in files]
        dataframes[dtype] = dfs  # list of DataFrames, one per file
    return dataframes


########################### Process DataFrames #################################


def drop_unecessary_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)

def rename_and_reoder_columns(df, new_order, new_names):
    if new_order == None:
        df.columns = new_names
        return df
    df = df[new_order]
    df.columns = new_names
    return df

def setup_time(df, datetime_col, format):
    """Function cleaning the data to make them exploitable"""
    df[datetime_col] = df[datetime_col].str.split(' - ').str[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
    df = df.set_index(datetime_col)
    df = df.resample('H').mean()
    return df.reset_index()
# End-of-file (EOF)

def fill_hourly_nans_by_rolling_mean(df, datetime_col, value_col, n_days=4):
    """
    Fill NaN values in a time series column with the mean of the same hour over the past n_days.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        datetime_col (str): Name of the timestamp column.
        value_col (str): Name of the value column with NaNs to fill.
        n_days (int): Number of previous days to average for filling (default: 4).
    
    Returns:
        pd.DataFrame: A copy of the input DataFrame with NaNs in value_col filled.
    """
    df = df.copy()
    
    # Ensure timestamp is datetime and set index
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    # Extract date and hour
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Pivot: rows = date, columns = hour
    pivot = df.pivot_table(values=value_col, index='date', columns='hour')

    # Compute rolling mean across previous n_days
    rolling_means = pivot.rolling(window=n_days, min_periods=1, center=True).mean()

    # Function to apply per row
    def fill_value(row):
        if pd.isna(row[value_col]):
            try:
                return rolling_means.loc[row['date'], row['hour']]
            except KeyError:
                return np.nan
        else:
            return row[value_col]

    # Apply filling logic
    df[value_col] = df.reset_index().apply(fill_value, axis=1).values

    # Drop helper columns
    df.drop(columns=['hour', 'date'], inplace=True)

    # Reset index to return to original format
    return df.reset_index()

def merge_df(dfs, on, how):
    """
    Merge a list of DataFrames on a common column.
    
    Parameters:
        dfs (List[pd.DataFrame]): List of DataFrames to merge.
        on (str): Column name to merge on.
        how (str): Type of merge ('inner', 'outer', 'left', 'right'). Default is 'inner'.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)
    return merged_df

def concat_data(dfs):
    dataframes = []
    for df in dfs:
            year = df['Date'].dt.year.min()  # get the year from the data itself
            dataframes.append((year, df))
            print('OUI')

    # Sort by year (oldest to newest)
    dataframes.sort(key=lambda x: x[0])

    # Extract only the DataFrames, now in the right order
    df_concat = pd.concat([df for _, df in dataframes], ignore_index=True)
    return df_concat

def concat_data2(folderpath):
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

def df_summary(df):
    print(f"Shape: {df.shape}")
    print("\nColumn Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst Rows:")
    print(df.head())
    print("\nLast Rows:")
    print(df.tail())
    print("-" * 40)