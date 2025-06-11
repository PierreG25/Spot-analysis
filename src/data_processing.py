"""module"""

import pandas as pd
import re
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
        dfs = [pd.read_csv(f) for f in files]
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
    return df
# End-of-file (EOF)

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
    print("-" * 40)