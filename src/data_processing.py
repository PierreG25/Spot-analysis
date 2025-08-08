import pandas as pd
import re
import numpy as np
from typing import List, Dict
import os


# ======================== LOAD CSV FILES INTO DATAFRAMES ========================

def load_epex_data(filepath):
    """
    Load a csv file into a DataFrame

    Args:
        filepath (): path
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(filepath)


def group_files_by_type(folder_path: str, types: List[str], extension: str = "") -> Dict[str, List[str]]:
    """
    Group files in a folder by type based on filename patterns

    Filenames are expected to follow the pattern:
        YYYY_<type>.<extension>
    where:
        - YYYY is a 4 digit year
        - <type> is one of the provided strings 'types'
        - <extension> is optional (e.g., ".csv")
    
    Args:
        folder_path (str): Path to the folder containing all the files
        types (List[str]): List of type strings to search in the filenames
        extensions (str): File extension to match

    Returns:
        Dict[str, List[str]]: Dictionary mapping each type to a list of matching file paths
    """
    grouped_files = {t: [] for t in types}
    ext_pattern = re.escape(extension) if extension else ''

    # Iterate trough all files in the target folder
    for f in os.listdir(folder_path):
        for t in types:
            pattern = re.compile(rf'^\d{{4}}_{t}{ext_pattern}$')
            # If filename matches the pattern, store its full path in the dictionary
            if pattern.match(f):
                grouped_files[t].append(os.path.join(folder_path, f))  # full path

    return grouped_files


def load_dataframes(grouped_files: Dict[str, List[str]]) -> Dict[str, List[pd.DataFrame]]:
    """
    Load CSV files into pandas DataFrames, grouped by types

    Args:
        grouped_files (Dict[str, List[str]]): Dictionary mapping each type to
            a list of file path. Typically the output of group_files_by_type
    
    Returns:
        Dict[str, List[pd.DataFrames]]: Dictionary mapping each type to a list
            of pandas DataFrames. Each DataFrame corresponds to one csv file
    
    Notes:
        - Missing or placeholder values such as 'N/A', 'n/a', 'NA', '-', '',
          and 'n/e' are automatically converted to NaN
    """
    dataframes = {}
    for dtype, files in grouped_files.items():
        # Read each file into a DataFrame, replacing certain placeholders with NaN
        dfs = [pd.read_csv(f, na_values=['N/A', 'n/a', 'NA', '-', '', 'n/e']) for f in files]
        dataframes[dtype] = dfs  # Assign list of DataFrames to the type key
    
    return dataframes


# ======================== PROCESS DATAFRAMES ========================


def drop_unecessary_columns(df, columns_to_drop):
    """
    Drop specified columns from a pandas DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_drop (list or array-like): List of columns names to remove

    Notes:
        - This function returns a new DataFrame; the original `df` is not modified
          unless `inplace=True` is explicitly used in `df.drop()`
    """
    return df.drop(columns=columns_to_drop)


def keep_necessary_columns(df, columns_to_keep):
    """
    Keep only the specified columns in a pandas DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_keep (list or array-like): List of columns name to retain

    Notes:
        - This function drops all columns not listed in `columns_to_keep`
        - The original DataFrame is not modified unless `inplace=True` is explicitly
          used in `df.drop()`
    """
    columns_names = df.columns
    columns_to_drop = [col for col in columns_names if col not in columns_to_keep]
    return df.drop(columns=columns_to_drop)


def filter_columns(df, columns: list, keep: bool):
    """
    Filter columns in a pandas DataFrame by keeping or dropping a given list

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list or array-like): List of column names to keep or drop
        keep (bool, optional): 
            - True (default): Keep only the specified columns
            - False: Drop the specified columns

    Returns:
        pd.DataFrame: New DataFrame with columns filtered according to `keep`

    Notes:
        - The original DataFrame is not modified unless `inplace=True` is explicitly
          passed to `df.drop()` (not supported in this helper)
    """
    if keep:
        # Keep only the specified columns → drop all others
        columns_to_drop = [col for col in df.columns if col not in columns]
    else:
        # Drop the specified columns directly
        columns_to_drop = columns

    # Return DataFrame without the unwanted columns
    return df.drop(columns=columns_to_drop)


def rename_and_reorder_columns(df, new_order, new_names):
    """
    Rename and optionally reorder columns in a pandas DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
        new_order (list or None):
            - If None: Keep current column order
            - If list: New column order specified by column names or indices
        new_name (list): List of new column names (must match number of columns)

    Returns:
        pd.DataFrame: DataFrame with columns renamed (and reordered if specified)
    
    Notes:
        - The length of `new_names` must match the number of columns after reordering
        - If `new_order` is provided, it must contain all columns to avoid KeyError
    """
    # If no reordering is requested, just rename columns
    if new_order == None:
        df.columns = new_names
        return df
    # Reoder and rename the columns
    df = df[new_order]
    df.columns = new_names
    return df


def setup_time(df, datetime_col, format):
    """
    Parse a datetime column, and resample the data hourly

    This function:
        1. Extracts the first part of the datetime string if it contains a ' - ' separator
        2. Parses the column into pandas datetime objects
        3. Sets the parsed datetime as the DataFrame index
        4. Resamples the data to hourly frequency, taking the mean for each hour
        5. Resets the index back to a regular column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        datetime_col (str): Column with datetime strings
        format (str): Datetime format string for parsing (e.g., "%Y-%m-%d %H:%M:%S")
    """
    df[datetime_col] = df[datetime_col].str.split(' - ').str[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
    df = df.set_index(datetime_col)
    df = df.resample('h').mean()
    return df.reset_index()


def fill_hourly_nans_by_rolling_mean(df, datetime_col, value_col, n_days=5):
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


def sort_and_concat(dfs):
    """
    Concatenate multiple DataFrames in chronological order based on the earliest year
    found in their 'Date' Column

    Args:
        dfs (list of pd.DataFrame): List of DataFrames, each containing a 'Date' column
            of datetime type
    
    Returns:
        pd.DataFrame: Single DataFrame containing all the rows from 'dfs' ordered by year
    """
    return pd.concat(
        sorted(dfs, key=lambda df: df['Date'].dt.year.min()),
        ignore_index=True
    )


def merge_df(dfs, on, how):
    """
    Merge a list of DataFrames on a common column
    
    Parameters:
        dfs (List[pd.DataFrame]): List of DataFrames to merge
        on (str): Column name to merge on
        how (str): Type of merge ('inner', 'outer', 'left', 'right'). Default is 'inner'
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)
    return merged_df


def df_summary(df):
    """
    Print a quick summary of a pandas DataFrame, including:
        - Shape (rows, columns)
        - Column data types
        - Missing values per column
        - First 5 rows
        - Last 5 rows

    Args:
        df (pd.DataFrame): The DataFrame to summarize

    Returns:
        None
    """
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