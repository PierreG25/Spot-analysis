import pandas as pd
import re
import numpy as np


# ============= EXTRACT PTDF COEFFICIENTS FROM CSV FILE ============= #

gen_path_be = 'data/raw/fbmc/generation/2025_generation_be_raw.csv'
gen_path_de = 'data/raw/fbmc/generation/2025_generation_de_raw.csv'
gen_path_fr = 'data/raw/fbmc/generation/2025_generation_fr_raw.csv'
gen_path_nl = 'data/raw/fbmc/generation/2025_generation_nl_raw.csv'

load_path_be = 'data/raw/fbmc/load/2025_load_be_raw.csv'
load_path_de = 'data/raw/fbmc/load/2025_load_de_raw.csv'
load_path_fr = 'data/raw/fbmc/load/2025_load_fr_raw.csv'
load_path_nl = 'data/raw/fbmc/load/2025_load_nl_raw.csv'

price_path_be = 'data/raw/fbmc/price/2025_price_be_raw.csv'
price_path_de = 'data/raw/fbmc/price/2025_price_de_raw.csv'
price_path_fr = 'data/raw/fbmc/price/2025_price_fr_raw.csv'
price_path_nl = 'data/raw/fbmc/price/2025_price_nl_raw.csv'

# ======================== FIXING SEQUENCE ISSUE WITHIN THE SPOT PRICE ======================== #
def sequence_selection(df, sequence = 'Sequence'):
    """Select the desired sequence 1 if the sequence column exists and does 
    not contain 'Without Sequence' otherwise return the original dataframe.
    """
    df = df.copy()
    if sequence not in df.columns:
        raise ValueError(f"Sequence column '{sequence}' not found in dataframe.")

    if 'Without Sequence' not in df[sequence].unique():
        df = df[df[sequence] == 'Sequence Sequence 1'].reset_index(drop=True)
        return df
    elif ('Without Sequence' in df[sequence].unique()) & (len(df[sequence].unique()) > 1):
        df = df[df[sequence] == 'Without Sequence'].reset_index(drop=True)
    
    print("No sequence selection applied, either because 'Without Sequence' was specified or because the sequence column is missing.")
    return df


# ======================== TIME SETUP FUNCTIONS ======================== #
DST_PATTERN = r"\(CET\)|\(CEST\)"

def flag_dst_rows(df, datetime_col = "MTU (CET/CEST)") -> pd.Series:
    """Flag rows that correspond to DST transition"""
    mask = df[datetime_col].str.contains(DST_PATTERN, regex=True)
    return mask

def downsample_to_15(df, value_col, datetime_col = "MTU (CET/CEST)"):
    """Downsample to 15-min resolution by forward filling each hour into 4 slots."""
    dt = pd.to_datetime(df[datetime_col])
    interval = (
        dt.sort_values()
        .drop_duplicates()
        .diff()
        .dropna()
        .mode()
        .iloc[0]
    )
    # print(f"Detected interval {interval} for {df} ")
    # print(interval == pd.Timedelta("15min"))
    # print(interval)
    # print(pd.Timedelta("15min"))
    if interval == pd.Timedelta("15min"):
        return df  # If already 15min, return the data as is
    
    df.to_csv('data/debug/df_before_downsample_debug.csv')
    group_col = df.columns.difference([datetime_col, value_col]).tolist()

    df = df.set_index(datetime_col)
    
    df_15min = (
        df.groupby(group_col)[value_col]
        .resample('15min')
        .ffill(limit=3)
        .reset_index()
    )
    df_15min.to_csv('data/debug/df_after_downsample_debug.csv')
    return df_15min


def setup_time(df, value_cols, datetime_col = "MTU (CET/CEST)", format = "mixed"):
    """Parse a datetime column, removing rows with DST transition hours."""
    # print(f"Original dataframe:\n{df.head()}")
    
    mask = flag_dst_rows(df)

    df[datetime_col] = (df[datetime_col].str.split(' - ')
                        .str[0].str
                        .replace(DST_PATTERN, "", regex=True)
                        .str.strip())
    
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
    # print(f"After parsing datetime:\n{df.head()}")

    day = df[datetime_col].dt.date
    day_with_dst = day[mask].unique()
    # print(f"Days with DST transitions:\n{day_with_dst}")
    
    # Remove rows with DST transition hours
    df = df.loc[~day.isin(day_with_dst)]
    # print(f"After removing DST transition rows:\n{df.head()}")
    
    print(f"Before downsampling: {len(df)} \n{df.head()}")
    df = downsample_to_15(df, value_cols)
    day = df[datetime_col].dt.date
    df = df.loc[~day.isin(day_with_dst)]
    print(f"After downsampling: {len(df)} \n{df.head()}")
    
    # Remove days with DST transition hours again after downsampling due to possible residuals

    return df


# ============= LOAD AND PREPROCESS DATA FUNCTIONS ============= #
# Function to load and preprocess generation data

def load_generation_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])

        df = setup_time(df, "Generation (MW)")

        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Generation (MW)": "Generation"})
        df = df.groupby(["Time", "Area"]).agg({"Generation": "sum"}).reset_index()
        df['Area'] = area_name
        df.to_csv('data/debug/gen_debug_' + f'{area_name}' + '.csv')
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Function to load and preprocess load data
def load_load_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
        df.drop(columns=['Day-ahead Total Load Forecast (MW)'], inplace=True)

        df = setup_time(df, "Actual Total Load (MW)") 
        df.to_csv('data/debug/load_debug_' + f'{area_name}' + '.csv')
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Actual Total Load (MW)": "Total load"})
        df = df[["Time", "Area", "Total load"]]
        df['Area'] = area_name
        # df.to_csv('data/debug/load_debug_' + f'{area_name}' + '.csv')
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def load_price_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
        df = sequence_selection(df, sequence = 'Sequence')
        df = setup_time(df, "Day-ahead Price (EUR/MWh)")
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Day-ahead Price (EUR/MWh)": "Price"})
        df['Area'] = area_name
        df = df[["Time", "Area", "Price"]]
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


# ============= CALCULATE METRICS AND CREATE MASTER DATASET ============= #
# Function to calculate net position and renewable share

def calculate_metrics(df_gen, df_load):
    df_gen.to_csv('data/raw/fbmc/generation/gen_debug.csv')
    df_load.to_csv('data/raw/fbmc/load/load_debug.csv')
    df = pd.merge(df_gen, df_load, on=['Time', 'Area'], suffixes=('_gen', '_load'))
    df['Net position'] = df['Generation'] - df['Total load']
    df['Renewable share'] = 0  # Placeholder as renewable data is not provided
    df['import/export flag'] = df['Net position'].apply(lambda x: 'import' if x < 0 else 'export')
    return df

# Main function to create master dataset
def create_master_dataset(gen_paths, load_paths, price_paths, areas):
    df_gen = load_generation_data(gen_paths, areas)
    df_load = load_load_data(load_paths, areas)
    df_price = load_price_data(price_paths, areas)
    print('OK')
    df_gen.to_csv('data/debug/gen_debug')
    df_load.to_csv('data/debug/load_debug')
    df_price.to_csv('data/debug/price_debug')

    master_df = calculate_metrics(df_gen, df_load)
    master_df = pd.merge(master_df, df_price, on=['Time', 'Area'], how='left')

    return master_df


# Define paths and areas
gen_paths = [gen_path_be, gen_path_de, gen_path_fr, gen_path_nl]
load_paths = [load_path_be, load_path_de, load_path_fr, load_path_nl]
price_paths = [price_path_be, price_path_de, price_path_fr, price_path_nl]

areas = ['BZN|BE', 'BZN|DE-LU', 'BZN|FR', 'BZN|NL']

# Create master dataset
master_dataset = create_master_dataset(gen_paths, load_paths, price_paths, areas)
master_dataset.to_csv('data/clean/fbmc/master_dataset_15min_WITH_PRICE.csv')
print(master_dataset.head())