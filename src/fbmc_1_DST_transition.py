import pandas as pd
import re
import numpy as np


### EXTRACT PTDF COEFFICIENTS FROM CSV FILE ###

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


### STANDARDIZE DATASETS TO 15MIN RESOLUTION + WINTER/SUMMER TIME ###


# def flag_dst_rows(df, datetime_col="MTU (CET/CEST)") -> pd.Series:
#     """Flag rows that correspond to DST transition"""
#     mask = df[datetime_col].str.contains(DST_PATTERN, regex=True)
#     return mask

# def downsample_to_15(df, value_cols, datetime_col="MTU (CET/CEST)"):
#     """Downsample to 15-min resolution by forward filling each hour into 4 slots for multiple value columns."""
    
#     # Ensure datetime_col is passed as a list
#     if isinstance(datetime_col, str):
#         datetime_col = [datetime_col]  # Convert to list if it's a string

#     # Ensure value_cols is a list
#     if isinstance(value_cols, str):
#         value_cols = [value_cols]  # Convert to list if it's a single string
    
#     # Check the time interval between consecutive rows
#     interval = pd.to_datetime(df[datetime_col[0]]).diff().mode()[0]
#     if interval == pd.Timedelta(minutes=15):
#         return df  # If already 15min, return the data as is

#     # Ensure `group_col` is defined correctly
#     group_col = df.columns.difference(datetime_col + value_cols).tolist()  # Ensure both are lists
    
#     # Set the datetime column as the index
#     df = df.set_index(datetime_col[0])  # Set the index with the first element in datetime_col list

#     # Resample and forward fill for each value column
#     for value_col in value_cols:
#         # Apply resampling
#         resampled_col = (
#             df.groupby(group_col)[value_col]
#             .resample('15min')
#             .ffill(limit=3)
#         )
        
#         # Align the resampled column with the original dataframe index
#         df[value_col] = resampled_col.reindex(df.index, method='ffill')  # Align with original index

#     df_15min = df.reset_index()  # Reset the index after processing
#     return df_15min


# def setup_time(df, value_cols, datetime_col="MTU (CET/CEST)", format="mixed"):
#     """Parse a datetime column, removing rows with DST transition hours."""
#     print(f"Original dataframe:\n{df.head()}")

#     # Flag DST rows
#     mask = flag_dst_rows(df, datetime_col)

#     # Remove timezone and split the datetime to clean format
#     df[datetime_col] = (df[datetime_col].str.split(' - ')
#                         .str[0].str
#                         .replace(DST_PATTERN, "", regex=True)
#                         .str.strip())

#     # Parse the datetime column to a standard format
#     df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
#     print(f"After parsing datetime:\n{df.head()}")

#     # Identify and remove rows that correspond to DST transitions
#     day = df[datetime_col].dt.date
#     day_with_dst = day[mask].unique()

#     # Remove rows with DST transition hours
#     df = df.loc[~day.isin(day_with_dst)]
#     print(f"After removing DST transition rows:\n{df.head()}")

#     # Downsample the data to 15-minute intervals
#     df = downsample_to_15(df, value_cols, datetime_col)
#     print(f"After downsampling:\n{df.head()}")

#     # Remove DST days again after downsampling due to possible residuals
#     day = df[datetime_col].dt.date
#     df = df.loc[~day.isin(day_with_dst)]

#     return df


DST_PATTERN = r"\(CET\)|\(CEST\)"

def flag_dst_rows(df, datetime_col = "MTU (CET/CEST)") -> pd.Series:
    """Flag rows that correspond to DST transition"""
    mask = df[datetime_col].str.contains(DST_PATTERN, regex=True)
    return mask

def downsample_to_15(df, value_col, datetime_col = "MTU (CET/CEST)"):
    """Downsample to 15-min resolution by forward filling each hour into 4 slots."""
    interval = pd.to_datetime(df[datetime_col]).diff().mode()[0]
    if interval == pd.Timedelta(minutes=15):
        return df  # If already 15min, return the data as is
    
    group_col = df.columns.difference([datetime_col, value_col]).tolist()

    df = df.set_index(datetime_col)
    
    df_15min = (
        df.groupby(group_col)[value_col]
        .resample('15min')
        .ffill(limit=3)
        .reset_index()
    )

    return df_15min

# def downsample_to_15(df, value_cols, datetime_col = "MTU (CET/CEST)"):
#     """Downsample to 15-min resolution by forward filling each hour into 4 slots for multiple value columns."""
    
#     interval = pd.to_datetime(df[datetime_col]).diff().mode()[0]
#     if interval == pd.Timedelta(minutes=15):
#         return df  # If already 15min, return the data as is

#     group_col = df.columns.difference([datetime_col] + value_cols).tolist()  # Ensure datetime_col is treated as a list

#     df = df.set_index(datetime_col)
    
#     # Loop through each value column and apply the resampling
#     for value_col in value_cols:
#         df[value_col] = (
#             df.groupby(group_col)[value_col]
#             .resample('15min')
#             .ffill(limit=3)
#         )

#     df_15min = df.reset_index()
#     return df_15min

def setup_time(df, value_cols, datetime_col = "MTU (CET/CEST)", format = "mixed"):
    """Parse a datetime column, removing rows with DST transition hours."""
    print(f"Original dataframe:\n{df.head()}")
    
    mask = flag_dst_rows(df)

    df[datetime_col] = (df[datetime_col].str.split(' - ')
                        .str[0].str
                        .replace(DST_PATTERN, "", regex=True)
                        .str.strip())
    
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
    print(f"After parsing datetime:\n{df.head()}")

    day = df[datetime_col].dt.date
    day_with_dst = day[mask].unique()
    # print(f"Days with DST transitions:\n{day_with_dst}")
    
    # Remove rows with DST transition hours
    df = df.loc[~day.isin(day_with_dst)]
    # print(f"After removing DST transition rows:\n{df.head()}")
    
    df = downsample_to_15(df, value_cols)
    # print(f"After downsampling:\n{df.head()}")
    
    # Remove days with DST transition hours again after downsampling due to possible residuals
    day = df[datetime_col].dt.date
    df = df.loc[~day.isin(day_with_dst)]

    return df

df_gen = setup_time(pd.read_csv(gen_path_fr, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"]), "Generation (MW)")

print('FINISHED PROCESSING GENERATION DATA')
df_gen.to_csv('data/raw/fbmc/generation/gen_fr_15min.csv')

# df_gen_be = pd.read_csv(gen_path_be, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
# df_gen_be_15min = to_15min_constant(df_gen_be)
# df_gen_be_15min.to_csv('data/raw/fbmc/generation/gen_be_15min.csv')

# df_gen_fr = pd.read_csv(gen_path_fr, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
# df_gen_fr_15min = to_15min_constant(df_gen_fr)
# df_gen_fr_15min.to_csv('data/raw/fbmc/generation/gen_fr_15min.csv')


# Function to load and preprocess generation data
def load_generation_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        print(path)
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])

        df = setup_time(df, "Generation (MW)")

        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Generation (MW)": "Generation"})
        df = df.groupby(["Time", "Area"]).agg({"Generation": "sum"}).reset_index()
        print(df[8450:8460])
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

print('FINISHED PROCESSING LOAD DATA')

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
def create_master_dataset(gen_paths, load_paths, areas):
    df_gen = load_generation_data(gen_paths, areas)
    df_load = load_load_data(load_paths, areas)
    master_df = calculate_metrics(df_gen, df_load)
    return master_df


# Define paths and areas
gen_paths = [gen_path_be, gen_path_de, gen_path_fr, gen_path_nl]
load_paths = [load_path_be, load_path_de, load_path_fr, load_path_nl]
areas = ['BZN|BE', 'BZN|DE-LU', 'BZN|FR', 'BZN|NL']

# Create master dataset
master_dataset = create_master_dataset(gen_paths, load_paths, areas)
master_dataset.to_csv('data/clean/fbmc/master_dataset_15min.csv')

# Save or use the master dataset
print(master_dataset.head())