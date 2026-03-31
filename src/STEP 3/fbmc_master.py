import pandas as pd
import numpy as np
from pathlib import Path


# ============= DEFINE PATHS ============= #

BASE_DIR = Path('data/raw/fbmc')
COUNTRIES = ['be', 'de', 'fr', 'nl', 'es']
YEARS = [2024, 2025]
DATA_TYPES = ['generation', 'load', 'price']

def get_paths(data_type, country, year):
    """Retourne les chemins pour un type de données, triés par année puis par pays."""
    return BASE_DIR / data_type / f'{year}_{data_type}_{country}_raw.csv'
    

# Utilisation
# gen_paths   = get_paths('generation')
# load_paths  = get_paths('load')
# price_paths = get_paths('price')

areas = ['BZN|BE', 'BZN|DE-LU', 'BZN|FR', 'BZN|NL', 'BZN|ES']

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

def downsample_to_15(df, value_col, datetime_col = "MTU (CET/CEST)", group_cols=None):
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

    if interval == pd.Timedelta("15min"):
        return df  # If already 15min, return the df as is

    if group_cols is None:
        candidate_cols = ["Area", "Production Type", "Sequence"]
        group_cols = [c for c in candidate_cols if c in df.columns]

    print(f"group_cols: {group_cols}")
    # This creates a 'dummy' row at the very end of each group (e.g., 00:00 of the next day)
    # so that resample('15min') is forced to fill the slots in between.
    group_maxes = df.groupby(group_cols)[datetime_col].max().reset_index()
    group_maxes[datetime_col] = group_maxes[datetime_col] + interval
    
    df_extended = pd.concat([df, group_maxes], ignore_index=True)

    # Calculate fill limit dynamically (e.g., 1h -> limit 3; 30m -> limit 1)
    ffill_limit = int(interval / pd.Timedelta('15min')) - 1

    df_extended = df_extended.set_index(datetime_col)
    
    df_15min = (
        df_extended.groupby(group_cols)[value_col]
        .resample('15min')
        .ffill(limit=ffill_limit)
        .reset_index()
    )
    print(f"df_15min columns after resample: {df_15min.columns}")

    # The dummy record is always the last row of each resampled group
    df_15min = df_15min.drop(df_15min.groupby(group_cols).tail(1).index)

    df_15min.to_csv('data/debug/df_after_downsample_debug.csv')
    return df_15min


def setup_time(df, value_cols, datetime_col = "MTU (CET/CEST)", format = "mixed"):
    """Parse a datetime column, removing rows with DST transition hours."""
    
    mask = flag_dst_rows(df)

    df[datetime_col] = (df[datetime_col].str.split(' - ')
                        .str[0].str
                        .replace(DST_PATTERN, "", regex=True)
                        .str.strip())
    
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)

    day = df[datetime_col].dt.date
    day_with_dst = day[mask].unique()
    
    # Remove rows with DST transition hours
    df = df.loc[~day.isin(day_with_dst)]
    print(f"df_setup_time columns: {df.columns}")
    df = downsample_to_15(df, value_cols)

    day = df[datetime_col].dt.date
    df = df.loc[~day.isin(day_with_dst)]
    
    # Remove days with DST transition hours again after downsampling due to possible residuals

    return df


# ============= LOAD AND PREPROCESS df FUNCTIONS ============= #
# Function to load and preprocess generation df

def _load_data(data_type: str, years: list[int], countries: list[str], transform_fn) -> pd.DataFrame:
    """Fonction générique : lit et concatène les années, puis applique la transformation par pays."""
    data_frames = []
    for country in countries:
        yearly_dfs = [
            pd.read_csv(
                get_paths(data_type, country, year),
                na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"]
            )
            for year in years
        ]
        df = pd.concat(yearly_dfs, ignore_index=True)
        df = transform_fn(df, country)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def load_generation_data(years: list[int] = YEARS, countries: list[str] = COUNTRIES) -> pd.DataFrame:
    def transform(df, country):
        df = setup_time(df, "Generation (MW)")
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Generation (MW)": "Generation"})
        df = df.groupby(["Time", "Area"]).agg({"Generation": "sum"}).reset_index()
        df.to_csv(f'data/debug/gen_debug_{country}.csv')
        return df
    return _load_data('generation', years, countries, transform)


def load_load_data(years: list[int] = YEARS, countries: list[str] = COUNTRIES) -> pd.DataFrame:
    def transform(df, country):
        df.drop(columns=['Day-ahead Total Load Forecast (MW)'], inplace=True)
        df = setup_time(df, "Actual Total Load (MW)")
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Actual Total Load (MW)": "Total load"})
        df = df[["Time", "Area", "Total load"]]
        df.to_csv(f'data/debug/load_debug_{country}.csv')
        return df
    return _load_data('load', years, countries, transform)


def load_price_data(years: list[int] = YEARS, countries: list[str] = COUNTRIES) -> pd.DataFrame:
    def transform(df, country):
        print(f"df_load_price_data columns for {country}: {df.columns.tolist()}")
        df = sequence_selection(df)
        df = setup_time(df, "Day-ahead Price (EUR/MWh)")
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Day-ahead Price (EUR/MWh)": "Price"})
        return df[["Time", "Area", "Price"]]
    return _load_data('price', years, countries, transform)


# ============= CALCULATE METRICS AND CREATE MASTER DATASET ============= #
# Function to calculate net position and renewable share

def calculate_metrics(df_gen, df_load):
    df = pd.merge(df_gen, df_load, on=['Time', 'Area'], suffixes=('_gen', '_load'))
    df['Net position'] = df['Generation'] - df['Total load']
    return df

# Main function to create master dataset
def create_master_dataset(areas):
    df_gen = load_generation_data()
    df_load = load_load_data()
    df_price = load_price_data()

    df_gen.to_csv('data/debug/gen_debug')
    df_load.to_csv('data/debug/load_debug')
    df_price.to_csv('data/debug/price_debug')

    master_df = calculate_metrics(df_gen, df_load)
    master_df = pd.merge(master_df, df_price, on=['Time', 'Area'], how='left')

    print(f"Areas in master_df before cleaning: {master_df['Area'].unique()}")
    master_df['Area'] = master_df['Area'].str.strip().str.split('|').str[1].str.split('-').str[0]

    return master_df


def build_country_spreads(
    df,
    epsilon = 0.1,
    ref_country = "FR",
    countries_code = ["BE", "DE", "NL", "ES"],
):
    """
    Return the dataset filtered on `country` with added spread columns
    versus each country in `countries_code`.

    Assumptions:
    - df contains columns: 'Time', 'Area', 'Price'
    - spread is computed as: Price(country) - Price(other_country)
    """

    data = df.copy()

    # Optional cleanup for exported CSVs
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")

    data["Time"] = pd.to_datetime(data["Time"])

    # Keep only the requested countries
    selected_areas = [ref_country] + countries_code
    data = data[data["Area"].isin(selected_areas)]

    # Wide format: one Price column per country, indexed by Time
    wide = (
        data.pivot_table(index="Time", columns="Area", values="Price", aggfunc="first")
        .rename_axis(None, axis=1)
        .reset_index()
    )

    # Keep only the base country rows from the original dataset
    result = data[data["Area"] == ref_country].copy()

    # Merge comparison country prices onto the base country rows
    cols_to_merge = ["Time"] + countries_code
    result = result.merge(wide[cols_to_merge], on="Time", how="left")

    # Rename merged country price columns and compute spreads
    for code in countries_code:
        result = result.rename(columns={code: f"Price_{code}"})
        result[f"spread_{ref_country}_{code}"] = result["Price"] - result[f"Price_{code}"]
        result['congestion_' + ref_country + '_' + code] = result[f"spread_{ref_country}_{code}"].abs() >= epsilon
        
        result.drop(columns=[f"Price_{code}"], inplace=True)

    result.drop(columns=["Area"], inplace=True)  # Drop the 'Area' column as it's now redundant

    return result


# Create NP dataset
if __name__ == "__main__":
    master_dataset = create_master_dataset(areas)
    master_dataset.to_csv('data/clean/STEP 3/NP_by_country.csv')

    filtered_master_dataset = build_country_spreads(master_dataset)
    filtered_master_dataset.to_csv('data/clean/STEP 3/XGBoost/NP_by_country_FR.csv', index=False)

    print("Master dataset created and saved to 'data/clean/STEP 3/NP_by_country.csv'")