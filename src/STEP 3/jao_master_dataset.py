import pandas as pd
import glob
import os
from pathlib import Path
from pandas.errors import EmptyDataError
from jao_downsampling import upsample_by_repetition
from jao_dst import remove_dst_rows, drop_timezone_info, filter_year

jao_raw_folder = "../epex-spot-analysis/data/raw/jao/shadow_prices/2025/"
START_YEAR = 2024
END_YEAR = 2025

def concatenate_files(folder_path, diff = ","):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dataframes = []

    for f in all_files:
        try:
            if os.path.getsize(f) == 0:
                print(f"Skipped empty file: {f}")
                continue
            dataframes.append(pd.read_csv(f, sep=diff))
        except EmptyDataError:
            print(f"Skipped invalid file: {f}")

    if not dataframes:
        raise ValueError("No valid files to concatenate.")

    return pd.concat(dataframes, ignore_index=True)


def keep_relevant_columns(df):
    df = df.copy()
    relevant_columns = [
        "dateTimeUtc",
        "cnecName",
        "ram",
        "hubFrom",
        "hubTo",
        "hub_BE",
        "hub_FR",
        "hub_DE",
        "hub_NL",
    ]
    df = df[relevant_columns]

    df.rename(columns={
        "dateTimeUtc": "Time",
        "cnecName": "CNEC Name",
        "ram": "RAM",
        "hub_BE": "BE",
        "hub_FR": "FR",
        "hub_DE": "DE",
        "hub_NL": "NL",
    }, inplace=True)
    print(df.columns)

    return df


def sort_time(df, datetime_col='Time'):
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.sort_values(datetime_col).reset_index(drop=True)

    return df


df_jao = concatenate_files(jao_raw_folder)
df_jao_clean = keep_relevant_columns(df_jao)
df_jao_clean = sort_time(df_jao_clean, 'Time')

# Upsampling by repetition (to get 15-min frequency)
df_jao_clean = upsample_by_repetition(df_jao_clean)

# DST handling
df_jao_clean = remove_dst_rows(df_jao_clean)
df_jao_clean = drop_timezone_info(df_jao_clean)
df_jao_clean = filter_year(df_jao_clean, START_YEAR, END_YEAR)

output_path = "../epex-spot-analysis/data/clean/STEP 3/shadow_prices/jao_clean_noDST_2025.csv"
df_jao_clean.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")