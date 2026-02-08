import pandas as pd
import glob
import os
from pathlib import Path

jao_raw_folder = "../epex-spot-analysis/data/raw/jao/shadow_prices/2025/"

def concatenate_files(folder_path, diff = ","):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    df = pd.concat((pd.read_csv(f, sep = diff) for f in all_files), ignore_index=True)
    print(f"Concatenated {len(all_files)}")

    return df

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
        "hub_BE": "BZN|BE",
        "hub_FR": "BZN|FR",
        "hub_DE": "BZN|DE-LU",
        "hub_NL": "BZN|NL",
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


output_path = "../epex-spot-analysis/data/clean/jao/shadow_prices/2025/jao_not_downsampled_2025.csv"
df_jao_clean.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")

# df_2025_01_01 = concatenate_files("../epex-spot-analysis/data/shadowPrices 2025-01-01 0000 - 2025-01-01 2300", diff=";")
# df_2025_01_01["DateTimeUtc"] = pd.to_datetime(df_2025_01_01["DateTimeUtc"])
# df_2025_01_01.sort_values("DateTimeUtc", inplace=True)
# df_2025_01_01.to_csv("../epex-spot-analysis/data/clean/jao/shadow_prices/2025/shadowPrices_test_2025_01_01.csv", index=False)