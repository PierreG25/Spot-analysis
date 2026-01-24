import pandas as pd
import re
import numpy as np


### EXTRACT PTDF COEFFICIENTS FROM CSV FILE ###

gen_path_be = 'data/raw/fmbc/generation/2025_generation_be_raw.csv'
gen_path_de = 'data/raw/fmbc/generation/2025_generation_de_raw.csv'
gen_path_fr = 'data/raw/fmbc/generation/2025_generation_fr_raw.csv'
gen_path_nl = 'data/raw/fmbc/generation/2025_generation_nl_raw.csv'

load_path_be = 'data/raw/fmbc/load/2025_load_be_raw.csv'
load_path_de = 'data/raw/fmbc/load/2025_load_de_raw.csv'
load_path_fr = 'data/raw/fmbc/load/2025_load_fr_raw.csv'
load_path_nl = 'data/raw/fmbc/load/2025_load_nl_raw.csv'

price_path_be = 'data/raw/fmbc/price/2025_price_be_raw.csv'
price_path_de = 'data/raw/fmbc/price/2025_price_de_raw.csv'
price_path_fr = 'data/raw/fmbc/price/2025_price_fr_raw.csv'
price_path_nl = 'data/raw/fmbc/price/2025_price_nl_raw.csv'


### STANDARDIZE DATASETS TO 15MIN RESOLUTION + WINTER/SUMMER TIME ###

import pandas as pd
import re

MTU_COL = "MTU (CET/CEST)"
GEN_COL = "Generation (MW)"

_TZ_RE = re.compile(r"\s*\((?:CET|CEST)\)\s*")

def parse_mtu_start_end(mtu: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse strings like:
      "01/01/2025 00:00:00 - 01/01/2025 01:00:00"
      "30/03/2025 01:00:00 (CET) - 30/03/2025 03:00:00 (CEST)"
    Strips "(CET)/(CEST)" labels and reads as dayfirst timestamps.
    """
    a, b = [x.strip() for x in str(mtu).split(" - ", 1)]
    a = _TZ_RE.sub("", a).strip()
    b = _TZ_RE.sub("", b).strip()
    start = pd.to_datetime(a, dayfirst=True)
    end = pd.to_datetime(b, dayfirst=True)
    return start, end

def normalize_mtu_dst(
    df_in: pd.DataFrame,
    mtu_col: str = MTU_COL,
    out_col: str = MTU_COL,
) -> pd.DataFrame:
    """
    DST/summer-winter time cleanup for any dataset:
    - Parses MTU interval strings (with or without CET/CEST tokens)
    - Produces a consistent "dd/mm/YYYY HH:MM:SS - dd/mm/YYYY HH:MM:SS" format
    - Adds 'start'/'end' datetime columns (naive, tz labels removed)

    Works regardless of the original resolution (15-min, hourly, 2-hour DST jump, etc.).
    """
    df = df_in.copy()

    se = df[mtu_col].apply(parse_mtu_start_end)
    df["start"] = se.apply(lambda x: x[0])
    df["end"] = se.apply(lambda x: x[1])

    df[out_col] = (
        df["start"].dt.strftime("%d/%m/%Y %H:%M:%S")
        + " - "
        + df["end"].dt.strftime("%d/%m/%Y %H:%M:%S")
    )
    return df

def to_15min_constant(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MTU intervals to 15-min intervals by repeating the same value
    for each 15-min slot covered by the interval (end is exclusive).
    Handles DST jump intervals as long as 'start'/'end' are correct.
    """
    # 1) normalize MTU strings + get start/end (DST-safe normalization)
    df = normalize_mtu_dst(df_in, mtu_col=MTU_COL, out_col=MTU_COL)
    df.to_csv('data/raw/fmbc/generation/gen_debug_before_cleanup.csv')

    # 2) numeric cleanup
    df[GEN_COL] = (
        df[GEN_COL]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df[GEN_COL] = pd.to_numeric(df[GEN_COL], errors="coerce")

    # 3) build 15-min starts for each row (end is exclusive)
    df["starts_15m"] = df.apply(
        lambda r: pd.date_range(r["start"], r["end"], freq="15min", inclusive="left"),
        axis=1,
    )

    # 4) explode into rows
    out = df.explode("starts_15m").rename(columns={"starts_15m": "start15"})

    # 5) rebuild MTU (15-min)
    out["end15"] = out["start15"] + pd.Timedelta(minutes=15)
    out[MTU_COL] = (
        out["start15"].dt.strftime("%d/%m/%Y %H:%M:%S")
        + " - "
        + out["end15"].dt.strftime("%d/%m/%Y %H:%M:%S")
    )

    return out[[MTU_COL, "Area", "Production Type", GEN_COL]].reset_index(drop=True)

df_gen_be = pd.read_csv(gen_path_be, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
df_gen_be_15min = to_15min_constant(df_gen_be)
df_gen_be_15min.to_csv('data/raw/fmbc/generation/gen_be_15min.csv')

df_gen_fr = pd.read_csv(gen_path_fr, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
df_gen_fr_15min = to_15min_constant(df_gen_fr)
df_gen_fr_15min.to_csv('data/raw/fmbc/generation/gen_fr_15min.csv')

print('FINISHED PROCESSING GENERATION DATA')

# Function to load and preprocess generation data
def load_generation_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        print(path)
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
        print(df.head())
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Generation (MW)": "Generation"})
        df = df.groupby(["Time", "Area"]).agg({"Generation": "sum"}).reset_index()
        print(df.head())
        df['Area'] = area_name
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Function to load and preprocess load data
def load_load_data(paths, areas):
    data_frames = []
    for path, area_name in zip(paths, areas):
        df = pd.read_csv(path, na_values=["N/A", "n/a", "NA", "-", "", " ", "  ", "\t", "n/e"])
        df = df.rename(columns={"MTU (CET/CEST)": "Time", "Actual Total Load (MW)": "Total load"})
        df = df[["Time", "Area", "Total load"]]
        df['Area'] = area_name
        print(df.head())
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Function to calculate net position and renewable share
def calculate_metrics(df_gen, df_load):
    df_gen.to_csv('data/raw/fmbc/generation/gen_debug.csv')
    df_load.to_csv('data/raw/fmbc/load/load_debug.csv')
    df = pd.merge(df_gen, df_load, on=['Time', 'Area'], suffixes=('_gen', '_load'))
    df['Net position'] = df['Generation'] - df['Total load']
    df['Renewable share'] = 0  # Placeholder as renewable data is not provided
    df['import/export flag'] = df['Net position'].apply(lambda x: 'import' if x < 0 else 'export')
    print(df.head())
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

# Save or use the master dataset
print(master_dataset.head())


