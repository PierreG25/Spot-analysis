import pandas as pd
import numpy as np
from fbmc_master import setup_time

gen_path_fr_2025 = 'data/raw/fbmc/generation/2025_generation_fr_raw.csv'
gen_path_fr_2024 = 'data/raw/fbmc/generation/2024_generation_fr_raw.csv'

df_gen_fr_2025 = pd.read_csv(gen_path_fr_2025)
df_gen_fr_2024 = pd.read_csv(gen_path_fr_2024)

df_gen_fr = pd.concat([df_gen_fr_2024, df_gen_fr_2025], ignore_index=True)

df_gen_fr = setup_time(df_gen_fr, "Generation (MW)")


def non_dispatchable_gen(df, non_dispatch_col = 'Renewable' , types = ["Other renewable",
                                        "Solar",
                                        "Marine",
                                        "Wind Offshore",
                                        "Wind Onshore",
                                        "Hydro Run-of-river and pondage"]):
    """
    Aggregate non-dispatchable generation 
    types into a single 'Renewable' column.
    """

    df = df.copy()
    df[non_dispatch_col] = df[types].sum(axis=1)
    df.drop(columns=types, inplace=True)

    return df


def build_generation_table(
    dataframe,
    datetime_col="MTU (CET/CEST)",
    production_type_col="Production Type",
    generation_col="Generation (MW)",
    fill_value=None,
    non_dispatch=True
):
    """
    Transform a long generation dataset into a wide yearly 15-minute dataset.

    Parameters
    ----------
    data : str or pd.DataFrame
        CSV path or already-loaded DataFrame.
    timestamp_col : str
        Column containing interval strings such as:
        '01/01/2025 00:00:00'
    production_type_col : str
        Column containing production types.
    generation_col : str
        Column containing generation values.
    fill_value : scalar or None
        Value used to fill missing entries after pivoting/reindexing.
        Keep as None to preserve NaN.

    Returns
    -------
    production_types : list[str]
        Sorted unique production types.
    wide_df : pd.DataFrame
        DataFrame with one row per timestamp and one column per production type.
    """
    df = dataframe.copy()


    # Convert generation to numeric; non-numeric values like 'n/e' become NaN
    df[generation_col] = pd.to_numeric(df[generation_col], errors="coerce")

    # Extract unique production types
    production_types = sorted(df[production_type_col].dropna().unique().tolist())
    print(f"Unique production types: {production_types}")

    # Group and pivot to wide format
    # min_count=1 preserves NaN when all values in a group are missing
    wide_df = (
        df.groupby([datetime_col, production_type_col])[generation_col]
        .sum(min_count=1)
        .unstack(production_type_col)
        .sort_index()
    )

    if fill_value is not None:
        wide_df = wide_df.fillna(fill_value)

    wide_df.columns.name = None
    wide_df = wide_df.reset_index()

    # Stable column order
    wide_df = wide_df[[datetime_col] + production_types]
    wide_df.rename(columns={datetime_col: "Time"}, inplace=True)    

    if non_dispatch:
        wide_df = non_dispatchable_gen(wide_df)

    wide_df.to_csv('data/clean/STEP 3/XGBoost/generation_by_type_fr.csv', index=False)

    return wide_df

if __name__ == "__main__":
    build_generation_table(df_gen_fr)