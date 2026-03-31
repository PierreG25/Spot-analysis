import pandas as pd

def remove_dst_rows(df, datetime_col = "Time"):
    df = df.copy()
    remove_days = pd.to_datetime(["2024-03-31", "2024-10-27", "2025-03-30", "2025-10-26"]).date
    df_without_dst =df[~df[datetime_col].dt.date.isin(remove_days)]
    print("Removing DST days:", remove_days)

    return df_without_dst


def drop_timezone_info(df, datetime_col = "Time"):
    """Remove timezone information from a datetime column."""
    df = df.copy()
    df[datetime_col] = df[datetime_col].dt.tz_localize(None)
    return df

def filter_year(df, start_year, end_year, datetime_col = "Time"):
    """Filter dataframe to only include rows from a specific year."""
    df = df.copy()

    df_filtered = df[df[datetime_col].dt.year.isin(range(start_year, end_year + 1))]
    print(f"Filtering for years between {start_year} and {end_year}")

    return df_filtered