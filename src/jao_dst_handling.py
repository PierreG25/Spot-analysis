import pandas as pd

def remove_dst_rows(df, datetime_col = "Time"):
    remove_days = pd.to_datetime(["2025-03-30", "2025-10-26"]).date
    df_without_dst =df[~df[datetime_col].dt.date.isin(remove_days)]
    print("Removing DST days:", remove_days)

    return df_without_dst


def drop_timezone_info(df, datetime_col = "Time"):
    """Remove timezone information from a datetime column."""
    df[datetime_col] = df[datetime_col].dt.tz_localize(None)
    return df