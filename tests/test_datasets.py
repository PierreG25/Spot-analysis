import pandas as pd


def check_time_resolution(df, datetime_col, resolution = '15min'):
    """Check if the datetime column has the expected time resolution"""
    times = pd.to_datetime(df[datetime_col])
    times = times.drop_duplicates().sort_values()

    prev = times.iloc[:-1].reset_index(drop=True)
    next = times.iloc[1:].reset_index(drop=True)

    diffs = next - prev
    expected_diff = pd.to_timedelta(resolution)

    is_15min = (diffs == expected_diff)
    crosses_day = (prev.dt.date != next.dt.date)
    not_15min_multiple = (diffs % expected_diff != pd.Timedelta(0))

    invalid = ~(is_15min | (crosses_day & not_15min_multiple))

    if invalid.any():
        invalid_times = pd.DataFrame({
            'previous_time': prev[invalid],
            'next_time': next[invalid],
            'difference': diffs[invalid]
        })
        raise ValueError(f"Datetime column '{datetime_col}' does not have the expected {resolution} resolution at the following intervals:\n{invalid_times}")
    print(f"All timestamps in column '{datetime_col}' have the expected {resolution} resolution.")


def check_no_dst_days(df, datetime_col):
    """Check that no rows correspond to DST transition days"""
    remove_days = pd.to_datetime(["2025-03-30", "2025-10-26"]).date
    mask = df[datetime_col].dt.date.isin(remove_days)

    if mask.any():
        dst_rows = df.loc[mask, datetime_col]
        raise ValueError(f"DataFrame contains rows corresponding to DST transition days in column '{datetime_col}':\n{dst_rows}")
    print(f"No rows in column '{datetime_col}' correspond to DST transition days.")


# ======================== TESTS ======================== #
path = '../data/clean/jao/shadow_prices/2025/jao_master_data_set_clean.csv'
df = pd.read_csv(path, parse_dates=['Time'])

check_time_resolution(df, 'Time')
check_no_dst_days(df, 'Time')