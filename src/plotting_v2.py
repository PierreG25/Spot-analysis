import matplotlib.pyplot as plt
import pandas as pd

def format_period(period, year):
    """
    Concatenate a period with a respective year.
    For example:
    - format_period(summer, 2025) returns 'summer-2025'
    - format_period(('01-02', '05-26'), 2025) returns ('2025-01-02', '2025-05-26')
    """
    year = str(year)

    # If period is a tuple of two MM-DD strings
    if isinstance(period, tuple) and len(period) == 2:
        start, end = period
        return (f"{year}-{start}", f"{year}-{end}")
    
    # If period is a season (string)
    elif isinstance(period, str):
        return f"{period}-{year}"
    
    else:
        raise ValueError("Invalid period format. Must be ('MM-DD', 'MM-DD') or 'season'.")


def standardize_time_period(period):
    """
    Function converting a time period into the right format for further analysis
    
    Accepts either:
    - A season name and year: one of "winter-YYYY", "spring-YYYY", "summer-YYYY", "autumn-YYYY"
    - A date range tuple: ('YYYY-MM-DD', 'YYYY-MM-DD')

    Returns a tuple of (start_date, end_date)
    
    Examples:
        process_period("summer-2024")
        process_period(("2024-06-01", "2024-09-01"))
    """
    season_map = {
        'spring': (lambda y: (pd.Timestamp(y, 3, 20).normalize(), pd.Timestamp(y, 6, 20).normalize())),
        'summer': (lambda y: (pd.Timestamp(y, 6, 21).normalize(), pd.Timestamp(y, 9, 22).normalize())),
        'autumn': (lambda y: (pd.Timestamp(y, 9, 23).normalize(), pd.Timestamp(y, 12, 20).normalize())),
        'winter': (lambda y: (pd.Timestamp(y, 12, 21).normalize(), pd.Timestamp(y+1, 3, 19).normalize()))
    }

    if isinstance(period, str):
        season, year = period.lower().split('-')
        year = int(year)
        period_func = season_map[season]

        if season in season_map:
            return period_func(year)
        raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")


def plot_avg_hourly_prices(df, start_year, end_year, period):
    """
    Function plotting the average hourly day ahead prices over a certain period of time
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for year in range(start_year, end_year+1):
        period_yearly = format_period(period, year)
        (start, end) = standardize_time_period(period_yearly)

        df_yearly = df[(df.index >= start) & (df.index <= end)]
        daily_avg = df_yearly.resample()

        x = daily_avg['Hour']
        y = daily_avg['Day-ahead Price (EUR/MWh)']
        ax.plot(x,y, label=str(year), alpha=0.8)
        ax.scatter(x,y, marker='+', color='r')
        print(f'{year} OK')

    ax.set_xlabel('Hours')
    ax.set_ylabel('Day-ahead prices')
    ax.set_title(f'Average day-ahead hourly prices from {start_year} to {end_year}')
    ax.legend()
    ax.grid(True)
    plt.show()