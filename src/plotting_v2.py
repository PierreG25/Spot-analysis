import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

def format_period(period, year):
    """
    Concatenate a period with a respective year.
    For example:
    - format_period(summer, 2025) returns 'summer-2025'
    - format_period(('01-02', '05-26'), 2025) returns ('2025-01-02', '2025-05-26')
    """

    season_map = {
        'spring': (lambda y: (f'{y}/3/20', f'{y}/6/20')),
        'summer': (lambda y: (f'{y}/6/21', f'{y}/9/22')),
        'autumn': (lambda y: (f'{y}/9/23', f'{y}/12/20')),
        'winter': (lambda y: (f'{y}/12/21', f'{y+1}/3/19'))
    }

    # If period is a tuple of two MM-DD strings
    if isinstance(period, tuple) and len(period) == 2:
        start, end = period
        return (f"{year}/{start}", f"{year}/{end}")
    
    # If period is a season (string)
    elif isinstance(period, str):
        year = int(year)
        period_func = season_map[period]

        if period in season_map:
            return period_func(year)
        raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")
    
    raise ValueError("Invalid period format")


def rolling_mean(df, start, end, wd):
    mask = (df.index >= start) & (df.index <= end)
    df_filtered = df[mask]
    daily_avg = df_filtered['Day-ahead Price (EUR/MWh)'].resample('D').mean()
    return (df_filtered['Date'].unique(), daily_avg.rolling(window = wd, center = True).mean())

def split_period(period):
    if isinstance(period,tuple):
        return f'{period[0]} - {period[1]}'
    return period

def extract_periodic_data(df, start_year, end_year, filter_period):
    filtered = []
    for year in range(start_year, end_year + 1):
        start_date, end_date = format_period(filter_period, year)
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered.append(df.loc[mask])
    return pd.concat(filtered)

########################################## Plots Functions ########################################################

######### Time serie plot

def plot_smooth_prices(df, start, end, window_days):
    df = df.set_index('MTU (CET/CEST)')
    x, y = rolling_mean(df, start, end, window_days)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(x, y)

    ax.set_xlabel('Dates')
    ax.set_ylabel('Day-ahead prices (EUR/MWh)')
    ax.set_title(f'Daily Average Electricity Prices ({start} - {end})\nwith {window_days}-Day Rolling Mean')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


######### Average hourly prices plot

def plot_avg_hourly_prices(df, start_year, end_year, period):
    """
    Function plotting the average hourly day ahead prices over a certain period of time
    """
    df = df.set_index('MTU (CET/CEST)')
    fig, ax = plt.subplots(figsize=(12, 6))

    for year in range(start_year, end_year+1):
        (start, end) = format_period(period, year)
        mask = (df.index >= start) & (df.index <= end)
        
        df_yearly = df[mask]
        daily_avg = df_yearly.groupby('Hour')['Day-ahead Price (EUR/MWh)'].mean().reset_index()

        x = daily_avg['Hour']
        y = daily_avg['Day-ahead Price (EUR/MWh)']
        ax.plot(x,y, label=str(year), alpha=0.8)
        ax.scatter(x,y, marker='+', color ='k', zorder=3)
        print(f'{year} OK')

    ax.set_xlabel('Hours')
    ax.set_ylabel('Day-ahead prices (EUR/MWh)')
    ax.set_title(f'Average day-ahead hourly prices ({start_year} - {end_year}) \nwithin {split_period(period)}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


######### Boxplot

def plot_boxplots_by_weekday(df, start_year, end_year, period):
    """
    Create boxplots of prices by day of the week, grouped by year.

    Parameters:
    - df: pandas DataFrame with datetime and price columns
    - start_year, end_year: int, range of years to include
    - datetime_column: str, name of datetime column
    - price_column: str, name of price column
    """

    # Convert to datetime and filter by year
    df = df.set_index('MTU (CET/CEST)')
    df = extract_periodic_data(df, start_year, end_year, period)

    # Aggregate to daily average price
    df_daily = df['Day-ahead Price (EUR/MWh)'].resample('D').mean().to_frame()
    df_daily['weekday'] = df_daily.index.day_name()
    df_daily['year'] = df_daily.index.year

    # Ensure consistent weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_daily['weekday'] = pd.Categorical(df_daily['weekday'], categories=weekday_order, ordered=True)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_daily, x='weekday', y='Day-ahead Price (EUR/MWh)', hue='year')

    plt.title(f"Electricity Prices by Day of the Week ({start_year} - {end_year}) \nwithin {split_period(period)}")
    plt.xlabel("Day of the Week")
    plt.ylabel("Average Daily Price (EUR/MWh)")
    plt.legend(title="Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


######### Heat map

def plot_heatmap(df, start_year, end_year, period):
    # Convert to datetime and filter by year
    df = df.set_index('MTU (CET/CEST)')
    df_concat = extract_periodic_data(df, start_year, end_year, period)

    df_concat['weekday'] = df_concat.index.day_name()
    df_concat['hour'] = df_concat.index.hour

    # Create pivot table of average prices
    heatmap_data = df_concat.pivot_table(
        index='weekday', columns='hour', values='Day-ahead Price (EUR/MWh)', aggfunc='mean'
    )

    # Ensure correct weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(weekday_order)

    # Plot the heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Average Price (EUR/MWh)'})
    plt.title(f"Average Electricity Price by Weekday and Hour ({start_year} - {end_year}) \nwithin {split_period(period)}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.show()