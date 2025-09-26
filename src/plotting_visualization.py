import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd

# ================================= HELPING FUNCTIONS =================================

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

        if period in season_map:
            return season_map[period](year)
        raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")
    
    raise ValueError("Invalid period format")


def get_season(month: int) -> str:
    """
    Return the season for a given month number

    Args:
        month (int): Month number

    Returns:
        str: Name of the season
    """
    if month not in range(1, 13):
        raise ValueError("Month must be an integer between 1 and 12.")
    
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
    

def ensure_datetime_index(df: pd.DataFrame, datetime_col: str ="Date") -> pd.DataFrame:
    """
    Ensure that the DataFrame has a datetime index named datetime_col

    Args:
        df (pd.DataFrame): Input DataFrame
        datetime_col (str): Name of the datetime column to set as index if not already

    Returns:
        pd.DataFrame: DataFrame indexed by datetime_col
    """
    if df.index.name != datetime_col:
        df = df.set_index(datetime_col)
    return df


def rolling_mean(df: pd.DataFrame, col: str, wd: int) -> pd.Series:
    """
    Compute a centered rolling mean over a daily resampled version of a column

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index
        col (str): Name of the column to compute the rolling mean on
        wd (int): Size of the rolling window (in days)
    
    Returns:
        pd.Series: A Series containing the smoothed daily values

    Notes:
        - The DataFrame contains a DatetimeIndex
    """
    daily_avg = df[col].resample('D').mean()
    return daily_avg.rolling(window = wd, center = True).mean()


def rolling_std(df, col, wd):
    """
    Compute a centered rolling standard deviation over a daily resampled version of a column

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex
        col (str): Name of the column to compute the rolling mean on
        wd (int): Size of the rolling window (in days)
    
    Returns:
        pd.Series: A Series containing the smoothed daily standard deviation values

    Notes:
        - The DataFrame contains a DatetimeIndex
    """
    daily_avg = df[col].resample('D').mean()
    return daily_avg.rolling(window = wd, center = True).std()


def split_period(period) -> str:
    """
    Convert a period value into a string representation

    If the input is a tuple (start, end), it returns a string formatted as "start - end".
    Otherwise, it returns the input unchanged

    Args:
        period (tuple or str): The period to process. Typically either:
            - A tuple (start, end) representing the range
            - A preformatted string
    
    Returns:
        str: String representation of the period
    """
    if isinstance(period,tuple):
        return f'{period[0]} - {period[1]}'
    return period


def filter_data(df, start_date, end_date):
    """
    Filter a DataFrame to include only rows within a specified date range

    The DataFrame must have a DatetimeIndex. Rows with the index values greater than
    or equal to 'start_date' and strictly less than 'end_date' are included

    Args:
        df (pd.DataFrame): Input DataFrame with DatetimeIndex
        start_date (str or pd.Timestamp): Start date of the filtering range (inclusive)
        end_date (str or pd>Timestamp): End date of the filtering range (exclusive)
    
    Returns:
        pd.DataFrame: Filtered DataFrame

    Notes:
        - The DataFrame contains a DatetimeIndex
    """
    mask = (df.index >= start_date) & (df.index < end_date)
    return df.loc[mask]


def extract_periodic_data(df, start_year, end_year, filter_period):
    """
    Extract and concatenate data slices from a DataFrame for repeated periods over
    multiple years

    For each year in [start_year, end_year], this function:
        - Uses `filter_period` and the year to get start and end dates (via `format_period`)
        - Filters the DataFrame for that period
        - Collects the filtered slices and concatenates them
    
    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex
        start_year (int): First year (inclusive)
        end_year (int): Last year (inclusive)
        filter_period (str or tuple of str):
            - If str, must be one of ['spring', 'summer', 'autumn', 'winter']
            - If tuple, should be (start_MM-DD, end_MM-DD)

    Returns:
        pd.DataFrame: Concatenated DataFrame containing filtered data for all
            specified years
    
    Notes:
        - The DataFrame contains a DatetimeIndex
    """
    filtered = []
    for year in range(start_year, end_year + 1):
        start_date, end_date = format_period(filter_period, year)
        mask = (df.index >= start_date) & (df.index < end_date)
        filtered.append(df.loc[mask])
    return pd.concat(filtered)


def extract_OHLC(df, col, index):
    """
    Extract OHLC (Open, High, Low, Close) aggreagated values for a specified column,
    goruped by the given index

    Args:
        df (pd.DataFrame): Input DataFrame
        col (str): Name of the column to aggregate
        index (str): Name of the column or index level to grouo by

    Returns:
        pd.DataFrame: DataFrame with columns ['Open', 'High', 'Low', 'Close] indexed by index
    """
    df_OHLC = df.groupby(index).agg(
        Open=(col,'first'),
        High=(col, 'max'),
        Low=(col, 'min'),
        Close=(col,'last')
    )
    return df_OHLC


def shift_date(date, x_days):
    """
    Sift a date string by a specified number of days

    Args:
        date (str): Date string in the format 'YYYY/MM/DD'
        x_days (int): Number of days to shift the date by (can be negative)
    
    Returns:
        str: New date sring shifted by x_days
    """
    date_obj = datetime.strptime(date, '%Y/%m/%d')
    # Shift the date by x_days using timedelta
    shifted_date = date_obj + timedelta(days=x_days)
    # Format the shifted date back into string
    return shifted_date.strftime('%Y/%m/%d')


# ================================= PRICES VISUALIZATION =================================

# ============= Time serie plot

def plot_smooth_prices(df, start, end, window_days, col='Price', raw_values=True, inversed_style=True):
    start_extended=shift_date(start, -window_days)
    end_extended=shift_date(end, window_days)

    print(start_extended)
    print(end_extended)

    df = ensure_datetime_index(df)
    df = filter_data(df, start_extended, end_extended)
    df['day'] = df.index.floor('D')

    y = rolling_mean(df, col, window_days)[window_days:-(window_days-1)]
    x = y.index
    std=rolling_std(df, col, window_days)[window_days:-(window_days-1)]
    upper_band = y + 2*std
    lower_band = y - 2*std

    print(len(x))
    print(len(y))
    print(len(upper_band))

    df = filter_data(df, start, end)

    print('ok')

    fig, axs = plt.subplots(2,1, figsize=(14,8), sharex=True)

    if raw_values is True:
        axs[0].plot(df.index, df[col], label='Raw prices')
    axs[0].plot(x, y, label='Smooth prices', color='r')
    axs[0].plot(x, upper_band, linestyle='--', color='gray')
    axs[0].plot(x, lower_band, linestyle='--', color='gray')
    axs[0].fill_between(x, upper_band, lower_band, color='gray', alpha=0.4, label='Volatility bands')
    axs[0].set_xlabel('Dates')
    axs[0].set_ylabel('Price (EUR/MWh)')
    axs[0].set_title(f'Daily Average Electricity Prices ({start} - {end})\nwith {window_days}-Day Rolling Mean')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month

    if inversed_style:
        inverse_colors = mpf.make_marketcolors(
        up="#b62826",    # Close < Open
        down="#0d7541",  # Close > Open
        edge='i', wick='i', volume='in'
        )
        candle_style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=inverse_colors,
        rc={'font.size': 10}
        )
    else:
        candle_style='charles'
    # axs[1] = sns.boxplot(x='day', y=col, data=df, showfliers=False, color='#00CC96')
    df_OHLC = extract_OHLC(df, col, 'day')
    mpf.plot(df_OHLC, type='candle', ax=axs[1], style=candle_style, show_nontrading=True)
    axs[1].set_ylabel('Price (EUR/MWh)')
    axs[1].set_title('Daily prices volatility')
    axs[1].yaxis.set_label_position("left")
    axs[1].yaxis.tick_left()
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(True)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('../figures/timeserie_prices.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ============= Price distribution

def plot_price_dist(df, start_year, end_year, price_col='Price', bins=20):
    df = ensure_datetime_index(df)
    df = filter_data(df, start_year, end_year).reset_index()

    df['bin'] = pd.cut(df[price_col], bins=bins)
    binned_dist = df.groupby('bin')[price_col].count()
    print(type(binned_dist))

    fig, ax=plt.subplots(figsize=(12,6))
    # binned_dist.plot(kind='bar')
    ax.bar(binned_dist.index.astype(str), binned_dist.values)
    ax.set_title(f'Average {price_col.capitalize()} by Binned {price_col.capitalize()}')
    ax.set_ylabel(f'{price_col.capitalize()}')
    ax.set_xlabel(f'{price_col.capitalize()} Bins')
    ax.grid(axis='y')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('../figures/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ============= Average hourly prices plot

def plot_avg_hourly_prices(df, start_year, end_year, period, col='Price'):
    """
    Function plotting the average hourly day ahead prices over a certain period of time
    """
    df = ensure_datetime_index(df)
    df['Hour'] = df.index.hour
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=(end_year - start_year + 1))

    for i, year in enumerate(range(start_year, end_year+1)):
        (start, end) = format_period(period, year)
        mask = (df.index >= start) & (df.index < end)
        
        df_yearly = df[mask]
        daily_avg = df_yearly.groupby('Hour')[col].mean().reset_index()

        x = daily_avg['Hour']
        y = daily_avg[col]
        ax.step(x, y, where='post', label=str(year), alpha=0.8, color=palette[i])
        print(f'{year} OK')

    ax.set_xlabel('Hours')
    ax.set_ylabel(col)
    ax.set_title(f'Average hourly prices ({start_year} - {end_year}) \nwithin {split_period(period)}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('../figures/daily_price_profile.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ============= Boxplot

def boxplot(df, period, col='Price'):
    """
    Create boxplots of prices by either weekdays, months or seasons, grouped by year.

    Parameters:
    - df: pandas DataFrame with datetime and price columns
    - datetime_column: str, name of datetime column
    - col: str, name of price column, by default equal to 'Price'
    """
    df = ensure_datetime_index(df)
    df = df[col].resample('D').mean().to_frame()    #Smooth out value to erase hourly outliers
    df['year'] = df.index.year

    if period=='weekdays':
        df['weekday'] = df.index.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
        x_axis = 'weekday'
        x_label = 'Weekdays'

    elif period=='months':
        df['month'] = df.index.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
        x_axis = 'month'
        x_label = 'Months'

    elif period=='seasons':
        df['season'] = df.index.month.map(get_season)
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        df['season'] = pd.Categorical(df['season'], season_order, ordered=True)
        x_axis = 'season'
        x_label = 'Seasons'

    else:
        raise ValueError('Wrong input, use weekdays, months, or seasons ')
    
    plt.figure(figsize=(16,6))
    sns.boxplot(data=df, x=x_axis, y=col, hue='year', palette='Reds')
    plt.xlabel(x_label)
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Electricity Prices')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('../figures/boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ============= Heat map

def plot_heatmap(df, start_year, end_year, period, col='Price'):
    # Convert to datetime and filter by year
    df = ensure_datetime_index(df)
    df_concat = extract_periodic_data(df, start_year, end_year, period)

    df_concat['weekday'] = df_concat.index.day_name()
    df_concat['hour'] = df_concat.index.hour

    # Create pivot table of average prices
    heatmap_data = df_concat.pivot_table(
        index='weekday', columns='hour', values=col, aggfunc='mean'
    )

    # Ensure correct weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(weekday_order)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Average Price (EUR/MWh)'})
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title(f"Average Price by Weekday and Hour ({start_year} - {end_year}) \nwithin {split_period(period)}")
    plt.tight_layout()
    plt.savefig('../figures/heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()