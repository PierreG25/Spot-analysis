import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import mplfinance as mpf
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

        if period in season_map:
            return season_map[period](year)
        raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")
    
    raise ValueError("Invalid period format")


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    

def ensure_datetime_index(df, datetime_col='Date'):
    """
    Ensure that the DataFrame has a datetime index named datetime_col.

    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_col (str): Name of the datetime column to set as index if not already.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime_col.
    """
    if df.index.name != datetime_col:
        df = df.set_index(datetime_col)
    return df


def rolling_mean(df, col, wd):
    daily_avg = df[col].resample('D').mean()
    return daily_avg.rolling(window = wd, center = True).mean()

def rolling_std(df, col, wd):
    daily_avg = df[col].resample('D').mean()
    return daily_avg.rolling(window = wd, center = True).std()


def split_period(period):
    if isinstance(period,tuple):
        return f'{period[0]} - {period[1]}'
    return period

def filter_data(df, start_date, end_date):
    mask = (df.index >= start_date) & (df.index < end_date)
    return df.loc[mask]

def extract_periodic_data(df, start_year, end_year, filter_period):
    filtered = []
    for year in range(start_year, end_year + 1):
        start_date, end_date = format_period(filter_period, year)
        mask = (df.index >= start_date) & (df.index < end_date)
        filtered.append(df.loc[mask])
    return pd.concat(filtered)

def extract_OHLC(df, col, index):
    df_OHLC = df.groupby(index).agg(
        Open=(col,'first'),
        High=(col, 'max'),
        Low=(col, 'min'),
        Close=(col,'last')
    )
    return df_OHLC

def shift_date(date_str, x_days):
    date_obj = datetime.strptime(date_str, '%Y/%m/%d')

    shifted_date = date_obj + timedelta(days=x_days)

    return shifted_date.strftime('%Y/%m/%d')

########################################## Plots Functions for prices visualization ########################################################

######### Time serie plot

def plot_smooth_prices(df, start, end, window_days, save_path, col='Price', raw_values=True):
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

    fig, axs = plt.subplots(2,1, figsize=(16,10), sharex=True)

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

    inverse_colors = mpf.make_marketcolors(
    up="#d64543",    # Close < Open
    down="#26a666",  # Close > Open
    edge='i', wick='i', volume='in'
)
    inverse_style = mpf.make_mpf_style(
    base_mpf_style='yahoo',
    marketcolors=inverse_colors,
    rc={'font.size': 10}
)
    # axs[1] = sns.boxplot(x='day', y=col, data=df, showfliers=False, color='#00CC96')
    df_OHLC = extract_OHLC(df, col, 'day')
    mpf.plot(df_OHLC, type='candle', ax=axs[1], style=inverse_style, show_nontrading=True)
    axs[1].set_ylabel('Price (EUR/MWh)')
    axs[1].set_title('Candlestick daily prices')
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
    if save_path:
        plt.savefig(save_path)
    plt.show()



######### Price distribution

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
    plt.show()


######### Average hourly prices plot

def plot_avg_hourly_prices(df, start_year, end_year, period, save_path, col='Price'):
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
    ax.set_title(f'Average day-ahead hourly prices ({start_year} - {end_year}) \nwithin {split_period(period)}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


######### Boxplot

def plot_boxplots_by_weekday(df, start_year, end_year, period, save_path, col='Price'):
    """
    Create boxplots of prices by day of the week, grouped by year.

    Parameters:
    - df: pandas DataFrame with datetime and price columns
    - start_year, end_year: int, range of years to include
    - datetime_column: str, name of datetime column
    - price_column: str, name of price column
    """

    # Convert to datetime and filter by year
    df = ensure_datetime_index(df)
    df = extract_periodic_data(df, start_year, end_year, period)

    # Aggregate to daily average price
    df_daily = df[col].resample('D').mean().to_frame()
    df_daily['weekday'] = df_daily.index.day_name()
    df_daily['year'] = df_daily.index.year

    # Ensure consistent weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_daily['weekday'] = pd.Categorical(df_daily['weekday'], categories=weekday_order, ordered=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_daily, x='weekday', y=col, hue='year')

    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Average Daily Price (EUR/MWh)")
    ax.set_title(f"Electricity Prices by Day of the Week ({start_year} - {end_year}) \nwithin {split_period(period)}")
    ax.legend(title="Year")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def boxplot(df, period, save_path, col='Price'):
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
    
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x=x_axis, y=col, hue='year')
    plt.xlabel(x_label)
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Electricity Prices')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

######### Heat map

def plot_heatmap(df, start_year, end_year, period, save_path, col='Price'):
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
    ax.set_title(f"Average Electricity Price by Weekday and Hour ({start_year} - {end_year}) \nwithin {split_period(period)}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


########################################## Plots Functions for the influence of external variables ########################################################



########################################## Plots Functions for anomalies ########################################################