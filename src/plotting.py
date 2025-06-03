from datetime import date, datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def process_time_period(period):
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
        'spring': (lambda y: (date(y, 3, 20), date(y, 6, 20))),
        'summer': (lambda y: (date(y, 6, 21), date(y, 9, 22))),
        'autumn': (lambda y: (date(y, 9, 23), date(y, 12, 20))),
        'winter': (lambda y: (date(y, 12, 21), date(y+1, 3, 19)))
    }

    if isinstance(period, str):
        season, year = period.lower().split('-')
        year = int(year)
        period_func = season_map[season]

        if season in season_map:
            return period_func(year)
        raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")
    if isinstance(period, tuple):
        start_date = datetime.strptime(period[0], "%Y-%m-%d")
        end_date = datetime.strptime(period[1], "%Y-%m-%d")
        return (start_date.date(), end_date.date())
    raise ValueError("Wrong input. Expected inputs: 'season-YYYY' or ('YYYY-MM-DD', 'YYYY-MM-DD')'")


def plot_price_hour(df, period):
    """
    Function plotting the average hourly day ahead prices over a certain period of time
    """
    (start, end) = process_time_period(period)

    df_period = df[(df['Date'] >= start) & (df['Date'] <= end)]
    daily_avg = df_period.groupby('Hour')['Day-ahead Price (EUR/MWh)'].mean().reset_index()

    x = daily_avg['Hour']
    y = daily_avg['Day-ahead Price (EUR/MWh)']

    peaks, _ = find_peaks(y)
    for i, x_peak in enumerate(peaks):
        if i==0:
            plt.axvspan(x_peak-1, x_peak+1, color='gray', alpha=0.15, label='Peak zone')
        plt.axvspan(x_peak-1, x_peak+1, color='gray', alpha=0.15)

    default_ticks = list(x)[0::4]
    combined_ticks = sorted(set(default_ticks + list(x.iloc[peaks])))

    plt.xticks(ticks=combined_ticks, labels=combined_ticks)
    plt.scatter(x, y, marker='+', color='r')
    plt.plot(x, y, linestyle='--', color='b',alpha=0.5)
    plt.title('Average Day-ahead hourly price')
    plt.xlabel('Hour')
    plt.ylabel('Day-ahead Price (EUR/MWh)')
    plt.legend()
    plt.show()