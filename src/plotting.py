import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def process_time_period(period):
    """
    Function converting a time period into the right format for further analysis
    
    Accepts either:
    - A season name: one of "winter", "spring", "summer", "autumn"
    - A date range tuple: ("YYYY-MM-DD", "YYYY-MM-DD")

    Returns a tuple of (start_date, end_date)
    
    Examples:
        process_period("summer")
        process_period(("2024-06-01", "2024-09-01"))
    """
    if isinstance(period, str):
        season = period.lower()
        if season == "winter":
            return ("2024-12-01", "2025-02-28")
        elif season == "spring":
            return ("2025-03-01", "2025-05-31")
        elif season == "summer":
            return ("2025-06-01", "2025-08-31")
        elif season == "autumn":
            return ("2025-09-01", "2025-11-30")
        else:
            raise ValueError("Wrong input, use one of: 'winter', 'spring', 'summer', 'autumn'")
    elif isinstance(period, tuple):
        return period
    else:
        raise ValueError("Wrong input. Expected inputs: 'seasons' or '(YYYY-MM-DD, YYYY-MM-DD'")


def plot_price_hour(df):
    """
    Function plotting the average hourly day ahead prices over a certain period of time
    """

    daily_avg = df.groupby('Hour')['Day-ahead Price (EUR/MWh)'].mean().reset_index()

    x = daily_avg['Hour']
    y = daily_avg['Day-ahead Price (EUR/MWh)']

    peaks, _ = find_peaks(y)
    for i in enumerate(peaks):
        if i==0:
            plt.axvspan(peaks[i]-1, peaks[i]+1, color='gray', alpha=0.15, label='Peak zone')
        else:
            plt.axvspan(peaks[i]-1, peaks[i]+1, color='gray', alpha=0.15)

    default_ticks = list(x)[0::4]
    combined_ticks = sorted(set(default_ticks + list(x.iloc[peaks])))

    plt.xticks(ticks=combined_ticks, labels=combined_ticks)
    plt.scatter(x, y, marker='+', color='r')
    plt.plot(x, y, linestyle='--', color='b',alpha=0.5)
    plt.title('Average Day-ahead hourly price')
    plt.xlabel('Hour')
    plt.ylabel('Day-ahead Price (EUR/MWh)')
    plt.legend()
    return