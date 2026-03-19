import pandas as pd
import numpy as np

interconexion_path = 'data/clean/STEP 3/interconnexion_FR_filtered.csv'
df_interco = pd.read_csv(interconexion_path, parse_dates=['Time'])

def stress_metrics(df, delta = 0.8,
                   flow_col = 'Flow_normalized',
                   ram_col = 'RAM'):
    """
    Calculate stress metrics based on the flow and RAM values. The function adds a new column 'stress' 
    which is the ratio of the absolute flow to the RAM. A line is considered stressed if this ratio exceeds 0.8.
    """

    df.groupby('Time').apply(lambda x: x.assign(stress = x[flow_col].abs() / x[ram_col],
                                                 stressed = (x[flow_col].abs() / x[ram_col]) >= delta))
    
    # We use groupby and apply to ensure that the stress metrics are calculated for each time step separately, 
    # which is important if there are multiple lines at the same time step. 
    # The assign function allows us to create new columns based on the calculations. 
    # The 'stress' column contains the ratio of the absolute flow to the RAM, and the 'stressed' column is a 
    # boolean indicating whether this ratio exceeds the specified delta threshold (0.8 in this case).

    # df = df.copy()
    
    # df['stress'] = df[flow_col].abs() / df[ram_col]
    # df['stressed'] = df['stress'] >= delta
    
    return df


def congestion(df,
                epsilon = 0.1,
                ref_country = 'FR',
                countries_code = ['BE', 'DE', 'NL']):
    """
    Calculate congestion metrics based on the spread values. A zone is considered congested 
    if the absolute value of the spread exceeds a certain threshold (epsilon). The function adds a new column 
    for each neighboring country indicating whether the line is congested or not.
    """
    df = df.copy()
    
    # Assuming 'congested' is a boolean column indicating whether the line is congested at each time step
    for country in countries_code:
        df['congestion_' + ref_country + '-' + country] = df['Spread ' + ref_country + '-' + country].abs() >= epsilon
        print(abs(df['Spread ' + ref_country + '-' + country]).head())
    
    # # Total congested hours
    # total_congested_hours = df['congested'].sum()
    
    # # Average congestion duration (in hours)
    # df['congestion_group'] = (df['congested'] != df['congested'].shift()).cumsum()
    # congestion_durations = df[df['congested']].groupby('congestion_group').size() * 0.25  # Assuming 15-minute intervals
    # average_congestion_duration = congestion_durations.mean() if not congestion_durations.empty else 0
    
    # # Maximum congestion duration (in hours)
    # max_congestion_duration = congestion_durations.max() if not congestion_durations.empty else 0
    
    # return {
    #     'total_congested_hours': total_congested_hours,
    #     'average_congestion_duration': average_congestion_duration,
    #     'max_congestion_duration': max_congestion_duration
    # }

    return df

congestion(stress_metrics(df_interco)).to_csv('data/clean/STEP 3/interco_congestion_metrics.csv', index=False)