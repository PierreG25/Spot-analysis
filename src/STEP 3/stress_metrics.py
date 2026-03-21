import pandas as pd
import numpy as np

interconexion_path = 'data/clean/STEP 3/interconnexion_FR_filtered.csv'
df_interco = pd.read_csv(interconexion_path, parse_dates=['Time'])


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
    
    return df


def stress_metrics_split(df,
                        countries_code = ['BE', 'DE', 'NL'],
                        stress_col='stress'):
    """
    Calcule le stress ligne par ligne, puis retourne un DataFrame
    avec un timestamp unique et le stress maximal pour chaque temps.
    """
    df = df.copy()

    for country in countries_code:
        mask = df['To'] == country
        df[f'stress_{country}'] = np.where(mask, df[stress_col], 0)
        # df[f'stressed_{country}'] = np.where(mask, stress >= delta, False)
    
    return df


def groupby_time_max_stress(
    df,
    delta=0.8,
    datetime_col='Time',
    keep_cols = ['Spread FR-BE', 'Spread FR-DE', 'Spread FR-NL'],
    countries_code=('BE', 'DE', 'NL')
):
    """
    Group the dataframe by time and calculate the maximum stress for each neighboring country.
    Also keep the first value of the spread columns for each time step.
    """
    # Calculate stress metrics for each line and split them into separate columns for each country
    df = stress_metrics_split(df)

    stress_cols = [f'stress_{country}' for country in countries_code]

    agg_dict = {
        **{col: 'max' for col in stress_cols},
        **{col: 'first' for col in keep_cols}
    }

    result = df.groupby(datetime_col, as_index=False).agg(agg_dict)

    for country in countries_code:
        result[f'stressed_{country}'] = result[f'stress_{country}'] >= delta

    result.to_csv('data/clean/STEP 3/XGBoost/interco_stress_metrics.csv', index=False)
    return result


interco_dataset_final = groupby_time_max_stress(df_interco)
