import pandas as pd
import numpy as np
from ptdf import flow_lines

path_zone_fundamental = "data/clean/STEP 3/NP_by_country.csv"

df_zone_fundamental = pd.read_csv(path_zone_fundamental, parse_dates=['Time'])
df_jao_lines = pd.read_csv('data/clean/STEP 3/shadow_prices/jao_clean_noDST_2025.csv', parse_dates=['Time']).sort_values('Time')

# ============= HELPING FUNCTIONS ============= #


def filter_country_congestion(df, country_code, 
                              hub_from = 'hubFrom', 
                              hub_to = 'hubTo'):
    """
    Filter the dataframe for a specific country by checking 
    the Origin and Destination of the flow lines
    """
    df = df.copy()
    if country_code == None:
        return df

    df_country = df[(df[hub_from].str.contains(country_code)) | (df[hub_to].str.contains(country_code))]
    return df_country


def filter_neighbors(df,
                     countries_code = ['BE', 'DE', 'NL'],
                     neighbors_col = 'To'):
    """
    Filter the dataframe to only keep the lines that are connected to the neighbors of the country of interest.
    """

    df = df.copy()

    if countries_code is None:
        return df
    
    df = df[(df[neighbors_col].isin(countries_code))]
    return df


def directionally_connexion(df, country_code,
                             hub_from = 'hubFrom',
                             hub_to = 'hubTo',
                             flow = 'Flow_MW'):
    """
    Normalize the flow values to be positive for exports and negative for imports, 
    based on the country of interest. Also create 'From' and 'To' columns to indicate 
    the direction of the flow with respect to the country of interest.
    """

    df['From'] = country_code
    df['To'] = np.where(df[hub_from] == country_code, df[hub_to], df[hub_from])
    df['Flow_normalized'] = np.where(df[hub_from] == country_code, df[flow], -df[flow])

    df = df.drop(columns = [flow, hub_from, hub_to])

    # df['export_constrained'] = (df[conges] & (df['Flow_normalized'] > 0))
    # df['import_constrained'] = (df[conges] & (df['Flow_normalized'] < 0))

    return df


# ============= SPREAD DATASET ============= #

def spread_dataset(df_zone, df_lines,
                datetime_col = 'Time',
                price_col = 'Price',
                zone_col = 'Area',
                ref_country = 'FR',
                zones = ['FR', 'BE', 'DE', 'NL'],
                neighbors = True):
    """
    Create a dataset that combines price spreads and line flows, 
    normalized by direction, for a specific reference country and its neighbors.
    """

    # Compute line flows
    df_flw = flow_lines(df_zone, df_lines)
  
    duplicates = df_zone[df_zone.duplicated(subset = [datetime_col, zone_col], keep=False)]

    # Sort them so you can see the conflicting rows next to each other
    if not duplicates.empty:
        print(duplicates.sort_values(by=[datetime_col, zone_col]))
        raise ValueError('Duplicates in the datetime col of the zonal dataset')

    df_zone_pivoted = df_zone.pivot(index = datetime_col, columns = zone_col, values = price_col)
    df_flow_spread = df_flw.merge(df_zone_pivoted, on = datetime_col, how = 'left')

    # Compute the price spread (compared to the ref country) for each country within zone
    for z in zones:
        df_flow_spread[f'Spread {ref_country}-{z}'] = df_flow_spread[f'{ref_country}'] - df_flow_spread[z]

    df_flow_spread.drop(columns=zones + [f'Spread {ref_country}-{ref_country}'], inplace=True)

    # Filter the dataframe to only have the lines with 
    # Origin or Destination being the country of reference
    df_flow_spread = filter_country_congestion(df_flow_spread, ref_country)

    # Addition of the price spread
    df_flow_spread_normalized = directionally_connexion(df_flow_spread, ref_country)

    if neighbors:
        df_flow_spread_normalized = filter_neighbors(df_flow_spread_normalized)
        df_flow_spread_normalized.to_csv(f'data/clean/STEP 3/interconnexion_{ref_country}_filtered.csv')

        return df_flow_spread_normalized

    df_flow_spread_normalized.to_csv(f'data/clean/STEP 3/interconnexion_{ref_country}.csv')

    return df_flow_spread_normalized

spread_dataset(df_zone_fundamental, df_jao_lines, ref_country = 'FR', neighbors = True)