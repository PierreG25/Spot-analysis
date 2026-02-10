import pandas as pd
import numpy as np

path_zone_fundamental = "data/clean/fbmc/master_dataset_15min_WITH_PRICE.csv"
path_flow = "data/clean/fbmc/line_flows_fbmc_real_ptdf.csv"

df_zone_fundamental = pd.read_csv(path_zone_fundamental)
df_flow = pd.read_csv(path_flow)

drop_columns = True

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
    df = df.copy()

    if countries_code == None:
        return df
    
    df = df[(df[neighbors_col].isin(countries_code))]
    return df


def directionally_congestion(df, country_code, 
                             hub_from = 'hubFrom', 
                             hub_to = 'hubTo', 
                             flow = 'Flow_MW',
                             conges = 'congested'):

    df['From'] = country_code
    df['To'] = np.where(df[hub_from] == country_code, df[hub_to], df[hub_from])
    df['Flow_normalized'] = np.where(df[hub_from] == country_code, df[flow], -df[flow])

    if drop_columns:
        df = df.drop(columns = [flow, hub_from, hub_to])

    df['export_constrained'] = (df[conges] & (df['Flow_normalized'] > 0))
    df['import_constrained'] = (df[conges] & (df['Flow_normalized'] < 0))

    return df


# ============= SPREAD DATASET ============= #

def spread_dataset(df_flw, df_zone, 
              datetime_col = 'Time',
              price_col = 'Price',
              zone_col = 'Area',
              ref_country = 'FR',
              zones = ['FR', 'BE', 'DE', 'NL'],
              neighbors = True):
    
    # Ensure datetime format for both dataframes
    df_zone[datetime_col] = pd.to_datetime(df_zone[datetime_col])
    df_flw[datetime_col] = pd.to_datetime(df_flw[datetime_col])

    # price_BE = df_zone[df_zone['Area'].str.contains('BE')][[datetime_col, price_col]]
    
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
    df_flow_spread_normalized = directionally_congestion(df_flow_spread, ref_country)

    if neighbors:
        df_flow_spread_normalized = filter_neighbors(df_flow_spread_normalized)
        df_flow_spread_normalized.to_csv(f'data/clean/fbmc/spread_dataset_{ref_country}_filterd.csv')

        return df_flow_spread_normalized

    df_flow_spread_normalized.to_csv(f'data/clean/fbmc/spread_dataset_{ref_country}.csv')

    return df_flow_spread_normalized
    

print(spread_dataset(df_flow, df_zone_fundamental).head())
