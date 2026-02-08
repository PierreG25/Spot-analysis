import pandas as pd
import numpy as np

path_zone_fundamental = "data/clean/fbmc/master_dataset_15min_WITH_PRICE.csv"
path_flow = "data/clean/fbmc/line_flows_fbmc_real_ptdf.csv"

df_zone_fundamental = pd.read_csv(path_zone_fundamental)
df_flow = pd.read_csv(path_flow)

drop_columns = True

# ============= HELPING FUNCTIONS ============= #

def add_price(df_flw, df_zone, 
              datetime_col = 'Time',
              price_col = 'Price',
              zone_col = 'Area',
              zones = ['FR', 'BE', 'DE', 'NL']):
    
    # Ensure datetime format for both dataframes
    df_zone[datetime_col] = pd.to_datetime(df_zone[datetime_col])
    df_flw[datetime_col] = pd.to_datetime(df_flw[datetime_col])

    # price_BE = df_zone[df_zone['Area'].str.contains('BE')][[datetime_col, price_col]]
    
    duplicates = df_zone[df_zone.duplicated(subset = [datetime_col, zone_col], keep=False)]

    # Sort them so you can see the conflicting rows next to each other
    print(duplicates.sort_values(by=[datetime_col, zone_col]))

    df_zone_pivoted = df_zone.pivot(index = datetime_col, columns = zone_col, values = price_col)
    print(df_zone_pivoted.head())
    df_final = df_flw.merge(df_zone_pivoted, on = datetime_col, how = 'left')

    return df_final

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


def normalize_direction(df, country_code, 
                        hub_from = 'hubFrom', 
                        hub_to = 'hubTo', 
                        flow = 'Flow_MW'):
    df_country = filter_country_congestion(df, country_code)

    df_country['From'] = country_code
    df_country['To'] = np.where(df_country[hub_from] == country_code, df_country[hub_to], df_country[hub_from])
    df_country['Flow_normalized'] = np.where(df_country[hub_from] == country_code, df_country[flow], -df_country[flow])

    if drop_columns:
        df_country = df_country.drop(columns = [flow, hub_from, hub_to])
    return df_country


def directionally_congestion(df, conges = 'congested', flow = 'Flow_normalized'):
    df['export_constrained'] = (df[conges] & (df[flow] > 0))
    df['import_constrained'] = (df[conges] & (df[flow] < 0))

    return df


# ============= SPREAD DATASET ============= #

print(add_price(df_flow, df_zone_fundamental))
exit()

df_flow_fr = filter_country_congestion(df_flow, 'FR')

df_flow_fr_normalized = normalize_direction(df_flow_fr, 'FR')
df_flow_fr_normalized = directionally_congestion(df_flow_fr_normalized)
print(df_flow_fr_normalized[8080:8090])
print(df_flow_fr_normalized['To'].unique())
print(df_flow_fr_normalized.head(15))
print(filter_neighbors(df_flow_fr_normalized).head())

# def spread_price(df_flw, df_price, 
#                  col_price = 'Price', 
#                  country_from = 'FR', 
#                  countries_to = ['BE', 'DE', 'NL']):
