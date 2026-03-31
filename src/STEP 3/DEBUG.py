import pandas as pd
import numpy as np

path_zone_fundamental = "data/clean/STEP 3/NP_by_country.csv"
df = pd.read_csv(path_zone_fundamental, parse_dates=['Time'])

print(df[df['Time'] == '2024-04-01 01:00:00'])
print(df.loc[df['Area'] == [], 'Time'])