import pandas as pd
import numpy as np


master_path = 'data/clean/fbmc/master_dataset_15min.csv'
df = pd.read_csv(master_path, parse_dates=['Time'])

def NP_matrix(df, datetime_col = 'Time', value_col = 'Net position'):
    pivot_df = df.pivot(index=datetime_col, columns='Area', values=value_col)
    print(pivot_df.head())
    return pivot_df

NP_matrix(df)