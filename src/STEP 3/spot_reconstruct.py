import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


path_1 = 'data/clean/STEP 3/XGBoost/generation_by_type_fr.csv'
path_2 = 'data/clean/STEP 3/XGBoost/NP_by_country_FR.csv'
path_3 = 'data/clean/STEP 3/XGBoost/interco_stress_metrics.csv'

df_gen = pd.read_csv(path_1, parse_dates=['Time'])
df_np = pd.read_csv(path_2, parse_dates=['Time'])
df_interco = pd.read_csv(path_3, parse_dates=['Time'])

# ============= PREPARE DATA ============= #

def time_cols(df):
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['month'] = df['Time'].dt.month
    return df


def lag_features(df, col='Price', lags=[4, 96, 672]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def prepare_data(df_gen, df_np, df_interco):
    df = pd.merge(df_np, df_gen, on='Time', how='inner')
    df = pd.merge(df, df_interco, on='Time', how='left').fillna({
        'stress_BE': 0,
        'stress_DE': 0,
        'stress_NL': 0,
        'stressed_BE': False,
        'stressed_DE': False,
        'stressed_NL': False
    })

    df = time_cols(df)
    df = lag_features(df)
    return df

if __name__ == "__main__":
    df = prepare_data(df_gen, df_np, df_interco)
    df.to_csv('data/clean/STEP 3/XGBoost/master_dataset_v2.csv', index=False)