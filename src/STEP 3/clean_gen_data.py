import pandas as pd
import numpy as np

gen_path_fr = 'data/raw/fbmc/generation/2025_generation_fr_raw.csv'

df_gen_fr = pd.read_csv(gen_path_fr)

print(df_gen_fr.columns)

def pivot_generation(df, datetime_col = 'MTU (CET/CEST)', area_col = 'Area', gen_col = 'Generation (MW)'):
    df = df.copy()

    df_pivot = df.pivot(index=datetime_col,
                        columns=area_col,
                        values=gen_col
                        ).fillna('Na').reset_index()
    return df_pivot

print(pivot_generation(df_gen_fr).head())