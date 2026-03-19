import pandas as pd
import numpy as np

def upsample_by_repetition(df):
    # 1. Load data
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'])

    # 2. Duplicate every row 4 times
    # df.index.repeat(4) creates an index like [0,0,0,0, 1,1,1,1, ...]
    df_expanded = df.loc[df.index.repeat(4)].reset_index(drop=True)

    # 3. Create minute offsets: 0, 15, 30, 45
    # We create a repeating pattern [0, 15, 30, 45] for the length of the new dataframe
    minute_offsets = np.tile([0, 15, 30, 45], len(df))
    
    # 4. Apply offsets to the Time column
    df_expanded['Time'] = df_expanded['Time'] + pd.to_timedelta(minute_offsets, unit='m')
    df_expanded = df_expanded.sort_values('Time').reset_index(drop=True)

    # 5. Save
    print(f"Success! Expanded {len(df)} rows to {len(df_expanded)} rows.")
    
    return df_expanded