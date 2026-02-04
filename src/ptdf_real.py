import pandas as pd
import numpy as np


master_path = 'data/clean/fbmc/master_dataset_15min.csv'
df = pd.read_csv(master_path, parse_dates=['Time']).sort_values('Time')
df_jao_flow = pd.read_csv('data/clean/jao/shadow_prices/2025/jao_clean_noDST_2025.csv', parse_dates=['Time']).sort_values('Time')

print(df.head())

# Fake PTDF data
# Lines: L1, L2, L3, L4
# Areas: FR, DE-LU, BE, NL
# PTDF coefficients and RAM values

df_np = df[['Time', 'Area', 'Net position']]

# Function to compute line flows
def flow_lines(df_np, df_ptdf, datetime_col = 'Time', zone = 'Area', value_col = 'Net position', line_col = 'CNEC Name'):
    """Compute line flows based on net positions and PTDF coefficients."""
    np_wide = (df_np
    .pivot(index=datetime_col, columns=zone, values=value_col)
    )

    print('np_wide:')
    print(np_wide.head())
    
    flows = []
    areas = df_np[zone].unique()

    for (t, line), row in df_ptdf.groupby([datetime_col, line_col]):
        # Net positions at time t
        print(t)
        print(line)

        np_t = np_wide.loc[t]

        # PTDF coefficients for this line
        ptdf = row[areas].iloc[0]

        # Flow computation
        flow = (ptdf * np_t).sum()

        flows.append({
            "Time": t,
            "Line": line,
            "Flow_MW": flow,
            "RAM": row["RAM"].iloc[0]
        })

    flows_df = pd.DataFrame(flows)
    # Compute utilization and congestion
    flows_df["utilization"] = flows_df["Flow_MW"].abs() / flows_df["RAM"]
    flows_df["congested"] = flows_df["utilization"] >= 1.0

    return flows_df

df_flow = flow_lines(df, df_jao_flow)
print(df_flow.head())
df_flow.to_csv('data/clean/fbmc/line_flows_fbmc_real_ptdf.csv', index=False)