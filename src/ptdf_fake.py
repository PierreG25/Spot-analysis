import pandas as pd
import numpy as np


master_path = 'data/clean/fbmc/master_dataset_15min.csv'
df = pd.read_csv(master_path, parse_dates=['Time']).sort_values('Time')

print(df.head())

# Fake PTDF data
# Lines: L1, L2, L3, L4
# Areas: FR, DE-LU, BE, NL
# PTDF coefficients and RAM values
ptdf_fake = {
    "Line": ['L1', 'L2', 'L3', 'L4'],
    "BZN|FR": [ 0.8, -0.5, -0.3,  0.0],
    "BZN|DE-LU": [-0.4,  0.9, -0.2, -0.3],
    "BZN|BE": [ 0.0, -0.6,  0.7, -0.1],
    "BZN|NL": [ 0.2,  0.2, -0.2, -0.2],
    "RAM": [1000, 800, 600, 400]
}

def create_fake_ptdf(ptdf, start = "2025-01-01 00:00:00", end = "2025-12-31 00:00:00", freq = "15min"):
    df = pd.DataFrame(ptdf)
    times = pd.date_range(start=start, end=end, freq="15min")
    
    df_ptdf = pd.concat([df.assign(Time=t) for t in times], ignore_index=True)

    # put Time as first column
    df_ptdf = df_ptdf[["Time"] + [c for c in df_ptdf.columns if c != "Time"]]

    # Remove DST days
    remove_days = pd.to_datetime(["2025-03-30", "2025-10-26"]).date
    print("Removing DST days:", remove_days)

    df_ptdf = df_ptdf[~df_ptdf["Time"].dt.date.isin(remove_days)]
    return df_ptdf

df_ptdf = create_fake_ptdf(ptdf_fake)
df_np = df[['Time', 'Area', 'Net position']]
print(df_ptdf)


remove_days = pd.to_datetime(["2025-03-30", "2025-10-26"])
mask = df_ptdf["Time"].dt.normalize().isin(remove_days)

if mask.any():
    raise ValueError("DST days not removed correctly")
print("DST days removed successfully")


# Function to compute line flows
def flow_lines(df_np, df_ptdf, datetime_col = 'Time', zone = 'Area', value_col = 'Net position', line_col = 'Line'):
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

df_flow = flow_lines(df, df_ptdf)
print(df_flow.head())
df_flow.to_csv('data/clean/fbmc/line_flows_fbmc_fake_ptdf.csv', index=False)