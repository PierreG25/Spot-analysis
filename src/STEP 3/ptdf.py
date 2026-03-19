import pandas as pd
import numpy as np


def flow_lines(df, df_ptdf,
               datetime_col = 'Time',
               zone = 'Area',
               value_col = 'Net position',
               line_col = 'CNEC Name',
               save = True):
    """Compute line flows based on net positions and PTDF coefficients."""
    df = df.copy()
    # We only keep the relevant columns for the spread dataset (we will merge with the price dataset later)
    df_np = df[['Time', 'Area', 'Net position']]

    np_wide = (df_np
    .pivot(index=datetime_col, columns=zone, values=value_col)
    )

    # print('np_wide:')
    # print(np_wide.head())
    
    flows = []
    areas = df_np[zone].unique()

    for (t, line), row in df_ptdf.groupby([datetime_col, line_col]):
        # Net positions at time t
        print(t)

        np_t = np_wide.loc[t]

        # PTDF coefficients for this line
        ptdf = row[areas].iloc[0]

        # Flow computation
        flow = (ptdf * np_t).sum()

        flows.append({
            "Time": t,
            "Line": line,
            "Flow_MW": flow,
            "hubFrom": row["hubFrom"].iloc[0],
            "hubTo": row["hubTo"].iloc[0],
            "RAM": row["RAM"].iloc[0]
        })

    flows_df = pd.DataFrame(flows)
    # Compute utilization and congestion
    flows_df["utilization"] = flows_df["Flow_MW"].abs() / flows_df["RAM"]
    # flows_df["congested"] = flows_df["utilization"] >= 1.0

    if save:
        flows_df.to_csv('data/clean/STEP 3/line_flows_ptdf.csv', index=False)
        print("Saved line flows to 'data/clean/STEP 3/line_flows_ptdf.csv'")

    return flows_df