import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotting_visualization import *

######################## Helping functions ##########################

cost_map = {
    'Renewables': 1,
    'Hydro': 5,
    'Nuclear': 10,
    'Lignite': 37.5,
    'Hard coal': 50,
    'Natural gas': 75,
    'Fuel oil': 125
}

def marginal_costs(generation_type):

    if generation_type in cost_map:
        return cost_map[generation_type]
    raise ValueError("Wrong input; use the generation technologies from the following list:" \
                    " solar, wind, nuclear, hydro, lignite, hard coal, natural gas, and fuel oil")

def plot_constant_func(x_start, x_end, value, generation_type):
    x = [x_start, x_end]
    y = np.full_like(x, value)

    plt.plot(x, y, label=f'{generation_type}')
    plt.fill_between(x, y, 0)
    plt.xlabel('Generation type tech')
    plt.ylabel('Spot prices (EUR/MWh)')
    plt.title('Merit Order Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

# plot_constant_func(10,100, marginal_costs('wind'), 'test1')
# plot_constant_func(100,200, marginal_costs('nuclear'), 'test2')
# plot_constant_func(200,300, marginal_costs('lignite'), 'test3')
# plt.show()

######################## Merit order curve ##########################

def merit_order_curve(df, date):
    df = ensure_datetime_index(df)
    df_merit = pd.DataFrame({
        'technology': [],
        'capacity': [],
        'marginal costs': [],
    })
    selected_row = df.loc[date]
    for value in selected_row.index:
        if value in cost_map:
            new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value)}
            df_merit.loc[len(df_merit)] = new_row
    df_merit = df_merit.sort_values(by='marginal costs')
    df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
    df_merit['previous capacity'] = df_merit['cumulative capacity'] - df_merit['capacity']

    plt.figure(figsize=(12, 6))
    for i, row in df_merit.iterrows():
        plt.fill_between(
            [row['previous capacity'], row['cumulative capacity']],
            [row['marginal costs'], row['marginal costs']],
            step='pre',
            y2=0,
            label=row['technology']
        )

    forecast_load = df.loc[date]['Forecasted load']
    marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
    spot_price = marginal_unit['marginal costs']

    x_intersect = forecast_load
    y_intersect = spot_price

    plt.plot([x_intersect, x_intersect], [0, y_intersect], color='r', linestyle='--')
    plt.plot([0, x_intersect], [y_intersect, y_intersect], color='r', linestyle='--')
    plt.scatter([x_intersect], [y_intersect], color='k', zorder=5)

    plt.xlabel('Cumulative Capacity (MW)')
    plt.ylabel('Marginal Cost (EUR/MWh)')
    plt.title('Merit Order Curve')
    plt.legend(loc=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def generation_tech_distrib(df1, start_year, end_year):
    df1 = ensure_datetime_index(df1)    # To include in filter_data function
    df1 = filter_data(df1, start_year, end_year)

    count_map = {
    'Renewables': 0,
    'Hydro': 0,
    'Nuclear': 0,
    'Lignite': 0,
    'Hard coal': 0,
    'Natural gas': 0,
    'Fuel oil': 0
}
    print(df1.index)
    for date in df1.index:
        print(date)   
        df_merit = pd.DataFrame({
            'technology': [],
            'capacity': [],
            'marginal costs': [],
        })
        selected_row = df1.loc[date]
        forecast_load = df1.loc[date]['Forecasted load']
        i=1
        for value in selected_row.index:
            print(i)
            # print(selected_row.index)
            if value in cost_map:
                new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value)}
                df_merit.loc[len(df_merit)] = new_row
            i+=1
        df_merit = df_merit.sort_values(by='marginal costs')
        df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
        df_merit['previous capacity'] = df_merit['cumulative capacity'] - df_merit['capacity']
        if not (df_merit['cumulative capacity'] >= forecast_load).any():
            raise ValueError(f'WARNING: forecasted load ({forecast_load}) is superior to the total capacity available ({df_merit['cumulative capacity'].iloc[-1]}) for {date}')
        marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
        marginal_tech = marginal_unit['technology']

        if marginal_tech in count_map:
            count_map[marginal_tech]+= 1
    
    print(count_map)
    # Total sum
    total = sum(count_map.values())

    # Step 1: Assign equal widths to all categories (or vary if needed)
    n = len(count_map)
    equal_width = 1.0 / n  # Total width is 1
    widths = [equal_width] * n

    # Step 2: Compute areas (value %) and corresponding heights
    areas = [v / total for v in count_map.values()]
    heights = [area / w for area, w in zip(areas, widths)]

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    left = 0
    for (label, area, width, height) in zip(count_map.keys(), areas, widths, heights):
        rect = plt.Rectangle((left, 0), width, height, label=f"{label} ({area*100:.1f}%)")
        ax.add_patch(rect)
        # Add label inside rectangle
        ax.text(left + width/2, height/2, label, ha='center', va='center', color='white', fontsize=9)
        left += width

    # Adjust plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(heights)*1.1)
    ax.axis('off')
    plt.title("Mekko Chart of Generation Technologies (by % of Count)")
    plt.show()