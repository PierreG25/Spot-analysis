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

    forecast_load = df.loc[date]['Load']
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

    df1=filter_data(df1, start_year, end_year)

    count_map = {
    'Renewables': 0,
    'Hydro': 0,
    'Nuclear': 0,
    'Lignite': 0,
    'Hard coal': 0,
    'Natural gas': 0,
    'Fuel oil': 0
}

    for date in df1.index:    
        df_merit = pd.DataFrame({
            'technology': [],
            'capacity': [],
            'marginal costs': [],
        })
        selected_row = df1.loc[date]
        forecast_load = df1.loc[date]['Load']
        for value in selected_row.index:
            if value in cost_map:
                new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value)}
                df_merit.loc[len(df_merit)] = new_row
        df_merit = df_merit.sort_values(by='marginal costs')
        df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
        df_merit['previous capacity'] = df_merit['cumulative capacity'] - df_merit['capacity']

        marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
        marginal_tech = marginal_unit['technology']

        if marginal_tech in count_map:
            count_map[marginal_tech]+= 1
    
    total = sum(count_map.values())
    x_start = 0
    colors = ['#6B6B6B', '#FF7F0E', '#8B0000', '#FDB813', '#00BFFF', "#2BFF00", "#EA00FF"]

    fig = go.Figure()

    # Compute rectangles with width × height = area share
    for i, (tech, count) in enumerate(count_map.items()):
        height = marginal_costs(tech)
        if height == 0:
            continue  # skip if height is 0 to avoid division by zero
        area = count / total
        width = area / height * 100  # scale for display
        fig.add_shape(
            type='rect',
            x0=x_start,
            x1=x_start + width,
            y0=0,
            y1=height,
            fillcolor='colors[i]',
            line=dict(width=1, color='black')
        )
        fig.add_annotation(
            x=x_start + width / 2,
            y=height + 3,
            text=f"{tech}<br>{count}x",
            showarrow=False,
            font=dict(size=11)
        )
        x_start += width

    fig.update_layout(
        title="Mekko Chart – Area ∝ Frequency Tech Sets Spot Price",
        xaxis=dict(title="Width (Scaled)", showgrid=False, zeroline=False),
        yaxis=dict(title="Marginal Cost (€/MWh)", showgrid=True, zeroline=False),
        width=900,
        height=500
    )

    fig.show()

    