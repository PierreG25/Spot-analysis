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

energy_colors = {
    "Renewable": "#2ca02c",
    "Hydro": "#1f77b4",
    "Nuclear": "#ffcc00",
    "Lignite": "#8c564b",
    "Hard coal": "#2c2c2c",
    "Natural gas": "#ff7f0e",
    "Fuel oil": "#d62728",
    "Others": "#d3d3d3"
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


def plot_donut(map_data, threshold_pct=3):
    total = sum(map_data.values())
    threshold_value = total * (threshold_pct / 100)

    # Group data
    large = {k: v for k, v in map_data.items() if v >= threshold_value}
    small = {k: v for k, v in map_data.items() if v < threshold_value}

    if small:
        large["Others"] = sum(small.values())

    labels = list(large.keys())
    sizes = list(large.values())

    # Colors (repeating if necessary)
    colors = [energy_colors[energy] for energy in labels]

    def autopct(pct):
        return f'{pct:.1f}%' if pct > threshold_pct else ''

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=autopct,
        wedgeprops=dict(width=0.3),
        pctdistance=0.85,
        textprops=dict(color="black", weight="bold", fontsize=10)
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.text(0, 0, 'Marginal\nEnergy Sources', ha='center', va='center', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.show()

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
    merit_order_limit = df_merit.iloc[-1,3] < forecast_load
    if merit_order_limit:
        print('SUP')
        marginal_unit = df_merit.iloc[-1]
        spot_price = marginal_unit['marginal costs']
        forecast_load = marginal_unit['cumulative capacity']    # This a bordeline case (Load > supply), thus we consider the last technology 
                                                                #to set the price and to plot we affect the load as the the total cumulative capacity 
    else:
        print('INF')
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

        merit_order_limit = df_merit.iloc[-1,3] < forecast_load
        if merit_order_limit:
            # raise ValueError(f'WARNING: forecasted load ({forecast_load}) is 
            # superior to the total capacity available ({df_merit['cumulative capacity'].iloc[-1]}) for {date}')
            marginal_unit = df_merit['cumulative capacity'].iloc[-1]
            marginal_tech = marginal_unit['technology']
        else: 
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


def donut_tech_dist_v1(df, start_year, end_year):
    df = ensure_datetime_index(df)    # To include in filter_data function
    df = filter_data(df, start_year, end_year)

    count_map = {
    'Renewables': 0,
    'Hydro': 0,
    'Nuclear': 0,
    'Lignite': 0,
    'Hard coal': 0,
    'Natural gas': 0,
    'Fuel oil': 0
}
    
    for date in df.index:
        print(date)   
        df_merit = pd.DataFrame({
            'technology': [],
            'capacity': [],
            'marginal costs': [],
        })
        selected_row = df.loc[date]
        forecast_load = selected_row['Forecasted load']

        i=1
        for value in selected_row.index:
            # print(i)
            # print(selected_row.index)
            if value in cost_map:
                new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value)}
                df_merit.loc[len(df_merit)] = new_row
            i+=1
        df_merit = df_merit.sort_values(by='marginal costs')
        df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
        df_merit['previous capacity'] = df_merit['cumulative capacity'] - df_merit['capacity']
        print(df_merit)

        merit_order_limit = df_merit.iloc[-1,3] < forecast_load
        print(forecast_load)
        if merit_order_limit:
            # raise ValueError(f'WARNING: forecasted load ({forecast_load}) is 
            # superior to the total capacity available ({df_merit['cumulative capacity'].iloc[-1]}) for {date}')
            marginal_unit = df_merit.iloc[-1]
            print(marginal_unit)
            marginal_tech = marginal_unit['technology']
            print('OK')
        else: 
            print(df_merit[df_merit['cumulative capacity'] >= forecast_load])
            marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
            marginal_tech = marginal_unit['technology']

        if marginal_tech in count_map:
            count_map[marginal_tech]+= 1
        print('OUIIIII')
        print(count_map)
    
    print(count_map)

    # Extract labels and values
    labels = list(count_map.keys())
    sizes = list(count_map.values())

    # Create pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3),  # This makes it a donut (hollow pie)
            pctdistance=0.85,  # moves percentage toward center of wedge
        textprops=dict(color="k", weight="bold")
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    plt.title('Energy Capacity by Source')
    plt.tight_layout()
    plt.show()



def donut_tech_dist(df, start_year, end_year):
    df = ensure_datetime_index(df)    # To include in filter_data function
    df = filter_data(df, start_year, end_year)
    exceed_supply=0

    count_map = {
    'Renewables': 0,
    'Hydro': 0,
    'Nuclear': 0,
    'Lignite': 0,
    'Hard coal': 0,
    'Natural gas': 0,
    'Fuel oil': 0
}
    
    for date in df.index: 
        df_merit = pd.DataFrame({
            'technology': [],
            'capacity': [],
            'marginal costs': [],
        })
        selected_row = df.loc[date]
        forecast_load = selected_row['Forecasted load']

        for value in selected_row.index:

            if value in cost_map:
                new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value)}
                df_merit.loc[len(df_merit)] = new_row

        df_merit = df_merit.sort_values(by='marginal costs')
        df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()

        merit_order_limit = df_merit.iloc[-1,3] < forecast_load     # Test to know if demand exceed supply, meaning if our current data and model are not enough to model correctly the electricity market
        if merit_order_limit:

            marginal_unit = df_merit.iloc[-1]
            marginal_tech = marginal_unit['technology']
            exceed_supply+=1
        else: 
            marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
            marginal_tech = marginal_unit['technology']

        if marginal_tech in count_map:
            count_map[marginal_tech]+= 1
    
    print(count_map)
    print(exceed_supply)

    plot_donut(count_map)