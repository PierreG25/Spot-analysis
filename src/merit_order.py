import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from plotting_visualization import *

######################## Helping functions ##########################

cost_map = {
    'Solar': 2.5,
    'Wind': 2.5,
    'Nuclear': 10,
    'Hydro': 12.5,
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
    selected_row = df.loc[date,:]
    for value in selected_row.index:
        if value in cost_map:
            new_row = {f'{value}', selected_row[value], marginal_costs(value)}
            df_merit.loc[len(df_merit)] = new_row
    
    df_merit = df_merit.sort_values(by='marginal costs')
    df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
    df_merit['previous capacity'] = df_merit['cumulative capacity'] - df['capacity']

    for i, row in df.iterrows():
        plt.fill_between(
            [row['previous capacity'], row['cumulative capacity']],
            [row['marginal costs'], row['marginal costs']],
            step='pre',
            y2=0,
            label=row['technology'],
            alpha=0.7
        )

        plt.xlabel('Cumulative Capacity (MW)')
        plt.ylabel('Marginal Cost (€/MWh)')
        plt.title('Merit Order Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()