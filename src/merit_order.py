import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from plotting_visualization import *

######################## Helping functions ##########################

cost_map = {
    'solar': 2.5,
    'wind': 2.5,
    'nuclear': 10,
    'hydro': 12.5,
    'lignite': 37.5,
    'hard coal': 50,
    'natural gas': 75,
    'fuel oil': 125
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
    cost_map = {
        'solar': 2.5,
        'wind': 2.5,
        'nuclear': 10,
        'hydro': 12.5,
        'lignite': 37.5,
        'hard coal': 50,
        'natural gas': 75,
        'fuel oil': 125
    }

    selected_row = df.loc[date,:]
