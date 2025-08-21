import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotting_visualization import *

# =================== HELPING FUNCTIONS ===================

# Marginal costs
COST_MAP = {    
    'Renewables': 1,
    'Hydro': 5,
    'Nuclear': 10,
    'Lignite': 37.5,
    'Hard coal': 50,
    'Natural gas': 75,
    'Fuel oil': 125
}

ENERGY_COLORS = {
    "Renewables": "#2ca02c",
    "Hydro": "#1f77b4",
    "Nuclear": "#ffcc00",
    "Lignite": "#8c564b",
    "Hard coal": "#2c2c2c",
    "Natural gas": "#ff7f0e",
    "Fuel oil": "#d62728",
    "Others": "#d3d3d3"
}


def marginal_costs(generation_type,cost_map):
    """
    Check if the energy is within the dictionary and return the associated marginal cost

    Args:
        generation_type (str):
        cost_map (dict): Marginal costs for each energy technology

    Returns:
        float: marginal cost of the selected energy technology
    """

    if generation_type in cost_map:
        return cost_map[generation_type]
    raise ValueError("Wrong input; use the generation technologies from the following list:" \
                    " solar, wind, nuclear, hydro, lignite, hard coal, natural gas, and fuel oil")


def plot_donut(map_data, threshold_pct=3):
    """
    Plot a filtered donut chart showing the predominant values in the data.

    Values whose percentage contribution is below the given threshold are
    grouped into a single slice labeled "Others".

    Args:
        map_data (dict): A dictionary where keys are category names (str) and
            values are their corresponding numerical amounts (int or float).
        threshold_pct (int or float): Percentage threshold below which
            categories are consolidated into the "Others" slice.
            Defaults to 3.

    Returns:
        None
    """
    total = sum(map_data.values())
    threshold_value = total * (threshold_pct / 100)

    # Group data
    large = {k: v for k, v in map_data.items() if v >= threshold_value}
    small = {k: v for k, v in map_data.items() if v < threshold_value}

    if small:
        large["Others"] = sum(small.values())

    labels = list(large.keys())
    sizes = list(large.values())

    # Colors
    colors = [ENERGY_COLORS[energy] for energy in labels]

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


# =================== MERIT ORDER CURVE ====================

def merit_order_curve(df, date):
    """
    Plot the merit order curve for a specific date and highlight the price 
    corresponding to the forecasted load.

    Simplifications & Assumptions:
        - Imports and exports of electricity are excluded from the analysis
        - Marginal costs are assumed constant for each generation technology
          due to limited data granularity
        - Actual available capacities are used for each generation technology,
          not D-1 forecasted capacities
        - If forecasted load exceeds total available supply at that time,
          the price is set based on the marginal cost of the last available
          generation source at its maximum capacity

    Args:
        df (pd.DataFrame): DataFrame containing at least the following columns:
            - 'Date'
            - 'Price'
            - 'Forecasted Load'
            - 'Load'
            - 'Renewables'
            - 'Hydro'
            - 'Nuclear'
            - 'Lignite'
            - 'Hard coal'
            - 'Natural gas'
            - 'Fuel Oil'
        date (str): Target date and time in the format "YYYY-MM-DD HH:MM:SS".

    Returns:
        None
    """

    df = ensure_datetime_index(df)
    selected_row = df.loc[date]

    # New DataFrame to analyse the merit order effect
    rows = []
    for tech in selected_row.index:
        if tech in COST_MAP:
            rows.append({
                'technology': tech,
                'capacity': selected_row[tech],
                'marginal costs': marginal_costs(tech, COST_MAP)
            })
    df_merit = pd.DataFrame(rows)

    df_merit = df_merit.sort_values(by='marginal costs')
    df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()
    df_merit['previous capacity'] = df_merit['cumulative capacity'] - df_merit['capacity']

    # Create a figure with wide rectangular for the merit order plot
    plt.figure(figsize=(12, 6))
    for i, row in df_merit.iterrows():
        plt.fill_between(
            [row['previous capacity'], row['cumulative capacity']],
            [row['marginal costs'], row['marginal costs']],
            step='pre',
            y2=0,
            label=row['technology'],
            color=ENERGY_COLORS.get(row['technology'], 'gray')
        )

    forecast_load = df.loc[date]['Forecasted load']

    #Test to see missing data in the load column (be careful for time change)
    if pd.isna(forecast_load):
        raise ValueError('Forecasted load is missing for the selected date. \
                         Becareful for time change')
    
    # Test if forecasted load > supply (i.e max capacity)
    merit_order_limit = forecast_load > df_merit.iloc[-1,3]
    if merit_order_limit:

        marginal_unit = df_merit.iloc[-1]
        spot_price = marginal_unit['marginal costs']
        forecast_load = marginal_unit['cumulative capacity']    
    else:

        marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
        spot_price = marginal_unit['marginal costs']
    
    # Draw lines to highlight the marginal unit and electricity price
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


def donut_tech_dist(df, start_year, end_year):
    """
    Analyze which energy technologies set the market price in the given period
    and plot a donut chart showing their relative share.

    Args: 
        df (pd.DataFrame): DataFrame containing at least the following columns:
            - 'Date'
            - 'Price'
            - 'Forecasted Load'
            - 'Load'
            - 'Renewables'
            - 'Hydro'
            - 'Nuclear'
            - 'Lignite'
            - 'Hard coal'
            - 'Natural gas'
            - 'Fuel Oil'
        start_year (str): Start year (inclusive) in 'YYYY' format
        end_yeat (str): End year (exclusive) in 'YYYY' format
    
    Returns:
        None
    """

    df = ensure_datetime_index(df)    # To include in filter_data function
    df = filter_data(df, start_year, end_year)
    exceed_supply = 0   # Number of occurences where load > supply

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
        selected_row = df.loc[date]
        forecast_load = selected_row['Forecasted load']

        # New DataFrame to analyse the merit order effect
        rows = []
        for tech in selected_row.index:
            if tech in COST_MAP:
                rows.append({
                    'technology': tech,
                    'capacity': selected_row[tech],
                    'marginal costs': marginal_costs(tech, COST_MAP)
                })
        df_merit = pd.DataFrame(rows)

        df_merit = df_merit.sort_values(by='marginal costs')
        df_merit['cumulative capacity'] = df_merit['capacity'].cumsum()

        # Test if forecasted load > supply (i.e max capacity)
        merit_order_limit = forecast_load > df_merit.iloc[-1,3]
        if merit_order_limit:

            marginal_unit = df_merit.iloc[-1]
            marginal_tech = marginal_unit['technology']     # Last available source at its max capacitiy set the price
            exceed_supply+=1
        else: 
            marginal_unit = df_merit[df_merit['cumulative capacity'] >= forecast_load].iloc[0]
            marginal_tech = marginal_unit['technology']

        if marginal_tech in count_map:
            count_map[marginal_tech]+= 1
    
    print(count_map)
    print(exceed_supply)

    plot_donut(count_map)


# ===================== LINEAR REGRESSION =====================


def dummies(df, start_year, end_year, test=True):
    """
    Prepare dataset for OLS regression:
    - Create dummy variables for hour, day, month, year
    - Drop first category to avoid multicollinearity
    """
    df=ensure_datetime_index(df)
    df = filter_data(df, start_year, end_year)
    df['hour']= df.index.hour
    df['dayofweek']= df.index.dayofweek
    df['month']= df.index.month
    df['year']= df.index.year

    for col in COST_MAP:
        df[col] = df[col]/1000  # Convert MW to GW

    df = pd.get_dummies(
        df, columns=['hour', 'dayofweek', 'month', 'year'],
        drop_first=True
    )
    df = df.astype(float)
    if test:
        df.to_excel('../data/df_ols.xlsx', index=True)

    return df


def run_ols(df, target, drivers):
    """
    Run OLS regression on the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing target and driver variables.
        target (str): The dependent variable (e.g., 'price').
        drivers (list of str): List of independent variables (e.g., ['load', 'wind', 'temperature']).

    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: The fitted OLS model.
    """
    print(drivers + [col for col in df.columns if col.startswith(('year_', 'month_', 'dayofweek_', 'hour_'))])
    X = df[drivers + [col for col in df.columns if col.startswith(('year_', 'month_', 'dayofweek_', 'hour_'))]]
    X = sm.add_constant(X)  # adds intercept
    y = df[target]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return results


def plot_coefficients(results, drivers):
    """
    Plots OLS coefficients of the considered drivers
    """
    coef = results.params[drivers]
    print("Constant (baseline price):", results.params["const"])
    print(coef)
    conf_int = results.conf_int().loc[drivers]

    plt.figure(figsize=(12,6))
    coef.plot(kind='bar', yerr=[coef - conf_int[0], conf_int[1] - coef], capsize=5)

    plt.title('OLS coefficients')
    plt.ylabel('EUR/MWh')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


# ===================== UNFINISHED MEKKO CHART =====================

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
            if value in COST_MAP:
                new_row = {'technology': f'{value}','capacity': selected_row[value],'marginal costs': marginal_costs(value, COST_MAP)}
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