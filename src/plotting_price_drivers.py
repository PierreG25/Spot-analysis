import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_partregress
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_visualization import *

######################## Helping functions ##########################

def extract_drivers_list(df, non_driver=['Date', 'Price']):
    df_columns=df.columns
    drivers = [col for col in df_columns if col not in non_driver]
    return drivers

######################## Drivers identification ##########################

def plot_scatter_price_driver(df, start_year, end_year, x_col, y_col='Price'):
    print(x_col)
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]

    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.3, color='b')
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.set_title(f'{x_col.capitalize()} vs {y_col.capitalize()}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


def multiple_plot_scatter_price_driver1(df, start_year, end_year, columns, y_col='Price', n_cols=2, figsize=(15, 10), title=None):
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]

    n_plots = len(columns)
    if n_plots == 0:
        print("No columns provided to plot.")
        return

    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division for rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for idx, feature in enumerate(columns):
        print(feature)
        sns.scatterplot( x=df[feature], y=df[y_col], ax=axes[idx], alpha=0.2, color='m')
        axes[idx].set_title(f"{y_col} vs {feature}")
        axes[idx].tick_params(axis='x', rotation=30)
        axes[idx].set_xlabel(f'{feature} (MW)')
        axes[idx].grid(True, linestyle='--', alpha=0.5)
    
    # Hide any unused subplots
    for j in range(n_plots, len(axes)):
        print("TEST")
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.show()


def plot_price_by_binned_driver(df, start_year, end_year, col, price_col='Price', bins=10):
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]

    df_temp = df[[col, price_col]].copy()
    df_temp['bin'] = pd.cut(df_temp[col], bins=bins)
    binned_means = df_temp.groupby('bin')[price_col].mean()

    fig, ax=plt.subplots(figsize=(12,6))
    binned_means.plot(kind='bar')
    ax.set_title(f'Average {price_col.capitalize()} by Binned {col.capitalize()}')
    ax.set_ylabel(f'{price_col.capitalize()}')
    ax.set_xlabel(f'{col.capitalize()} Bins')
    ax.grid(axis='y')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

######################## Multivariate Driver Screening ##########################

def plot_correlation_matrix(df, cols, start_year, end_year):
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    ax.set_title('Correlation matrix')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_partial_regression(df, start_year, end_year, target, driver, controls, figsize=(8,6), title=None):
    """
    Plot a partial regression plot showing the relationship between the driver and the target,
    after controlling for other variables.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset containing target, driver, and control variables.
    target : str
        The dependent variable (e.g., 'price').
    driver : str
        The independent variable of interest (e.g., 'load').
    controls : list of str
        The other explanatory variables to control for (e.g., ['wind', 'temperature']).
    figsize : tuple
        Size of the plot.
    title : str
        Optional custom title.

    Returns:
    --------
    None (displays plot)
    """
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]
    # Prepare X and y
    X = df[[driver] + controls]
    X = sm.add_constant(X)  # adds intercept
    y = df[target]

    # Fit the full model
    model = sm.OLS(y, X).fit()

    # Create partial regression plot
    fig = plt.figure(figsize=figsize)
    sm.graphics.plot_partregress(endog=target,
                                 exog_i=driver,
                                 exog_others=controls,
                                 data=df,
                                 obs_labels=False,
                                 ax=fig.add_subplot(111),
                                 alpha=0.5)

    # Title
    if title is None:
        title = f"Partial Regression Plot: {driver} vs {target}"
    plt.title(title)
    plt.xlabel(f"{driver} (adjusted)")
    plt.ylabel(f"{target} (adjusted)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multiple_partial_regressions(df, target, drivers, controls, start_year=None, end_year=None, figsize=(6, 4)):
    """
    Plot partial regression plots for multiple drivers against a target variable,
    controlling for other specified variables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all relevant data.
    target : str
        Name of the dependent variable column.
    drivers : list of str
        List of driver variable names to plot partial regressions for.
    controls : list of str
        List of control variable names to adjust for.
    start_year : int or str, optional
        Start year (inclusive) to filter data. If None, no lower bound.
    end_year : int or str, optional
        End year (inclusive) to filter data. If None, no upper bound.
    figsize : tuple, optional
        Size of each individual plot (width, height).

    Returns:
    --------
    None
    """
    # Filter by year if 'Date' column and years provided
    if ('Date' in df.columns) and (start_year is not None or end_year is not None):
        df = df.copy()
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        if start_year is not None:
            df = df[df['Year'] >= int(start_year)]
        if end_year is not None:
            df = df[df['Year'] <= int(end_year)]
    
    # Drop rows with missing data in relevant columns
    cols_to_check = [target] + drivers + controls
    df_clean = df.dropna(subset=cols_to_check)

    n_drivers = len(drivers)
    ncols = 2
    nrows = (n_drivers + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axes = axes.flatten() if n_drivers > 1 else [axes]

    for i, driver in enumerate(drivers):
        ax = axes[i]

        # Prepare exog variables: driver + controls with constant intercept
        exog_vars = [driver] + controls
        X = df_clean[exog_vars]
        X = sm.add_constant(X)
        y = df_clean[target]

        # Partial regression plot: y vs driver controlling for controls
        plot_partregress(endog=y, exog_i=df_clean[driver], exog_others=df_clean[controls],
                         ax=ax, obs_labels=False)
        ax.set_title(f"Partial Regression: {driver} vs {target}")

    # Hide any unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

######################## Model-Based Driver Importance ##########################