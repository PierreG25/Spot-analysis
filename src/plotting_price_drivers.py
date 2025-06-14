import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import pandas as pd
import math
from plotting_visualization import *

######################## Helping functions ##########################

def extract_drivers_list(df, non_driver=['Date', 'Day-ahead Price (EUR/MWh)']):
    df_columns=df.columns
    drivers = [col for col in df_columns if col not in non_driver]
    return drivers

######################## Drivers identification ##########################

def plot_scatter_price_driver(df, start_year, end_year, x_col, y_col='Day-ahead Price (EUR/MWh)'):
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


def multiple_plot_scatter_price_driver1(df, start_year, end_year, columns, y_col='Day-ahead Price (EUR/MWh)', n_cols=2, figsize=(15, 10), title=None):
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


def plot_price_by_binned_driver(df, start_year, end_year, col, price_col='Day-ahead Price (EUR/MWh)', bins=10):
    df = ensure_datetime_index(df)
    mask = (df.index >= start_year) & (df.index < end_year)
    df = df.loc[mask]

    df_temp = df[[col, price_col]].copy()
    df_temp['bin'] = pd.cut(df_temp[col], bins=bins)
    binned_means = df_temp.groupby('bin')[price_col].mean()

    fig, ax=plt.subplots(figsize=(12,6))
    binned_means.plot(kind='bar')
    ax.set_title(f'Average {price_col.capitalize()} by Binned {col.capitalize()}')
    ax.set_ylabel(f'{price_col.capitalize()} (€)')
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

######################## Model-Based Driver Importance ##########################