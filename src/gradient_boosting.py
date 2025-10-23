import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotting_visualization import *

# =================== HELPING FUNCTIONS ===================

def create_lag_features(df, target_col='Price'):
    """
    Creates lag features for the target column.
    """
    # Time features
    df["hour"] = df["Date"].dt.hour
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Lags

    df["price_lag_24"] = df[target_col].shift(-24)

    df = df.drop(df.index[-24:])

    return df

# =================== XGBOOST MODEL ===================


def train_test_split_time(df, target_col='Price', train_size=0.8):
    """
    Splits the dataframe into training and testing sets based on time.
    """
    split_index = int(len(df) * train_size)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    return X_train, X_test, y_train, y_test


def train_test_split_time2(df, target_col='Price', train_size=0.995):
    """
    Splits the dataframe into training and testing sets based on time.
    """
    split_index = int(len(df) * train_size)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    split_index_val = int(len(train) * train_size)
    val = train.iloc[split_index_val:]
    train = train.iloc[:split_index_val]
    
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost model with early stopping
    """
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse"
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns MAE and RMSE
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    return y_pred, rmse, mae


# =================== CHARTS ===================


def plot_forecast(df_test, y_test, y_pred):
    plt.figure(figsize=(15,6))
    plt.plot(df_test.tail(len(y_test)).index + pd.Timedelta(hours=24), y_test, label='Actual', color='blue')
    plt.plot(df_test.tail(len(y_test)).index + pd.Timedelta(hours=24), y_pred, label='Forecast', color='orange')
    plt.legend()
    plt.title('Market Price Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price (EUR/MWh)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'../figures/forecast_xgboost.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_error_distribution(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, edgecolor='k', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error (EUR/MWh)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()


def plot_parity(y_test, y_pred):
    plt.figure(figsize=(8,8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", alpha=0.8)
    plt.xlabel("Actual Price (EUR/MWh)")
    plt.ylabel("Predicted Price (EUR/MWh)")
    plt.title("Parity Plot: Actual vs Predicted")
    plt.savefig(f'../figures/parity.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()