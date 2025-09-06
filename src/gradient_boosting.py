import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =================== HELPING FUNCTIONS ===================

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
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    return y_pred, rmse, mae


# =================== CHARTS ===================


def plot_forecast(df_test, y_test, y_pred):
    plt.figure(figsize=(15,6))


def plot_error_distribution(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, edgecolor='k', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error [EUR/MWh]')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()


def plot_parity(y_test, y_pred):
    plt.figure(figsize=(8,8))