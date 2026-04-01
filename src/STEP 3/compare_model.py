import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from spot_reconstruct import temporal_split, get_xy


TARGET      = "Price"
DATE_COL    = "Time"

model_path_1 ='figures/STEP 3/XGBoost/full/final_xgboost_model.json'
model_path_2 ='figures/STEP 3/XGBoost/fr_only/final_xgboost_model.json'

features_path_1 = 'figures/STEP 3/XGBoost/full/final_xgboost_model_features.json'
features_path_2 = 'figures/STEP 3/XGBoost/fr_only/final_xgboost_model_features.json'

# ══════════════════════════════════════════════════════════════════════════════
# LOADING MODELS AND FEATURE LISTS
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_list(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_saved_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODELS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def models_comparison(df,
    model1_path,
    model2_path,
    features1_path,
    features2_path,
    label1="Model 1",
    label2="Model 2",
    zoom_start=None,
    zoom_end=None):

    # Load models and feature lists
    model1 = load_saved_model(model1_path)
    model2 = load_saved_model(model2_path)

    features1 = load_feature_list(features1_path)
    features2 = load_feature_list(features2_path)

    # Implement X_test for both models

    test = temporal_split(df)[2]

    X_test1 = get_xy(test,  features1)[0]
    X_test2 = get_xy(test,  features2)[0]
    y_test = test[TARGET].copy()

    # Predictions
    y_pred1 = model1.predict(X_test1)
    y_pred2 = model2.predict(X_test2)

    # Plotting predictions vs actual values
    plt.figure(figsize=(18, 6))
    plt.plot(test[DATE_COL], y_test, label="Réel", linewidth=1.2, alpha=0.9)
    plt.plot(test[DATE_COL], y_pred1, label=label1, linewidth=1.0, alpha=0.8)
    plt.plot(test[DATE_COL], y_pred2, label=label2, linewidth=1.0, alpha=0.8)

    plt.title("Comparison of 2 models vs actual values")
    plt.ylabel("Price (€/MWh)")
    plt.xlabel("Time")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig("figures/STEP 3/XGBoost/models_comparison.png", dpi=300)
    plt.show()

    # Zoom on a specific period (ex: 1 month)

    if zoom_start is not None or zoom_end is not None:
        df_plot = test.copy()
        df_plot["pred1"] = y_pred1
        df_plot["pred2"] = y_pred2

        if zoom_start is not None:
            df_plot = df_plot[df_plot[DATE_COL] >= pd.to_datetime(zoom_start)]
        if zoom_end is not None:
            df_plot = df_plot[df_plot[DATE_COL] <= pd.to_datetime(zoom_end)]

        plt.figure(figsize=(18, 6))
        plt.plot(df_plot[DATE_COL], df_plot[TARGET], label="Réel", linewidth=1.5)
        plt.plot(df_plot[DATE_COL], df_plot["pred1"], label=label1, linewidth=1.2)
        plt.plot(df_plot[DATE_COL], df_plot["pred2"], label=label2, linewidth=1.2)

        plt.title("Zoom test")
        plt.ylabel("Price (€/MWh)")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/STEP 3/XGBoost/models_comparison_zoom.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/clean/STEP 3/XGBoost/master_dataset_v2.csv", parse_dates=[DATE_COL])

    models_comparison(df, model_path_1, model_path_2, features_path_1, features_path_2,
                      label1="FR-only model", label2="Full model")