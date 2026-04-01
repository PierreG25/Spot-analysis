import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap
import optuna
import json
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy import stats
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from fbmc_master import sequence_selection, setup_time, downsample_to_15

save_plots_path = 'figures/STEP 3/XGBoost'

path_1 = 'data/clean/STEP 3/XGBoost/generation_by_type_fr.csv'
path_2 = 'data/clean/STEP 3/XGBoost/NP_by_country_FR.csv'
path_3 = 'data/clean/STEP 3/XGBoost/interco_stress_metrics.csv'

path_fr_price_24 = 'data/raw/fbmc/price/2024_price_fr_raw.csv'

df_gen = pd.read_csv(path_1, parse_dates=['Time'])
df_np = pd.read_csv(path_2, parse_dates=['Time'])
df_interco = pd.read_csv(path_3, parse_dates=['Time'])


# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Full" # 2 options: "fr_only", "Full"

COMPUTE_DATA = True
OPTIUNA_TUNING = False
SHAP_ANALYSIS = True

DATA_PATH   = "data/clean/STEP 3/XGBoost/master_dataset_v2.csv"
TARGET      = "Price"
DATE_COL    = "Time"
N_SPLITS    = 5                    # folds pour la validation croisée temporelle
N_TRIALS    = 80                   # nombre d'essais Optuna
RANDOM_SEED = 42
plt.style.use("seaborn-v0_8-whitegrid")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPARE THE DATA
# ══════════════════════════════════════════════════════════════════════════════

def add_cols_price(df_1, df_2, col='Price 2024'):
    """
    Add a new column to df_1 with the day-ahead price from df_2, after processing and smoothing it.
    """

    df_2 = sequence_selection(df_2)
    df_2 = setup_time(df_2, "Day-ahead Price (EUR/MWh)")
    df_2 = df_2.rename(columns={"MTU (CET/CEST)": "Time", "Day-ahead Price (EUR/MWh)": col})
    df_2[col] = df_2[col].rolling(window=672, center=True, min_periods=1).mean()  # Simple moving average sur 96 périodes (24h)

    df_2.to_csv('data/clean/STEP 3/XGBoost/price_2024_15min.csv', index=False)

    df_1[col] = df_2[col]

    return df_1

def drop_cols(df, cols):
    if cols is None or len(cols) == 0:
        return df
    
    df = df.copy()
    df.drop(columns=cols, inplace=True)
    return df

def keep_cols(df, cols):
    if cols is None or len(cols) == 0:
        return df
    
    df = df.copy()
    df = df[cols]
    return df


def time_cols(df):
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['month'] = df['Time'].dt.month
    return df


def lag_features(df, col='Price', lags=[96, 672]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def prepare_data(df_gen,
                 df_np,
                 df_interco):
    df = pd.merge(df_np, df_gen, on='Time', how='inner')
    df = pd.merge(df, df_interco, on='Time', how='left').fillna({
        'stress_BE': 0,
        'stress_DE': 0,
        'stress_NL': 0,
        'stressed_BE': False,
        'stressed_DE': False,
        'stressed_NL': False
    })

    df = time_cols(df)
    df = lag_features(df)

    # Residual load implementation
    df['res_load'] = df['Total load'] - df['Renewable']
    df.drop(columns=['Total load', 'Renewable'], inplace=True)

    if EXPERIMENT_NAME == "fr_only":
        cols_to_drop = ['stressed_NL', 'congestion_FR_NL', 'spread_FR_NL', 'stress_NL',
                        'stressed_BE', 'congestion_FR_BE', 'spread_FR_BE', 'stress_BE',
                        'stressed_DE', 'congestion_FR_DE', 'spread_FR_DE', 'stress_DE',
                        'congestion_FR_ES', 'spread_FR_ES', 'Net position']

        df = drop_cols(df, cols_to_drop)

    df.to_csv('data/clean/STEP 3/XGBoost/master_dataset_v2.csv', index=False)
    print("Data prepared and saved to 'master_dataset_v2.csv'")
    return df


def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
 
    print(f"** Dataset chargé : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
    print(f"   Période : {df[DATE_COL].min()} → {df[DATE_COL].max()}")
    print(f"   Prix — min: {df[TARGET].min():.1f} | max: {df[TARGET].max():.1f} "
          f"| mean: {df[TARGET].mean():.1f} €/MWh\n")
    return df
 

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL  (70% train | 15% val | 15% test)
# ══════════════════════════════════════════════════════════════════════════════
 
def temporal_split(df):
    """Temporal split : 70% train | 15% val | 15% test"""
    n = len(df)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)

    train = df.iloc[:i_val].copy()
    val   = df.iloc[i_val:i_test].copy()
    test  = df.iloc[i_test:].copy()

    print(f"** Split temporel")
    print(f"   Train : {train[DATE_COL].min().date()} → {train[DATE_COL].max().date()} "
            f"({len(train):,} obs)")
    print(f"   Val   : {val[DATE_COL].min().date()} → {val[DATE_COL].max().date()} "
            f"({len(val):,} obs)")
    print(f"   Test  : {test[DATE_COL].min().date()} → {test[DATE_COL].max().date()} "
            f"({len(test):,} obs)\n")
    return train, val, test
 
 
def get_xy(df, features):
    X = df[features]
    y = df[TARGET]
    return X, y
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════
 
def mape(y_true, y_pred, eps=1e-3):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
 
def evaluate(y_true, y_pred, label=""):
    """Compute and print MAE, RMSE, MAPE and R² metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mpe  = mape(y_true.values, y_pred)
    print(f"  [{label}]  MAE={mae:.2f} €/MWh | RMSE={rmse:.2f} | "
          f"MAPE={mpe:.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mpe, "R2": r2}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. BASELINE  (paramètres par défaut)
# ══════════════════════════════════════════════════════════════════════════════
 
def run_baseline(X_train, y_train, X_val, y_val):
    """
    Train a baseline XGBoost model with default parameters and early stopping on validation set.
    """
    print("── Baseline XGBoost (params by default) ──")
    model = XGBRegressor(missing=np.inf,
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    pred_val = model.predict(X_val)
    evaluate(y_val, pred_val, "Baseline Val")
    print()
    return model
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING — Optuna + TimeSeriesSplit
# ══════════════════════════════════════════════════════════════════════════════
 
def tune_hyperparams(X_train_full, y_train_full):
    """
    Optuna tunin using time-series cross-validation on the full train+val set to maximize out-of-fold RMSE.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
 
    def objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 300, 1500),
            "max_depth"         : trial.suggest_int("max_depth", 3, 10),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight"  : trial.suggest_int("min_child_weight", 1, 20),
            "gamma"             : trial.suggest_float("gamma", 0, 5),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 0.5, 5),
            "random_state"      : RANDOM_SEED,
            "n_jobs"            : -1,
            "eval_metric"       : "rmse",
        }
 
        fold_maes = []
        for train_idx, val_idx in tscv.split(X_train_full):
            X_f_tr, y_f_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_f_val, y_f_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
 
            m = XGBRegressor(missing=np.inf, **params, early_stopping_rounds=30)
            m.fit(X_f_tr, y_f_tr,
                  eval_set=[(X_f_val, y_f_val)],
                  verbose=False)
            fold_maes.append(np.sqrt(mean_squared_error(y_f_val, m.predict(X_f_val))))
 
        return np.mean(fold_maes)
 
    print(f"── Optuna tuning ({N_TRIALS} trials × {N_SPLITS} folds) ──")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
 
    best = study.best_params
    print(f"\n** Meilleurs paramètres (RMSE CV = {study.best_value:.2f} €/MWh)")
    for k, v in best.items():
        print(f"   {k}: {v}")
    print()
    return best, study


def save_best_params(best_params, file_path = f"{save_plots_path}/{EXPERIMENT_NAME}/best_params.json"):
    with open(file_path, "w") as f:
        json.dump(best_params, f)
    print(f"Best parameters saved to {file_path}")


def load_best_params(file_path = f"{save_plots_path}/{EXPERIMENT_NAME}/best_params.json"):
    try:
        with open(file_path, "r") as f:
            best_params = json.load(f)
        print(f"Best parameters loaded from {file_path}")
        return best_params
    except FileNotFoundError:
        print(f"No saved parameters found at {file_path}. Running optimization...")
        return None
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. MODÈLE FINAL
# ══════════════════════════════════════════════════════════════════════════════
 
def train_final_model(best_params, X_train, y_train, X_val, y_val):
    """
    Entraîne le modèle final avec early stopping sur la validation.
    """
    print("── Entraînement du modèle final ──")
    model = XGBRegressor(missing=np.inf,
        **best_params,
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )
    print(f"   Meilleur n_estimators (early stopping) : {model.best_iteration}\n")
    return model
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7. ÉVALUATION COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════
 
def full_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test,
                    df_test,
                    path = f"{save_plots_path}/{EXPERIMENT_NAME}/"):
    print("── Final model evaluation ──")
    metrics = {}
    for X, y, lbl in [(X_train, y_train, "Train"),
                      (X_val,   y_val,   "Val  "),
                      (X_test,  y_test,  "Test ")]:
        metrics[lbl] = evaluate(y, model.predict(X), lbl)
    print()
 
    pred_test = model.predict(X_test)
    residuals = y_test.values - pred_test

    mean_r  = np.mean(residuals)
    std_r   = np.std(residuals)
    skew_r  = stats.skew(residuals)
 
    fig, axes = plt.subplots(1,2, figsize=(12, 6))
    fig.suptitle("Évaluation du modèle XGBoost — Prix spot FR", fontsize=14, fontweight="bold")

    # (a) Actual vs Predicted — time series
    ax = axes[0]
    ax.plot(df_test[DATE_COL].values, y_test.values, label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(df_test[DATE_COL].values, pred_test,     label="Forecast", alpha=0.7, linewidth=0.8)
    ax.set_title(f"Market Price : Actual vs Forecast (Test) — RMSE={metrics['Test ']['RMSE']:.2f} €/MWh")
    ax.set_ylabel("€/MWh")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # (b) Scatter Actual vs Predicted
    ax = axes[1]
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    ax.scatter(y_test, pred_test, alpha=0.3, s=5, color="steelblue")
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Prix réel (€/MWh)")
    ax.set_ylabel("Prix prédit (€/MWh)")
    ax.set_title(f"Scatter — R²={metrics['Test ']['R2']:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{path}evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   → Graphique sauvegardé : evaluation.png\n")


    # --------- Residual analysis ---------
    
    fig, axes = plt.subplots(1,2, figsize=(12, 6))
    fig.suptitle("Residual analysis", fontsize=14, fontweight="bold")

    # (c) Distribution des résidus

    ax = axes[0]
    x_range = np.linspace(residuals.min(), residuals.max(), 300)

    ax.hist(residuals, bins=80, density = True, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.plot(x_range, stats.norm.pdf(x_range, mean_r, std_r),
            color="tomato", linewidth=1.6, linestyle="--", label="Normal dist. (μ, σ)")
    ax.plot(x_range, stats.gaussian_kde(residuals)(x_range),
            color="navy", linewidth=1.4, label="KDE")
    ax.axvline(0,      color="black",      linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_xlabel("Residuals (€/MWh)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residuals distribution — RMSE={metrics['Test ']['RMSE']:.2f} €/MWh")
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.text(
        0.97, 0.97,
        f"μ  = {mean_r:+.2f} €/MWh\n"
        f"σ  = {std_r:.2f} €/MWh\n"
        f"Skewness = {skew_r:.3f}\n",
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgray", alpha=0.85)
    )
 
    # (d) Q-Q plot

    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    se     = (1 / stats.norm.pdf(osm)) * np.sqrt(osr * (1 - osr) / len(residuals))
    # ax.fill_between(osm, slope * osm + intercept - 1.96 * se * slope,
    #                      slope * osm + intercept + 1.96 * se * slope,
    #                 alpha=0.15, color="tomato", label="IC 95 %")
    ax.scatter(osm, osr, s=4, color="steelblue", alpha=0.5, label="Quantiles obs.")
    ax.plot(osm, slope * osm + intercept,
            color="tomato", linewidth=1.5, linestyle="--", label="Normal dist. (μ, σ)")

    ax.set_xlabel("Theoretical quantiles (Normal dist.)")
    ax.set_ylabel("Observed quantiles (residuals)")
    ax.set_title("Q-Q plot of residuals")
    ax.legend(fontsize=8.5, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{path}residuals_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   → Graphique sauvegardé : residuals_analysis.png\n")

    print("── Résumé statistique des résidus ──")
    print(f"   Moyenne       : {mean_r:+.3f} €/MWh  (biais — idéal = 0)")
    print(f"   Écart-type    : {std_r:.3f} €/MWh")
    print(f"   Skewness      : {skew_r:.3f}  (>0 = queue droite = sous-estim. spikes)")

    # (e) Heatmap mean residuals by hour and month
    if "hour" in df_test.columns and "month" in df_test.columns:
        heatmap_data = df_test.copy()
        heatmap_data["residual"] = residuals
        pivot = heatmap_data.pivot_table(index="hour", columns="month", values="residual", aggfunc="mean")

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="RdBu_r", center=0, linewidths=0.3, annot=False)
        plt.title("Mean residuals by hour and month", fontsize=12)
        plt.xlabel("Month")
        plt.ylabel("Hour")
        plt.tight_layout()
        plt.savefig(f"{path}residuals_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("   → Graphique sauvegardé : residuals_heatmap.png\n")

    # --------- Spikes analysis ---------

    # (f) Spikes analysis
    spikes_analysis(df_test, y_test, pred_test, metrics)

    return pred_test, residuals, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 8. SPIKES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def spikes_analysis(df_test,
                    y_test,
                    pred_test,
                    metrics,
                    spike_q=0.95,
                    path = f"{save_plots_path}/{EXPERIMENT_NAME}/"):
    
    y_array = y_test.values
    residuals = y_array - pred_test

    # Détection des spikes positifs et negatifs
    threshold_pos = np.quantile(y_array, spike_q)
    threshold_neg = np.quantile(y_array, 1 - spike_q)

    spike_mask = (y_array >= threshold_pos) | (y_array <= threshold_neg)
    normal_mask = ~spike_mask

    print(f"   Seuil Q{int(spike_q*100)} : {threshold_pos:.1f} €/MWh")
    print(f"   Heures spike  : {spike_mask.sum()/4} ({spike_mask.mean()*100:.1f} %)")
    print(f"   Heures normal : {normal_mask.sum()/4}\n")

    # Real vs Predicted — Spikes vs Normal

    fig, axes = plt.subplots(1,2, figsize=(12, 6))
    fig.suptitle(
        f"Spike analysis — threshold Q{int(spike_q*100)} = {threshold_pos:.1f} €/MWh",
        fontsize=13, fontweight="bold"
    )
 
    ax = axes[0]

    # Surlignage des zones spike (groupes consécutifs)
    spike_series = pd.Series(spike_mask, index=df_test[DATE_COL].values)
    changes      = spike_series.ne(spike_series.shift()).cumsum()
    first_span   = True
    for _, grp in spike_series.groupby(changes):
        if grp.iloc[0]:
            ax.axvspan(grp.index[0], grp.index[-1],
                       color="tomato", alpha=0.20, zorder=1,
                       label=f"Spike (Q{int(spike_q*100)}+)" if first_span else "_nolegend_")
            first_span = False
 
    ax.plot(df_test[DATE_COL], y_array,    color="black",     linewidth=0.9,
            alpha=0.85, label="Actual",   zorder=3)
    ax.plot(df_test[DATE_COL], pred_test, color="steelblue", linewidth=0.9,
            alpha=0.80, label="Forecast", zorder=2)
 
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title(
        f"Actual vs Forecast — "
        f"MAE global : {metrics['Test ']['MAE']:.2f} €/MWh  |  "
        f"RMSE global : {metrics['Test ']['RMSE']:.2f} €/MWh"
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=9, framealpha=0.8)
 
    # Sous-plot résidus — colorés rouge si spike, bleu si normal
    ax2 = axes[1]
    colors = ["tomato" if s else "steelblue" for s in spike_mask]
    ax2.bar(df_test[DATE_COL], residuals, color=colors, width=0.04, alpha=0.55)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residual (€/MWh)")
    ax2.set_xlabel("Time")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
 
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{path}spike_actual_vs_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   → Graphique sauvegardé : spike_actual_vs_forecast.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ANALYSE SHAP — price drivers
# ══════════════════════════════════════════════════════════════════════════════
 
def shap_analysis(model, X_train, X_test, features, path = f"{save_plots_path}/{EXPERIMENT_NAME}/"):
    print("── Analyse SHAP ──")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # ShapValues object
 
    # ── (a) Summary plot — importance globale ──────────────────────────────
    # Importance globale = moyenne des |SHAP values|
    importance = np.abs(shap_values.values).mean(axis=0)

    # DataFrame pour trier
    imp_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": importance
    }).sort_values("importance", ascending=False).head(20)

    # Pour avoir les features les plus importantes de gauche à droite
    imp_df = imp_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(imp_df["feature"], imp_df["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean(|SHAP value|)")
    plt.xlabel("Features")
    plt.title("SHAP — Importance globale des features", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{path}shap_importance_vertical.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── (b) Beeswarm plot — direction d'impact ────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.title("SHAP — Direction et magnitude d'impact", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{path}shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── (c) Dependence plots — top 4 features ────────────────────────────
    importances = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=features
    ).sort_values(ascending=False)
    top4 = importances.index[:4].tolist()
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SHAP Dependence Plots — Top 4 features", fontsize=13, fontweight="bold")
    for ax, feat in zip(axes.flatten(), top4):
        shap.dependence_plot(feat, shap_values.values, X_test,
                             ax=ax, show=False)
        ax.set_title(feat, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{path}shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── (d) Heatmap SHAP par heure  ───────────────────────────────────────
    # if "hour" in features:
    #     shap_df = pd.DataFrame(shap_values.values, columns=features)
    #     shap_df["hour"] = X_test["hour"].values
    #     heatmap_data = shap_df.groupby("hour")[top4].mean()
 
    #     plt.figure(figsize=(12, 5))
    #     sns.heatmap(heatmap_data.T, cmap="RdBu_r", center=0,
    #                 linewidths=0.3, annot=False)
    #     plt.title("SHAP moyen par heure — Top 4 features", fontsize=12)
    #     plt.xlabel("Heure")
    #     plt.tight_layout()
    #     plt.savefig(f"{path}shap_heatmap_hour.png", dpi=150, bbox_inches="tight")
    #     plt.show()
 
    print("   → Graphiques SHAP sauvegardés\n")
 
    # Tableau récapitulatif
    summary = importances.reset_index()
    summary.columns = ["Feature", "SHAP importance (mean |shap|)"]
    summary["Rank"] = range(1, len(summary) + 1)
    print(summary.to_string(index=False))
    return shap_values, importances


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
 
    # ── 1. Chargement ────────────────────────────────────────────────────────
    if COMPUTE_DATA:
        df = prepare_data(df_gen, df_np, df_interco)
    else:
        df = load_and_prepare(DATA_PATH)
 
    # Définition des features (toutes les colonnes sauf Time et Price)
    FEATURES = [c for c in df.columns if c not in [DATE_COL, TARGET]]
 
    # ── 2. Split ─────────────────────────────────────────────────────────────
    train, val, test = temporal_split(df)
 
    X_train, y_train = get_xy(train, FEATURES)
    X_val,   y_val   = get_xy(val,   FEATURES)
    X_test,  y_test  = get_xy(test,  FEATURES)
 
    # Concat train+val for tuning
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
 
    # ── 3. Baseline ──────────────────────────────────────────────────────────
    baseline = run_baseline(X_train, y_train, X_val, y_val)
 
    # ── 4. Tuning Optuna ─────────────────────────────────────────────────────
    if OPTIUNA_TUNING:
        best_params, study = tune_hyperparams(X_tv, y_tv)
        save_best_params(best_params)
    else:
        best_params = load_best_params()
        if best_params is None:
            best_params, study = tune_hyperparams(X_tv, y_tv)
            save_best_params(best_params)
 
    # ── 5. Final model  ──────────────────────────────────────────────────────
    final_model = train_final_model(best_params, X_train, y_train, X_val, y_val)
 
    # ── 6. Evaluation ────────────────────────────────────────────────────────
    pred_test, residuals, metrics = full_evaluation(
        final_model, X_train, y_train, X_val, y_val, X_test, y_test, test
    )
    
    final_model.save_model(f"{save_plots_path}/{EXPERIMENT_NAME}/final_xgboost_model.json")

    with open(f"{save_plots_path}/{EXPERIMENT_NAME}/final_xgboost_model.json".replace(".json", "_features.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    with open(f"{save_plots_path}/{EXPERIMENT_NAME}/final_xgboost_model.json".replace(".json", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "best_iteration": int(final_model.best_iteration) if final_model.best_iteration is not None else None,
        "features_count": len(FEATURES),
        "train_start": str(train[DATE_COL].min()),
        "train_end": str(train[DATE_COL].max()),
        "val_start": str(val[DATE_COL].min()),
        "val_end": str(val[DATE_COL].max()),
        "test_start": str(test[DATE_COL].min()),
        "test_end": str(test[DATE_COL].max()),
    }
    with open(f"{save_plots_path}/{EXPERIMENT_NAME}/final_xgboost_model.json".replace(".json", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── 7. SHAP ──────────────────────────────────────────────────────────────
    if SHAP_ANALYSIS:
        shap_values, importances = shap_analysis(final_model, X_train, X_test, FEATURES)
 
    print("\n🎯 Pipeline terminé.")
