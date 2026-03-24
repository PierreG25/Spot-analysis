import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap
import optuna
import json
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


path_1 = 'data/clean/STEP 3/XGBoost/generation_by_type_fr.csv'
path_2 = 'data/clean/STEP 3/XGBoost/NP_by_country_FR.csv'
path_3 = 'data/clean/STEP 3/XGBoost/interco_stress_metrics.csv'

df_gen = pd.read_csv(path_1, parse_dates=['Time'])
df_np = pd.read_csv(path_2, parse_dates=['Time'])
df_interco = pd.read_csv(path_3, parse_dates=['Time'])


# ── Config ────────────────────────────────────────────────────────────────────
COMPUTE_DATA = False
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

def time_cols(df):
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['month'] = df['Time'].dt.month
    return df


def lag_features(df, col='Price', lags=[96, 672]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def prepare_data(df_gen, df_np, df_interco):
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

    df.to_csv('data/clean/STEP 3/XGBoost/master_dataset_v2.csv', index=False)
    print("Data prepared and saved to 'master_dataset_v2.csv'")
    return df


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
 
    print(f"** Dataset chargé : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
    print(f"   Période : {df[DATE_COL].min()} → {df[DATE_COL].max()}")
    print(f"   Prix — min: {df[TARGET].min():.1f} | max: {df[TARGET].max():.1f} "
          f"| mean: {df[TARGET].mean():.1f} €/MWh\n")
    return df
 

# ══════════════════════════════════════════════════════════════════════════════
# 2. SCENARIOS - STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════

 
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
    model = XGBRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
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
    Optuna tunin using time-series cross-validation on the full train+val set to maximize out-of-fold MAE.
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
            "eval_metric"       : "mae",
        }
 
        fold_maes = []
        for train_idx, val_idx in tscv.split(X_train_full):
            X_f_tr, y_f_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_f_val, y_f_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
 
            m = XGBRegressor(**params, early_stopping_rounds=30)
            m.fit(X_f_tr, y_f_tr,
                  eval_set=[(X_f_val, y_f_val)],
                  verbose=False)
            fold_maes.append(mean_absolute_error(y_f_val, m.predict(X_f_val)))
 
        return np.mean(fold_maes)
 
    print(f"── Optuna tuning ({N_TRIALS} trials × {N_SPLITS} folds) ──")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
 
    best = study.best_params
    print(f"\n✅ Meilleurs paramètres (MAE CV = {study.best_value:.2f} €/MWh)")
    for k, v in best.items():
        print(f"   {k}: {v}")
    print()
    return best, study


def save_best_params(best_params, file_path="data/clean/STEP 3/XGBoost/best_params.json"):
    with open(file_path, "w") as f:
        json.dump(best_params, f)
    print(f"Best parameters saved to {file_path}")


def load_best_params(file_path="data/clean/STEP 3/XGBoost/best_params.json"):
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
    model = XGBRegressor(
        **best_params,
        early_stopping_rounds=50,
        eval_metric="mae",
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
                    df_test: pd.DataFrame):
    print("── Évaluation finale ──")
    metrics = {}
    for X, y, lbl in [(X_train, y_train, "Train"),
                      (X_val,   y_val,   "Val  "),
                      (X_test,  y_test,  "Test ")]:
        metrics[lbl] = evaluate(y, model.predict(X), lbl)
    print()
 
    pred_test = model.predict(X_test)
    residuals = y_test.values - pred_test
 
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Évaluation du modèle XGBoost — Prix spot FR", fontsize=14, fontweight="bold")
 
    # (a) Actual vs Predicted — time series
    ax = axes[0, 0]
    ax.plot(df_test[DATE_COL].values, y_test.values, label="Réel", alpha=0.7, linewidth=0.8)
    ax.plot(df_test[DATE_COL].values, pred_test,     label="Prédit", alpha=0.7, linewidth=0.8)
    ax.set_title("Série temporelle : Réel vs Prédit (Test)")
    ax.set_ylabel("€/MWh")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

 
    # (b) Scatter Actual vs Predicted
    ax = axes[0, 1]
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    ax.scatter(y_test, pred_test, alpha=0.3, s=5, color="steelblue")
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Prix réel (€/MWh)")
    ax.set_ylabel("Prix prédit (€/MWh)")
    ax.set_title(f"Scatter — R²={metrics['Test ']['R2']:.4f}")
 
    # (c) Distribution des résidus
    ax = axes[1, 0]
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Résidu (€/MWh)")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des résidus — MAE={metrics['Test ']['MAE']:.2f} €/MWh")
 
    # (d) Résidus vs Prix réel
    ax = axes[1, 1]
    ax.scatter(y_test, residuals, alpha=0.3, s=5, color="darkorange")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Prix réel (€/MWh)")
    ax.set_ylabel("Résidu (€/MWh)")
    ax.set_title("Résidus vs Prix réel")
 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   → Graphique sauvegardé : evaluation.png\n")
    return pred_test, residuals, metrics
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8. ANALYSE SHAP — drivers du prix
# ══════════════════════════════════════════════════════════════════════════════
 
def shap_analysis(model, X_train, X_test, features):
    print("── Analyse SHAP ──")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # ShapValues object
 
    # ── (a) Summary plot — importance globale ──────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP — Importance globale des features", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── (b) Beeswarm plot — direction d'impact ────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.title("SHAP — Direction et magnitude d'impact", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
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
    plt.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── (d) Heatmap SHAP par heure  ───────────────────────────────────────
    if "hour" in features:
        shap_df = pd.DataFrame(shap_values.values, columns=features)
        shap_df["hour"] = X_test["hour"].values
        heatmap_data = shap_df.groupby("hour")[top4].mean()
 
        plt.figure(figsize=(12, 5))
        sns.heatmap(heatmap_data.T, cmap="RdBu_r", center=0,
                    linewidths=0.3, annot=False)
        plt.title("SHAP moyen par heure — Top 4 features", fontsize=12)
        plt.xlabel("Heure")
        plt.tight_layout()
        plt.savefig("shap_heatmap_hour.png", dpi=150, bbox_inches="tight")
        plt.show()
 
    print("   → Graphiques SHAP sauvegardés\n")
 
    # Tableau récapitulatif
    summary = importances.reset_index()
    summary.columns = ["Feature", "SHAP importance (mean |shap|)"]
    summary["Rank"] = range(1, len(summary) + 1)
    print(summary.to_string(index=False))
    return shap_values, importances
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 9. ANALYSE PAR RÉGIME  (segmentation des performances)
# ══════════════════════════════════════════════════════════════════════════════
 
def regime_analysis(df_test: pd.DataFrame, y_test, pred_test):
    print("\n── Analyse par régime de marché ──")
    df_res = df_test.copy()
    df_res["pred"]     = pred_test
    df_res["residual"] = y_test.values - pred_test
    df_res["abs_err"]  = np.abs(df_res["residual"])
 
    results = {}
 
    # Par heure
    if "hour" in df_res.columns:
        r = df_res.groupby("hour")["abs_err"].mean()
        results["MAE par heure"] = r
 
    # Par mois
    if "month" in df_res.columns:
        r = df_res.groupby("month")["abs_err"].mean()
        results["MAE par mois"] = r
 
    # Par jour de la semaine
    if "dayofweek" in df_res.columns:
        r = df_res.groupby("dayofweek")["abs_err"].mean()
        results["MAE par jour"] = r
 
    # Par quartile de prix réel (performance sur les spikes)
    df_res["price_quartile"] = pd.qcut(y_test, q=4,
                                       labels=["Q1 (bas)", "Q2", "Q3", "Q4 (haut)"])
    results["MAE par quartile de prix"] = df_res.groupby("price_quartile")["abs_err"].mean()
 
    # Par congestion (si colonnes présentes)
    for cong_col in ["congestion_FR_BE", "congestion_FR_DE", "congestion_FR_NL"]:
        if cong_col in df_res.columns:
            r = df_res.groupby(cong_col)["abs_err"].mean()
            results[f"MAE | {cong_col}"] = r
 
    # Affichage
    for title, series in results.items():
        print(f"\n  {title}")
        print(series.to_string())
 
    # Graphique : MAE par heure & par quartile prix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance par régime de marché", fontsize=13, fontweight="bold")
 
    if "MAE par heure" in results:
        results["MAE par heure"].plot(kind="bar", ax=axes[0], color="steelblue")
        axes[0].set_title("MAE par heure de la journée")
        axes[0].set_xlabel("Heure")
        axes[0].set_ylabel("MAE (€/MWh)")
 
    results["MAE par quartile de prix"].plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("MAE par quartile de prix réel")
    axes[1].set_xlabel("Quartile")
    axes[1].set_ylabel("MAE (€/MWh)")
 
    plt.tight_layout()
    plt.savefig("regime_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n   → Graphique sauvegardé : regime_analysis.png")
 
 
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
 
    # ── 7. SHAP ──────────────────────────────────────────────────────────────
    if SHAP_ANALYSIS:
        shap_values, importances = shap_analysis(final_model, X_train, X_test, FEATURES)
 
    # ── 8. Regimes ───────────────────────────────────────────────────────────
    regime_analysis(test, y_test, pred_test)
 
    print("\n🎯 Pipeline terminé.")
