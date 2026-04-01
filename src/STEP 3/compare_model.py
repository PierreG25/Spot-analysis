import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from xgboost import XGBRegressor
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
    label1="Model Full",
    label2="Model FR only",
    zoom_start=None,
    zoom_end=None,
    cong=True,
    cong_cols=None,
    min_period_hours=48,
    max_zoom_windows=3,
    context_hours=48):

    # Load models and feature lists
    model1 = load_saved_model(model1_path)
    model2 = load_saved_model(model2_path)

    features1 = load_feature_list(features1_path)
    features2 = load_feature_list(features2_path)

    # Implement X_test for both models

    test = temporal_split(df)[2].reset_index(drop=True)

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

    # ── Zoom automatique sur périodes de congestion (cong=True) ───────────────

    if cong:
        _CONG_COLORS = {"FR_BE": "#a8d8ea", "FR_DE": "#ffb347",
                        "FR_NL": "#b0e57c", "FR_ES": "#c9a0dc"}
 
        # Colonnes par défaut si non fournies
        if cong_cols is None:
            cong_cols = ["congestion_FR_BE", "congestion_FR_DE",
                         "congestion_FR_NL", "congestion_FR_ES"]
 
        cong_existing = [c for c in cong_cols if c in test.columns]
 
        if not cong_existing:
            print("   [cong] Aucune colonne de congestion trouvée — vérifier cong_cols.")

        else:
            # Détection des périodes : au moins 1 interconnexion active
            cong_mask = test[cong_existing].sum(axis=1) >= 1
            changes   = cong_mask.ne(cong_mask.shift()).cumsum()
 
            periods = []
            for _, group in cong_mask.groupby(changes):
                if group.iloc[0] and len(group) >= min_period_hours:
                    periods.append((group.index[0], group.index[-1]))
 
            print(f"   [cong] Périodes détectées (>= {min_period_hours}h) : {len(periods)}")
            for i, (s, e) in enumerate(periods):
                active = [c.replace("congestion_", "") for c in cong_existing
                          if test[c].iloc[s:e+1].any()]
                print(f"     [{i+1}] {test[DATE_COL].iloc[s].strftime('%Y-%m-%d %Hh')} → "
                      f"{test[DATE_COL].iloc[e].strftime('%Y-%m-%d %Hh')} "
                      f"({e-s+1}h)  |  {', '.join(active)}")
 
            if not periods:
                print("   [cong] Aucune période — réduire min_period_hours.")
            else:
                # Les max_zoom_windows périodes les plus longues, retri chronologique
                windows = sorted(
                    sorted(periods, key=lambda x: x[1]-x[0], reverse=True)[:max_zoom_windows],
                    key=lambda x: x[0]
                )
 
                fig, axes = plt.subplots(len(windows), 1,
                                         figsize=(18, 5 * len(windows)),
                                         squeeze=False)
                fig.suptitle(
                    f"{label1} vs {label2} — zoom périodes de congestion",
                    fontsize=13, fontweight="bold"
                )
 
                for i, (idx_s, idx_e) in enumerate(windows):
                    ax = axes[i][0]
 
                    w_s = max(0, idx_s - context_hours)
                    w_e = min(len(test) - 1, idx_e + context_hours)
 
                    dates  = test[DATE_COL].iloc[w_s:w_e+1].reset_index(drop=True)
                    actual = y_test.iloc[w_s:w_e+1].values
                    p1     = y_pred1[w_s:w_e+1]
                    p2     = y_pred2[w_s:w_e+1]
 
                    # Surlignage fin par interconnexion
                    for cong_col in cong_existing:
                        key      = cong_col.replace("congestion_", "")
                        col_data = test[cong_col].iloc[w_s:w_e+1].reset_index(drop=True)
                        sub_chg  = col_data.ne(col_data.shift()).cumsum()
                        for _, grp in col_data.groupby(sub_chg):
                            if grp.iloc[0] == 1:
                                ax.axvspan(dates.iloc[grp.index[0]],
                                           dates.iloc[grp.index[-1]],
                                           color=_CONG_COLORS.get(key, "gray"),
                                           alpha=0.18, zorder=1)
 
                    # Surlignage global de la période sélectionnée
                    ax.axvspan(dates.iloc[idx_s - w_s], dates.iloc[idx_e - w_s],
                               color="gold", alpha=0.12, zorder=0,
                               label="Période sélectionnée")
 
                    # Tracés
                    ax.plot(dates, actual, color="black",     linewidth=1.2,
                            label="Réel", zorder=3)
                    ax.plot(dates, p1,     color="steelblue", linewidth=1.0,
                            alpha=0.85, label=label1, zorder=2)
                    ax.plot(dates, p2,     color="tomato",    linewidth=1.0,
                            linestyle="--", alpha=0.85, label=label2, zorder=2)
 
                    # MAE dans la zone de congestion
                    sl_s = idx_s - w_s
                    sl_e = idx_e - w_s + 1
                    mae1 = np.mean(np.abs(actual[sl_s:sl_e] - p1[sl_s:sl_e]))
                    mae2 = np.mean(np.abs(actual[sl_s:sl_e] - p2[sl_s:sl_e]))
                    ax.text(
                        0.01, 0.97,
                        f"Zone congestion  —  "
                        f"MAE {label1} : {mae1:.1f} €/MWh  |  "
                        f"MAE {label2} : {mae2:.1f} €/MWh  |  "
                        f"Δ : {mae2 - mae1:+.1f} €/MWh",
                        transform=ax.transAxes, fontsize=8.5,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="lightgray", alpha=0.85)
                    )
 
                    # Titre
                    active_labels = [
                        c.replace("congestion_", "") for c in cong_existing
                        if test[c].iloc[idx_s:idx_e+1].any()
                    ]
                    ax.set_title(
                        f"Période {i+1} — "
                        f"{test[DATE_COL].iloc[idx_s].strftime('%d %b %Y %Hh')} → "
                        f"{test[DATE_COL].iloc[idx_e].strftime('%d %b %Y %Hh')} "
                        f"({idx_e - idx_s + 1}h)   |   Actives : {', '.join(active_labels)}",
                        fontsize=10
                    )
                    ax.set_ylabel("Price (€/MWh)")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Hh"))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                    fig.autofmt_xdate(rotation=30, ha="right")
 
                    if i == 0:
                        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.8)
 
                    # Légende couleurs par liaison
                    from matplotlib.patches import Patch
                    cong_legend = [
                        Patch(facecolor=_CONG_COLORS.get(c.replace("congestion_", ""), "gray"),
                              alpha=0.6, label=c.replace("congestion_", ""))
                        for c in cong_existing
                        if test[c].iloc[idx_s:idx_e+1].any()
                    ]
                    if cong_legend:
                        ax.legend(handles=cong_legend, loc="lower right",
                                  fontsize=7.5, framealpha=0.7,
                                  title="Congestion par liaison", title_fontsize=7.5)
 
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig("figures/STEP 3/XGBoost/models_comparison_cong_zoom.png", dpi=300)
                plt.show()
                print("   → Graphique sauvegardé : models_comparison_cong_zoom.png\n")
        



if __name__ == "__main__":
    df = pd.read_csv("data/clean/STEP 3/XGBoost/master_dataset_v2.csv", parse_dates=[DATE_COL])

    models_comparison(df, model_path_1, model_path_2, features_path_1, features_path_2, cong_cols=["congestion_FR_BE", "congestion_FR_DE"])