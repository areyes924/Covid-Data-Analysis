
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# Load master dataset (path anchored to this script's location so it works
# regardless of the current working directory it's run from)
DATA_PATH = Path(__file__).resolve().parent.parent / "master_dataset" / "master_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Puerto Rico is missing voting/policy columns (no electoral votes / not covered
# by this policy dataset) - drop it rather than impute
df = df[df["state"] != "Puerto Rico"].reset_index(drop=True)

target_col = "deaths_per_100k"

# excluded: cases_per_100k and death_rate are mechanically tied to deaths_per_100k
# (deaths_per_100k = cases_per_100k * death_rate), so they'd leak the target
# rather than reveal real drivers. biden_pct is dropped since it's a near-perfect
# mirror of trump_pct (collinear pair, keeping both just splits their weight).
feature_cols = [
    "pct_full_max",
    "pct_full_asof_2022-03-01",
    "pct_full_growth_pp_2021-09-01_to_2022-09-01",
    "trump_pct",
    "FMNOENF",
    "AGEPRIORITY",
    "stayhome_never_adopted",
    "stayhome_days_since_earliest_adoption",
    "stayhome_duration_days",
    "clbsns_never_adopted",
    "clbsns_days_since_earliest_adoption",
    "clbsns_duration_days",
    "fm_all_never_adopted",
    "FM_ALL_days_since_earliest_adoption",
    "population_65_plus_percent",
    "hs_pct",
    "bachelors_pct",
    "advanced_pct",
    "median_household_income_2019",
    "white_non_latino_percent_2020",
]

X = df[feature_cols]
y = df[target_col]

loo = LeaveOneOut()
n_samples, n_features = X.shape

# out-of-fold predictions, one slot per state, filled in as each fold predicts
# the state it held out
enet_oof_preds = np.zeros(n_samples)
rf_oof_preds = np.zeros(n_samples)

# per-fold coefficients / importances, stacked into a (n_folds x n_features)
# matrix so we can average down the columns afterwards
enet_coefs = np.zeros((n_samples, n_features))
rf_importances = np.zeros((n_samples, n_features))

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # fit the scaler on the training fold only - the held-out state must not
    # influence the mean/std used to standardize it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TODO: fit ElasticNetCV on X_train_scaled/y_train, predict X_test_scaled,
    # store the prediction in enet_oof_preds[test_idx] and enet.coef_ in enet_coefs[test_idx]
    enet = ElasticNetCV(cv=5, random_state=0)
    enet.fit(X_train_scaled, y_train)
    enet_oof_preds[test_idx] = enet.predict(X_test_scaled)
    enet_coefs[test_idx] = enet.coef_

    # TODO: fit RandomForestRegressor on X_train/y_train (trees don't need scaling),
    # predict X_test, store in rf_oof_preds[test_idx] and
    # rf.feature_importances_ in rf_importances[test_idx]
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)
    rf_oof_preds[test_idx] = rf.predict(X_test)
    rf_importances[test_idx] = rf.feature_importances_
    
# TODO: once the loop finishes, compare enet_oof_preds/rf_oof_preds against y
# with r2_score/mean_squared_error/mean_absolute_error, and average enet_coefs /
# rf_importances down axis=0 to rank features
r2_score(y, enet_oof_preds)
np.sqrt(mean_squared_error(y, enet_oof_preds))
mean_absolute_error(y, enet_oof_preds)

print("Elastic Net Results:")
print(f"R2 Score: {r2_score(y, enet_oof_preds)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, enet_oof_preds))}")
print(f"MAE: {mean_absolute_error(y, enet_oof_preds)}")

print("Random Forest Results:")
print(f"R2 Score: {r2_score(y, rf_oof_preds)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, rf_oof_preds))}")
print(f"MAE: {mean_absolute_error(y, rf_oof_preds)}")

enet_coef_mean = pd.Series(enet_coefs.mean(axis=0), index=feature_cols)
enet_coef_mean = enet_coef_mean.reindex(enet_coef_mean.abs().sort_values(ascending=False).index)

rf_importance_mean = pd.Series(rf_importances.mean(axis=0), index=feature_cols)
rf_importance_mean = rf_importance_mean.sort_values(ascending=False)

print("\nElastic Net mean coefficients (ranked by |coef|):")
print(enet_coef_mean)

print("\nRandom Forest mean feature importances (ranked):")
print(rf_importance_mean)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.family": "sans-serif",
})

# --- predicted vs. actual, one panel per model ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, preds, name in zip(axes, [enet_oof_preds, rf_oof_preds], ["Elastic Net", "Random Forest"]):
    ax.scatter(y, preds, s=28, color=BLUE, alpha=0.75, edgecolors="none")
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    ax.plot(lims, lims, color=INK_MUTED, linestyle="--", linewidth=1, zorder=0)
    ax.set_title(f"{name}: predicted vs. actual", color=INK_PRIMARY, fontsize=11)
    ax.set_xlabel("Actual deaths per 100k")
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Predicted deaths per 100k")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "predicted_vs_actual.png", dpi=150)

# --- Random Forest feature importances (magnitude only -> single hue) ---
fig, ax = plt.subplots(figsize=(8, 7))
ranked = rf_importance_mean.sort_values(ascending=True)
ax.barh(ranked.index, ranked.values, color=BLUE, height=0.6)
ax.set_xlabel("Mean feature importance")
ax.set_title("Random Forest feature importances (LOOCV mean)", color=INK_PRIMARY, fontsize=11)
ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "rf_feature_importances.png", dpi=150)

# --- Elastic Net coefficients (signed -> diverging blue/red) ---
fig, ax = plt.subplots(figsize=(8, 7))
ranked = enet_coef_mean.reindex(enet_coef_mean.abs().sort_values(ascending=True).index)
colors = [BLUE if v >= 0 else RED for v in ranked.values]
ax.barh(ranked.index, ranked.values, color=colors, height=0.6)
ax.axvline(0, color=BASELINE, linewidth=1)
ax.set_xlabel("Mean coefficient (standardized features)")
ax.set_title("Elastic Net coefficients (LOOCV mean)", color=INK_PRIMARY, fontsize=11)
ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Positive"),
    plt.Rectangle((0, 0), 1, 1, color=RED, label="Negative"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False, labelcolor=INK_SECONDARY)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "enet_coefficients.png", dpi=150)

print(f"\nPlots saved to {PLOTS_DIR}")
