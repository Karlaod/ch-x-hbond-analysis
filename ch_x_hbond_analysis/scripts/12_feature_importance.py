"""
12_feature_importance.py
Feature importance for full and ablated models via permutation importance and SHAP.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import permutation_importance
from plot_config import *

DATA_DIR = os.path.dirname(DATA_PATH)

# Feature display-name map

DISPLAY_NAMES = {
    "h_x_distance":                    "H···X Distance",
    "c_h_x_angle":                     "C–H···X Angle",
    "donor_hybridisation_sp2":         "Donor sp²",
    "donor_hybridisation_sp3":         "Donor sp³",
    "acceptor_hybridisation_sp2":      "Acceptor sp²",
    "acceptor_hybridisation_sp3":      "Acceptor sp³",
    "structure_type_organometallic":   "Organometallic",
}

def display(name):
    return DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())

# Load data and models

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

clf_full = joblib.load(os.path.join(MODELS_PATH, "full_feature_model.pkl"))
clf_abl  = joblib.load(os.path.join(MODELS_PATH, "no_distance_model.pkl"))

X_test_abl = X_test.drop(columns=["h_x_distance"])

# Permutation importance helpers

def _perm_importance(clf, X, y, label):
    print(f"Permutation importance — {label} …")
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42,
                                    scoring="roc_auc")
    df = pd.DataFrame({
        "feature":          X.columns,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["display"] = df["feature"].map(display)
    print(f"\n  {label} — ranked features:")
    for _, row in df.iterrows():
        print(f"    {row['rank']}. {row['display']:<25}  "
              f"{row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    print()
    return df


perm_full = _perm_importance(clf_full, X_test,     y_test, "Full Model")
perm_abl  = _perm_importance(clf_abl,  X_test_abl, y_test, "No Distance Model")

# SHAP values

RNG = np.random.default_rng(42)

def _shap_values(clf, X, label):
    print(f"SHAP values — {label} …")
    idx      = RNG.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X.iloc[idx].reset_index(drop=True)
    explainer   = shap.TreeExplainer(clf)
    shap_vals   = explainer.shap_values(X_sample)
    # GradientBoostingClassifier returns a single array (log-odds)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({
        "feature":       X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"]    = df.index + 1
    df["display"] = df["feature"].map(display)
    print(f"  {label} — SHAP done ({len(X_sample)} samples).")
    return df, shap_vals, X_sample


shap_full_df, shap_full_vals, X_full_sample = _shap_values(clf_full, X_test,     "Full Model")
shap_abl_df,  shap_abl_vals,  X_abl_sample  = _shap_values(clf_abl,  X_test_abl, "No Distance Model")
print()

# Save tables

ensure_dirs()

save_table(perm_full[["feature","importance_mean","importance_std","rank"]],
           "permutation_importance_full")
save_table(perm_abl[["feature","importance_mean","importance_std","rank"]],
           "permutation_importance_ablation")
save_table(shap_full_df[["feature","mean_abs_shap","rank"]],
           "shap_mean_abs_full")
save_table(shap_abl_df[["feature","mean_abs_shap","rank"]],
           "shap_mean_abs_ablation")

# Figure helpers

def _horiz_bar(ax, features, values, errors, xlabel, title, colour=ACCENT_SLATE):
    """Horizontal ranked bar chart, most important at top."""
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, xerr=errors, color=colour, edgecolor="none",
            error_kw={"ecolor": "#555555", "linewidth": 1, "capsize": 3})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


# (a) Permutation importance — Full Model

fig, ax = plt.subplots(figsize=(8, 5))
_horiz_bar(ax,
           perm_full["display"].tolist(),
           perm_full["importance_mean"].tolist(),
           perm_full["importance_std"].tolist(),
           "Mean Accuracy Decrease",
           "Feature Importance — Full Model (Permutation)")
plt.tight_layout()
save_figure(fig, "12a_permutation_importance_full")
plt.close(fig)

# (b) Permutation importance — No Distance Model

fig, ax = plt.subplots(figsize=(8, 5))
_horiz_bar(ax,
           perm_abl["display"].tolist(),
           perm_abl["importance_mean"].tolist(),
           perm_abl["importance_std"].tolist(),
           "Mean Accuracy Decrease",
           "Feature Importance — No Distance Model (Permutation)")
plt.tight_layout()
save_figure(fig, "12b_permutation_importance_ablation")
plt.close(fig)

# (c) SHAP importance — Full Model (manual bar chart)

fig, ax = plt.subplots(figsize=(10, 6))
_horiz_bar(ax,
           shap_full_df["display"].tolist(),
           shap_full_df["mean_abs_shap"].tolist(),
           [0] * len(shap_full_df),
           "Mean |SHAP Value|",
           "SHAP Feature Importance — Full Model")
plt.tight_layout()
save_figure(fig, "12c_shap_importance_full")
plt.close(fig)

# (d) SHAP importance — No Distance Model

fig, ax = plt.subplots(figsize=(10, 6))
_horiz_bar(ax,
           shap_abl_df["display"].tolist(),
           shap_abl_df["mean_abs_shap"].tolist(),
           [0] * len(shap_abl_df),
           "Mean |SHAP Value|",
           "SHAP Feature Importance — No Distance Model")
plt.tight_layout()
save_figure(fig, "12d_shap_importance_ablation")
plt.close(fig)

# (e) SHAP beeswarm — Full Model

# Rename columns to display names for the plot
X_full_display = X_full_sample.rename(columns=DISPLAY_NAMES)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_full_vals, X_full_display, show=False)
plt.title("SHAP Value Distribution — Full Model")
plt.tight_layout()
save_figure(fig, "12e_shap_beeswarm_full")
plt.close(fig)

# (f) Side-by-side permutation importance panel

x_max = max(
    (perm_full["importance_mean"] + perm_full["importance_std"]).max(),
    (perm_abl["importance_mean"]  + perm_abl["importance_std"]).max(),
) * 1.15

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

_horiz_bar(ax_l,
           perm_full["display"].tolist(),
           perm_full["importance_mean"].tolist(),
           perm_full["importance_std"].tolist(),
           "Mean Accuracy Decrease",
           "Full Model")
_horiz_bar(ax_r,
           perm_abl["display"].tolist(),
           perm_abl["importance_mean"].tolist(),
           perm_abl["importance_std"].tolist(),
           "Mean Accuracy Decrease",
           "No Distance Model")

ax_l.set_xlim(0, x_max)
ax_r.set_xlim(0, x_max)

fig.suptitle("Feature Importance Comparison: Full vs No Distance",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "12f_importance_comparison_panel")
plt.close(fig)

# Console summary

print("=" * 60)
print("FEATURE IMPORTANCE SUMMARY")
print("=" * 60)
print()
print("  Permutation importance (Full Model, ranked by mean):")
for _, row in perm_full.iterrows():
    print(f"    {row['rank']}. {row['display']:<25}  {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
print()
print("  Mean |SHAP| (Full Model, ranked):")
for _, row in shap_full_df.iterrows():
    print(f"    {row['rank']}. {row['display']:<25}  {row['mean_abs_shap']:.4f}")
print()
print("  Saved tables: permutation_importance_full/ablation, shap_mean_abs_full/ablation")
print("  Saved figures: 12a–12f")
print("=" * 60)
