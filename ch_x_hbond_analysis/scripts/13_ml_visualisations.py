"""
13_ml_visualisations.py
Assembles publication-ready panels from saved ML result figures, plus a fresh
model comparison bar chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_config import *

def _load(filename):
    """Load a saved PNG from RESULTS_FIGURES."""
    return mpimg.imread(os.path.join(RESULTS_FIGURES, filename + ".png"))


def _place(ax, filename, title):
    """Show a saved PNG on ax with a subtitle."""
    ax.imshow(_load(filename))
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")


# ---------------------------------------------------------------------------
# (a) ML Summary Panel — 2×2
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

_place(axes[0, 0], "10a_confusion_matrix_baseline",  "Full Model — Confusion Matrix")
_place(axes[0, 1], "11a_confusion_matrix_ablation",  "No Distance — Confusion Matrix")
_place(axes[1, 0], "10b_roc_curve_baseline",         "Full Model — ROC Curve")
_place(axes[1, 1], "11b_roc_comparison",             "ROC Comparison")

fig.suptitle("Machine Learning Classification Summary", fontsize=15, fontweight="bold")
plt.tight_layout(pad=2.0)
save_figure(fig, "13a_ml_summary_panel")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Feature Importance Panel — 2×3
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(20, 14))

_place(axes[0, 0], "12a_permutation_importance_full",      "Permutation Importance — Full Model")
_place(axes[0, 1], "12b_permutation_importance_ablation",  "Permutation Importance — No Distance")
_place(axes[0, 2], "12f_importance_comparison_panel",      "Permutation Importance Comparison")
_place(axes[1, 0], "12c_shap_importance_full",             "SHAP Importance — Full Model")
_place(axes[1, 1], "12d_shap_importance_ablation",         "SHAP Importance — No Distance")
_place(axes[1, 2], "12e_shap_beeswarm_full",               "SHAP Beeswarm — Full Model")

fig.suptitle("Feature Importance Analysis", fontsize=15, fontweight="bold")
plt.tight_layout(pad=2.0)
save_figure(fig, "13b_feature_importance_panel")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Model Comparison Bar Chart — created fresh
# ---------------------------------------------------------------------------

comparison = pd.read_csv(
    os.path.join(os.path.dirname(RESULTS_TABLES), "tables", "model_comparison.csv")
)

BAR_METRICS = ["Accuracy", "Precision (Br)", "Recall (Br)", "F1 (Br)", "ROC AUC"]
full_vals = [float(comparison.loc[comparison["Metric"] == m, "Full Model"].values[0])
             for m in BAR_METRICS]
abl_vals  = [float(comparison.loc[comparison["Metric"] == m, "No Distance"].values[0])
             for m in BAR_METRICS]

x     = np.arange(len(BAR_METRICS))
bar_w = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars_full = ax.bar(x - bar_w / 2, full_vals, width=bar_w,
                   color=HALOGEN_COLOURS["Cl"], label="Full Model",    edgecolor="none")
bars_abl  = ax.bar(x + bar_w / 2, abl_vals,  width=bar_w,
                   color=HALOGEN_COLOURS["Br"], label="No Distance",   edgecolor="none")

for bar in list(bars_full) + list(bars_abl):
    h = bar.get_height()
    ax.annotate(f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(BAR_METRICS)
ax.set_ylabel("Score")
ax.set_ylim(0.2, 0.85)
ax.set_title("Classification Performance: Full Model vs Ablation")
ax.legend(title="Model")
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout(pad=2.0)
save_figure(fig, "13c_performance_comparison")
plt.close(fig)

print("ML visualisation panels saved.")
print("  13a_ml_summary_panel.png")
print("  13b_feature_importance_panel.png")
print("  13c_performance_comparison.png")
