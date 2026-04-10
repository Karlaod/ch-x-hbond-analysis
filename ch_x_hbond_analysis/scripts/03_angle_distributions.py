"""
03_angle_distributions.py
Analyses and visualises C-H···X angle distributions for C-H···Cl and C-H···Br contacts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_config import *

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

cl = df.loc[df["halogen"] == "Cl", "c_h_x_angle"].values
br = df.loc[df["halogen"] == "Br", "c_h_x_angle"].values

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

def _stats(arr, halogen):
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "halogen": halogen,
        "count":   len(arr),
        "mean":    arr.mean(),
        "median":  np.median(arr),
        "std":     arr.std(),
        "min":     arr.min(),
        "max":     arr.max(),
        "Q1":      q1,
        "Q3":      q3,
        "IQR":     q3 - q1,
    }

stats_df = pd.DataFrame([_stats(cl, "Cl"), _stats(br, "Br")]).round(4)
save_table(stats_df, "angle_summary_stats")

# ---------------------------------------------------------------------------
# (a) KDE comparison — frequency y-axis, median lines, text box
# ---------------------------------------------------------------------------

x_range = np.linspace(100, 180, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

for halogen, colour in HALOGEN_COLOURS.items():
    data = df.loc[df["halogen"] == halogen, "c_h_x_angle"].values
    kde  = gaussian_kde(data, bw_method=0.08)
    y_vals = kde(x_range) * len(data) * 1.0   # bin_width = 1.0 degree
    ax.plot(x_range, y_vals, color=colour, linewidth=2.5, label=halogen)
    ax.fill_between(x_range, y_vals, alpha=0.2, color=colour)

cl_med = np.median(cl)
br_med = np.median(br)
ax.axvline(cl_med, color=HALOGEN_COLOURS["Cl"], linestyle="--", linewidth=2, alpha=0.8)
ax.axvline(br_med, color=HALOGEN_COLOURS["Br"], linestyle="--", linewidth=2, alpha=0.8)

textstr = f"Median Cl: {cl_med:.1f}°\nMedian Br: {br_med:.1f}°"
props   = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#CCCCCC")
ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=11, fontweight="bold",
        verticalalignment="top", horizontalalignment="left", bbox=props)

ax.set_title("C–H···X Angle Distribution by Halogen")
ax.set_xlabel("C–H···X Angle (°)")
ax.set_ylabel("Frequency")
ax.set_xlim(100, 180)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save_figure(fig, "03a_angle_kde_comparison")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Normalised KDE — probability density, shape comparison
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for halogen, colour in HALOGEN_COLOURS.items():
    data  = df.loc[df["halogen"] == halogen, "c_h_x_angle"].values
    kde_y = gaussian_kde(data, bw_method=0.05)(x_range)
    ax.plot(x_range, kde_y, color=colour, linewidth=2, label=halogen)
    ax.fill_between(x_range, kde_y, alpha=0.15, color=colour)

ax.set_title("Normalised C–H···X Angle Distribution")
ax.set_xlabel("C–H···X Angle (°)")
ax.set_ylabel("Probability Density")
ax.set_xlim(100, 180)
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save_figure(fig, "03b_angle_kde_normalised")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Horizontal box plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))

bp = ax.boxplot(
    [cl, br],
    vert=False,
    tick_labels=["Cl", "Br"],
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.5},
    boxprops={"linewidth": 1},
    whiskerprops={"linewidth": 1},
    capprops={"linewidth": 1},
    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
)

for patch, colour in zip(bp["boxes"], [HALOGEN_COLOURS["Cl"], HALOGEN_COLOURS["Br"]]):
    patch.set_facecolor(colour)
    patch.set_alpha(0.6)

for arr, y_pos, colour in [(cl, 1, HALOGEN_COLOURS["Cl"]),
                            (br, 2, HALOGEN_COLOURS["Br"])]:
    ax.scatter(arr.mean(), y_pos, marker="D", color=colour,
               zorder=5, s=40, edgecolors="black", linewidths=0.5)

ax.set_title("C–H···X Angle by Halogen")
ax.set_xlabel("C–H···X Angle (°)")
ax.grid(False)
ax.grid(axis="x", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "03c_angle_boxplot")
plt.close(fig)

# ---------------------------------------------------------------------------
# (d) Vertical violin plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

parts = ax.violinplot(
    [cl, br],
    positions=[1, 2],
    vert=True,
    showmedians=False,
    showextrema=False,
)

for body, colour in zip(parts["bodies"], [HALOGEN_COLOURS["Cl"], HALOGEN_COLOURS["Br"]]):
    body.set_facecolor(colour)
    body.set_alpha(0.7)
    body.set_edgecolor("none")

for arr, x_pos in [(cl, 1), (br, 2)]:
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr      = q3 - q1
    whisk_lo = max(arr.min(), q1 - 1.5 * iqr)
    whisk_hi = min(arr.max(), q3 + 1.5 * iqr)
    ax.vlines(x_pos, whisk_lo, whisk_hi, color="black", linewidth=1)
    ax.vlines(x_pos, q1, q3,             color="black", linewidth=3)
    ax.scatter(x_pos, med, color="white", zorder=5, s=25,
               edgecolors="black", linewidths=0.8)

ax.set_title("C–H···X Angle Distribution by Halogen")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Cl", "Br"])
ax.set_ylabel("C–H···X Angle (°)")
ax.set_xlim(0.4, 2.6)
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "03d_angle_violin")
plt.close(fig)

print("Angle distribution figures and table saved.")
