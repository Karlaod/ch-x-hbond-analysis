"""
02_distance_distributions.py
Analyses and visualises H···X distance distributions for C-H···Cl and C-H···Br contacts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
from plot_config import *

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

cl = df.loc[df["halogen"] == "Cl", "h_x_distance"].values
br = df.loc[df["halogen"] == "Br", "h_x_distance"].values

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

def _stats(arr, halogen):
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "halogen":  halogen,
        "count":    len(arr),
        "mean":     arr.mean(),
        "median":   np.median(arr),
        "std":      arr.std(),
        "min":      arr.min(),
        "max":      arr.max(),
        "Q1":       q1,
        "Q3":       q3,
        "IQR":      q3 - q1,
    }

stats_df = pd.DataFrame([_stats(cl, "Cl"), _stats(br, "Br")]).round(4)
save_table(stats_df, "distance_summary_stats")

# ---------------------------------------------------------------------------
# Shared KDE evaluation helper
# ---------------------------------------------------------------------------

x_min  = df["h_x_distance"].min() - 0.05
x_max  = df["h_x_distance"].max() + 0.05
x_grid = np.linspace(x_min, x_max, 500)


def _kde_curve(arr):
    return gaussian_kde(arr)(x_grid)


# --- (a) KDE comparison with reference lines --------------------------------

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 6))

for halogen, colour in HALOGEN_COLOURS.items():
    data = df.loc[df["halogen"] == halogen, "h_x_distance"].values
    kde = gaussian_kde(data, bw_method=0.05)
    x_range = np.linspace(2.2, 3.6, 1000)
    y_vals = kde(x_range) * len(data) * 0.02
    ax.plot(x_range, y_vals, color=colour, linewidth=2.5, label=halogen)
    ax.fill_between(x_range, y_vals, alpha=0.2, color=colour)

cl_med = df.loc[df["halogen"] == "Cl", "h_x_distance"].median()
br_med = df.loc[df["halogen"] == "Br", "h_x_distance"].median()

ax.axvline(cl_med, color=HALOGEN_COLOURS["Cl"], linestyle="--", linewidth=2, alpha=0.8)
ax.axvline(br_med, color=HALOGEN_COLOURS["Br"], linestyle="--", linewidth=2, alpha=0.8)

textstr = f"Median Cl: {cl_med:.3f} Å\nMedian Br: {br_med:.3f} Å"
props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#CCCCCC")
ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=11, fontweight="bold",
        verticalalignment="top", horizontalalignment="right", bbox=props)

ax.axvline(2.95, color="#555555", linestyle=":", linewidth=1.5)
ax.axvline(3.05, color="#555555", linestyle=":", linewidth=1.5)
ax.annotate("vdW Cl\n2.95 Å", xy=(2.95, ax.get_ylim()[1] * 0.55),
            fontsize=9, fontweight="bold", color="#333333", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))
ax.annotate("vdW Br\n3.05 Å", xy=(3.05, ax.get_ylim()[1] * 0.40),
            fontsize=9, fontweight="bold", color="#333333", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))

ax.set_title("H···X Distance Distribution by Halogen")
ax.set_xlabel("H···X Distance (Å)")
ax.set_ylabel("Frequency")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_figure(fig, "02a_distance_kde_comparison")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Normalised KDE comparison
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(10, 6))
ax  = fig.add_subplot(111)

for arr, halogen, colour in [(cl, "Cl", HALOGEN_COLOURS["Cl"]),
                              (br, "Br", HALOGEN_COLOURS["Br"])]:
    kde_y = _kde_curve(arr)
    ax.plot(x_grid, kde_y, color=colour, linewidth=2, label=halogen)
    ax.fill_between(x_grid, kde_y, alpha=0.15, color=colour)

ax.set_title("Normalised H···X Distance Distribution")
ax.set_xlabel("H···X Distance (Å)")
ax.set_ylabel("Probability Density")
ax.legend(title="Halogen")
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "02b_distance_kde_normalised")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Horizontal box plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(10, 4))
ax  = fig.add_subplot(111)

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

# Mean diamond markers
for arr, y_pos, colour in [(cl, 1, HALOGEN_COLOURS["Cl"]),
                            (br, 2, HALOGEN_COLOURS["Br"])]:
    ax.scatter(arr.mean(), y_pos, marker="D", color=colour,
               zorder=5, s=40, edgecolors="black", linewidths=0.5)

ax.set_title("H···X Distance by Halogen")
ax.set_xlabel("H···X Distance (Å)")
ax.grid(False)
ax.grid(axis="x", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "02c_distance_boxplot")
plt.close(fig)

# ---------------------------------------------------------------------------
# (d) Vertical violin plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111)

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

# Inner box plot (thin black lines) with median dot
for arr, x_pos in [(cl, 1), (br, 2)]:
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr      = q3 - q1
    whisk_lo = max(arr.min(), q1 - 1.5 * iqr)
    whisk_hi = min(arr.max(), q3 + 1.5 * iqr)
    ax.vlines(x_pos, whisk_lo, whisk_hi, color="black", linewidth=1)
    ax.vlines(x_pos, q1, q3,             color="black", linewidth=3)
    ax.scatter(x_pos, med, color="white", zorder=5, s=25,
               edgecolors="black", linewidths=0.8)

ax.set_title("H···X Distance Distribution by Halogen")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Cl", "Br"])
ax.set_ylabel("H···X Distance (Å)")
ax.set_xlim(0.4, 2.6)
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "02d_distance_violin")
plt.close(fig)

print("Distance distribution figures and table saved.")
