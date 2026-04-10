"""
05_hybridisation_analysis.py
Analyses how H···X distance and C-H···X angle vary with donor carbon
hybridisation (sp, sp2, sp3) for each halogen.

sp has only 23 contacts (21 Cl, 2 Br). Shown as a rug plot in KDE panels
and included in bar charts and summary tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_config import *

# ---------------------------------------------------------------------------
# Load data — all hybridisations included
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

HYB_LABELS = {"sp": "sp", "sp2": "sp²", "sp3": "sp³"}
HYB_ORDER  = ["sp", "sp2", "sp3"]

# ---------------------------------------------------------------------------
# Summary statistics tables (all hybridisations)
# ---------------------------------------------------------------------------

def _group_stats(value_col):
    return (
        df.groupby(["halogen", "donor_hybridisation"])[value_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
        )
        .round(4)
        .reset_index()
    )

dist_stats  = _group_stats("h_x_distance")
angle_stats = _group_stats("c_h_x_angle")

save_table(dist_stats,  "hybridisation_distance_stats")
save_table(angle_stats, "hybridisation_angle_stats")

# ---------------------------------------------------------------------------
# Helper: look up a stat for a specific halogen × hybridisation
# ---------------------------------------------------------------------------

def _stat(stats_df, halogen, hyb, col):
    row = stats_df[(stats_df["halogen"] == halogen) &
                   (stats_df["donor_hybridisation"] == hyb)]
    return float(row[col].values[0]) if len(row) else np.nan

# ---------------------------------------------------------------------------
# Figure (a) & (b) — three KDE subplots (sp², sp³, sp rug)
# ---------------------------------------------------------------------------

def _kde_panel(axes, y_col, x_range, xlabel):
    """
    Three side-by-side subplots: sp², sp³ KDE comparison; sp rug plot.
    axes: list of three Axes objects.
    """
    x_grid = np.linspace(x_range[0], x_range[1], 1000)

    panel_specs = [
        ("sp2", "sp² Donors",      False),
        ("sp3", "sp³ Donors",      False),
        ("sp",  "sp Donors (n=23)", True),
    ]

    for i, (ax, (hyb, subtitle, is_sp)) in enumerate(zip(axes, panel_specs)):
        if is_sp:
            # Rug plot — individual ticks for each contact
            for halogen, colour in HALOGEN_COLOURS.items():
                vals = df[(df["donor_hybridisation"] == "sp") &
                          (df["halogen"] == halogen)][y_col].values
                if len(vals) > 0:
                    ax.plot(vals, np.zeros_like(vals), "|",
                            color=colour, markersize=18, markeredgewidth=2,
                            label=f"{halogen} (n={len(vals)})")
            ax.set_ylim(-0.05, 0.5)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.legend(fontsize=9, loc="upper right")
        else:
            for halogen, colour in HALOGEN_COLOURS.items():
                vals = df[(df["donor_hybridisation"] == hyb) &
                          (df["halogen"] == halogen)][y_col].values
                if len(vals) < 5:
                    continue
                kde_y = gaussian_kde(vals, bw_method=0.1)(x_grid)
                ax.plot(x_grid, kde_y, color=colour, linewidth=2.5,
                        label=halogen)
                ax.fill_between(x_grid, kde_y, alpha=0.2, color=colour)
                med = np.median(vals)
                ax.axvline(med, color=colour, linestyle="--",
                           linewidth=1.5, alpha=0.8)
            ax.set_ylabel("Density" if i == 0 else "")
            if i == 0:
                ax.legend(title="Halogen", loc="upper left")

        ax.set_xlim(x_range)
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.grid(False)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

# --- (a) Distance ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
_kde_panel(axes, "h_x_distance", (2.2, 3.6), "H···X Distance (Å)")
fig.suptitle("Effect of Donor Hybridisation on H···X Distance",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "05a_distance_by_hybridisation")
plt.close(fig)

# --- (b) Angle ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
_kde_panel(axes, "c_h_x_angle", (100, 180), "C–H···X Angle (°)")
fig.suptitle("Effect of Donor Hybridisation on C–H···X Angle",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "05b_angle_by_hybridisation")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure (c) & (d) — grouped bar chart: median ± IQR per hybridisation
# ---------------------------------------------------------------------------

def _median_bar(ax, stats_df, ylabel, title):
    """
    Grouped bar chart: median bar height, IQR as asymmetric error bars.
    Groups = hybridisation (sp, sp², sp³), hue = Cl / Br.
    """
    x       = np.arange(len(HYB_ORDER))
    bar_w   = 0.3

    for i_h, halogen in enumerate(["Cl", "Br"]):
        colour  = HALOGEN_COLOURS[halogen]
        offset  = (i_h - 0.5) * bar_w

        medians   = [_stat(stats_df, halogen, h, "median") for h in HYB_ORDER]
        lower_err = [_stat(stats_df, halogen, h, "median") -
                     _stat(stats_df, halogen, h, "Q1")     for h in HYB_ORDER]
        upper_err = [_stat(stats_df, halogen, h, "Q3") -
                     _stat(stats_df, halogen, h, "median") for h in HYB_ORDER]

        ax.bar(
            x + offset, medians,
            width=bar_w, label=halogen,
            color=colour, edgecolor="none",
            yerr=[lower_err, upper_err],
            capsize=4,
            error_kw={"ecolor": "#333333", "linewidth": 1, "capthick": 1},
        )

        # Median value labels above each bar
        for xi, med, uerr in zip(x + offset, medians, upper_err):
            if not np.isnan(med):
                ax.text(xi, med + uerr + ax.get_ylim()[1] * 0.01,
                        f"{med:.3f}", ha="center", va="bottom",
                        fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([HYB_LABELS[h] for h in HYB_ORDER])
    ax.set_xlabel("Donor Hybridisation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Halogen")
    ax.grid(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

# --- (c) Median distance ---
fig, ax = plt.subplots(figsize=(8, 5))
_median_bar(ax, dist_stats, "Median H···X Distance (Å)",
            "Median H···X Distance by Donor Hybridisation")
ax.set_ylim(2.8, 3.3)
plt.tight_layout()
save_figure(fig, "05c_median_distance_comparison")
plt.close(fig)

# --- (d) Median angle ---
fig, ax = plt.subplots(figsize=(8, 5))
_median_bar(ax, angle_stats, "Median C–H···X Angle (°)",
            "Median C–H···X Angle by Donor Hybridisation")
ax.set_ylim(100, 150)
plt.tight_layout()
save_figure(fig, "05d_median_angle_comparison")
plt.close(fig)

print("Hybridisation analysis figures and tables saved.")
