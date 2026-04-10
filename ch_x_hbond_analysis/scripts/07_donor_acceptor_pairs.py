"""
07_donor_acceptor_pairs.py
Analyses C–H···X interactions by donor–acceptor hybridisation pair combinations.
Heatmaps show contact counts and median H···X distances for each Cl / Br subset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plot_config import *

PROJECT_CMAP = LinearSegmentedColormap.from_list(
    "project_blue",
    ["#FFFFFF", "#C8DDE5", "#A3C4D0", "#4A90A4", "#2D6073", "#1B3A4B"],
    N=256,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

HYB_ORDER  = ["sp", "sp2", "sp3"]
HYB_LABELS = {"sp": "sp", "sp2": "sp²", "sp3": "sp³"}

# ---------------------------------------------------------------------------
# Build pivot tables
# ---------------------------------------------------------------------------

def _pivot_count(halogen):
    sub = df[df["halogen"] == halogen]
    return (
        sub.groupby(["donor_hybridisation", "acceptor_hybridisation"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=HYB_ORDER, columns=HYB_ORDER, fill_value=0)
    )


def _pivot_median(halogen):
    sub = df[df["halogen"] == halogen]
    return (
        sub.groupby(["donor_hybridisation", "acceptor_hybridisation"])["h_x_distance"]
        .median()
        .unstack()
        .reindex(index=HYB_ORDER, columns=HYB_ORDER)
    )


count_cl  = _pivot_count("Cl")
count_br  = _pivot_count("Br")
median_cl = _pivot_median("Cl")
median_br = _pivot_median("Br")

# ---------------------------------------------------------------------------
# Save tables (raw labels for data integrity)
# ---------------------------------------------------------------------------

save_table(count_cl.reset_index(),  "pair_counts_cl")
save_table(count_br.reset_index(),  "pair_counts_br")
save_table(median_cl.reset_index(), "pair_median_distance_cl")
save_table(median_br.reset_index(), "pair_median_distance_br")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relabel(pivot):
    """Rename sp/sp2/sp3 → display labels (sp, sp², sp³) for both axes."""
    return pivot.rename(index=HYB_LABELS, columns=HYB_LABELS)


def _heatmap_count(ax, pivot, title):
    sns.heatmap(
        _relabel(pivot),
        ax=ax,
        annot=True,
        fmt=",d",
        cmap=PROJECT_CMAP,
        linewidths=0.5,
        linecolor="#E0E0E0",
        cbar_kws={"label": "Count"},
    )
    ax.set_title(title)
    ax.set_xlabel("Acceptor Hybridisation")
    ax.set_ylabel("Donor Hybridisation")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def _heatmap_median(ax, pivot, title):
    sns.heatmap(
        _relabel(pivot),
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=PROJECT_CMAP.reversed(),
        linewidths=0.5,
        linecolor="#E0E0E0",
        cbar_kws={"label": "Median H···X Distance (Å)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Acceptor Hybridisation")
    ax.set_ylabel("Donor Hybridisation")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


# ---------------------------------------------------------------------------
# (a) Count heatmap — Cl
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_heatmap_count(ax, count_cl, "C–H···Cl Contact Counts by Hybridisation Pair")
plt.tight_layout()
save_figure(fig, "07a_pair_count_heatmap_cl")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Count heatmap — Br
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_heatmap_count(ax, count_br, "C–H···Br Contact Counts by Hybridisation Pair")
plt.tight_layout()
save_figure(fig, "07b_pair_count_heatmap_br")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Median distance heatmap — Cl
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_heatmap_median(ax, median_cl, "Median H···Cl Distance by Hybridisation Pair (Å)")
plt.tight_layout()
save_figure(fig, "07c_pair_median_distance_cl")
plt.close(fig)

# ---------------------------------------------------------------------------
# (d) Median distance heatmap — Br
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_heatmap_median(ax, median_br, "Median H···Br Distance by Hybridisation Pair (Å)")
plt.tight_layout()
save_figure(fig, "07d_pair_median_distance_br")
plt.close(fig)

# ---------------------------------------------------------------------------
# (e) 2×2 panel
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

_heatmap_count( axes[0, 0], count_cl,  "C–H···Cl: Contact Counts")
_heatmap_count( axes[0, 1], count_br,  "C–H···Br: Contact Counts")
_heatmap_median(axes[1, 0], median_cl, "C–H···Cl: Median H···X Distance (Å)")
_heatmap_median(axes[1, 1], median_br, "C–H···Br: Median H···X Distance (Å)")

fig.suptitle("Donor–Acceptor Hybridisation Pair Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "07e_pair_analysis_panel")
plt.close(fig)

print("Donor–acceptor pair heatmaps and tables saved.")
