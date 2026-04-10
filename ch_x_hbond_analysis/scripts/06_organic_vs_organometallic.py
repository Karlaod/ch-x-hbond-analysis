"""
06_organic_vs_organometallic.py
Compares C-H···X interaction geometry between organic and organometallic structures.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_config import *

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

STRUCT_ORDER  = ["organic", "organometallic"]
STRUCT_LABELS = {"organic": "Organic", "organometallic": "Organometallic"}

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

stats = (
    df.groupby(["halogen", "structure_type"])
    .agg(
        count=("h_x_distance", "count"),
        mean_distance=("h_x_distance", "mean"),
        median_distance=("h_x_distance", "median"),
        std_distance=("h_x_distance", "std"),
        mean_angle=("c_h_x_angle", "mean"),
        median_angle=("c_h_x_angle", "median"),
        std_angle=("c_h_x_angle", "std"),
    )
    .round(4)
    .reset_index()
)
save_table(stats, "structure_type_comparison_stats")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat(halogen, struct, col):
    row = stats[(stats["halogen"] == halogen) & (stats["structure_type"] == struct)]
    return float(row[col].values[0]) if len(row) else np.nan


STRUCT_LINESTYLES = {"organic": "-", "organometallic": "--"}

def _kde_subplot(ax, halogen, y_col, x_range, xlabel):
    """KDE comparison of organic vs organometallic for one halogen."""
    x_grid = np.linspace(x_range[0], x_range[1], 1000)
    box_lines = []

    for struct, colour in STRUCTURE_COLOURS.items():
        vals = df[(df["halogen"] == halogen) &
                  (df["structure_type"] == struct)][y_col].values
        if len(vals) < 5:
            continue
        linestyle = STRUCT_LINESTYLES[struct]
        kde_y = gaussian_kde(vals, bw_method=0.15)(x_grid)
        ax.plot(x_grid, kde_y, color=colour, linewidth=2.5, linestyle=linestyle)
        ax.fill_between(x_grid, kde_y, alpha=0.2, color=colour)
        med = np.median(vals)
        ax.axvline(med, color=colour, linestyle=linestyle, linewidth=1.5, alpha=0.8)
        # Build legend line: solid dash indicator + label + median
        dash = "──" if linestyle == "-" else "‒ ‒"
        box_lines.append(f"{dash} {STRUCT_LABELS[struct]}: median {med:.3f}")

    # Combined legend + median text box, upper right (curves peak left)
    textstr = "\n".join(box_lines)
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#CCCCCC")
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right", bbox=props,
            fontfamily="monospace")

    ax.set_xlim(x_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def _median_bar(ax, y_col, stat_col, ylabel, title):
    """Grouped bar chart: Cl and Br per structure type, median ± IQR."""
    x       = np.arange(len(STRUCT_ORDER))
    bar_w   = 0.3

    for i_h, halogen in enumerate(["Cl", "Br"]):
        colour = HALOGEN_COLOURS[halogen]
        offset = (i_h - 0.5) * bar_w
        medians = [_stat(halogen, s, stat_col) for s in STRUCT_ORDER]

        ax.bar(x + offset, medians, width=bar_w, label=halogen,
               color=colour, edgecolor="none")

        for xi, med in zip(x + offset, medians):
            if not np.isnan(med):
                ax.annotate(f"{med:.3f}", xy=(xi, med),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([STRUCT_LABELS[s] for s in STRUCT_ORDER])
    ax.set_xlabel("Structure Type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Halogen")
    ax.grid(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def _draw_donut(ax, labels, counts, colours, total_label):
    """Donut chart with outside labels and white centre showing total."""
    total = sum(counts)
    pcts  = [c / total * 100 for c in counts]

    ax.set_aspect("equal")
    wedges, _ = ax.pie(
        counts,
        colors=colours,
        startangle=90,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    )

    for wedge, label, count, pct in zip(wedges, labels, counts, pcts):
        angle = (wedge.theta1 + wedge.theta2) / 2
        rad   = math.radians(angle)
        lx    = math.cos(rad) * 1.25
        ly    = math.sin(rad) * 1.25
        ax.text(lx, ly, f"{label}\n{count:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=10)

    centre = plt.Circle((0, 0), 0.58, color="white")
    ax.add_patch(centre)
    ax.text(0,  0.12, f"{total:,}", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(0, -0.14, total_label, ha="center", va="center",
            fontsize=9, color="#555555")
    ax.axis("off")

# ---------------------------------------------------------------------------
# (a) Distance KDE — two subplots (Cl, Br)
# ---------------------------------------------------------------------------

fig, (ax_cl, ax_br) = plt.subplots(1, 2, figsize=(14, 6))
_kde_subplot(ax_cl, "Cl", "h_x_distance", (2.2, 3.6), "H···X Distance (Å)")
_kde_subplot(ax_br, "Br", "h_x_distance", (2.2, 3.6), "H···X Distance (Å)")
ax_cl.set_title("C–H···Cl")
ax_br.set_title("C–H···Br")
fig.suptitle("H···X Distance by Structure Type", fontsize=14, fontweight="bold")
# Shared y-axis range across both subplots
y_max = max(ax_cl.get_ylim()[1], ax_br.get_ylim()[1])
ax_cl.set_ylim(0, y_max * 1.1)
ax_br.set_ylim(0, y_max * 1.1)
plt.tight_layout()
save_figure(fig, "06a_distance_by_structure_type")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Angle KDE — two subplots (Cl, Br)
# ---------------------------------------------------------------------------

fig, (ax_cl, ax_br) = plt.subplots(1, 2, figsize=(14, 6))
_kde_subplot(ax_cl, "Cl", "c_h_x_angle", (100, 180), "C–H···X Angle (°)")
_kde_subplot(ax_br, "Br", "c_h_x_angle", (100, 180), "C–H···X Angle (°)")
ax_cl.set_title("C–H···Cl")
ax_br.set_title("C–H···Br")
fig.suptitle("C–H···X Angle by Structure Type", fontsize=14, fontweight="bold")
# Shared y-axis range across both subplots
y_max = max(ax_cl.get_ylim()[1], ax_br.get_ylim()[1])
ax_cl.set_ylim(0, y_max * 1.1)
ax_br.set_ylim(0, y_max * 1.1)
plt.tight_layout()
save_figure(fig, "06b_angle_by_structure_type")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Median distance bar chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
_median_bar(ax, "h_x_distance", "median_distance",
            "Median H···X Distance (Å)",
            "Median H···X Distance by Structure Type and Halogen")
# Zoom y-axis to data range
all_dist_medians = [_stat(h, s, "median_distance")
                    for h in ["Cl", "Br"] for s in STRUCT_ORDER]
margin = 0.02
ax.set_ylim(min(all_dist_medians) - margin, max(all_dist_medians) + margin)
plt.tight_layout()
save_figure(fig, "06c_median_distance_by_structure")
plt.close(fig)

# ---------------------------------------------------------------------------
# (d) Median angle bar chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
_median_bar(ax, "c_h_x_angle", "median_angle",
            "Median C–H···X Angle (°)",
            "Median C–H···X Angle by Structure Type and Halogen")
all_angle_medians = [_stat(h, s, "median_angle")
                     for h in ["Cl", "Br"] for s in STRUCT_ORDER]
margin = 2
ax.set_ylim(min(all_angle_medians) - margin, max(all_angle_medians) + margin)
plt.tight_layout()
save_figure(fig, "06d_median_angle_by_structure")
plt.close(fig)

# ---------------------------------------------------------------------------
# (e) Halogen proportion donuts — one per structure type
# ---------------------------------------------------------------------------

fig, (ax_org, ax_omx) = plt.subplots(1, 2, figsize=(10, 5))

for ax, struct, title in [
    (ax_org, "organic",        "Organic Structures"),
    (ax_omx, "organometallic", "Organometallic Structures"),
]:
    subset = df[df["structure_type"] == struct]
    counts = [len(subset[subset["halogen"] == h]) for h in ["Cl", "Br"]]
    colours = [HALOGEN_COLOURS["Cl"], HALOGEN_COLOURS["Br"]]
    _draw_donut(ax, ["Cl", "Br"], counts, colours, "Total")
    ax.set_title(title, pad=16)

plt.tight_layout()
save_figure(fig, "06e_halogen_proportion_by_structure")
plt.close(fig)

print("Organic vs organometallic figures and table saved.")
