"""
01_dataset_overview.py
Complete summary of the extracted C-H···X-C contact dataset.
Produces console output, summary tables, and overview figures.
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plot_config import *

# Load data

df = pd.read_csv(DATA_PATH)
n_total = len(df)

# Helper

def _fmt(n):
    """Format an integer with comma thousands separator."""
    return f"{n:,}"


def _summary(series, name):
    """Return a DataFrame with count and percentage for each value of *series*."""
    counts = series.value_counts()
    pct = (counts / n_total * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage": pct}).rename_axis(name).reset_index()


# 1. Compute summaries

halogen_summary      = _summary(df["halogen"],             "halogen")
structure_summary    = _summary(df["structure_type"],      "structure_type")
hybrid_summary       = _summary(df["donor_hybridisation"], "donor_hybridisation")

# Cross-tabulations (counts)
xtab_struct = pd.crosstab(df["halogen"], df["structure_type"])
xtab_hybrid = pd.crosstab(df["halogen"], df["donor_hybridisation"])

# Reorder hybridisation columns where present
hyb_order = [h for h in ("sp", "sp2", "sp3") if h in xtab_hybrid.columns]
xtab_hybrid = xtab_hybrid[hyb_order]

# 2. Console summary

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total contacts:       {_fmt(n_total)}")
print(f"Unique structures:    {_fmt(df['refcode'].nunique())}")

print("\n--- By halogen ---")
for _, row in halogen_summary.iterrows():
    print(f"  {row['halogen']:4s}  {_fmt(row['count']):>10s}  ({row['percentage']:.1f}%)")

print("\n--- By structure type ---")
for _, row in structure_summary.iterrows():
    print(f"  {row['structure_type']:15s}  {_fmt(row['count']):>10s}  ({row['percentage']:.1f}%)")

print("\n--- By donor hybridisation ---")
for _, row in hybrid_summary.iterrows():
    print(f"  {row['donor_hybridisation']:4s}  {_fmt(row['count']):>10s}  ({row['percentage']:.1f}%)")

print("\n--- Halogen × structure type ---")
print(xtab_struct.to_string())

print("\n--- Halogen × donor hybridisation ---")
print(xtab_hybrid.to_string())
print("=" * 60)

# 3. Save tables

save_table(halogen_summary,   "halogen_summary")
save_table(structure_summary, "structure_type_summary")
save_table(hybrid_summary,    "hybridisation_summary")
save_table(xtab_struct.reset_index(),  "halogen_by_structure")
save_table(xtab_hybrid.reset_index(), "halogen_by_hybridisation")

# 4. Figures

# Shared label map for hybridisation
hyb_labels = {"sp": "sp", "sp2": "sp²", "sp3": "sp³"}

bar_width = 0.3  # used in (d) and (e)


def _draw_donut(ax, labels, counts, colours):
    """
    Render a donut chart onto *ax*.
    Labels appear outside the ring: name, comma count, percentage.
    White centre shows total count and 'Total'.
    """
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
        ax.text(
            lx, ly,
            f"{label}\n{count:,}\n({pct:.1f}%)",
            ha="center", va="center",
            fontsize=10,
        )

    centre = plt.Circle((0, 0), 0.58, color="white")
    ax.add_patch(centre)
    ax.text(0,  0.10, f"{total:,}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0, -0.14, "Total",      ha="center", va="center", fontsize=10, color="#555555")
    ax.axis("off")


def _bar_labels(ax, rects, y_pad_frac=0.01):
    """Add comma-formatted count labels above each vertical bar."""
    y_pad = ax.get_ylim()[1] * y_pad_frac
    for rect in rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, h + y_pad,
            f"{int(h):,}",
            ha="center", va="bottom", fontsize=8,
        )


# --- (a) Halogen donut ------------------------------------------------------

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111)
_draw_donut(
    ax,
    labels  = halogen_summary["halogen"].tolist(),
    counts  = halogen_summary["count"].tolist(),
    colours = [HALOGEN_COLOURS[h] for h in halogen_summary["halogen"]],
)
ax.set_title("Distribution of Contacts by Halogen Identity", pad=16)
plt.tight_layout()
save_figure(fig, "01a_contacts_by_halogen")
plt.close(fig)

# --- (b) Structure type donut -----------------------------------------------

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111)
_draw_donut(
    ax,
    labels  = structure_summary["structure_type"].tolist(),
    counts  = structure_summary["count"].tolist(),
    colours = [STRUCTURE_COLOURS.get(s, CATEGORICAL_PALETTE[i])
               for i, s in enumerate(structure_summary["structure_type"])],
)
ax.set_title("Distribution of Contacts by Structure Type", pad=16)
plt.tight_layout()
save_figure(fig, "01b_contacts_by_structure_type")
plt.close(fig)

# --- (c) Hybridisation — donut (left) + log-scale bar (right) ---------------

# Ordered sp → sp2 → sp3 for both subplots
hyb_order_all = [h for h in ("sp", "sp2", "sp3") if h in hybrid_summary["donor_hybridisation"].values]
hyb_df        = hybrid_summary.set_index("donor_hybridisation").loc[hyb_order_all].reset_index()
hyb_colours   = [HYBRID_COLOURS[h] for h in hyb_df["donor_hybridisation"]]

fig, (ax_donut, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

# Left: donut — custom labelling because sp (0.0%) and sp2 (3.8%) are tiny
hyb_total = hyb_df["count"].sum()
hyb_pcts  = (hyb_df["count"] / hyb_total * 100).tolist()
hyb_cnts  = hyb_df["count"].tolist()
hyb_keys  = hyb_df["donor_hybridisation"].tolist()   # ["sp", "sp2", "sp3"]

ax_donut.set_aspect("equal")
wedges, _ = ax_donut.pie(
    hyb_cnts,
    colors=hyb_colours,
    startangle=90,
    wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
)

_arrow = dict(arrowstyle="-", color="#666666", lw=0.8)

for wedge, key, count, pct in zip(wedges, hyb_keys, hyb_cnts, hyb_pcts):
    angle = (wedge.theta1 + wedge.theta2) / 2
    rad   = math.radians(angle)
    label_text = f"{hyb_labels[key]}\n{count:,}\n({pct:.1f}%)"

    if key == "sp3":
        # Large segment — standard outside placement
        lx = math.cos(rad) * 1.25
        ly = math.sin(rad) * 1.25
        ax_donut.text(lx, ly, label_text, ha="center", va="center", fontsize=10)
    else:
        # Arrow tip at ring midpoint (radius 0.8)
        tx = math.cos(rad) * 0.8
        ty = math.sin(rad) * 0.8
        # sp2 → upper right; sp → above sp2 (30 pt gap enforced via xytext offset)
        if key == "sp2":
            lx, ly = 1.55, 0.90
        else:  # sp
            lx, ly = 1.55, 1.40
        ax_donut.annotate(
            label_text,
            xy=(tx, ty), xycoords="data",
            xytext=(lx, ly), textcoords="data",
            ha="center", va="center", fontsize=10,
            arrowprops=_arrow,
        )

centre = plt.Circle((0, 0), 0.58, color="white")
ax_donut.add_patch(centre)
ax_donut.text(0,  0.10, f"{hyb_total:,}", ha="center", va="center", fontsize=13, fontweight="bold")
ax_donut.text(0, -0.14, "Total",           ha="center", va="center", fontsize=10, color="#555555")
ax_donut.axis("off")
ax_donut.set_title("Hybridisation Proportions", pad=16)

# Right: horizontal bar, log scale, top-to-bottom = sp3, sp2, sp
hyb_df_rev  = hyb_df.iloc[::-1].reset_index(drop=True)
bar_colours = [HYBRID_COLOURS[h] for h in hyb_df_rev["donor_hybridisation"]]

bars = ax_bar.barh(
    [hyb_labels[h] for h in hyb_df_rev["donor_hybridisation"]],
    hyb_df_rev["count"],
    color=bar_colours,
    edgecolor="none",
    height=0.5,
)
for bar, count in zip(bars, hyb_df_rev["count"]):
    ax_bar.text(
        bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
        f"{count:,}",
        va="center", ha="left", fontsize=9,
    )
ax_bar.set_xscale("log")
ax_bar.set_title("Hybridisation Counts (log scale)", pad=16)
ax_bar.set_xlabel("Number of Contacts (log scale)")
ax_bar.set_ylabel("Donor Hybridisation")
ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_bar.grid(False)
ax_bar.grid(axis="x", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax_bar.set_axisbelow(True)

fig.suptitle("Distribution of Contacts by Donor Hybridisation", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save_figure(fig, "01c_contacts_by_hybridisation")
plt.close(fig)

# --- (d) Halogen × structure type (grouped bar) -----------------------------

struct_cols   = xtab_struct.columns.tolist()
struct_labels = [s.capitalize() for s in struct_cols]
halogens_list = xtab_struct.index.tolist()
x             = range(len(struct_cols))

fig = plt.figure(figsize=(8, 5))
ax  = fig.add_subplot(111)

for i, halogen in enumerate(halogens_list):
    offset = (i - len(halogens_list) / 2 + 0.5) * bar_width
    rects  = ax.bar(
        [xi + offset for xi in x],
        xtab_struct.loc[halogen],
        width=bar_width,
        label=halogen,
        color=HALOGEN_COLOURS.get(halogen, CATEGORICAL_PALETTE[i]),
        edgecolor="none",
    )
    _bar_labels(ax, rects)

ax.set_title("C–H···X Contacts by Halogen and Structure Type")
ax.set_xlabel("Structure Type")
ax.set_ylabel("Number of Contacts")
ax.set_xticks(list(x))
ax.set_xticklabels(struct_labels)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend(title="Halogen")
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "01d_halogen_by_structure_type")
plt.close(fig)

# --- (e) Halogen × donor hybridisation (grouped bar, log y-axis) ------------

hyb_order     = [h for h in ("sp", "sp2", "sp3") if h in xtab_hybrid.columns]
hyb_xlabels   = [hyb_labels[h] for h in hyb_order]
xtab_hybrid   = xtab_hybrid[hyb_order]
halogens_list = xtab_hybrid.index.tolist()
x             = range(len(hyb_order))

fig = plt.figure(figsize=(8, 5))
ax  = fig.add_subplot(111)

for i, halogen in enumerate(halogens_list):
    offset = (i - len(halogens_list) / 2 + 0.5) * bar_width
    rects  = ax.bar(
        [xi + offset for xi in x],
        xtab_hybrid.loc[halogen],
        width=bar_width,
        label=halogen,
        color=HALOGEN_COLOURS.get(halogen, CATEGORICAL_PALETTE[i]),
        edgecolor="none",
    )
    # Use point-offset annotations so labels sit correctly on a log scale
    for rect in rects:
        h = rect.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

ax.set_yscale("log")
ax.set_ylim(1, 500_000)
ax.set_title("C–H···X Contacts by Halogen and Donor Hybridisation")
ax.set_xlabel("Donor Hybridisation")
ax.set_ylabel("Number of Contacts (log scale)")
ax.set_xticks(list(x))
ax.set_xticklabels(hyb_xlabels)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend(title="Halogen")
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "01e_halogen_by_hybridisation")
plt.close(fig)

print("\nAll tables and figures saved.")
