"""
04_distance_vs_angle.py
2D density plots showing the relationship between H···X distance and C-H···X angle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from plot_config import *

PROJECT_CMAP = LinearSegmentedColormap.from_list(
    "project_blue", ["#FFFFFF", "#C8DDE5", "#A3C4D0", "#4A90A4", "#2D6073", "#1B3A4B"], N=256
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

cl = df[df["halogen"] == "Cl"][["h_x_distance", "c_h_x_angle"]].values
br = df[df["halogen"] == "Br"][["h_x_distance", "c_h_x_angle"]].values

# ---------------------------------------------------------------------------
# Shared axis ranges (consistent across all panels)
# ---------------------------------------------------------------------------

x_lo = df["h_x_distance"].min() - 0.05
x_hi = df["h_x_distance"].max() + 0.05
y_lo, y_hi = 100, 180

# Grid for KDE evaluation
xx, yy = np.mgrid[x_lo:x_hi:200j, y_lo:y_hi:200j]
grid   = np.vstack([xx.ravel(), yy.ravel()])

# ---------------------------------------------------------------------------
# KDE evaluation helper
# ---------------------------------------------------------------------------

def _kde_values(data):
    """Return KDE density evaluated on the shared (xx, yy) grid."""
    kde = gaussian_kde(data.T)
    return kde(grid).reshape(xx.shape)


# Pre-compute KDE for both halogens (used in both panel and individual plots)
zz_cl = _kde_values(cl)
zz_br = _kde_values(br)

# Shared colour scale limits for contourf (so both panels are comparable)
vmin = min(zz_cl.min(), zz_br.min())
vmax = max(zz_cl.max(), zz_br.max())
levels = np.linspace(vmin, vmax, 16)   # 15 intervals → 16 boundaries

# ---------------------------------------------------------------------------
# Helper: draw a single contour panel onto an existing Axes
# ---------------------------------------------------------------------------

def _draw_contour(ax, zz, title, colourbar_label="Density"):
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap=PROJECT_CMAP)
    cb = plt.colorbar(cf, ax=ax)
    cb.set_label(colourbar_label)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("H···X Distance (Å)")
    ax.set_ylabel("C–H···X Angle (°)")
    ax.set_title(title)


def _draw_hexbin(ax, data, title):
    hb = ax.hexbin(
        data[:, 0], data[:, 1],
        gridsize=40, cmap=PROJECT_CMAP,
        extent=(x_lo, x_hi, y_lo, y_hi),
    )
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Count")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("H···X Distance (Å)")
    ax.set_ylabel("C–H···X Angle (°)")
    ax.set_title(title)


# ---------------------------------------------------------------------------
# (a) 2×2 panel
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

_draw_contour(axes[0, 0], zz_cl, "C–H···Cl Density Contour")
_draw_contour(axes[0, 1], zz_br, "C–H···Br Density Contour")
_draw_hexbin(axes[1, 0], cl, "C–H···Cl Hexbin")
_draw_hexbin(axes[1, 1], br, "C–H···Br Hexbin")

# Restore all four spines on panel subplots for framing
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(True)

# Row labels in place of suptitle
fig.text(0.01, 0.75, "Density Contour", va="center", ha="left",
         rotation=90, fontsize=13, fontweight="bold")
fig.text(0.01, 0.27, "Hexbin", va="center", ha="left",
         rotation=90, fontsize=13, fontweight="bold")

plt.tight_layout()
save_figure(fig, "04a_distance_vs_angle_panel")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Cl contour — individual
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_draw_contour(ax, zz_cl, "C–H···Cl Distance vs Angle Density Contour")
plt.tight_layout()
save_figure(fig, "04b_distance_vs_angle_contour_cl")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Br contour — individual
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
_draw_contour(ax, zz_br, "C–H···Br Distance vs Angle Density Contour")
plt.tight_layout()
save_figure(fig, "04c_distance_vs_angle_contour_br")
plt.close(fig)

print("Distance vs angle figures saved.")
