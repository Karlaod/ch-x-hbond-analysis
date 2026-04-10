"""
plot_config.py
Shared styling, colour palettes, paths, and save helpers for all analysis
and visualisation scripts in this project. Import at the top of each script:

    from plot_config import *
"""

import os
import matplotlib.pyplot as plt

# Paths

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_PATH       = os.path.join(_PROJECT_DIR, "data",            "ch_x_contacts.csv")
RESULTS_FIGURES = os.path.join(_PROJECT_DIR, "results",         "figures")
RESULTS_TABLES  = os.path.join(_PROJECT_DIR, "results",         "tables")
MODELS_PATH     = os.path.join(_PROJECT_DIR, "models")


def ensure_dirs():
    """Create output directories if they do not already exist."""
    for path in (RESULTS_FIGURES, RESULTS_TABLES, MODELS_PATH):
        os.makedirs(path, exist_ok=True)


# Colour palettes

HALOGEN_COLOURS = {
    "Cl": "#1B3A4B",
    "Br": "#4A90A4",
}

STRUCTURE_COLOURS = {
    "organic":        "#4A90A4",
    "organometallic": "#1B3A4B",
}

HYBRID_COLOURS = {
    "sp":  "#A3C4D0",
    "sp2": "#4A90A4",
    "sp3": "#1B3A4B",
}

# General categorical palette — blue gradient, dark to light
CATEGORICAL_PALETTE = [
    "#1B3A4B", "#2D6073", "#4A90A4", "#6DB0C4",
    "#A3C4D0", "#C8DDE5", "#E0EDF2",
]

# Single accent colour for single-category plots
ACCENT_SLATE = "#1B3A4B"

# Matplotlib style

plt.rcParams.update({
    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],

    # Axes titles and labels
    "axes.titlesize":        14,
    "axes.titleweight":      "bold",
    "axes.labelsize":        12,

    # Tick labels
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,

    # Legend
    "legend.fontsize":       10,

    # Figure
    "figure.dpi":            300,
    "figure.figsize":        (8, 5),
    "figure.facecolor":      "white",

    # Axes background
    "axes.facecolor":        "white",

    # Grid
    "axes.grid":             True,
    "grid.color":            "#E0E0E0",
    "grid.linewidth":        0.5,
    "axes.axisbelow":        True,   # grid behind plot elements

    # Spines: hide top and right
    "axes.spines.top":       False,
    "axes.spines.right":     False,

    # Save
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",
})

# Save helpers

def save_figure(fig, filename):
    """
    Save *fig* to RESULTS_FIGURES as both PNG and PDF.

    Parameters
    ----------
    fig      : matplotlib.figure.Figure
    filename : str  — base name without extension, e.g. "distance_distribution"
    """
    ensure_dirs()
    base = os.path.join(RESULTS_FIGURES, filename)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")


def save_table(df, filename):
    """
    Save *df* to RESULTS_TABLES as CSV.

    Parameters
    ----------
    df       : pandas.DataFrame
    filename : str  — base name without extension, e.g. "summary_stats"
    """
    ensure_dirs()
    path = os.path.join(RESULTS_TABLES, filename + ".csv")
    df.to_csv(path, index=False)
