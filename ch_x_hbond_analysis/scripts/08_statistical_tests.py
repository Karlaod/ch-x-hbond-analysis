"""
08_statistical_tests.py
Statistical tests comparing C–H···Cl and C–H···Br interaction geometries.
No figures — tables only.

Note: with ~290K data points almost all p-values will be < 0.001.
Effect sizes (Cohen's d) are more informative than p-values at this sample size.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from plot_config import *

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

cl = df[df["halogen"] == "Cl"]
br = df[df["halogen"] == "Br"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Pooled Cohen's d."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else np.nan


def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


results = []

def _record(comparison, test, stat, p, effect_size=np.nan):
    results.append({
        "comparison":   comparison,
        "test":         test,
        "statistic":    round(float(stat), 4),
        "p_value":      float(p),
        "effect_size":  round(float(effect_size), 4) if not np.isnan(effect_size) else np.nan,
        "significance": sig_label(p),
    })

def _mw(comparison, a, b):
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    _record(comparison, "Mann-Whitney U", stat, p)

def _ks(comparison, a, b):
    stat, p = ks_2samp(a, b)
    _record(comparison, "KS two-sample", stat, p)

def _mw_ks_d(comparison, a, b):
    """Mann-Whitney U + KS + Cohen's d for a pair."""
    _mw(comparison, a, b)
    _ks(comparison, a, b)
    d = cohens_d(a, b)
    # Attach Cohen's d to a separate row
    _record(comparison, "Cohen's d", d, 1.0, effect_size=d)


# ---------------------------------------------------------------------------
# 1 & 2. Cl vs Br — distance and angle
# ---------------------------------------------------------------------------

_mw_ks_d("Cl vs Br | H···X Distance",   cl["h_x_distance"].values, br["h_x_distance"].values)
_mw_ks_d("Cl vs Br | C–H···X Angle",    cl["c_h_x_angle"].values,  br["c_h_x_angle"].values)

# ---------------------------------------------------------------------------
# 3 & 4. Within each halogen — Organic vs Organometallic
# ---------------------------------------------------------------------------

for hal, subset in [("Cl", cl), ("Br", br)]:
    org  = subset[subset["structure_type"] == "organic"]
    omx  = subset[subset["structure_type"] == "organometallic"]
    _mw(f"{hal} | Organic vs Organometallic | Distance", org["h_x_distance"].values, omx["h_x_distance"].values)
    _mw(f"{hal} | Organic vs Organometallic | Angle",    org["c_h_x_angle"].values,  omx["c_h_x_angle"].values)

# ---------------------------------------------------------------------------
# 5 & 6. Within each halogen — sp² vs sp³
# ---------------------------------------------------------------------------

for hal, subset in [("Cl", cl), ("Br", br)]:
    sp2 = subset[subset["donor_hybridisation"] == "sp2"]
    sp3 = subset[subset["donor_hybridisation"] == "sp3"]
    _mw(f"{hal} | sp² vs sp³ | Distance", sp2["h_x_distance"].values, sp3["h_x_distance"].values)
    _mw(f"{hal} | sp² vs sp³ | Angle",    sp2["c_h_x_angle"].values,  sp3["c_h_x_angle"].values)

# ---------------------------------------------------------------------------
# Compile results
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

print("=" * 72)
print("STATISTICAL TESTS — C–H···X INTERACTION GEOMETRY")
print("=" * 72)
print()
print("NOTE: n ~ 290,000. At this sample size nearly all tests will reach")
print("p < 0.001. Prioritise effect sizes (Cohen's d) over p-values.")
print()

current_comparison = None
for _, row in results_df.iterrows():
    if row["comparison"] != current_comparison:
        current_comparison = row["comparison"]
        print(f"  {current_comparison}")

    es_str = f"  effect size = {row['effect_size']:.4f}" if not np.isnan(row["effect_size"]) else ""
    print(f"    {row['test']:<22}  stat = {row['statistic']:>14.4f}  "
          f"p = {row['p_value']:.2e}  {row['significance']}{es_str}")
print()
print("=" * 72)

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

save_table(results_df, "statistical_tests_summary")

cl_vs_br = results_df[results_df["comparison"].str.startswith("Cl vs Br")]
save_table(cl_vs_br, "cl_vs_br_summary")

print("Statistical test tables saved.")
