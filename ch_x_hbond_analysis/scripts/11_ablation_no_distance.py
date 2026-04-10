"""
11_ablation_no_distance.py
Retrains the classifier WITHOUT h_x_distance to test whether angular and
environment features alone can distinguish C–H···Cl from C–H···Br.
Compares results against the full-feature baseline (script 10).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from plot_config import *

PROJECT_CMAP = LinearSegmentedColormap.from_list(
    "project_blue",
    ["#FFFFFF", "#C8DDE5", "#A3C4D0", "#4A90A4", "#2D6073", "#1B3A4B"],
    N=256,
)

DATA_DIR = os.path.dirname(DATA_PATH)

# ---------------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------------

X_train_full = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test_full  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train      = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test       = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

# ---------------------------------------------------------------------------
# Drop h_x_distance
# ---------------------------------------------------------------------------

DROP_COL = "h_x_distance"
X_train  = X_train_full.drop(columns=[DROP_COL])
X_test   = X_test_full.drop(columns=[DROP_COL])

print(f"Dropped feature : {DROP_COL}")
print(f"Remaining features ({len(X_train.columns)}):")
for f in X_train.columns:
    print(f"  {f}")
print()

# ---------------------------------------------------------------------------
# Sample weights — identical to script 10
# ---------------------------------------------------------------------------

n_train   = len(y_train)
n_cl      = int((y_train == 0).sum())
n_br      = int((y_train == 1).sum())
weight_cl = n_train / (2 * n_cl)
weight_br = n_train / (2 * n_br)
sample_weights = np.where(y_train == 0, weight_cl, weight_br)

# ---------------------------------------------------------------------------
# Train ablation model
# ---------------------------------------------------------------------------

print("Training ablation GradientBoostingClassifier (no distance) …")
clf_ablation = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
clf_ablation.fit(X_train, y_train, sample_weight=sample_weights)
print("Training complete.")

# ---------------------------------------------------------------------------
# Ablation predictions
# ---------------------------------------------------------------------------

y_pred_abl  = clf_ablation.predict(X_test)
y_prob_abl  = clf_ablation.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Full model predictions (for ROC comparison)
# ---------------------------------------------------------------------------

clf_full   = joblib.load(os.path.join(MODELS_PATH, "full_feature_model.pkl"))
y_prob_full = clf_full.predict_proba(X_test_full)[:, 1]
y_pred_full = clf_full.predict(X_test_full)

# ---------------------------------------------------------------------------
# Metrics — ablation
# ---------------------------------------------------------------------------

acc_abl     = accuracy_score(y_test, y_pred_abl)
auc_abl     = roc_auc_score(y_test, y_prob_abl)
cm_abl      = confusion_matrix(y_test, y_pred_abl)
report_abl  = classification_report(y_test, y_pred_abl, target_names=["Cl", "Br"],
                                    output_dict=True)

# Metrics — full (re-derived for parity)
acc_full    = accuracy_score(y_test, y_pred_full)
auc_full    = roc_auc_score(y_test, y_prob_full)
report_full = classification_report(y_test, y_pred_full, target_names=["Cl", "Br"],
                                    output_dict=True)

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

ensure_dirs()
joblib.dump(clf_ablation, os.path.join(MODELS_PATH, "no_distance_model.pkl"))

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

report_df = pd.DataFrame(report_abl).T.reset_index().rename(columns={"index": "class"})
save_table(report_df, "ablation_classification_report")

cm_df = pd.DataFrame(
    cm_abl,
    index=["Actual Cl", "Actual Br"],
    columns=["Predicted Cl", "Predicted Br"],
)
save_table(cm_df.reset_index().rename(columns={"index": ""}), "ablation_confusion_matrix")

# Comparison table
metrics = ["Accuracy", "Precision (Cl)", "Recall (Cl)", "F1 (Cl)",
           "Precision (Br)", "Recall (Br)", "F1 (Br)", "ROC AUC"]
full_vals = [
    acc_full,
    report_full["Cl"]["precision"], report_full["Cl"]["recall"], report_full["Cl"]["f1-score"],
    report_full["Br"]["precision"], report_full["Br"]["recall"], report_full["Br"]["f1-score"],
    auc_full,
]
abl_vals = [
    acc_abl,
    report_abl["Cl"]["precision"], report_abl["Cl"]["recall"], report_abl["Cl"]["f1-score"],
    report_abl["Br"]["precision"], report_abl["Br"]["recall"], report_abl["Br"]["f1-score"],
    auc_abl,
]
comparison_df = pd.DataFrame({
    "Metric":       metrics,
    "Full Model":   [round(v, 4) for v in full_vals],
    "No Distance":  [round(v, 4) for v in abl_vals],
    "Difference":   [round(a - f, 4) for f, a in zip(full_vals, abl_vals)],
})
save_table(comparison_df, "model_comparison")

# ---------------------------------------------------------------------------
# (a) Confusion matrix — ablation
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_abl,
    ax=ax,
    annot=True,
    fmt=",d",
    cmap=PROJECT_CMAP,
    xticklabels=["Cl", "Br"],
    yticklabels=["Cl", "Br"],
    linewidths=0.5,
    linecolor="#E0E0E0",
    cbar_kws={"label": "Count"},
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Ablation Classifier Confusion Matrix (No Distance)")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
save_figure(fig, "11a_confusion_matrix_ablation")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) ROC curve comparison
# ---------------------------------------------------------------------------

fpr_full, tpr_full, _ = roc_curve(y_test, y_prob_full)
fpr_abl,  tpr_abl,  _ = roc_curve(y_test, y_prob_abl)
auc_diff = auc_abl - auc_full

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_full, tpr_full, color=HALOGEN_COLOURS["Cl"], linewidth=2.5,
        label=f"Full model (AUC = {auc_full:.4f})")
ax.plot(fpr_abl,  tpr_abl,  color=HALOGEN_COLOURS["Br"], linewidth=2.5,
        label=f"No distance (AUC = {auc_abl:.4f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=1.2, linestyle="--",
        label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve: Full Model vs No Distance")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(axis="both", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.legend(loc="lower right")

sign  = "+" if auc_diff >= 0 else ""
props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#CCCCCC")
ax.text(0.95, 0.28, f"ΔAUC = {sign}{auc_diff:.4f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="right", bbox=props)

plt.tight_layout()
save_figure(fig, "11b_roc_comparison")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Metric comparison bar chart
# ---------------------------------------------------------------------------

bar_metrics = ["Accuracy", "Precision (Br)", "Recall (Br)", "F1 (Br)", "ROC AUC"]
idx_map     = {m: i for i, m in enumerate(metrics)}
full_bar    = [full_vals[idx_map[m]] for m in bar_metrics]
abl_bar     = [abl_vals[idx_map[m]]  for m in bar_metrics]

x     = np.arange(len(bar_metrics))
bar_w = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_full = ax.bar(x - bar_w / 2, full_bar, width=bar_w,
                   color=HALOGEN_COLOURS["Cl"], label="Full Model", edgecolor="none")
bars_abl  = ax.bar(x + bar_w / 2, abl_bar,  width=bar_w,
                   color=HALOGEN_COLOURS["Br"], label="No Distance", edgecolor="none")

for bar in list(bars_full) + list(bars_abl):
    h = bar.get_height()
    ax.annotate(f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

all_vals = full_bar + abl_bar
y_lo = max(0, min(all_vals) - 0.05)
y_hi = min(1.0, max(all_vals) + 0.08)
ax.set_ylim(y_lo, y_hi)

ax.set_xticks(x)
ax.set_xticklabels(bar_metrics)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend(title="Model")
ax.grid(False)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, "11c_model_comparison")
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("ABLATION STUDY — FULL MODEL vs NO DISTANCE")
print("=" * 65)
print(f"  {'Metric':<22}  {'Full Model':>10}  {'No Distance':>11}  {'Diff':>8}")
print(f"  {'-'*22}  {'-'*10}  {'-'*11}  {'-'*8}")
for _, row in comparison_df.iterrows():
    sign = "+" if row["Difference"] >= 0 else ""
    print(f"  {row['Metric']:<22}  {row['Full Model']:>10.4f}  {row['No Distance']:>11.4f}  {sign}{row['Difference']:>7.4f}")
print()
print("  Ablation classification report (no distance):")
print(classification_report(y_test, y_pred_abl, target_names=["Cl", "Br"]))
print("  Saved:")
print("    models/no_distance_model.pkl")
print("    tables/ablation_classification_report.csv")
print("    tables/ablation_confusion_matrix.csv")
print("    tables/model_comparison.csv")
print("    figures/11a_confusion_matrix_ablation.png")
print("    figures/11b_roc_comparison.png")
print("    figures/11c_model_comparison.png")
print("=" * 65)
