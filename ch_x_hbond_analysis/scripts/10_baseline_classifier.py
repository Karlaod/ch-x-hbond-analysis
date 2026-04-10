"""
10_baseline_classifier.py
Gradient-boosted classifier to distinguish C–H···Cl from C–H···Br interactions
using all engineered features. Handles class imbalance via sample weights.
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

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

feature_names = pd.read_csv(os.path.join(DATA_DIR, "feature_names.csv"))["feature"].tolist()

# ---------------------------------------------------------------------------
# Sample weights — compensate for 85/15 class imbalance
# ---------------------------------------------------------------------------

n_train   = len(y_train)
n_cl      = int((y_train == 0).sum())
n_br      = int((y_train == 1).sum())
weight_cl = n_train / (2 * n_cl)
weight_br = n_train / (2 * n_br)

sample_weights = np.where(y_train == 0, weight_cl, weight_br)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Training GradientBoostingClassifier …")
clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
clf.fit(X_train, y_train, sample_weight=sample_weights)
print("Training complete.")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

y_pred      = clf.predict(X_test)
y_prob      = clf.predict_proba(X_test)[:, 1]   # probability of Br (class 1)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm      = confusion_matrix(y_test, y_pred)
report  = classification_report(y_test, y_pred, target_names=["Cl", "Br"], output_dict=True)

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

ensure_dirs()
joblib.dump(clf, os.path.join(MODELS_PATH, "full_feature_model.pkl"))

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
save_table(report_df, "baseline_classification_report")

cm_df = pd.DataFrame(cm, index=["Actual Cl", "Actual Br"],
                         columns=["Predicted Cl", "Predicted Br"])
save_table(cm_df.reset_index().rename(columns={"index": ""}),
           "baseline_confusion_matrix")

# ---------------------------------------------------------------------------
# (a) Confusion matrix heatmap
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
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
ax.set_title("Baseline Classifier Confusion Matrix")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
save_figure(fig, "10a_confusion_matrix_baseline")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) ROC curve
# ---------------------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=HALOGEN_COLOURS["Cl"], linewidth=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=1.2, linestyle="--", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Baseline Classifier ROC Curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(axis="both", color="#E0E0E0", linewidth=0.5, alpha=0.3)

props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#CCCCCC")
ax.text(0.62, 0.12, f"AUC = {roc_auc:.4f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="left", bbox=props)

ax.legend(loc="lower right")
plt.tight_layout()
save_figure(fig, "10b_roc_curve_baseline")
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("BASELINE CLASSIFIER — RESULTS SUMMARY")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}")
print(f"  ROC AUC   : {roc_auc:.4f}")
print()
print("  Classification report:")
print(classification_report(y_test, y_pred, target_names=["Cl", "Br"]))
print("  Confusion matrix (rows = Actual, cols = Predicted):")
print(f"               Pred Cl    Pred Br")
print(f"  Actual Cl    {cm[0,0]:>8,}   {cm[0,1]:>8,}")
print(f"  Actual Br    {cm[1,0]:>8,}   {cm[1,1]:>8,}")
print()
print("  Saved:")
print("    models/full_feature_model.pkl")
print("    tables/baseline_classification_report.csv")
print("    tables/baseline_confusion_matrix.csv")
print("    figures/10a_confusion_matrix_baseline.png")
print("    figures/10b_roc_curve_baseline.png")
print("=" * 60)
