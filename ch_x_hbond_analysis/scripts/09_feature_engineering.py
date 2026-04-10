"""
09_feature_engineering.py
Prepares the dataset for ML classification (Cl vs Br).
Outputs: scaled train/test splits, saved scaler and label encoder.
No figures — data artefacts only.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from plot_config import *

DATA_DIR = os.path.dirname(DATA_PATH)

# Load data

df = pd.read_csv(DATA_PATH)

# Feature matrix

NUMERIC_COLS     = ["h_x_distance", "c_h_x_angle"]
CATEGORICAL_COLS = ["donor_hybridisation", "acceptor_hybridisation", "structure_type"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS

X_raw = df[FEATURE_COLS].copy()

# One-hot encode categoricals (drop first to avoid multicollinearity)
X_encoded = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=True)

feature_names = list(X_encoded.columns)

# Target encoding — Cl = 0, Br = 1

le = LabelEncoder()
y = le.fit_transform(df["halogen"])          # ['Br', 'Cl'] → sorted → Br=0, Cl=1
# Ensure Cl=0, Br=1 regardless of LabelEncoder sort order
if le.classes_[0] == "Cl":                   # Cl → 0, Br → 1  ✓
    pass
else:                                         # Br → 0, Cl → 1: flip
    y = 1 - y
    le.classes_ = le.classes_[::-1]

# Train / test split — stratified 80/20

X_train_enc, X_test_enc, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Scale numerical features only

scaler = StandardScaler()

X_train = X_train_enc.copy()
X_test  = X_test_enc.copy()

X_train[NUMERIC_COLS] = scaler.fit_transform(X_train_enc[NUMERIC_COLS])
X_test[NUMERIC_COLS]  = scaler.transform(X_test_enc[NUMERIC_COLS])

# Save model artefacts

ensure_dirs()
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
joblib.dump(le,     os.path.join(MODELS_PATH, "label_encoder.pkl"))

# Save data splits as CSV

X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv( os.path.join(DATA_DIR, "X_test.csv"),  index=False)

pd.Series(y_train, name="halogen").to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
pd.Series(y_test,  name="halogen").to_csv(os.path.join(DATA_DIR, "y_test.csv"),  index=False)

pd.DataFrame({"feature": feature_names}).to_csv(
    os.path.join(DATA_DIR, "feature_names.csv"), index=False
)

# Console summary

n_total  = len(df)
n_feat   = len(feature_names)
n_train  = len(y_train)
n_test   = len(y_test)

cl_train = int((y_train == 0).sum())
br_train = int((y_train == 1).sum())
cl_test  = int((y_test  == 0).sum())
br_test  = int((y_test  == 1).sum())

print("=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"  Total samples      : {n_total:,}")
print(f"  Features (encoded) : {n_feat}")
print()
print("  Feature names:")
for name in feature_names:
    print(f"    {name}")
print()
print(f"  Train set : {n_train:,}  ({n_train / n_total * 100:.1f}%)")
print(f"  Test set  : {n_test:,}  ({n_test  / n_total * 100:.1f}%)")
print()
print("  Class distribution (Cl=0, Br=1):")
print(f"    Train — Cl: {cl_train:,}  ({cl_train / n_train * 100:.1f}%)  "
      f"Br: {br_train:,}  ({br_train / n_train * 100:.1f}%)")
print(f"    Test  — Cl: {cl_test:,}  ({cl_test  / n_test  * 100:.1f}%)  "
      f"Br: {br_test:,}  ({br_test  / n_test  * 100:.1f}%)")
print()
print("  NOTE: Class imbalance (~85% Cl / ~15% Br) will be handled")
print("  in the classifier via class_weight='balanced', not resampling.")
print()
print("  Saved artefacts:")
print(f"    models/scaler.pkl")
print(f"    models/label_encoder.pkl")
print(f"    data/X_train.csv, X_test.csv")
print(f"    data/y_train.csv, y_test.csv")
print(f"    data/feature_names.csv")
print("=" * 60)
