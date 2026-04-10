# Data Mining Molecular Crystal Structures to Identify and Classify Geometric Patterns in C–H···Cl and C–H···Br Interactions

## Overview
Large-scale crystallographic data mining of the Cambridge Structural Database (CSD) to analyse weak hydrogen bonding interactions of the type C–H···X–C (X = Cl, Br). The project combines descriptive statistical analysis with supervised machine learning to characterise how chlorine and bromine function as weak hydrogen bond acceptors in molecular crystal structures.

## Research Questions
1. How do the geometric characteristics of C–H···Cl and C–H···Br interactions vary with halogen identity and chemical environment?
2. Can a machine learning classifier distinguish C–H···Cl from C–H···Br interactions based solely on geometric and environment features, and which features are most discriminative?

## Dataset
- **Source:** Cambridge Structural Database (CSD 2025)
- **Total contacts:** 290,499 (246,850 Cl, 43,649 Br)
- **Geometric criteria:** H···Cl 2.30–3.40 Å, H···Br 2.40–3.50 Å, C–H···X angle 100–180°
- **C–H normalisation:** Bond lengths normalised to 1.083 Å
- **Classification:** Structures classified as organic or organometallic

## Project Structure
ch_x_hbond_analysis/
├── data/                  → Raw and processed datasets
├── results/
│   ├── figures/           → All plots and visualisations
│   └── tables/            → Summary statistics and metrics (CSV)
├── scripts/               → All analysis and ML scripts
├── models/                → Trained classifiers and fitted transformers
├── .gitignore
└── README.md

## Scripts

### Configuration
| Script | Description |
|--------|-------------|
| `plot_config.py` | Shared styling, colour palettes (navy blue theme), file paths, and save helpers imported by all scripts |
| `00_extract_ch_x_contacts.py` | Extracts C–H···X–C contacts from the CSD using the CSD Python API with geometric filtering, hydrogen normalisation, hybridisation classification, and structure type assignment |

### Part 1 — Statistical Analysis
| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `01_dataset_overview.py` | Contact counts and breakdowns by halogen, structure type, and hybridisation | Donut charts, grouped bar charts, summary CSVs |
| `02_distance_distributions.py` | H···X distance analysis with KDE, box, and violin plots for Cl vs Br | Distance KDEs with median and vdW reference lines, summary statistics |
| `03_angle_distributions.py` | C–H···X angle analysis with KDE, box, and violin plots for Cl vs Br | Angle KDEs with median lines, summary statistics |
| `04_distance_vs_angle.py` | 2D density landscape of distance vs angle | Contour plots and hexbin panels for each halogen |
| `05_hybridisation_analysis.py` | Effect of donor hybridisation (sp, sp², sp³) on distance and angle | KDE panels by hybridisation, median comparison bar charts |
| `06_organic_vs_organometallic.py` | Comparison of interaction geometry between organic and organometallic structures | KDE overlays by structure type, median comparison charts, proportion donuts |
| `07_donor_acceptor_pairs.py` | Analysis of donor–acceptor hybridisation pair combinations | Count and median distance heatmaps for each halogen |
| `08_statistical_tests.py` | Mann-Whitney U, Kolmogorov-Smirnov, and Cohen's d tests | Statistical comparison tables (all tests p < 0.001; distance Cohen's d = 0.079, angle Cohen's d = 0.289) |

### Part 2 — Machine Learning
| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `09_feature_engineering.py` | One-hot encoding, scaling, stratified 80/20 train/test split | Processed CSVs in data/, scaler and encoder in models/ |
| `10_baseline_classifier.py` | Gradient boosted classifier on all 7 features with class-weight balancing | Accuracy 0.727, ROC AUC 0.791, confusion matrix, ROC curve |
| `11_ablation_no_distance.py` | Retrained classifier without H···X distance feature | Accuracy 0.645, ROC AUC 0.721 (ΔAUC = −0.070), comparison metrics |
| `12_feature_importance.py` | Permutation importance and SHAP analysis on both models | Permutation: distance ranks 1st; SHAP: organometallic ranks 1st; hybridisation features rank low |
| `13_ml_visualisations.py` | Combined summary panels and performance comparison charts | ML summary panel, feature importance panel, comparison bar chart |

## Key Findings
- **Distance:** Cl and Br show statistically significant but practically small distance differences (Cohen's d = 0.079)
- **Angle:** More meaningful angular divergence (Cohen's d = 0.289); Br contacts cluster at lower angles
- **Classification:** Full model achieves AUC 0.791; removing distance reduces AUC by only 0.070 to 0.721
- **Feature hierarchy:** H···X distance is the most discriminative feature by permutation importance; structure type (organometallic) produces the largest individual SHAP contributions
- **Ablation insight:** Residual AUC of 0.721 confirms that angle, hybridisation, and structure type carry real discriminative signal independent of the trivial size difference between Cl and Br

## Models
| File | Description |
|------|-------------|
| `full_feature_model.pkl` | Gradient boosted classifier trained on all 7 features |
| `no_distance_model.pkl` | Gradient boosted classifier trained without H···X distance |
| `scaler.pkl` | Fitted StandardScaler for numerical features |
| `label_encoder.pkl` | Fitted LabelEncoder for halogen target variable |

## Requirements
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
shap

## Author
Karla O'Donnell
