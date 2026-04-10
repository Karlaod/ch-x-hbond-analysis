# Data Mining Molecular Crystal Structures to Identify and Classify Geometric Patterns in C–H···Cl and C–H···Br Interactions

## Overview
Large-scale crystallographic data mining of the Cambridge Structural Database (CSD) to analyse weak hydrogen bonding interactions of the type C–H···X–C (X = Cl, Br). The project combines descriptive statistical analysis with supervised machine learning to characterise how chlorine and bromine function as weak hydrogen bond acceptors in molecular crystal structures.

## Research Questions
1. How do the geometric characteristics of C–H···Cl and C–H···Br interactions vary with halogen identity and chemical environment?
2. Can a machine learning classifier distinguish C–H···Cl from C–H···Br interactions based solely on geometric and environment features, and which features are most discriminative?

## Dataset
- Source: Cambridge Structural Database (CSD 2025)
- 290,499 accepted C–H···X contacts (246,850 Cl, 43,649 Br)
- Geometric criteria: H···Cl 2.30–3.40 Å, H···Br 2.40–3.50 Å, C–H···X angle 100–180°
- C–H bond lengths normalised to 1.083 Å
- Structures classified as organic or organometallic

## Project Structure
ch_x_hbond_analysis/
├── data/                → Raw extracted contact data
├── results/
│   ├── figures/         → All plots and visualisations
│   └── tables/          → Summary statistics and metrics
├── scripts/
│   ├── 00–08            → Part 1: Statistical analysis
│   └── 09–13            → Part 2: Machine learning
├── models/              → Trained classifiers and fitted transformers
└── README.md

## Scripts

### Part 1 — Statistical Analysis
| Script | Purpose | Output |
|--------|---------|--------|
| 00_extract_ch_x_contacts.py | CSD data extraction | data/ |
| 01_dataset_overview.py | Contact counts and breakdowns | results/tables/ |
| 02_distance_distributions.py | H···X distance histograms and KDEs | results/figures/ |
| 03_angle_distributions.py | C–H···X angle histograms and KDEs | results/figures/ |
| 04_distance_vs_angle.py | 2D density plots | results/figures/ |
| 05_hybridisation_analysis.py | Donor hybridisation breakdowns | results/figures/ |
| 06_organic_vs_organometallic.py | Structure type comparisons | results/figures/ |
| 07_donor_acceptor_pairs.py | Pair combination heatmaps | results/figures/ |
| 08_statistical_tests.py | KS and Mann-Whitney U tests | results/tables/ |

### Part 2 — Machine Learning
| Script | Purpose | Output |
|--------|---------|--------|
| 09_feature_engineering.py | Encoding, scaling, train/test split | data/ + models/ |
| 10_baseline_classifier.py | Full-feature classification | models/ + results/ |
| 11_ablation_no_distance.py | Classification without distance | models/ + results/ |
| 12_feature_importance.py | SHAP and permutation importance | results/ |
| 13_ml_visualisations.py | ROC curves, SHAP plots | results/figures/ |

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
Dublin City University