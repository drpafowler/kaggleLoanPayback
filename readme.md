# Loan Payback Prediction: Automated Stacking Pipeline

This repository contains a modular, end-to-end machine learning pipeline designed for the Season 5 Episode 11 Competition in Kaggle's Playground. It utilizes stacking to combine the strengths of GBDTs, Neural Networks, and Linear models.

---

## Directory Structure

```text
├── data/                   # Competition data & standardized fold definitions
│   ├── train.csv           # Training set
│   ├── test.csv            # Test set (for leaderboard)
│   ├── sample_sub.csv      # Submission template
│   └── fold_ids.csv        # Shared K-Fold IDs (Auto-generated)
├── oof/                    # Out-of-Fold & Test predictions (.npy format)
├── submissions/            # CSV files for individual models and final blend
├── controller.py           # Master Pipeline Controller
├── ensemble.py             # Correlation & diversity report
├── blender.py              # Ridge Regression Super-Learner (The Stacker)
└── [model_name].py         # Individual model scripts (LGBM, XGB, etc.)


