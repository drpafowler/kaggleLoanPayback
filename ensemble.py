import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
DATA_DIR = 'data'
OOF_DIR = 'oof'
TARGET = 'loan_paid_back'

# Load the true labels from the training set
train = pd.read_csv(f'{DATA_DIR}/train.csv')
y_true = train[TARGET].values

# =========================================================================================
# 2. LOAD OOF PREDICTIONS
# =========================================================================================
# Get all OOF files in the directory
oof_files = sorted(glob.glob(os.path.join(OOF_DIR, '*_oof.npy')))

if not oof_files:
    print(f"No OOF files found in {OOF_DIR}!")
else:
    print(f"Found {len(oof_files)} OOF files.")

# Load each file into a dictionary
oof_dict = {}
for f in oof_files:
    model_name = os.path.basename(f).replace('_oof.npy', '')
    oof_dict[model_name] = np.load(f)

# Create a DataFrame of all OOF predictions
oof_df = pd.DataFrame(oof_dict)

# =========================================================================================
# 3. INDIVIDUAL MODEL PERFORMANCE
# =========================================================================================
print("\n--- Individual Model AUC Scores ---")
scores = {}
for col in oof_df.columns:
    score = roc_auc_score(y_true, oof_df[col])
    scores[col] = score
    print(f"{col:<20} : {score:.5f}")

# Sort scores to see the best performer
best_model = max(scores, key=scores.get)
print(f"\nBest Individual Model: {best_model} ({scores[best_model]:.5f})")

# =========================================================================================
# 4. CORRELATION ANALYSIS
# =========================================================================================
corr_matrix = oof_df.corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
plt.title('OOF Prediction Correlation Matrix')
plt.tight_layout()
plt.savefig('model_correlation_heatmap.png')
print("\n✅ Correlation heatmap saved as 'model_correlation_heatmap.png'")

# =========================================================================================
# 5. SIMPLE BLEND TEST
# =========================================================================================
# Check if a simple average improves the score
simple_average = oof_df.mean(axis=1)
blend_score = roc_auc_score(y_true, simple_average)

print(f"\nSimple Mean Blend AUC: {blend_score:.5f}")
improvement = blend_score - scores[best_model]
print(f"Improvement over best model: {improvement:+.5f}")

# =========================================================================================
# 6. DIVERSITY INSIGHTS
# =========================================================================================
print("\n--- Strategy Recommendations ---")
# Find highly correlated pairs
high_corr = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_matrix.iloc[i, j] > 0.98:
            high_corr.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

if high_corr:
    print("⚠️ High Correlation Alert (> 0.98):")
    for m1, m2, val in high_corr:
        print(f"  - {m1} & {m2}: {val:.4f} (Consider dropping one or tuning for diversity)")
else:
    print("✅ Models show good diversity (all correlations < 0.98).")

print("\nReady for stacking with the Ridge Regression Blender!")