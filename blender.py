import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
DATA_DIR = 'data'
OOF_DIR = 'oof'
SUB_DIR = 'submissions'
TARGET = 'loan_paid_back'
ID = 'id'

# =========================================================================================
# 2. LOAD DATA
# =========================================================================================
# Load training labels
train = pd.read_csv(f'{DATA_DIR}/train.csv')
y_true = train[TARGET].values

# Get all OOF and Test prediction files
# We sort them to ensure the columns are in the exact same order for both matrices
oof_files = sorted(glob.glob(os.path.join(OOF_DIR, '*_oof.npy')))
test_files = sorted(glob.glob(os.path.join(OOF_DIR, '*_test.npy')))

if len(oof_files) != len(test_files):
    raise ValueError("Mismatch between number of OOF files and Test files!")

print(f"Found {len(oof_files)} models to stack:")
for f in oof_files:
    print(f" - {os.path.basename(f).replace('_oof.npy', '')}")

# =========================================================================================
# 3. CREATE META-FEATURES
# =========================================================================================
# Stack the predictions as columns in our meta-feature matrix
X_train_meta = np.column_stack([np.load(f) for f in oof_files])
X_test_meta = np.column_stack([np.load(f) for f in test_files])

# =========================================================================================
# 4. TRAIN RIDGE BLENDER
# =========================================================================================
print("\nTraining Ridge Regression Blender...")

# alpha is the regularization strength. 1.0 is a safe default.
# If you want to automate this, you could use RidgeCV(alphas=[0.1, 1.0, 10.0])
blender = Ridge(alpha=1.0, random_state=42)
blender.fit(X_train_meta, y_true)

# Look at the weights assigned to each model
print("\n--- Model Weights (Coefficients) ---")
model_names = [os.path.basename(f).replace('_oof.npy', '') for f in oof_files]
weights = pd.Series(blender.coef_, index=model_names)
print(weights.sort_values(ascending=False))

# =========================================================================================
# 5. EVALUATE & GENERATE SUBMISSION
# =========================================================================================
# Predict on training set to check final CV
final_oof_preds = blender.predict(X_train_meta)
final_oof_preds = np.clip(final_oof_preds, 0, 1) # Ensure probabilities are valid
final_cv_score = roc_auc_score(y_true, final_oof_preds)

print(f"\n{'='*40}")
print(f"FINAL STACKED CV AUC: {final_cv_score:.6f}")
print(f"{'='*40}")

# Predict on test set
final_test_preds = blender.predict(X_test_meta)
final_test_preds = np.clip(final_test_preds, 0, 1)

# Create submission
sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
sample_sub[TARGET] = final_test_preds
output_path = f'{SUB_DIR}/submission_final_ridge_stack.csv'
sample_sub.to_csv(output_path, index=False)

print(f"\n✅ Final stacked submission saved to: {output_path}")