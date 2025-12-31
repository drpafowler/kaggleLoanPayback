import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
MODEL_NAME = "cat_v1"
DATA_DIR = 'data'
OOF_DIR = 'oof'
SUB_DIR = 'submissions'

# Ensure directories exist
for folder in [DATA_DIR, OOF_DIR, SUB_DIR]:
    os.makedirs(folder, exist_ok=True)

# Competition specific columns
TARGET = 'loan_paid_back'
ID = 'id'
SEED = 42
N_SPLITS = 5

# =========================================================================================
# 2. LOAD DATA & STANDARDIZE FOLDS
# =========================================================================================
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

# Use the EXACT same folds as the LightGBM and XGBoost models
fold_path = f'{DATA_DIR}/fold_ids.csv'
if os.path.exists(fold_path):
    print("--- Loading existing standard fold IDs ---")
    folds = pd.read_csv(fold_path)
    train = train.merge(folds, on=ID)
else:
    # This block only runs if you haven't run the LGBM or XGB scripts yet
    print("--- Creating new standard fold IDs ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
        train.loc[val_idx, 'fold'] = fold
    train[[ID, 'fold']].to_csv(fold_path, index=False)

# =========================================================================================
# 3. PREPROCESSING
# =========================================================================================
# CatBoost requires categorical features to be identified explicitly.
# We fill NaNs with a string placeholder as CatBoost handles them well this way.
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    train[col] = train[col].fillna("None").astype(str)
    test[col] = test[col].fillna("None").astype(str)

X = train.drop([ID, TARGET, 'fold'], axis=1)
y = train[TARGET]
X_test = test.drop([ID], axis=1)

# =========================================================================================
# 4. CROSS-VALIDATION LOOP
# =========================================================================================
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

for fold in range(N_SPLITS):
    print(f"\nTraining Fold {fold}...")
    
    # Split Data
    X_tr, y_tr = X[train['fold'] != fold], y[train['fold'] != fold]
    X_val, y_val = X[train['fold'] == fold], y[train['fold'] == fold]
    
    # Create CatBoost Pools (Optimized data structures for CatBoost)
    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    
    # Initialize CatBoost
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric='AUC',
        random_seed=SEED,
        verbose=200,      # Print progress every 200 iterations
        early_stopping_rounds=100
    )
    
    # Fit Model
    model.fit(train_pool, eval_set=val_pool)
    
    # Predict Out-of-Fold
    batch_oof = model.predict_proba(X_val)[:, 1]
    oof_preds[train['fold'] == fold] = batch_oof
    
    # Track performance
    score = roc_auc_score(y_val, batch_oof)
    fold_scores.append(score)
    print(f"Fold {fold} AUC: {score:.5f}")
    
    # Predict Test (Averaging over folds)
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# =========================================================================================
# 5. SAVE RESULTS
# =========================================================================================
overall_cv = roc_auc_score(y, oof_preds)
print(f"\n{'='*30}\nOVERALL CV AUC: {overall_cv:.5f}\n{'='*30}")

# Save OOF and Test predictions for the Blender
np.save(f'{OOF_DIR}/{MODEL_NAME}_oof.npy', oof_preds)
np.save(f'{OOF_DIR}/{MODEL_NAME}_test.npy', test_preds)

# Save Submission file
submission = sample_sub.copy()
submission[TARGET] = test_preds
sub_filename = f'{SUB_DIR}/submission_{MODEL_NAME}_CV{overall_cv:.4f}.csv'
submission.to_csv(sub_filename, index=False)

print(f"✅ Saved CatBoost OOFs to /{OOF_DIR} and submission to /{SUB_DIR}")