import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
MODEL_NAME = "logistic_v1"
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

# Load the exact same folds used for your GBDT models
fold_path = f'{DATA_DIR}/fold_ids.csv'
if os.path.exists(fold_path):
    print("--- Loading existing standard fold IDs ---")
    folds = pd.read_csv(fold_path)
    train = train.merge(folds, on=ID)
else:
    print("Warning: fold_ids.csv not found. Creating new folds (Ensure other models use these).")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
        train.loc[val_idx, 'fold'] = fold
    train[[ID, 'fold']].to_csv(fold_path, index=False)

# =========================================================================================
# 3. PREPROCESSING PIPELINE
# =========================================================================================
# Identify columns
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Numeric: Impute missing and Scale (Critical for Logistic Regression)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Impute missing and One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# =========================================================================================
# 4. CROSS-VALIDATION LOOP
# =========================================================================================
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

# Prepare Test data features once
X_test = test.drop([ID], axis=1)

for fold in range(N_SPLITS):
    print(f"\nTraining Fold {fold}...")
    
    # Split
    train_df = train[train['fold'] != fold]
    val_df = train[train['fold'] == fold]
    
    X_tr, y_tr = train_df.drop([ID, TARGET, 'fold'], axis=1), train_df[TARGET]
    X_val, y_val = val_df.drop([ID, TARGET, 'fold'], axis=1), val_df[TARGET]
    
    # Build the full Pipeline: Preprocess -> Model
    # Logistic Regression is used for binary classification (0/1)
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1))
    ])
    
    # Fit
    clf.fit(X_tr, y_tr)
    
    # Predict probabilities (index 1 for 'loan_paid_back')
    batch_oof = clf.predict_proba(X_val)[:, 1]
    oof_preds[train['fold'] == fold] = batch_oof
    
    # Track AUC
    score = roc_auc_score(y_val, batch_oof)
    fold_scores.append(score)
    print(f"Fold {fold} AUC: {score:.5f}")
    
    # Predict Test
    test_preds += clf.predict_proba(X_test)[:, 1] / N_SPLITS

# =========================================================================================
# 5. SAVE RESULTS
# =========================================================================================
overall_cv = roc_auc_score(train[TARGET], oof_preds)
print(f"\n{'='*30}\nLOGISTIC REGRESSION OVERALL CV AUC: {overall_cv:.5f}\n{'='*30}")

# Save OOF and Test predictions for the Blender
np.save(f'{OOF_DIR}/{MODEL_NAME}_oof.npy', oof_preds)
np.save(f'{OOF_DIR}/{MODEL_NAME}_test.npy', test_preds)

# Save standard submission file
submission = sample_sub.copy()
submission[TARGET] = test_preds
sub_filename = f'{SUB_DIR}/submission_{MODEL_NAME}_CV{overall_cv:.4f}.csv'
submission.to_csv(sub_filename, index=False)

print(f"✅ Saved Logistic Regression OOFs to /{OOF_DIR} and submission to /{SUB_DIR}")