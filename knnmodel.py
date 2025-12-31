import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
MODEL_NAME = "knn_v1"
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

# Load the exact same folds used for your other models
fold_path = f'{DATA_DIR}/fold_ids.csv'
if os.path.exists(fold_path):
    print("--- Loading existing standard fold IDs ---")
    folds = pd.read_csv(fold_path)
    train = train.merge(folds, on=ID)
else:
    print("Error: fold_ids.csv not found. Run a baseline GBDT script first!")

# =========================================================================================
# 3. PREPROCESSING PIPELINE (Crucial for KNN)
# =========================================================================================
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Numeric: Impute and Standardize (KNN is distance-based, scaling is mandatory)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: One-Hot Encoding is preferred for KNN to maintain equidistant features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

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
    print(f"\nTraining Fold {fold} (Note: KNN can be slow on large datasets)...")
    
    # Split
    train_df = train[train['fold'] != fold]
    val_df = train[train['fold'] == fold]
    
    X_tr, y_tr = train_df.drop([ID, TARGET, 'fold'], axis=1), train_df[TARGET]
    X_val, y_val = val_df.drop([ID, TARGET, 'fold'], axis=1), val_df[TARGET]
    
    # Initialize KNN
    # n_neighbors=50 provides a smoother decision boundary for large data
    model = KNeighborsClassifier(
        n_neighbors=50, 
        weights='distance', 
        metric='minkowski', 
        p=2, 
        n_jobs=-1
    )
    
    # Build full Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit
    clf.fit(X_tr, y_tr)
    
    # Predict OOF
    batch_oof = clf.predict_proba(X_val)[:, 1]
    oof_preds[train['fold'] == fold] = batch_oof
    
    # Track performance
    score = roc_auc_score(y_val, batch_oof)
    fold_scores.append(score)
    print(f"Fold {fold} AUC: {score:.5f}")
    
    # Predict Test
    test_preds += clf.predict_proba(X_test)[:, 1] / N_SPLITS

# =========================================================================================
# 5. SAVE RESULTS
# =========================================================================================
overall_cv = roc_auc_score(train[TARGET], oof_preds)
print(f"\n{'='*30}\nKNN OVERALL CV AUC: {overall_cv:.5f}\n{'='*30}")

# Save OOF and Test predictions for the Blender
np.save(f'{OOF_DIR}/{MODEL_NAME}_oof.npy', oof_preds)
np.save(f'{OOF_DIR}/{MODEL_NAME}_test.npy', test_preds)

# Save standard submission file
submission = sample_sub.copy()
submission[TARGET] = test_preds
sub_filename = f'{SUB_DIR}/submission_{MODEL_NAME}_CV{overall_cv:.4f}.csv'
submission.to_csv(sub_filename, index=False)

print(f"✅ Saved KNN OOFs and submission.")