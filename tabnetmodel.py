import os
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
MODEL_NAME = "tabnet_v1"
DATA_DIR = 'data'
OOF_DIR = 'oof'
SUB_DIR = 'submissions'

os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = 'loan_paid_back'
ID = 'id'
SEED = 42
N_SPLITS = 5

# =========================================================================================
# 2. LOAD DATA & FOLDS
# =========================================================================================
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

fold_path = f'{DATA_DIR}/fold_ids.csv'
train = train.merge(pd.read_csv(fold_path), on=ID)

# =========================================================================================
# 3. TABNET PREPROCESSING
# =========================================================================================
# TabNet requires categorical features to be label encoded
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
cat_idxs = []
cat_dims = []

for i, col in enumerate(train.drop([ID, TARGET, 'fold'], axis=1).columns):
    if col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        cat_idxs.append(i)
        cat_dims.append(len(le.classes_))

X = train.drop([ID, TARGET, 'fold'], axis=1).values
y = train[TARGET].values
X_test = test.drop([ID], axis=1).values

# =========================================================================================
# 4. CROSS-VALIDATION LOOP
# =========================================================================================
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold in range(N_SPLITS):
    print(f"\n--- Training TabNet Fold {fold} ---")
    
    X_tr, y_tr = X[train['fold'] != fold], y[train['fold'] != fold]
    X_val, y_val = X[train['fold'] == fold], y[train['fold'] == fold]
    
    clf = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        optimizer_params=dict(lr=2e-2),
        # CHANGE THESE TWO LINES:
        scheduler_fn=torch.optim.lr_scheduler.StepLR, 
        scheduler_params={"step_size": 50, "gamma": 0.9},
        seed=SEED
    )
    
    clf.fit(
        X_train=X_tr, y_train=y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=100, patience=10,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0, drop_last=False
    )
    
    batch_oof = clf.predict_proba(X_val)[:, 1]
    oof_preds[train['fold'] == fold] = batch_oof
    test_preds += clf.predict_proba(X_test)[:, 1] / N_SPLITS

# =========================================================================================
# 5. SAVE RESULTS
# =========================================================================================
overall_auc = roc_auc_score(y, oof_preds)
print(f"TabNet Overall CV AUC: {overall_auc:.5f}")

np.save(f'{OOF_DIR}/{MODEL_NAME}_oof.npy', oof_preds)
np.save(f'{OOF_DIR}/{MODEL_NAME}_test.npy', test_preds)

submission = sample_sub.copy()
submission[TARGET] = test_preds
submission.to_csv(f'{SUB_DIR}/submission_{MODEL_NAME}.csv', index=False)