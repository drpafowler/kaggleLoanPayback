import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

# =========================================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================================
MODEL_NAME = "mlp_v1"
DATA_DIR = 'data'
OOF_DIR = 'oof'
SUB_DIR = 'submissions'

for folder in [DATA_DIR, OOF_DIR, SUB_DIR]:
    os.makedirs(folder, exist_ok=True)

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
if os.path.exists(fold_path):
    train = train.merge(pd.read_csv(fold_path), on=ID)
else:
    print("Error: Run baseline script first to generate fold_ids.csv")

# =========================================================================================
# 3. NN PREPROCESSING (Scaling & Label Encoding)
# =========================================================================================
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# 3a. Scale Numerical Data
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols].fillna(0))
test[num_cols] = scaler.transform(test[num_cols].fillna(0))

# 3b. Label Encode Categoricals for Embeddings
cat_dims = []
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    cat_dims.append(len(le.classes_))

# =========================================================================================
# 4. MODEL ARCHITECTURE (MLP with Embeddings)
# =========================================================================================
def build_model(num_features, cat_dims):
    # Input for numerical features
    num_input = layers.Input(shape=(num_features,), name='num_input')
    
    # Inputs and Embeddings for categorical features
    cat_inputs = []
    cat_embs = []
    for dim in cat_dims:
        inp = layers.Input(shape=(1,))
        # Embedding size: min(50, (dim+1)//2) is a good rule of thumb
        emb = layers.Embedding(dim, min(50, (dim + 1) // 2))(inp)
        emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)
        cat_inputs.append(inp)
        cat_embs.append(emb)
    
    # Combine everything
    x = layers.Concatenate()([num_input] + cat_embs)
    
    # Deep Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=[num_input] + cat_inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    return model

# =========================================================================================
# 5. CROSS-VALIDATION LOOP
# =========================================================================================
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# Prepare Test data for prediction
test_inputs = [test[num_cols].values] + [test[col].values for col in cat_cols]

for fold in range(N_SPLITS):
    print(f"\n--- Training Fold {fold} ---")
    
    train_idx = train[train['fold'] != fold].index
    val_idx = train[train['fold'] == fold].index
    
    # Format inputs for NN: [num_array, cat1_array, cat2_array...]
    X_tr = [train.loc[train_idx, num_cols].values] + [train.loc[train_idx, col].values for col in cat_cols]
    X_val = [train.loc[val_idx, num_cols].values] + [train.loc[val_idx, col].values for col in cat_cols]
    y_tr, y_val = train.loc[train_idx, TARGET].values, train.loc[val_idx, TARGET].values
    
    model = build_model(len(num_cols), cat_dims)
    
    # Train
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), 
              epochs=20, batch_size=1024, verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    
    # Predict
    batch_oof = model.predict(X_val).flatten()
    oof_preds[val_idx] = batch_oof
    
    fold_auc = roc_auc_score(y_val, batch_oof)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")
    
    test_preds += model.predict(test_inputs).flatten() / N_SPLITS

# =========================================================================================
# 6. SAVE OUTPUTS
# =========================================================================================
overall_auc = roc_auc_score(train[TARGET], oof_preds)
print(f"\nOverall Neural Network CV AUC: {overall_auc:.5f}")

np.save(f'{OOF_DIR}/{MODEL_NAME}_oof.npy', oof_preds)
np.save(f'{OOF_DIR}/{MODEL_NAME}_test.npy', test_preds)

submission = sample_sub.copy()
submission[TARGET] = test_preds
submission.to_csv(f'{SUB_DIR}/submission_{MODEL_NAME}.csv', index=False)
print("✅ Saved Deep MLP OOFs and submission.")