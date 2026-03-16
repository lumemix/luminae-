import os
import argparse
import joblib
import numpy as np
import pandas as pd
import sys
import io
import yaml

# --- SKLEARN & METRICS ---
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN

# --- MODELS ---
from lightgbm import LGBMClassifier
import lightgbm as lgbm 
from xgboost import XGBClassifier

# --- PLOTTING ---
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- GLOBAL CONFIGURATION ---
CSV_PATH = config["hematologynet_train_csv"]
MODELS_DIR = config["hematologynet_models_dir"]

# -------------------
# Bucket mapping (multi-label)
# -------------------
BUCKETS = {
    0: ["Acute_MI", "Ischemic_HD"],
    1: ["Arrhythmia"],
    2: ["Heart_Failure", "Cardiomyopathy"],
    3: ["Cerebrovascular"],
    4: ["Arterial_Embolism", "Peripheral_Vascular", "Pulmonary_Embolism", "Venous_Thrombo"],
    5: ["Valvular_HD", "Endocarditis", "Pericardial"],
    6: ["Congenital_HD", "Aortic_Disease", "Pulmonary_Heart"],
    7: ["Other_CVD", "Other_Vascular", "Hypertension"]
}

# Features 
DEMO_COLS = ['age', 'gender', 'race']
DRUG_COLS = ["ACE_Inhibitors", "Beta_Blockers", "Statins", "Anticoagulants", "Diuretics", "Calcium_Channel_Blockers"]
LEAKY_COLS = [
    "subject_id","hadm_id","admittime","dischtime",
    "drg_codes","drg_severity_raw","drg_mortality_raw",
    "hospital_expire_flag","label",
    "Acute_MI","Ischemic_HD","Arrhythmia","Heart_Failure","Cardiomyopathy",
    "Cerebrovascular","Arterial_Embolism","Peripheral_Vascular",
    "Pulmonary_Embolism","Venous_Thrombo","Valvular_HD","Endocarditis",
    "Pericardial","Congenital_HD","Aortic_Disease","Pulmonary_Heart",
    "Other_CVD","Other_Vascular","Hypertension"
]

# -------------------
# 1. SOTA Feature Engineering
# -------------------
def add_sota_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SOTA clinical, demographic, and drug features."""
    print("Applying SOTA Feature Engineering...")
    X = df.copy()

    # --- 1. Demographic Engineering ---
    if 'age' in X.columns:
        bins = [0, 45, 65, 75, 85, 120]
        labels = ['<45', '45-64', '65-74', '75-84', '85+']
        X['Age_Group'] = pd.cut(X['age'], bins=bins, labels=labels, right=False, ordered=True)
    
    if 'race' in X.columns:
        non_white_races = ['BLACK/AFRICAN AMERICAN', 'ASIAN', 'HISPANIC/LATINO', 'OTHER', 'MULTI RACE ETHNICITY']
        X['Non_White_Race'] = X['race'].astype(str).str.upper().apply(lambda x: 1 if any(r in x for r in non_white_races) else 0).astype('int32')

    # --- 2. Drug & Hospital Engineering ---
    drug_cols_available = [c for c in DRUG_COLS if c in X.columns]
    if drug_cols_available:
        X['Polypharmacy_Score'] = X[drug_cols_available].sum(axis=1).astype('int32')
        X['High_Risk_Polypharmacy'] = (X['Polypharmacy_Score'] >= 4).astype('int32')
    
    if 'Hospital_Mortality' in X.columns:
        X['Hospital_Mortality'] = X['Hospital_Mortality'].fillna(0).astype('int32')

    # --- 3. LAB & VITAL RATIOS ---
    if all(c in X.columns for c in ['BUN_mean', 'Creatinine_mean']):
        X['BUN_Creatinine_Ratio'] = (X['BUN_mean'] / X['Creatinine_mean']).replace([np.inf, -np.inf], 0).astype('float32')
        
    if all(c in X.columns for c in ['Potassium_mean', 'Sodium_mean']):
        X['Potassium_Sodium_Ratio'] = (X['Potassium_mean'] / X['Sodium_mean']).replace([np.inf, -np.inf], 0).astype('float32')

    if all(c in X.columns for c in ['HeartRate_mean', 'SystolicBP_mean']):
        X['Shock_Index'] = (X['HeartRate_mean'] / X['SystolicBP_mean']).replace([np.inf, -np.inf], 0).astype('float32')
        
    if all(c in X.columns for c in ['Temperature_mean', 'WBC_mean']):
        X['Infection_Severity_Proxy'] = (X['Temperature_mean'] * X['WBC_mean']).replace([np.inf, -np.inf], 0).astype('float32')

    # Imputation preparation
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    
    for col in object_cols:
        X[col] = X[col].astype("category").cat.add_categories('Missing').fillna('Missing')
    
    for col in cat_cols:
         X[col] = X[col].cat.add_categories('Missing').fillna('Missing')

    return X

# -------------------
# Multi-label target construction
# -------------------
def build_multilabel_targets(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.DataFrame(0, index=df.index, columns=list(BUCKETS.keys()), dtype=np.int32)
    for bucket_id, cols in BUCKETS.items():
        # Check if columns exist to avoid KeyError
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            mask = (df[valid_cols].sum(axis=1) > 0)
            y.loc[mask, bucket_id] = 1
    return y

# -------------------
# Data loader
# -------------------
def load_dataset(csv_path, sample_rows=None, return_full=False):
    nrows = None if (sample_rows is None or sample_rows == 0) else sample_rows
    print(f"Reading CSV (rows={'ALL' if nrows is None else nrows})...")

    try:
        df = pd.read_csv(csv_path, nrows=nrows)
    except UnicodeDecodeError:
        print("Standard UTF-8 decode failed. Trying 'latin-1'...")
        df = pd.read_csv(csv_path, nrows=nrows, encoding='latin-1')

    # Build multi-label targets
    Y = build_multilabel_targets(df)

    # Features: drop diagnosis flags + leaky cols
    diag_cols = [c for cols in BUCKETS.values() for c in cols]
    drop_cols = [c for c in diag_cols + LEAKY_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')

    # Apply SOTA Feature Engineering
    X = add_sota_features(X)

    # Encode categorical to integer codes
    cat_cols_to_encode = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_enc = X.copy()
    for col in cat_cols_to_encode:
        if X_enc[col].dtype.name != 'category':
            X_enc[col] = X_enc[col].astype("category")
        X_enc[col] = X_enc[col].cat.codes.astype("int32")

    # Downcast numerics + Robust Median Imputation
    for col in X_enc.columns:
        if X_enc[col].dtype in ['float64', 'float32']:
            X_enc[col] = X_enc[col].astype("float32")
            median_val = X_enc[col].median()
            # FIX: Handle cases where column is all NaN
            if pd.isna(median_val):
                X_enc[col] = X_enc[col].fillna(0)
            else:
                X_enc[col] = X_enc[col].fillna(median_val)
                
        elif X_enc[col].dtype in ['int64', 'int32']:
            X_enc[col] = X_enc[col].astype("int32").fillna(-1)

    print("Loaded shape (Final Feature Set):", X_enc.shape)

    if return_full:
        # Correctly returns 3 values
        return X_enc, Y, list(X_enc.columns)
    else:
        # Returns 5 values
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_enc, Y, test_size=0.2, random_state=42
        )
        return X_train, Y_train, X_val, Y_val, list(X_enc.columns)
    
# -------------------
# 2. Targeted resampling helper
# -------------------
def oversample_per_label(X_train, y_train_label, bucket_id: int, strategy: str, weight: float = 1.0):
    pos = int(y_train_label.sum())
    
    # Safety check: Cannot SMOTE if fewer than 6 samples (k_neighbors=5 default)
    if strategy is None or pos < 6:
        return X_train, y_train_label

    try:
        # Step 1: SMOTE
        smote_sampler = SMOTE(random_state=42, sampling_strategy=min(1.0, weight), k_neighbors=5)
        X_res, y_res = smote_sampler.fit_resample(X_train, y_train_label)
        
        if strategy == "hybrid":
            try:
                adasyn_sampler = ADASYN(random_state=42, sampling_strategy=min(1.0, weight), n_neighbors=5)
                X_res, y_res = adasyn_sampler.fit_resample(X_res, y_res)
            except ValueError:
                pass # Fallback to SMOTE result if ADASYN fails
        
        return X_res, y_res

    except ValueError as e:
        print(f"[Bucket {bucket_id}] Oversampling warning: {e}")
        return X_train, y_train_label

# -------------------
# 3. One-vs-rest LightGBM (Statistically Robust)
# -------------------
def train_ovr_lightgbm(X_train, Y_train, X_val, Y_val, resample_strategy=None):
    models = {}
    preds_prob = pd.DataFrame(index=Y_val.index, columns=Y_val.columns, dtype=np.float32)
    
    params = {
        'n_estimators': 3000,           
        'learning_rate': 0.03,          
        'num_leaves': 64,              
        'max_depth': 8,                
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    dummy_stderr = io.StringIO()

    for bucket_id in Y_train.columns:
        # Resample logic
        X_tr, y_tr = oversample_per_label(
            X_train, Y_train[bucket_id].values, int(bucket_id), resample_strategy, 1.0
        )
        
        if y_tr.sum() == 0:
            preds_prob[bucket_id] = 0.0
            continue

        # Split for calibration
        X_a, X_b, y_a, y_b = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

        # LightGBM handles imbalance natively better than manual SMOTE in high dimensions
        # 'is_unbalance': True automatically calculates weights based on imbalance
        base = LGBMClassifier(
            objective="binary",
            is_unbalance=True, 
            **params
        )
        
        # Capture C++ output
        original_stderr = sys.stderr
        sys.stderr = dummy_stderr 
        
        try:
            base.fit(X_a, y_a, eval_set=[(X_b, y_b)], callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)])
        except Exception:
            # Fallback if callbacks fail
            base.fit(X_a, y_a)
        finally:
            sys.stderr = original_stderr
        
        # Calibrate (Isotonic is robust for large samples, Sigmoid safer for small)
        method = "isotonic" if len(y_b) > 1000 else "sigmoid"
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
        cal.fit(X_b, y_b)

        models[bucket_id] = cal
        preds_prob[bucket_id] = cal.predict_proba(X_val)[:, 1]
        
    return models, preds_prob

# -------------------
# One-vs-rest XGBoost (Statistically Robust + Weighted)
# -------------------
def train_ovr_xgboost_sota(X_train, Y_train, X_val, Y_val):
    models = {}
    preds_prob = pd.DataFrame(index=Y_val.index, columns=Y_val.columns, dtype=np.float32)

    params = {
        'n_estimators': 3000,           
        'learning_rate': 0.05,          
        'max_depth': 6,                
        'subsample': 0.8,              
        'colsample_bytree': 0.8,        
        'objective': "binary:logistic",
        'n_jobs': -1,
        'random_state': 42,
        'eval_metric': 'logloss' # Explicitly set to avoid warnings
    }
    
    for bucket_id in Y_train.columns:
        y_tr = Y_train[bucket_id].values
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        
        if pos_count == 0:
            preds_prob[bucket_id] = 0.0
            continue
            
        # DYNAMIC WEIGHTING: Automatically fights imbalance for Class 0 vs Class 5
        scale_pos_weight = neg_count / pos_count
        print(f"[Bucket {bucket_id}] N={len(y_tr)} | Pos={pos_count} | Weight={scale_pos_weight:.2f}")
        
        # Split for calibration
        X_a, X_b, y_a, y_b = train_test_split(X_train, y_tr, test_size=0.2, random_state=42)

        base = XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
        
        # No early stopping to avoid version conflicts, training full capacity
        base.fit(X_a, y_a, verbose=False)
        
        method = "isotonic" if len(y_b) > 1000 else "sigmoid"
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
        cal.fit(X_b, y_b)

        models[bucket_id] = cal
        preds_prob[bucket_id] = cal.predict_proba(X_val)[:, 1]

    return models, preds_prob

# -------------------
# Evaluation Utils
# -------------------
def ensemble_probs(probs_list):
    return sum(probs_list) / float(len(probs_list))

def save_models(models: dict, mode: str):
    out_dir = os.path.join(MODELS_DIR, f"{mode}_multilabel_models")
    os.makedirs(out_dir, exist_ok=True)
    for bucket_id, model in models.items():
        joblib.dump(model, os.path.join(out_dir, f"{mode}_bucket_{bucket_id}.pkl"))

def optimize_thresholds(preds_prob: pd.DataFrame, Y_val: pd.DataFrame, grid=None):
    if grid is None: grid = np.linspace(0.1, 0.6, 11)
    best_t = {}
    for b in Y_val.columns:
        y_true = Y_val[b].values; p = preds_prob[b].values
        best_f1, best = -1.0, 0.5
        for t in grid:
            y_pred = (p >= t).astype(np.int32)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1: best_f1, best = f1, t
        best_t[b] = best
    return best_t

def apply_thresholds(preds_prob: pd.DataFrame, thresholds: dict):
    P = np.zeros_like(preds_prob.values, dtype=np.int32)
    for i, b in enumerate(preds_prob.columns):
        P[:, i] = (preds_prob[b].values >= thresholds[b]).astype(np.int32)
    return P

def evaluate_final(Y_val, preds, probs, X_val, model_dict, feature_names):
    print("\n--- Final Classification Report ---")
    print(classification_report(Y_val.values, preds, digits=3, zero_division=0))
    
    # Prevalence Visualization
    real_counts = Y_val.sum(axis=0)
    pred_counts = pd.Series(preds.sum(axis=0), index=Y_val.columns)
    
    print("\nPrevalence Comparison:")
    print(pd.DataFrame({'Real': real_counts, 'Predicted': pred_counts}))

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="xgboost_sota",
                        choices=["lightgbm", "xgboost", "xgboost_sota", "ensemble"])
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--resample-strategy", type=str, default=None,
                        choices=[None, "smote", "adasyn", "hybrid"])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    # --- FIX 1: Set return_full=True to receive 3 values ---
    X, Y, feature_names = load_dataset(CSV_PATH, sample_rows=args.sample, return_full=True)
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    # Create stratification label (argmax is imperfect for multilabel but sufficient for splitting)
    y_strat = np.argmax(Y.values, axis=1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_strat)):
        print(f"\n=== Fold {fold+1}/{args.folds} ===")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        if args.mode == "lightgbm":
            models, probs = train_ovr_lightgbm(X_train, Y_train, X_val, Y_val, args.resample_strategy)
        elif args.mode in ["xgboost", "xgboost_sota"]:
            models, probs = train_ovr_xgboost_sota(X_train, Y_train, X_val, Y_val)
        elif args.mode == "ensemble":
            # Hybrid Ensemble: Combines Balanced LightGBM with Weighted XGBoost
            print("Training Ensemble Part 1: LightGBM...")
            models_lgb, probs_lgb = train_ovr_lightgbm(X_train, Y_train, X_val, Y_val)
            print("Training Ensemble Part 2: XGBoost...")
            models_xgb, probs_xgb = train_ovr_xgboost_sota(X_train, Y_train, X_val, Y_val)
            probs = ensemble_probs([probs_lgb, probs_xgb])
            models = {**models_lgb} # Saving one set for reference
        
        # Optimize Thresholds
        th = optimize_thresholds(probs, Y_val)
        preds = apply_thresholds(probs, th)

        evaluate_final(Y_val, preds, probs, X_val, models, feature_names)