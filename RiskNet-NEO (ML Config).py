import os
import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CSV_PATH   = config["risknet_train_csv"]
MODELS_DIR = config["risknet_models_dir"]

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

# Columns to drop to prevent leakage (keep clinical features only)
LEAKY_COLS = ["subject_id", "hadm_id", "hospital_expire_flag"]

# -------------------
# Multi-label target construction
# -------------------
def build_multilabel_targets(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.DataFrame(0, index=df.index, columns=list(BUCKETS.keys()), dtype=np.int32)
    for bucket_id, cols in BUCKETS.items():
        mask = (df[cols].sum(axis=1) > 0)
        y.loc[mask, bucket_id] = 1
    return y

# -------------------
# Data loader
# -------------------
def load_dataset(csv_path, sample_rows=None):
    nrows = None if (sample_rows is None or sample_rows == 0) else sample_rows
    print(f"Reading CSV (rows={'ALL' if nrows is None else nrows})...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print("Loaded shape:", df.shape)

    # Build multi-label targets
    Y = build_multilabel_targets(df)

    # Features: drop diagnosis flags + leaky cols
    diag_cols = [c for cols in BUCKETS.values() for c in cols]
    drop_cols = [c for c in diag_cols + LEAKY_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Cast categoricals properly
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")

    # Downcast numerics to save memory
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("int32")

    # Encode categorical to integer codes for model compatibility
    X_enc = X.copy()
    for col in cat_cols:
        X_enc[col] = X_enc[col].cat.codes.astype("int32")

    # Train/validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_enc, Y, test_size=0.2, random_state=42
    )

    print("Label prevalence (train):")
    print(Y_train.sum(axis=0).sort_index())
    print("Label prevalence (val):")
    print(Y_val.sum(axis=0).sort_index())

    return X_train, Y_train, X_val, Y_val, cat_cols

# -------------------
# Targeted resampling helper per label
# -------------------
def oversample_per_label(X_train, y_train_label, strategy: str, weight: float = 1.0):
    if strategy is None:
        return X_train, y_train_label

    try:
        if strategy == "smote":
            sampler = SMOTE(random_state=42, sampling_strategy=min(1.0, weight))
            return sampler.fit_resample(X_train, y_train_label)

        if strategy == "adasyn":
            sampler = ADASYN(random_state=42, sampling_strategy=min(1.0, weight))
            return sampler.fit_resample(X_train, y_train_label)

        if strategy == "hybrid":
            X_res, y_res = SMOTE(random_state=42, sampling_strategy=min(1.0, weight)).fit_resample(X_train, y_train_label)
            # Try ADASYN, but fall back to SMOTE if it fails
            try:
                X_res, y_res = ADASYN(random_state=42, sampling_strategy=min(1.0, weight)).fit_resample(X_res, y_res)
            except ValueError as e:
                print(f"ADASYN failed for this bucket (falling back to SMOTE): {e}")
            return X_res, y_res

    except ValueError as e:
        # If oversampler fails (e.g. ratio issue), just return original data
        print(f"Oversampling skipped due to error: {e}")
        return X_train, y_train_label

    return X_train, y_train_label

# -------------------
# One-vs-rest multi-label training (LightGBM + probability calibration)
# -------------------
def train_ovr_lightgbm(X_train, Y_train, X_val, Y_val, resample_strategy=None, resample_weights=None):
    if resample_weights is None:
        # Default: oversample more aggressively for rare buckets (4,5,6)
        resample_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.5, 5: 1.8, 6: 1.5, 7: 0.8}

    models = {}
    preds_prob = pd.DataFrame(index=Y_val.index, columns=Y_val.columns, dtype=np.float32)

    for bucket_id in Y_train.columns:
        pos = int(Y_train[bucket_id].sum())
        print(f"[LightGBM] Bucket {bucket_id} (positives={pos})")
        X_tr, y_tr = oversample_per_label(X_train, Y_train[bucket_id].values, resample_strategy, resample_weights.get(bucket_id, 1.0))

        # Split for calibration
        X_a, X_b, y_a, y_b = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

        base = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            class_weight=None,
            objective="binary",
            n_jobs=-1
        )
        base.fit(X_a, y_a)

        cal = CalibratedClassifierCV(base, method="isotonic")
        cal.fit(X_b, y_b)

        models[bucket_id] = cal
        preds_prob[bucket_id] = cal.predict_proba(X_val)[:, 1]

    return models, preds_prob

# -------------------
# One-vs-rest multi-label training (XGBoost + probability calibration)
# -------------------
def train_ovr_xgboost(X_train, Y_train, X_val, Y_val, resample_strategy=None, resample_weights=None):
    if resample_weights is None:
        resample_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.5, 5: 1.8, 6: 1.5, 7: 0.8}

    models = {}
    preds_prob = pd.DataFrame(index=Y_val.index, columns=Y_val.columns, dtype=np.float32)

    for bucket_id in Y_train.columns:
        pos = int(Y_train[bucket_id].sum())
        print(f"[XGBoost] Bucket {bucket_id} (positives={pos})")
        X_tr, y_tr = oversample_per_label(X_train, Y_train[bucket_id].values, resample_strategy, resample_weights.get(bucket_id, 1.0))

        # Split for calibration
        X_a, X_b, y_a, y_b = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

        base = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            n_jobs=-1,
            reg_lambda=1.0
        )
        base.fit(X_a, y_a, eval_metric="logloss", verbose=False)

        cal = CalibratedClassifierCV(base, method="isotonic")
        cal.fit(X_b, y_b)

        models[bucket_id] = cal
        preds_prob[bucket_id] = cal.predict_proba(X_val)[:, 1]

    return models, preds_prob

# -------------------
# Ensembling (average calibrated probabilities)
# -------------------
def ensemble_probs(probs_list):
    # Average a list of DataFrames with identical columns/index
    avg = sum(probs_list) / float(len(probs_list))
    return avg

# -------------------
# Save models
# -------------------
def save_models(models: dict, mode: str):
    out_dir = os.path.join(MODELS_DIR, f"{mode}_multilabel_models")
    os.makedirs(out_dir, exist_ok=True)
    for bucket_id, model in models.items():
        joblib.dump(model, os.path.join(out_dir, f"{mode}_bucket_{bucket_id}.pkl"))
    print(f"Saved {len(models)} models to {out_dir}")

# -------------------
# Threshold grid evaluation
# -------------------
def evaluate_thresholds_grid(preds_prob, Y_val, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    for t in thresholds:
        preds = (preds_prob.values >= t).astype(np.int32)
        macro_f1 = f1_score(Y_val.values, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(Y_val.values, preds, average="micro", zero_division=0)
        print(f"Global threshold {t:.2f} → Macro F1: {macro_f1:.3f} | Micro F1: {micro_f1:.3f}")

# -------------------
# Multi-label Top-k accuracy (strict)
# -------------------
def topk_accuracy_multi_label(y_true, y_prob, k=1):
    correct = 0
    for i in range(y_true.shape[0]):
        topk_idx = np.argsort(y_prob[i])[-k:]
        if y_true[i, topk_idx].sum() > 0:
            correct += 1
    return correct / y_true.shape[0]

# -------------------
# Inflated Top-k accuracy (argmax simplification)
# -------------------
def inflated_topk(y_true, y_prob, k=1):
    y_true_single = y_true.argmax(axis=1)
    correct = 0
    for i in range(y_true.shape[0]):
        topk_idx = np.argsort(y_prob[i])[-k:]
        if y_true_single[i] in topk_idx:
            correct += 1
    return correct / y_true.shape[0]

# -------------------
# Final evaluation with visuals and extra metrics
# -------------------
def evaluate_final(Y_val, preds, probs, X_val, model_dict, feature_names, example_bucket=0):
    # Macro/Micro F1
    macro_f1 = f1_score(Y_val.values, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_val.values, preds, average="micro", zero_division=0)
    print(f"Final — Macro F1: {macro_f1:.3f} | Micro F1: {micro_f1:.3f}")
    print(classification_report(Y_val.values, preds, digits=3, zero_division=0))

    # Strict multi-label Top-1 and Top-5
    top1_acc = topk_accuracy_multi_label(Y_val.values, probs.values, k=1)
    top5_acc = topk_accuracy_multi_label(Y_val.values, probs.values, k=5)
    print(f"Strict Multi-label Top-1 Accuracy: {top1_acc:.3f}")
    print(f"Strict Multi-label Top-5 Accuracy: {top5_acc:.3f}")

    # Inflated single-label Top-1 and Top-5
    infl_top1 = inflated_topk(Y_val.values, probs.values, k=1)
    infl_top5 = inflated_topk(Y_val.values, probs.values, k=5)
    print(f"Inflated (argmax) Top-1 Accuracy: {infl_top1:.3f}")
    print(f"Inflated (argmax) Top-5 Accuracy: {infl_top5:.3f}")

    # ROC-AUC (macro average across classes)
    try:
        roc_auc = roc_auc_score(Y_val.values, probs.values, average="macro")
        print(f"Macro ROC-AUC: {roc_auc:.3f}")
    except Exception as e:
        print(f"ROC-AUC could not be computed: {e}")

    # Brier score (mean across classes)
    try:
        brier_scores = []
        for i, col in enumerate(Y_val.columns):
            brier = brier_score_loss(Y_val.iloc[:, i], probs.values[:, i])
            brier_scores.append(brier)
        mean_brier = np.mean(brier_scores)
        print(f"Mean Brier Score: {mean_brier:.5f}")
    except Exception as e:
        print(f"Brier score could not be computed: {e}")

    # Prediction vs Real prevalence
    real_counts = Y_val.sum(axis=0)
    pred_counts = pd.Series(preds.sum(axis=0), index=Y_val.columns)

    plt.figure(figsize=(8,5))
    plt.bar(real_counts.index.astype(str), real_counts.values, alpha=0.6, label="Real")
    plt.bar(pred_counts.index.astype(str), pred_counts.values, alpha=0.6, label="Predicted")
    plt.title("Prediction vs Real Prevalence per Bucket")
    plt.xlabel("Bucket")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    # Unified confusion matrix (multi-label → single-label via argmax)
    y_true_single = np.argmax(Y_val.values, axis=1)
    y_pred_single = np.argmax(probs.values, axis=1)

    cm = confusion_matrix(y_true_single, y_pred_single)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=Y_val.columns, yticklabels=Y_val.columns)
    plt.title("Unified Confusion Matrix Across Buckets")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # SHAP feature importance for one bucket
    clf = model_dict[example_bucket]
    base_estimator = clf.base_estimator if hasattr(clf, "base_estimator") else clf
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X_val[feature_names])
    shap.summary_plot(shap_values, X_val[feature_names], feature_names=feature_names)

# -------------------
# Per-class threshold optimization
# -------------------
def optimize_thresholds(preds_prob: pd.DataFrame, Y_val: pd.DataFrame, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.6, 11)

    best_t = {}
    for b in Y_val.columns:
        y_true = Y_val[b].values
        p = preds_prob[b].values
        best_f1, best = -1.0, 0.5
        for t in grid:
            y_pred = (p >= t).astype(np.int32)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, t
        best_t[b] = best
        print(f"Bucket {b}: best threshold {best:.2f}, F1 {best_f1:.3f}")
    return best_t

# -------------------
# Apply thresholds
# -------------------
def apply_thresholds(preds_prob: pd.DataFrame, thresholds: dict):
    P = np.zeros_like(preds_prob.values, dtype=np.int32)
    for i, b in enumerate(preds_prob.columns):
        P[:, i] = (preds_prob[b].values >= thresholds[b]).astype(np.int32)
    return P

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "ensemble"])
    parser.add_argument("--sample", type=int, default=50000, help="Number of rows to sample (0 = full dataset)")
    parser.add_argument("--resample-strategy", type=str, default="hybrid",
                        choices=[None, "smote", "adasyn", "hybrid"],
                        help="Apply oversampling strategy per label")
    parser.add_argument("--threshold-grid-min", type=float, default=0.1)
    parser.add_argument("--threshold-grid-max", type=float, default=0.6)
    parser.add_argument("--threshold-grid-steps", type=int, default=11)
    args = parser.parse_args()

    X_train, Y_train, X_val, Y_val, cat_cols = load_dataset(CSV_PATH, sample_rows=args.sample)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.mode == "lightgbm":
        models_lgb, probs_lgb = train_ovr_lightgbm(X_train, Y_train, X_val, Y_val,
                                                   resample_strategy=args.resample_strategy)
        save_models(models_lgb, "lightgbm")
        evaluate_thresholds_grid(probs_lgb, Y_val)
        grid = np.linspace(args.threshold_grid_min, args.threshold_grid_max, args.threshold_grid_steps)
        th_lgb = optimize_thresholds(probs_lgb, Y_val, grid=grid)
        preds_lgb = apply_thresholds(probs_lgb, th_lgb)
        evaluate_final(Y_val, preds_lgb, probs_lgb, X_val, models_lgb, list(X_train.columns), example_bucket=0)

    elif args.mode == "xgboost":
        # The script was incomplete here; we complete it similarly.
        models_xgb, probs_xgb = train_ovr_xgboost(X_train, Y_train, X_val, Y_val, resample_strategy=args.resample_strategy)
        save_models(models_xgb, "xgboost")
        evaluate_thresholds_grid(probs_xgb, Y_val)
        grid = np.linspace(args.threshold_grid_min, args.threshold_grid_max, args.threshold_grid_steps)
        th_xgb = optimize_thresholds(probs_xgb, Y_val, grid=grid)
        preds_xgb = apply_thresholds(probs_xgb, th_xgb)
        evaluate_final(Y_val, preds_xgb, probs_xgb, X_val, models_xgb, list(X_train.columns), example_bucket=0)

    elif args.mode == "ensemble":
        # Train both and average probabilities
        models_lgb, probs_lgb = train_ovr_lightgbm(X_train, Y_train, X_val, Y_val, resample_strategy=args.resample_strategy)
        models_xgb, probs_xgb = train_ovr_xgboost(X_train, Y_train, X_val, Y_val, resample_strategy=args.resample_strategy)
        probs_ens = ensemble_probs([probs_lgb, probs_xgb])
        # For saving, maybe just keep one set as representative (LGB)
        save_models(models_lgb, "ensemble_lgb")
        evaluate_thresholds_grid(probs_ens, Y_val)
        grid = np.linspace(args.threshold_grid_min, args.threshold_grid_max, args.threshold_grid_steps)
        th_ens = optimize_thresholds(probs_ens, Y_val, grid=grid)
        preds_ens = apply_thresholds(probs_ens, th_ens)
        evaluate_final(Y_val, preds_ens, probs_ens, X_val, models_lgb, list(X_train.columns), example_bucket=0)