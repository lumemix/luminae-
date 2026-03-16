# -*- coding: utf-8 -*-
"""
ProgressionNet Forecasting Pipeline (HOSP-only, Tabular EHR, no neural nets)
- Events × Horizons: MI/HF/Deterioration/Discharge/Death over 24h, 48h, 7d, 30d
- Features: Manuscript thresholds (labs/vitals), abnormal burden, kinetics, clinical ratios
- Models: LightGBM binary heads + isotonic calibration
- Artifacts: models, calibrators, thresholds, feature dictionary, imputation report, calibration plots
"""

import os, io, sys, json, argparse, joblib
import numpy as np
import pandas as pd
from datetime import timedelta
import yaml

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
import lightgbm as lgbm

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------- Config ----------
CSV_PATH = config["progressionnet_train_csv"]
MODELS_DIR = config["progressionnet_models_dir"]

PATIENT_COL = "subject_id"
TIME_COL = "admittime"  # parseable to datetime, monotone per patient

HORIZONS = {
    "24h": timedelta(hours=24),
    "48h": timedelta(hours=48),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}

EVENTS = ["Acute_MI","Heart_Failure","Deterioration","Discharge","Death"]  # HOSP-only

# Base features
DEMO_COLS = ['age', 'gender', 'race']
DRUG_COLS = ["ACE_Inhibitors","Beta_Blockers","Statins","Anticoagulants","Diuretics","Calcium_Channel_Blockers"]
LAB_MEAN_COLS = [
    'Hemoglobin_mean','Hematocrit_mean','WBC_mean','Platelet Count_mean','INR_mean','PT_mean','aPTT_mean',
    'Sodium_mean','Potassium_mean','Chloride_mean','Calcium_mean','Magnesium_mean',
    'Creatinine_mean','BUN_mean','Troponin_mean','CK-MB_mean','BNP_mean','NT-proBNP_mean',
    'Glucose_mean','Lactate_mean','CRP_mean'
]
VITAL_MEAN_COLS = ['HeartRate_mean','RespiratoryRate_mean','SpO2_mean','Temperature_mean',
                   'SystolicBP_mean','DiastolicBP_mean','MeanBP_mean']

# Drop known leaky fields
LEAKY_COLS_BASE = [
    "dischtime", "hadm_id",
    "drg_codes","drg_severity_raw","drg_mortality_raw",
    "hospital_expire_flag","label",
    # Comorbidity target-like flags (to avoid leakage)
    "Acute_MI","Ischemic_HD","Arrhythmia","Heart_Failure","Cardiomyopathy",
    "Cerebrovascular","Arterial_Embolism","Peripheral_Vascular",
    "Pulmonary_Embolism","Venous_Thrombo","Valvular_HD","Endocarditis",
    "Pericardial","Congenital_HD","Aortic_Disease","Pulmonary_Heart",
    "Other_CVD","Other_Vascular","Hypertension",
]

# Manuscript thresholds (low/high)
LAB_THRESH = {
    "Hemoglobin_mean": (12, None), "Hematocrit_mean": (36, None),
    "WBC_mean": (4, 11), "Platelet Count_mean": (150, 450),
    "INR_mean": (None, 1.2), "PT_mean": (None, 15), "aPTT_mean": (None, 40),
    "Sodium_mean": (135, 145), "Potassium_mean": (3.5, 5.0), "Chloride_mean": (98, 107),
    "Calcium_mean": (8.5, 10.5), "Magnesium_mean": (1.7, 2.2),
    "Creatinine_mean": (None, 1.3), "BUN_mean": (None, 20),
    "Troponin_mean": (None, 0.04), "CK-MB_mean": (None, 5),
    "BNP_mean": (None, 100), "NT-proBNP_mean": (None, 125),
    "Glucose_mean": (70, 140), "Lactate_mean": (None, 2),
    "CRP_mean": (None, 5),
}
VITAL_THRESH = {
    "HeartRate_mean": (50, 120), "RespiratoryRate_mean": (12, 25),
    "SpO2_mean": (90, None), "Temperature_mean": (36.0, 38.0),
    "SystolicBP_mean": (90, 180), "DiastolicBP_mean": (50, 110),
    "MeanBP_mean": (60, 120),
}
# Abnormal fraction definitions
ABN_FRACTION_DEFS = {
    "SpO2_mean": ("low", 90),
    "HeartRate_mean": ("high", 120),
    "SystolicBP_mean": ("low", 90),
    "Glucose_mean": ("both", (70, 180)),
}

# ---------- Utilities ----------

def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ensure a column is parsed as datetime and contains no invalid values."""
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found in DataFrame.")
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} contains non-parsable values.")
    return df

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute a/b safely, replacing infinities with NaN."""
    r = a / b
    return r.replace([np.inf, -np.inf], np.nan)

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived clinical features, ratios, and missingness flags."""
    X = df.copy()

    # Age bins
    if "age" in X.columns:
        bins = [0, 45, 65, 75, 85, 120]
        labels = ["<45", "45-64", "65-74", "75-84", "85+"]
        X["Age_Group"] = pd.cut(X["age"], bins=bins, labels=labels, right=False, ordered=True)

    # Non-white proxy
    if "race" in X.columns:
        non_white = [
            "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO",
            "OTHER", "MULTI RACE ETHNICITY"
        ]
        X["Non_White_Race"] = (
            X["race"].astype(str).str.upper().apply(lambda x: 1 if any(r in x for r in non_white) else 0)
        ).astype("int32")

    # Polypharmacy
    drug_cols = [c for c in DRUG_COLS if c in X.columns]
    if drug_cols:
        X["Polypharmacy_Score"] = X[drug_cols].sum(axis=1).astype("int32")
        X["High_Risk_Polypharmacy"] = (X["Polypharmacy_Score"] >= 4).astype("int32")

    # Clinical ratios
    if {"BUN_mean", "Creatinine_mean"}.issubset(X.columns):
        X["BUN_Creatinine_Ratio"] = safe_ratio(X["BUN_mean"], X["Creatinine_mean"]).astype("float32")
    if {"HeartRate_mean", "SystolicBP_mean"}.issubset(X.columns):
        X["Shock_Index"] = safe_ratio(X["HeartRate_mean"], X["SystolicBP_mean"]).astype("float32")
    if {"MeanBP_mean", "HeartRate_mean"}.issubset(X.columns):
        X["MAP_HR_Coupling"] = safe_ratio(X["MeanBP_mean"], X["HeartRate_mean"]).astype("float32")
    if {"Platelet Count_mean", "WBC_mean"}.issubset(X.columns):
        X["PLR"] = safe_ratio(X["Platelet Count_mean"], X["WBC_mean"]).astype("float32")
    if {"Hemoglobin_mean", "Hematocrit_mean"}.issubset(X.columns):
        X["HHCC"] = safe_ratio(X["Hemoglobin_mean"], X["Hematocrit_mean"]).astype("float32")

    # Missingness flags
    for col in LAB_MEAN_COLS + VITAL_MEAN_COLS:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype("int32")

    # Category handling
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.add_categories(["Missing"]).fillna("Missing")

    return X

def encode_and_impute(X: pd.DataFrame, protected=None):
    """
    Encode categorical features and impute numeric features.
    Protected columns (IDs, time) are left untouched.
    """
    X_enc = X.copy()
    protected = protected or []

    # Encode categories (skip protected IDs/time)
    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
        if col in protected:
            continue
        if X_enc[col].dtype.name != "category":
            X_enc[col] = X_enc[col].astype("category")
        X_enc[col] = X_enc[col].cat.codes.astype("int32")

    # Impute numerics, collect report (skip protected IDs/time)
    impute_report = {}
    for col in X_enc.columns:
        if col in protected:
            continue
        if X_enc[col].dtype in ["float64", "float32"]:
            s = pd.to_numeric(X_enc[col], errors="coerce")
            med = s.median()
            impute_report[col] = {
                "type": "float",
                "median": float(med),
                "missing_pct": float(s.isna().mean())
            }
            X_enc[col] = s.astype("float32").fillna(med)
        elif X_enc[col].dtype in ["int64", "int32"]:
            s = pd.to_numeric(X_enc[col], errors="coerce")
            impute_report[col] = {
                "type": "int",
                "fill": -1,
                "missing_pct": float(s.isna().mean())
            }
            X_enc[col] = s.fillna(-1).astype("int32")

    return X_enc, impute_report

# ---------- Dataset loader ----------

def load_progression_dataset(csv_path, sample_rows=200000):
    # Rows to read
    nrows = 200000 if (sample_rows is None or sample_rows == 200000) else sample_rows
    print(f"Reading CSV (rows={'ALL' if nrows is None else nrows})...")
    df = pd.read_csv(csv_path, nrows=nrows, encoding="latin-1")
    print("Loaded shape:", df.shape)

    # Ensure datetime
    df = ensure_datetime(df, TIME_COL)

    # --- Detect patient/admission IDs and build protected list ---
    candidate_ids = ["subject_id", "hadm_id", "stay_id"]
    present_ids = [c for c in candidate_ids if c in df.columns]
    if not present_ids:
        raise ValueError(
            f"No patient/admission ID found. Expected one of {candidate_ids} in columns: {list(df.columns)}"
        )

    # Prefer subject_id for patient-level grouping
    PATIENT_COL = "subject_id" if "subject_id" in present_ids else present_ids[0]

    protected = list(present_ids) + [TIME_COL]  # keep IDs + admittime for grouping/labels

    # --- Build leaky set and drop while preserving protected ---
    leaky_cols = set(LEAKY_COLS_BASE)
    leaky_cols |= {c for c in df.columns if c.startswith("last_")}
    leaky_cols |= {c for c in df.columns if c.endswith("_abnormal")}
    leaky_cols |= {c for c in df.columns if ("time_above" in c or "time_below" in c)}

    drop_cols = [c for c in leaky_cols if c in df.columns and c not in protected]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Sanity check: ensure protected columns remain
    for col in protected:
        if col not in X.columns:
            raise ValueError(f"Protected column {col} missing from working frame after drop")

    # --- Feature engineering ---
    X = add_base_features(X)
    X = add_abnormal_flags(X)
    X = add_kinetics_and_burden(X, window_n=3)

    # --- Encode + impute ---
    X_enc, impute_report = encode_and_impute(X, protected=protected)

    # Exclude protected (IDs + time) from features
    feature_names = [c for c in X_enc.columns if c not in protected]
    X_enc = X_enc[feature_names]

    # --- Add engineered time features ---
    if TIME_COL in df.columns and np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        X_enc["admit_hour"] = df[TIME_COL].dt.hour
        X_enc["admit_dayofweek"] = df[TIME_COL].dt.dayofweek
        X_enc["days_since_admit"] = (df[TIME_COL] - df[TIME_COL].min()).dt.days
        feature_names.extend(["admit_hour", "admit_dayofweek", "days_since_admit"])

    # --- Build labels on original df (with IDs/time intact) ---
    labels, censored = build_time_based_labels(df)

    # Patient-level split stratified by any 48h event
    unique_ids = df[PATIENT_COL].unique()
    any48 = pd.Series(0, index=df.index, dtype="int32")
    for ev in EVENTS:
        # align indexes to avoid warnings
        any48 = any48 | labels[(ev, "48h")].reindex(df.index, fill_value=0)

    patient_strat = any48.groupby(df[PATIENT_COL]).max()

    if patient_strat.nunique() == 1:
        # Fallback if stratify has only one class
        train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    else:
        train_ids, val_ids = train_test_split(
            unique_ids,
            test_size=0.2,
            random_state=42,
            stratify=patient_strat.loc[unique_ids],
        )

    train_mask = df[PATIENT_COL].isin(train_ids)
    val_mask = df[PATIENT_COL].isin(val_ids)

    X_train = X_enc.loc[train_mask].reset_index(drop=True)
    X_val = X_enc.loc[val_mask].reset_index(drop=True)

    y_train, y_val = {}, {}
    cens_train, cens_val = {}, {}

    for ev in EVENTS:
        for hk in HORIZONS.keys():
            y_train[(ev, hk)] = labels[(ev, hk)].loc[train_mask].reset_index(drop=True)
            y_val[(ev, hk)] = labels[(ev, hk)].loc[val_mask].reset_index(drop=True)

    for hk in HORIZONS.keys():
        cens_train[hk] = censored[hk].loc[train_mask].reset_index(drop=True)
        cens_val[hk] = censored[hk].loc[val_mask].reset_index(drop=True)

    # Return only JSON-safe types (lists, dicts, primitives)
    return (
        X_train,
        X_val,
        y_train,
        y_val,
        cens_train,
        cens_val,
        list(feature_names),
        impute_report,
    )

# ---------- Abnormal flags and kinetics ----------

def add_abnormal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add abnormal low/high flags for labs and vitals based on thresholds."""
    X = df.copy()

    # Labs
    for col, (low, high) in LAB_THRESH.items():
        if col not in X.columns:
            continue
        if low is not None:
            X[f"{col}_low_flag"] = (X[col] <= low).astype("int32")
        if high is not None:
            X[f"{col}_high_flag"] = (X[col] >= high).astype("int32")

    # Vitals
    for col, (low, high) in VITAL_THRESH.items():
        if col not in X.columns:
            continue
        if low is not None:
            X[f"{col}_low_flag"] = (X[col] <= low).astype("int32")
        if high is not None:
            X[f"{col}_high_flag"] = (X[col] >= high).astype("int32")

    return X


def add_kinetics_and_burden(df: pd.DataFrame, window_n: int = 3) -> pd.DataFrame:
    """
    Add kinetics (last, delta, slope, volatility) and abnormal burden features.
    Operates patient-by-patient using available ID column and TIME_COL.
    """
    X = df.copy()

    # Auto-detect patient ID column
    candidate_ids = ["subject_id", "hadm_id", "stay_id"]
    id_col = next((c for c in candidate_ids if c in X.columns), None)
    if id_col is None or TIME_COL not in X.columns:
        raise ValueError(f"Required patient ID and/or {TIME_COL} not found in DataFrame")

    # Sort by patient and time
    X.sort_values([id_col, TIME_COL], inplace=True)

    # Features to compute kinetics on
    feature_cols = [c for c in LAB_MEAN_COLS + VITAL_MEAN_COLS if c in X.columns]

    def per_patient(g: pd.DataFrame) -> pd.DataFrame:
        t = g[TIME_COL]
        dt_hours = (t - t.shift(1)).dt.total_seconds() / 3600.0
        dt_hours = dt_hours.replace(0, np.nan)

        for col in feature_cols:
            last = g[col].shift(1)
            g[f"{col}_last"] = last
            g[f"{col}_delta"] = g[col] - last
            g[f"{col}_slope"] = g[f"{col}_delta"] / dt_hours
            g[f"{col}_vol"] = g[col].rolling(window=window_n, min_periods=2).std()
        return g

    # Apply per patient
    X = X.groupby(id_col, group_keys=False).apply(per_patient)

    # Abnormal burden for selected signals
    for col, defn in ABN_FRACTION_DEFS.items():
        if col not in X.columns:
            continue
        if defn[0] == "low":
            flag = (X[col] <= defn[1]).astype("int32")
        elif defn[0] == "high":
            flag = (X[col] >= defn[1]).astype("int32")
        elif defn[0] == "both":
            lo, hi = defn[1]
            flag = ((X[col] <= lo) | (X[col] >= hi)).astype("int32")
        else:
            continue

        fname = f"{col}_abn_flag"
        X[fname] = flag
        X[f"{col}_abn_burden"] = (
            X.groupby(id_col)[fname]
            .rolling(window=window_n, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return X

# ---------- Time-based labels (HOSP-only) ----------

def build_time_based_labels(df: pd.DataFrame):
    """
    Create binary labels per event × horizon.
    Requires:
      - Discharge time: dischtime
      - Death time: death_time (derived from hospital_expire_flag & dischtime)
      - MI/HF proxies: threshold crossings of Troponin/CK-MB and BNP/NT-proBNP
      - Deterioration composite: escalate within horizon using burden + kinetics
    """
    X = df.copy()
    X = ensure_datetime(X, TIME_COL)
    X.sort_values([PATIENT_COL, TIME_COL], inplace=True)

    # Death time derivation
    if "death_time" not in X.columns and "hospital_expire_flag" in X.columns and "dischtime" in X.columns:
        X["death_time"] = np.where(
            X["hospital_expire_flag"] == 1,
            pd.to_datetime(X["dischtime"], errors='coerce'),
            pd.NaT
        )

    labels = {}
    censored = {hk: pd.Series(False, index=X.index) for hk in HORIZONS.keys()}
    last_time = X.groupby(PATIENT_COL)[TIME_COL].transform("max")

    # Precompute MI/HF event times by threshold crossing (proxy)
    mi_signal = (
        ((X.get('Troponin_mean', pd.Series(np.nan, index=X.index)) >= 0.04)) |
        ((X.get('CK-MB_mean', pd.Series(np.nan, index=X.index)) >= 5))
    )
    hf_signal = (
        ((X.get('BNP_mean', pd.Series(np.nan, index=X.index)) >= 100)) |
        ((X.get('NT-proBNP_mean', pd.Series(np.nan, index=X.index)) >= 125))
    )

    # Helper: first time signal is true per patient
    def first_true_time(signal_series: pd.Series) -> pd.Series:
        times = X[TIME_COL]
        # For each patient, find earliest index where signal is true
        first_idx = signal_series.groupby(X[PATIENT_COL]).apply(
            lambda z: z.idxmax() if z.max() == 1 else -1
        )
        ev_time = pd.Series(pd.NaT, index=X.index)
        for pid, idx in first_idx.items():
            if idx != -1:
                ev_time.loc[idx] = times.loc[idx]
        # Forward fill patient-wise to have a single event time per patient
        ev_time = ev_time.groupby(X[PATIENT_COL]).transform('min')
        return ev_time

    X['mi_event_time'] = first_true_time(mi_signal)
    X['hf_event_time'] = first_true_time(hf_signal)
    X['discharge_time'] = pd.to_datetime(X['dischtime'], errors='coerce') if 'dischtime' in X.columns else pd.NaT

    # Deterioration composite: any two escalation criteria
    det_flags = pd.DataFrame(index=X.index)
    det_flags['SpO2_burden_high'] = (X.get('SpO2_mean_abn_burden', pd.Series(0, index=X.index)) > 0.5).astype('int32')
    det_flags['HR_burden_high']   = (X.get('HeartRate_mean_abn_burden', pd.Series(0, index=X.index)) > 0.5).astype('int32')
    det_flags['SBP_burden_high']  = (X.get('SystolicBP_mean_abn_burden', pd.Series(0, index=X.index)) > 0.3).astype('int32')
    det_flags['Lactate_high']     = (X.get('Lactate_mean', pd.Series(np.nan, index=X.index)) >= 2.0).astype('int32')
    det_flags['Cr_rising']        = (X.get('Creatinine_mean_delta', pd.Series(np.nan, index=X.index)) > 0).astype('int32')
    det_flags['BUNCr_rising']     = (X.get('BUN_Creatinine_Ratio_delta', pd.Series(np.nan, index=X.index)) > 0).astype('int32')
    det_flags['Glucose_dysg']     = (
        ((X.get('Glucose_mean', pd.Series(np.nan, index=X.index)) <= 70) |
         (X.get('Glucose_mean', pd.Series(np.nan, index=X.index)) >= 180))
    ).astype('int32')
    X['det_score_now'] = det_flags.sum(axis=1).astype('int32')

    # Build labels per horizon
    for hk, h_td in HORIZONS.items():
        censored[hk] = (X[TIME_COL] + h_td > last_time)

        labels[('Acute_MI', hk)] = ((X['mi_event_time'] > X[TIME_COL]) &
                                    (X['mi_event_time'] <= X[TIME_COL] + h_td)).astype('int32')
        labels[('Heart_Failure', hk)] = ((X['hf_event_time'] > X[TIME_COL]) &
                                         (X['hf_event_time'] <= X[TIME_COL] + h_td)).astype('int32')
        labels[('Discharge', hk)] = ((X['discharge_time'] > X[TIME_COL]) &
                                     (X['discharge_time'] <= X[TIME_COL] + h_td)).astype('int32')
        labels[('Death', hk)] = ((X.get('death_time', pd.Series(pd.NaT, index=X.index)) > X[TIME_COL]) &
                                 (X.get('death_time', pd.Series(pd.NaT, index=X.index)) <= X[TIME_COL] + h_td)).astype('int32')
        labels[('Deterioration', hk)] = (
            (X['det_score_now'] >= 2) &
            (~labels[('Discharge', hk)].astype(bool))
        ).astype('int32')

    return labels, censored

# ---------- Training, calibration, thresholds ----------
def train_binary_head_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost variant for LUMINAE progression forecasting.
    Includes Isotonic Calibration to match the LGBM head logic.
    """
    from xgboost import XGBClassifier
    from sklearn.calibration import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

    # Calculate scale_pos_weight for the extreme imbalance (the "dozen" deaths)
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

    # Initialize and train XGBoost
    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Get raw probabilities
    prob_val_raw = model.predict_proba(X_val)[:, 1]

    # Isotonic Calibration (Aligns raw scores with actual clinical risk)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(prob_val_raw, y_val)
    prob_val_calibrated = iso.transform(prob_val_raw)

    # Calculate Metrics for the console
    metrics = {
        "roc_auc": roc_auc_score(y_val, prob_val_calibrated),
        "brier": brier_score_loss(y_val, prob_val_calibrated),
        "avg_precision": average_precision_score(y_val, prob_val_calibrated)
    }

    return model, iso, prob_val_calibrated, metrics

def train_binary_head_lgbm(X_train, y_train, X_val, y_val):
    pos = int(y_train.sum()); neg = len(y_train) - pos
    spw = (neg / max(1, pos)) if pos > 0 else 1.0
    model = LGBMClassifier(
        objective='binary',
        n_estimators=800, learning_rate=0.05,
        max_depth=6, num_leaves=48,
        subsample=0.7, colsample_bytree=0.7,
        min_child_samples=100, reg_alpha=0.3, reg_lambda=0.3,
        random_state=42, n_jobs=-1,
        class_weight={0: 1.0, 1: spw}
    )
    dummy_stderr = io.StringIO(); original_stderr = sys.stderr; sys.stderr = dummy_stderr
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)]
        )
    finally:
        sys.stderr = original_stderr

    prob_val = model.predict_proba(X_val)[:, 1].astype('float32')

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(prob_val, y_val)
    prob_val_cal = iso.predict(prob_val).astype('float32')

    # Metrics
    auc = roc_auc_score(y_val, prob_val_cal) if len(np.unique(y_val)) > 1 else np.nan
    apr = average_precision_score(y_val, prob_val_cal) if len(np.unique(y_val)) > 1 else np.nan
    brier = brier_score_loss(y_val, prob_val_cal)

    return model, iso, prob_val_cal, {"roc_auc": auc, "avg_precision": apr, "brier": brier}


def optimize_threshold_binary(prob, y_true, mode="f1", alert_budget=None):
    if mode == "f1":
        best_f1, best_th = -1, 0.5
        for th in np.arange(0.05, 0.96, 0.01):
            y_pred = (prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1: best_f1, best_th = f1, th
        return float(best_th), float(best_f1)
    elif mode == "budget":
        if alert_budget is None or not (0 < alert_budget < 1):
            raise ValueError("alert_budget must be in (0,1).")
        th = np.quantile(prob, 1 - alert_budget)
        y_pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return float(th), float(f1)
    else:
        raise ValueError("mode must be 'f1' or 'budget'")


# ---------- Evaluation ----------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

def eval_head(y_true, prob, th, title="", results_store=None):
    """
    Evaluate a binary head: fix for 'index out of bounds' error using np.clip.
    """
    y_pred = (prob >= th).astype(int)
    print(f"{title} — Classification report:")
    print(classification_report(y_true, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Event","Event"], yticklabels=["No Event","Event"])
    plt.title(f"{title} — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

    # Macro metrics
    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, prob)
        metrics["avg_precision"] = average_precision_score(y_true, prob)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        print("ROC AUC:", metrics["roc_auc"])
        print("Average precision:", metrics["avg_precision"])
        print("Macro F1:", metrics["macro_f1"])
        print("Macro Recall:", metrics["macro_recall"])
        print("Macro Precision:", metrics["macro_precision"])
    metrics["brier"] = brier_score_loss(y_true, prob)
    print("Brier score:", metrics["brier"])

    # --- FIX START: Robust Bucketing ---
    bins = np.linspace(0, 1, 6)  # 5 buckets: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    # digitize returns 1-based index; we subtract 1. 
    # clip ensures any value (especially 1.0) stays within index 0-4.
    bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, len(bins) - 2)
    # --- FIX END ---

    df = pd.DataFrame({"prob": prob, "y_true": y_true, "y_pred": y_pred, "bin": bin_ids})
    print(f"\n{title} — Macro metrics per probability bucket:")
    bucket_stats = []
    
    for b in sorted(df["bin"].unique()):
        sub = df[df["bin"] == b]
        if len(sub) == 0: 
            continue
        
        # Safe metric calculation for buckets that might only contain one class
        f1b = f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
        recb = recall_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
        precb = precision_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
        
        print(f"Bucket {b} ({bins[b]:.2f}-{bins[b+1]:.2f}), n={len(sub)}")
        print("   F1:", f1b)
        print("   Recall:", recb)
        print("   Precision:", precb)
        bucket_stats.append({"bucket": b, "n": len(sub), "f1": f1b, "recall": recb, "precision": precb})
    
    metrics["bucket_stats"] = bucket_stats

    if results_store is not None:
        results_store.append({"title": title, "y_true": y_true, "y_pred": y_pred, "prob": prob, "metrics": metrics})

    return metrics


def save_calibration_plot(y_true, prob, out_path, title="Calibration"):
    """
    Save a calibration plot with robust binning.
    """
    bins = np.linspace(0, 1, 11) # 10 buckets
    # Apply the same clip fix here to prevent index errors in plotting
    bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, len(bins) - 2)
    
    df = pd.DataFrame({"prob": prob, "y": y_true, "bin": bin_ids})
    agg = df.groupby("bin").agg(
        prob_mean=("prob","mean"),
        y_rate=("y","mean")
    ).dropna()

    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'k--',label='Perfect')
    plt.plot(agg["prob_mean"], agg["y_rate"], marker='o', label='Model')
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Global aggregation ----------
# ---------- Global Evaluation & Orchestration ----------

def report_global_metrics(all_results):
    """
    Combine predictions across all heads and report global performance.
    """
    if not all_results:
        print("\n[Global Eval] No results to aggregate.")
        return

    # Concatenate all validation predictions
    y_true_global = np.concatenate([r["y_true"] for r in all_results])
    y_pred_global = np.concatenate([r["y_pred"] for r in all_results])
    prob_global   = np.concatenate([r["prob"]   for r in all_results])

    print("\n" + "="*50)
    print("=== GLOBAL EVALUATION ACROSS ALL HEADS ===")
    print("="*50)
    print(classification_report(y_true_global, y_pred_global, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_true_global, y_pred_global)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Event","Event"], yticklabels=["No Event","Event"])
    plt.title("LUMINAE: Global Progression Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

    print(f"Global ROC AUC:         {roc_auc_score(y_true_global, prob_global):.4f}")
    print(f"Global Avg Precision:   {average_precision_score(y_true_global, prob_global):.4f}")
    print(f"Global Macro F1:        {f1_score(y_true_global, y_pred_global, average='macro', zero_division=0):.4f}")
    print(f"Global Brier Score:     {brier_score_loss(y_true_global, prob_global):.4f}")

def train_all_heads(X_train, X_val, y_train, y_val, out_dir, threshold_mode="f1", alert_budget=None):
    """
    Orchestration with Atomic Saving: Persists each of the 18 models uniquely and immediately.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    all_results = [] 

    for ev in EVENTS:
        for hk in HORIZONS.keys():
            y_tr = y_train.get((ev, hk))
            y_va = y_val.get((ev, hk))

            if y_tr is None or y_va is None:
                continue

            # Skip heads with no variation (e.g., zero deaths in a specific window)
            if (len(np.unique(y_va)) < 2):
                print(f"Skipping {ev}-{hk}: No variation in target labels.")
                continue

            head_key = f"{ev}_{hk}"
            print(f"\n>>> Processing Head: {head_key} | Train Pos: {int(y_tr.sum())} | Val Pos: {int(y_va.sum())}")

            for algo in ["lgbm", "xgboost"]:
                try:
                    print(f"    Training {algo.upper()} variant...")
                    
                    if algo == "lgbm":
                        model, iso, prob_val, m = train_binary_head_lgbm(X_train, y_tr, X_val, y_va)
                    else:
                        model, iso, prob_val, m = train_binary_head_xgboost(X_train, y_tr, X_val, y_va)

                    th, f1v = optimize_threshold_binary(prob_val, y_va, mode=threshold_mode, alert_budget=alert_budget)
                    
                    # Calculate final metrics for this specific head/algo
                    y_pred = (prob_val >= th).astype(int)
                    head_metrics = {
                        "threshold": float(th),
                        "f1": float(f1v),
                        "roc_auc": float(m["roc_auc"]) if not np.isnan(m["roc_auc"]) else None,
                        "brier": float(m["brier"]),
                        "macro_f1": float(f1_score(y_va, y_pred, average="macro", zero_division=0))
                    }

                    # --- ATOMIC SAVING ---
                    # Save model and calibrator as a single artifact for the head
                    artifact = {"model": model, "calibrator": iso, "threshold": th, "metrics": head_metrics}
                    save_path = os.path.join(out_dir, f"model_{head_key}_{algo}.joblib")
                    joblib.dump(artifact, save_path)
                    
                    results[f"{head_key}-{algo}"] = head_metrics
                    
                    # Store for global reporting (Includes the fixed eval_head logic)
                    eval_head(y_va, prob_val, th, title=f"{head_key}-{algo}", results_store=all_results)
                    
                    print(f"    [SAVED] {save_path}")

                except Exception as e:
                    print(f"    [CRITICAL ERROR] Failed {head_key}-{algo}: {str(e)}")
                    results[f"{head_key}-{algo}"] = {"error": str(e)}

    # Save metadata summary
    with open(os.path.join(out_dir, "head_results.json"), "w") as fh:
        json.dump(to_json_safe(results), fh, indent=2)

    if all_results:
        report_global_metrics(all_results)

    return results

# ---------- JSON-safe helper ----------

def to_json_safe(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)): return str(obj)
    if isinstance(obj, dict): return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json_safe(v) for v in obj]
    return obj

# ---------- Main Execution ----------

import sys, traceback, faulthandler, argparse
faulthandler.enable()

def main():
    print("\n" + "="*50)
    print(" LUMINAE PROGRESSION NETWORK: FORECASTING ENGINE ")
    print("="*50 + "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200000, help="Rows to sample (0=full)")
    parser.add_argument("--tag-dir", type=str, default="ProgressionNet_Final", help="Output directory name")
    parser.add_argument("--threshold-mode", type=str, default="f1", choices=["f1","budget"])
    parser.add_argument("--alert-budget", type=float, default=None)
    args = parser.parse_args()

    out_dir = os.path.join(MODELS_DIR, args.tag_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading MIMIC-IV Progression Cohort (Sampling: {args.sample})...")
    X_train, X_val, y_train, y_val, cens_train, cens_val, feature_names, impute_report = \
        load_progression_dataset(CSV_PATH, sample_rows=args.sample)

    # 2. Persist Metadata
    with open(os.path.join(out_dir, "feature_names.json"), "w") as fh:
        json.dump(feature_names, fh, indent=2)
    
    # 3. Train and Save
    print("\nStarting Multi-Task Training Loop...")
    results = train_all_heads(
        X_train, X_val, y_train, y_val, out_dir,
        threshold_mode=args.threshold_mode, alert_budget=args.alert_budget
    )

    # 4. Final Summary
    summary = {
        "status": "complete",
        "timestamp": str(pd.Timestamp.now()),
        "config": vars(args),
        "heads": results
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(to_json_safe(summary), fh, indent=2)

    print(f"\nSUCCESS: All artifacts saved to {out_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL ERROR DURING EXECUTION:")
        traceback.print_exc()
        input("\nPress Enter to close...")