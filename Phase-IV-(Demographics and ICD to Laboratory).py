import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inputs
D_LAB = config["phase4_labitems"]
LABEVENTS = config["phase4_labevents"]
STAGE3 = config["phase4_stage3"]
OUTPUT = config["phase4_output"]
TEMP_AGG = config["phase4_temp"]

CHUNKSIZE = 1_000_000  # adjust as needed

# --- Load lab dictionary ---
d_lab = pd.read_csv(D_LAB, dtype=str)[["itemid","label"]]

# --- Curated CVD-relevant labs ---
CVD_LABS = [
    "Hemoglobin","Hematocrit","WBC","Platelet Count",
    "INR","PT","aPTT",
    "Sodium","Potassium","Chloride","Calcium","Magnesium",
    "Creatinine","BUN",
    "Troponin","CK-MB","BNP","NT-proBNP",
    "Glucose","Lactate","CRP"
]

# --- Abnormal thresholds ---
ABNORMAL_THRESHOLDS = {
    "Hemoglobin": {"low": 12, "high": None},
    "Hematocrit": {"low": 36, "high": None},
    "WBC": {"low": 4, "high": 11},
    "Platelet Count": {"low": 150, "high": 450},
    "INR": {"low": None, "high": 1.2},
    "PT": {"low": None, "high": 15},
    "aPTT": {"low": None, "high": 40},
    "Sodium": {"low": 135, "high": 145},
    "Potassium": {"low": 3.5, "high": 5.0},
    "Chloride": {"low": 98, "high": 107},
    "Calcium": {"low": 8.5, "high": 10.5},
    "Magnesium": {"low": 1.7, "high": 2.2},
    "Creatinine": {"low": None, "high": 1.3},
    "BUN": {"low": None, "high": 20},
    "Troponin": {"low": None, "high": 0.04},
    "CK-MB": {"low": None, "high": 5},
    "BNP": {"low": None, "high": 100},
    "NT-proBNP": {"low": None, "high": 125},
    "Glucose": {"low": 70, "high": 140},
    "Lactate": {"low": None, "high": 2},
    "CRP": {"low": None, "high": 5},
}

def abnormal_flag(lab, val):
    if pd.isna(val) or lab not in ABNORMAL_THRESHOLDS:
        return 0
    low = ABNORMAL_THRESHOLDS[lab]["low"]
    high = ABNORMAL_THRESHOLDS[lab]["high"]
    if low is not None and val < low:
        return 1
    if high is not None and val > high:
        return 1
    return 0

# Remove temp file if exists
if os.path.exists(TEMP_AGG):
    os.remove(TEMP_AGG)

# --- Process in chunks ---
for i, chunk in enumerate(pd.read_csv(LABEVENTS, chunksize=CHUNKSIZE, dtype=str, low_memory=False)):
    # Merge labels
    chunk = chunk.merge(d_lab, on="itemid", how="left")
    chunk = chunk[chunk["label"].isin(CVD_LABS)].copy()
    if chunk.empty:
        continue

    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
    chunk["abnormal_flag"] = chunk.apply(lambda r: abnormal_flag(r["label"], r["valuenum"]), axis=1)

    # Aggregate values
    agg_values = (
        chunk.groupby(["subject_id","hadm_id","label"])["valuenum"]
            .agg(["min","max","mean","std"])
            .unstack(level="label")
    )
    agg_values.columns = [f"{lab}_{stat}" for stat, lab in agg_values.columns]
    agg_values = agg_values.reset_index()

    # Aggregate abnormal flags
    agg_flags = (
        chunk.groupby(["subject_id","hadm_id","label"])["abnormal_flag"]
            .max()
            .unstack(fill_value=0)
            .reset_index()
    )
    agg_flags.columns = ["subject_id","hadm_id"] + [f"{lab}_abnormal" for lab in agg_flags.columns[2:]]

    # Merge values + flags
    agg = agg_values.merge(agg_flags, on=["subject_id","hadm_id"], how="left")

    # Append to temp file
    if not os.path.exists(TEMP_AGG):
        agg.to_csv(TEMP_AGG, index=False, mode="w")
    else:
        agg.to_csv(TEMP_AGG, index=False, mode="a", header=False)

    print(f"✅ Processed chunk {i+1}, rows written: {len(agg)}")

# --- Collapse temp aggregates across chunks ---
all_agg = pd.read_csv(TEMP_AGG)
final_agg = (
    all_agg.groupby(["subject_id","hadm_id"]).max().reset_index()
)

# --- Merge with Stage-III ---
stage3 = pd.read_csv(STAGE3)
merged = stage3.merge(final_agg, on=["subject_id","hadm_id"], how="left")

# Save final Phase-IV dataset
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

merged.to_csv(OUTPUT, index=False)

print("💾 Phase-IV dataset saved to:", OUTPUT)
print("📊 Shape:", merged.shape)