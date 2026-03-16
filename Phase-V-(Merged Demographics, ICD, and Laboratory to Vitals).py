import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inputs
CHARTEVENTS = config["phase5_chartevents"]
D_ITEMS = config["phase5_d_items"]
STAGE4 = config["phase5_stage4"]
OUTPUT = config["phase5_output"]
TEMP_AGG = config["phase5_temp"]

CHUNKSIZE = 1_000_000

# --- Load d_items dictionary ---
d_items = pd.read_csv(D_ITEMS, dtype=str)[["itemid","label"]]

# --- Curated vitals ---
VITALS = [
    "Heart Rate","Respiratory Rate","SpO2","Temperature Celsius",
    "Non Invasive Blood Pressure systolic","Non Invasive Blood Pressure diastolic",
    "Non Invasive Blood Pressure mean"
]

# --- Abnormal thresholds ---
ABNORMAL_THRESHOLDS = {
    "Heart Rate": {"low": 50, "high": 120},
    "Respiratory Rate": {"low": 12, "high": 25},
    "SpO2": {"low": 90, "high": None},
    "Temperature Celsius": {"low": 36.0, "high": 38.0},
    "Non Invasive Blood Pressure systolic": {"low": 90, "high": 180},
    "Non Invasive Blood Pressure diastolic": {"low": 50, "high": 110},
    "Non Invasive Blood Pressure mean": {"low": 60, "high": 120},
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
for i, chunk in enumerate(pd.read_csv(CHARTEVENTS, chunksize=CHUNKSIZE, dtype=str, low_memory=False)):
    # Merge labels
    chunk = chunk.merge(d_items, on="itemid", how="left")
    chunk = chunk[chunk["label"].isin(VITALS)].copy()
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

# --- Merge with Stage-IV dataset ---
stage4 = pd.read_csv(STAGE4)
merged = stage4.merge(final_agg, on=["subject_id","hadm_id"], how="left")

# Save final Phase-V dataset
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

merged.to_csv(OUTPUT, index=False)

print("💾 Phase-V dataset saved to:", OUTPUT)
print("📊 Shape:", merged.shape)