import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inputs
CORE = config["phase3_core"]
DIAG = config["phase3_diag"]
OUTPUT = config["phase3_output"]

# --- Load Stage-I core (already admission-level) ---
core = pd.read_csv(CORE)

# --- Load Stage-II cleaned diagnoses ---
diag = pd.read_csv(DIAG)

# Collapse diagnoses per admission: count buckets
bucket_counts = (
    diag.groupby(["subject_id","hadm_id","bucket"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
)

# Merge with Stage-I core
merged = core.merge(bucket_counts, on=["subject_id","hadm_id"], how="left")

# Fill NaN bucket counts with 0
bucket_cols = [c for c in merged.columns if c not in core.columns]
merged[bucket_cols] = merged[bucket_cols].fillna(0).astype(int)

# Save
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

merged.to_csv(OUTPUT, index=False)

print("💾 Merged dataset saved to:", OUTPUT)
print("📊 Shape:", merged.shape)
print("🧾 Bucket columns added:", bucket_cols)