import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------
# Paths
# -------------------------
temporal_path = config["phase12_temporal"]
static_path   = config["phase12_static"]
output_dir    = config["phase12_output_dir"]
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 1. Load datasets
# -------------------------
print("Loading temporal and static datasets...")
temporal_df = pd.read_csv(temporal_path, low_memory=False)
static_df   = pd.read_csv(static_path, low_memory=False)
print(f" Temporal shape: {temporal_df.shape}")
print(f" Static shape:   {static_df.shape}")

# -------------------------
# 2. Ensure merge keys have the same dtype
# -------------------------
for df in (temporal_df, static_df):
    df["subject_id"] = df["subject_id"].astype(str)
    df["hadm_id"]    = df["hadm_id"].astype(str)

# -------------------------
# 3. Merge on subject_id + hadm_id
# -------------------------
print("Merging temporal + static...")
final_master = static_df.merge(temporal_df, on=["subject_id","hadm_id"], how="left")
print(f" Final merged shape: {final_master.shape}")

# -------------------------
# 4. Save final dataset
# -------------------------
out_path = os.path.join(output_dir, "phase_xii_master.csv")
final_master.to_csv(out_path, index=False)
print(f"✅ Phase-XII master dataset saved to: {out_path}")