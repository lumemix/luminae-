import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------
# Paths
# -------------------------
phase_v_file = config["phase5_output"]
labitems_path = config["phase4_labitems"]
labevents_path = config["phase4_labevents"]
output_dir = config["phase9_output_dir"]
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 1. Define CVD-relevant labs
# -------------------------
cvd_labs = {
    "Hemoglobin", "Hematocrit", "Platelet Count", "WBC",
    "Creatinine", "Sodium", "Potassium", "Chloride", "Magnesium",
    "Glucose", "Lactate", "PT", "INR",
    "Troponin", "BNP", "NT-proBNP"
}

# -------------------------
# 2. Map lab itemids
# -------------------------
labitems = pd.read_csv(labitems_path)
labitems_cvd = labitems[labitems["label"].isin(cvd_labs)]
cvd_itemids = set(labitems_cvd["itemid"].unique())
labitems_cvd.to_csv(os.path.join(output_dir, "labitems_cvd.csv"), index=False)

# -------------------------
# 3. Stream labevents and aggregate per chunk
# -------------------------
agg_out_path = os.path.join(output_dir, "temporal_labs_chunked.csv")
first_write = True
chunk_id = 0

for chunk in pd.read_csv(labevents_path, chunksize=500000):
    chunk_id += 1
    filtered = chunk[chunk["itemid"].isin(cvd_itemids)]
    if filtered.empty:
        print(f"Chunk {chunk_id}: no CVD labs, skipping")
        continue

    grouped = filtered.groupby(["subject_id", "hadm_id", "itemid"])["valuenum"].agg(
        ["last", "min", "max", "mean", "std"]
    ).reset_index()

    grouped.to_csv(agg_out_path, mode="a", index=False, header=first_write)
    first_write = False
    print(f"Chunk {chunk_id}: wrote {len(grouped)} aggregated rows")

print("✅ Finished streaming labevents. Aggregates written to:", agg_out_path)

# -------------------------
# 4. Combine all chunked aggregates into wide format
# -------------------------
agg_all = pd.read_csv(agg_out_path)

# map itemid -> label
agg_all = agg_all.merge(labitems_cvd[["itemid", "label"]], on="itemid", how="left")

# pivot to wide
temporal_wide = agg_all.pivot_table(
    index=["subject_id", "hadm_id"],
    columns="label",
    values=["last", "min", "max", "mean", "std"],
    aggfunc="mean"  # if multiple chunks contributed
)

temporal_wide.columns = [f"{stat}_{lab}" for stat, lab in temporal_wide.columns]
temporal_wide = temporal_wide.reset_index()

# -------------------------
# 5. Merge with Phase-V file
# -------------------------
phase_v = pd.read_csv(phase_v_file)
merged = phase_v.merge(temporal_wide, on=["subject_id", "hadm_id"], how="left")

# -------------------------
# 6. Save final merged dataset
# -------------------------
final_out = os.path.join(output_dir, "phase_vi_merged.csv")
merged.to_csv(final_out, index=False)
print("✅ Final merged dataset saved:", merged.shape)