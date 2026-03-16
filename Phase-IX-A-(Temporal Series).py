import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------
# Paths
# -------------------------
temporal_vitals_chunked = config["phase10_vitals_chunked"]
temporal_labs_chunked = config["phase10_labs_chunked"]
labitems_path = config["phase10_labitems"]
d_items_path = config["phase10_d_items"]
output_dir = config["phase10_output_dir"]
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 1. Load item dictionaries
# -------------------------
labitems = pd.read_csv(labitems_path)
d_items = pd.read_csv(d_items_path)

# -------------------------
# 2. Pivot labs
# -------------------------
labs = pd.read_csv(temporal_labs_chunked)
labs = labs.merge(labitems[["itemid", "label"]], on="itemid", how="left")

labs_wide = labs.pivot_table(
    index=["subject_id", "hadm_id"],
    columns="label",
    values=["last", "min", "max", "mean", "std"],
    aggfunc="mean"
)
labs_wide.columns = [f"{stat}_{lab}" for stat, lab in labs_wide.columns]
labs_wide = labs_wide.reset_index()

# -------------------------
# 3. Pivot vitals
# -------------------------
vitals = pd.read_csv(temporal_vitals_chunked)
vitals = vitals.merge(d_items[["itemid", "label"]], on="itemid", how="left")

vitals_wide = vitals.pivot_table(
    index=["subject_id", "hadm_id"],
    columns="label",
    values=["last", "min", "max", "mean", "std"],
    aggfunc="mean"
)
vitals_wide.columns = [f"{stat}_{lab}" for stat, lab in vitals_wide.columns]
vitals_wide = vitals_wide.reset_index()

# -------------------------
# 4. Merge labs + vitals
# -------------------------
temporal_all = labs_wide.merge(vitals_wide, on=["subject_id", "hadm_id"], how="outer")

# -------------------------
# 5. Save Phase-IX temporal dataset
# -------------------------
temporal_all_out = os.path.join(output_dir, "temporal_all_wide.csv")
temporal_all.to_csv(temporal_all_out, index=False)

print("✅ Phase-IX temporal dataset saved:", temporal_all.shape)