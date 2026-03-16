import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT = config["phase1_input"]
OUTPUT = config["phase1_output"]

CHUNKSIZE = 500_000

# Drug buckets
DRUG_BUCKETS = {
    "ACE_Inhibitors": ["lisinopril","enalapril","captopril","ramipril"],
    "Beta_Blockers": ["metoprolol","atenolol","propranolol","carvedilol"],
    "Statins": ["atorvastatin","simvastatin","rosuvastatin","pravastatin"],
    "Anticoagulants": ["warfarin","heparin","apixaban","rivaroxaban","dabigatran"],
    "Diuretics": ["furosemide","hydrochlorothiazide","spironolactone"],
    "Calcium_Channel_Blockers": ["amlodipine","diltiazem","verapamil"],
}

# Remove old output if exists
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

# Process in chunks
for i, chunk in enumerate(pd.read_csv(
    INPUT,
    chunksize=CHUNKSIZE,
    dtype=str,          # everything as string for safety
    engine="python",    # tolerant parser
    on_bad_lines="skip" # skip malformed rows
)):
    # Keep only demographics, timestamps, DRGs, pharmacy
    keep_cols = [
        "subject_id","hadm_id","admittime","dischtime","admission_type",
        "insurance","marital_status","race","gender","anchor_age",
        "hospital_expire_flag",
        "drg_code","drg_severity","drg_mortality",
        "drug","route_rx"
    ]
    chunk = chunk[keep_cols]

    # Pre-map drug buckets at row level
    drug_lower = chunk["drug"].str.lower().fillna("")
    for bucket, keys in DRUG_BUCKETS.items():
        pattern = "|".join(keys)
        chunk[bucket] = drug_lower.str.contains(pattern, regex=True).astype(int)

    # Define aggregation functions
    agg_funcs = {
        "admittime": "first",
        "dischtime": "first",
        "admission_type": "first",
        "insurance": "first",
        "marital_status": "first",
        "race": "first",
        "gender": "first",
        "anchor_age": "first",
        "hospital_expire_flag": "first",
        "drg_code": lambda x: list(set(x.dropna())),
        "drg_severity": lambda x: list(set(x.dropna())),
        "drg_mortality": lambda x: list(set(x.dropna())),
        "drug": "nunique",
        "route_rx": "nunique",
    }
    # Add bucket aggregations
    for bucket in DRUG_BUCKETS:
        agg_funcs[bucket] = "max"

    # Collapse per admission
    grouped = chunk.groupby(["subject_id","hadm_id"]).agg(agg_funcs).reset_index()

    # Rename aggregated columns for clarity
    grouped = grouped.rename(columns={
        "drug": "n_drugs",
        "route_rx": "n_routes",
        "drg_code": "drg_codes",
        "drg_severity": "drg_severity_raw",
        "drg_mortality": "drg_mortality_raw"
    })

    # Append to disk
    if not os.path.exists(OUTPUT):
        grouped.to_csv(OUTPUT, index=False, mode="w")
    else:
        grouped.to_csv(OUTPUT, index=False, mode="a", header=False)

    print(f"✅ Processed chunk {i+1}, rows written: {len(grouped)}")

print("\n💾 Final collapsed file saved to:", OUTPUT)