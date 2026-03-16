import pandas as pd
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT = config["phase2_input"]
OUTPUT = config["phase2_output"]

CHUNKSIZE = 1_000_000

# --- Makeshift 19 CVD buckets ---
CVD_BUCKETS = {
    "Acute_MI": {"icd9": range(410, 415), "icd10": ["I21","I22"]},
    "Ischemic_HD": {"icd9": [414], "icd10": ["I20","I23","I24","I25"]},
    "Heart_Failure": {"icd9": [428], "icd10": ["I50"]},
    "Arrhythmia": {"icd9": range(426, 428), "icd10": ["I47","I48","I49"]},
    "Hypertension": {"icd9": [401,402,403,404,405], "icd10": ["I10","I11","I12","I13","I15"]},
    "Cerebrovascular": {"icd9": range(430, 439), "icd10": ["I60","I61","I62","I63","I64","I65","I66","I67","I68","I69"]},
    "Peripheral_Vascular": {"icd9": [443], "icd10": ["I70","I71","I72","I73","I74","I77"]},
    "Pulmonary_Heart": {"icd9": [415,416,417], "icd10": ["I26","I27"]},
    "Valvular_HD": {"icd9": range(394, 398), "icd10": ["I34","I35","I36","I37","I38","I39"]},
    "Cardiomyopathy": {"icd9": [425], "icd10": ["I42","I43"]},
    "Endocarditis": {"icd9": [421], "icd10": ["I33","I38","I39"]},
    "Pericardial": {"icd9": [420,423], "icd10": ["I30","I31","I32"]},
    "Congenital_HD": {"icd9": range(745, 747), "icd10": ["Q20","Q21","Q22","Q23","Q24","Q25","Q26"]},
    "Other_CVD": {"icd9": range(390, 399), "icd10": ["I51","I52"]},
    "Aortic_Disease": {"icd9": [441], "icd10": ["I71"]},
    "Venous_Thrombo": {"icd9": [453], "icd10": ["I80","I81","I82"]},
    "Arterial_Embolism": {"icd9": [444], "icd10": ["I74"]},
    "Pulmonary_Embolism": {"icd9": [415], "icd10": ["I26"]},
    "Other_Vascular": {"icd9": [447,448], "icd10": ["I77","I78","I79"]},
}

def bucket_icd(icd_code, icd_version):
    """Return list of buckets this ICD belongs to."""
    buckets = []
    if not icd_code or not icd_version:
        return buckets
    s = str(icd_code).upper().strip()
    v = str(icd_version).strip()
    if v == "9":
        if s[:3].isdigit():
            code = int(s[:3])
            for b, rules in CVD_BUCKETS.items():
                if "icd9" in rules and code in rules["icd9"]:
                    buckets.append(b)
    elif v == "10":
        for b, rules in CVD_BUCKETS.items():
            if "icd10" in rules and any(s.startswith(pref) for pref in rules["icd10"]):
                buckets.append(b)
    return buckets

# Remove old output if exists
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

bucket_counts = {b: 0 for b in CVD_BUCKETS}

# Process in chunks
for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNKSIZE, dtype=str)):
    keep_cols = ["subject_id","hadm_id","icd_code","icd_version"]
    chunk = chunk[keep_cols].copy()

    rows = []
    for _, r in chunk.iterrows():
        bks = bucket_icd(r["icd_code"], r["icd_version"])
        for b in bks:
            rows.append({
                "subject_id": r["subject_id"],
                "hadm_id": r["hadm_id"],
                "icd_code": r["icd_code"],
                "icd_version": r["icd_version"],
                "bucket": b
            })
            bucket_counts[b] += 1

    if rows:
        out_df = pd.DataFrame(rows)
        if not os.path.exists(OUTPUT):
            out_df.to_csv(OUTPUT, index=False, mode="w")
        else:
            out_df.to_csv(OUTPUT, index=False, mode="a", header=False)

    print(f"✅ Processed chunk {i+1}, rows written: {len(rows)}")

print("\n💾 Cleaned diagnoses saved to:", OUTPUT)
print("\n📊 Bucket survival counts:")
for b, c in bucket_counts.items():
    print(f"{b:20s}: {c}")