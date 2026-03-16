import os
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Paths ---
input_csv = config["phase6_input"]
output_dir = config["phase6_output_dir"]
os.makedirs(output_dir, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(input_csv)

# --- Drop identifiers / leakage columns ---
drop_cols = [
    "subject_id", "hadm_id", "admittime", "dischtime",
    "drg_codes", "drg_severity_raw", "drg_mortality_raw"
]
df = df.drop(columns=drop_cols, errors="ignore")

# --- Define input features ---
input_cols = [
    "insurance", "marital_status", "race", "gender", "anchor_age",
    "hospital_expire_flag",
    "n_drugs", "n_routes",
    "ACE_Inhibitors", "Beta_Blockers", "Statins",
    "Anticoagulants", "Diuretics", "Calcium_Channel_Blockers"
]

# --- Original 19 ICD buckets ---
orig_targets = [
    "Acute_MI", "Aortic_Disease", "Arrhythmia", "Arterial_Embolism",
    "Cardiomyopathy", "Cerebrovascular", "Congenital_HD", "Endocarditis",
    "Heart_Failure", "Hypertension", "Ischemic_HD", "Other_CVD",
    "Other_Vascular", "Pericardial", "Peripheral_Vascular",
    "Pulmonary_Embolism", "Pulmonary_Heart", "Valvular_HD", "Venous_Thrombo"
]

# --- Collapse mapping: 19 → 8 super-classes ---
collapse_map = {
    "Hypertension": "Hypertension",
    "Ischemic_HD": "IschemicHD",
    "Acute_MI": "IschemicHD",

    "Pulmonary_Heart": "PulmonaryHD",
    "Pulmonary_Embolism": "PulmonaryHD",

    "Heart_Failure": "HF_Cardiomyopathy",
    "Cardiomyopathy": "HF_Cardiomyopathy",

    "Arrhythmia": "Arrhythmias",

    "Cerebrovascular": "Cerebrovascular",

    "Aortic_Disease": "ArterialDiseases",
    "Arterial_Embolism": "ArterialDiseases",
    "Peripheral_Vascular": "ArterialDiseases",

    # Everything else → OtherCirculatory
    "Congenital_HD": "OtherCirculatory",
    "Endocarditis": "OtherCirculatory",
    "Other_CVD": "OtherCirculatory",
    "Other_Vascular": "OtherCirculatory",
    "Pericardial": "OtherCirculatory",
    "Valvular_HD": "OtherCirculatory",
    "Venous_Thrombo": "OtherCirculatory"
}

new_targets = [
    "Hypertension", "IschemicHD", "PulmonaryHD", "HF_Cardiomyopathy",
    "Arrhythmias", "Cerebrovascular", "ArterialDiseases", "OtherCirculatory"
]

# --- Clean and one-hot encode categorical columns ---
categorical_cols = ["insurance", "marital_status", "race", "gender"]

# Force to string, replace NaN, strip whitespace
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("UNKNOWN").str.strip()

df_inputs = df[input_cols].copy()
df_encoded = pd.get_dummies(df_inputs, columns=categorical_cols, drop_first=False)

# Force all columns to numeric (fixes object dtypes from messy categories)
df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

# --- Extract arrays ---
X = df_encoded.to_numpy(dtype=np.float32)
y_orig = df[orig_targets].to_numpy(dtype=np.int8)

# --- Collapse labels into 8 classes ---
y_new = np.zeros((y_orig.shape[0], len(new_targets)), dtype=np.int8)
new_class_index = {cls: i for i, cls in enumerate(new_targets)}

for old_idx, col in enumerate(orig_targets):
    new_label = collapse_map[col]
    new_idx = new_class_index[new_label]
    y_new[:, new_idx] |= y_orig[:, old_idx]

y = y_new
target_cols = new_targets

# --- Normalize inputs (z-score) ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save normalization parameters
np.save(os.path.join(output_dir, "mean.npy"), scaler.mean_)
np.save(os.path.join(output_dir, "scale.npy"), scaler.scale_)

# --- Label map (JSON) ---
label_map = {i: col for i, col in enumerate(target_cols)}
with open(os.path.join(output_dir, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ label_map.json written with", len(target_cols), "classes")

# --- Class distribution ---
class_counts = y.sum(axis=0)
class_distribution = {col: int(count) for col, count in zip(target_cols, class_counts)}

with open(os.path.join(output_dir, "class_distribution.txt"), "w") as f:
    for col, count in class_distribution.items():
        f.write(f"{col}: {count}\n")

# --- Class weights (inverse frequency) ---
total = class_counts.sum()
class_weights = {i: total / (len(target_cols) * count) if count > 0 else 0.0
                 for i, count in enumerate(class_counts)}
np.save(os.path.join(output_dir, "class_weights.npy"), class_weights)

# --- Stratified split ---
# Collapse multi-label y into a single dominant class index for stratification
y_strat = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_strat
)

# --- Shard function ---
def save_shards(X_data, y_data, split_name, shard_size=50000):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    num_shards = int(np.ceil(len(X_data) / shard_size))
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(X_data))
        np.save(os.path.join(split_dir, f"X_shard_{i}.npy"), X_data[start:end])
        np.save(os.path.join(split_dir, f"y_shard_{i}.npy"), y_data[start:end])
    print(f"✅ {split_name}: {num_shards} shards saved")

# --- Save shards ---
save_shards(X_train, y_train, "train")
save_shards(X_test, y_test, "test")

print("Class distribution:", class_distribution)
print("Class weights saved to class_weights.npy")