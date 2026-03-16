import os
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Paths ---
input_csv = config["phase7_input"]
output_dir = config["phase7_output_dir"]
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
    "n_drugs", "n_routes",
    "ACE_Inhibitors", "Beta_Blockers", "Statins",
    "Anticoagulants", "Diuretics", "Calcium_Channel_Blockers"
]

# --- Define ICD-only targets ---
target_cols = [
    "Acute_MI", "Aortic_Disease", "Arrhythmia", "Arterial_Embolism",
    "Cardiomyopathy", "Cerebrovascular", "Congenital_HD", "Endocarditis",
    "Heart_Failure", "Hypertension", "Ischemic_HD", "Other_CVD",
    "Other_Vascular", "Pericardial", "Peripheral_Vascular",
    "Pulmonary_Embolism", "Pulmonary_Heart", "Valvular_HD", "Venous_Thrombo"
]

# --- One-hot encode categorical columns ---
categorical_cols = ["insurance", "marital_status", "race", "gender"]
df_inputs = df[input_cols].copy()
df_encoded = pd.get_dummies(df_inputs, columns=categorical_cols, drop_first=False)

# --- Safety check: ensure all features are numeric ---
non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("⚠️ Non-numeric columns found, dropping:", non_numeric)
    df_encoded = df_encoded.drop(columns=non_numeric)

# --- Extract arrays ---
X = df_encoded.to_numpy(dtype=np.float32)
y = df[target_cols].to_numpy(dtype=np.int8)

# --- Normalize inputs (z-score) ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save normalization parameters
np.save(os.path.join(output_dir, "mean.npy"), scaler.mean_)
np.save(os.path.join(output_dir, "scale.npy"), scaler.scale_)

# --- Label map ---
label_map = {i: col for i, col in enumerate(target_cols)}
with open(os.path.join(output_dir, "label_map.txt"), "w") as f:
    for i, col in label_map.items():
        f.write(f"{i}: {col}\n")

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

# --- Stratified split (dominant ICD) ---
y_strat = np.argmax(y, axis=1)

X_train, X_val, y_train, y_val = train_test_split(
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
save_shards(X_train, y_train, "Training")
save_shards(X_val, y_val, "Validation")

print("Class distribution:", class_distribution)
print("Class weights saved to class_weights.npy")