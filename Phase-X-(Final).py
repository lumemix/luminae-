import pandas as pd
import numpy as np
import os
import yaml
from scipy.stats import iqr
from sklearn.linear_model import LinearRegression

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------
# Paths
# -------------------------
d_items_path = config["phase11_d_items"]
chartevents_path = config["phase11_chartevents"]
d_labitems_path = config["phase11_d_labitems"]
labevents_path = config["phase11_labevents"]
output_dir = config["phase11_output_dir"]
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def compute_slope(values):
    if len(values) < 2:
        return np.nan
    try:
        X = np.arange(len(values)).reshape(-1,1)
        y = np.array(values)
        model = LinearRegression().fit(X, y)
        return model.coef_[0]
    except:
        return np.nan

def pseudo_lagged_delta(values):
    if len(values) < 4:
        return np.nan
    q1 = int(len(values) * 0.25)
    q3 = int(len(values) * 0.75)
    early = np.mean(values[:q1]) if q1 > 0 else values[0]
    late = np.mean(values[q3:]) if q3 < len(values) else values[-1]
    return late - early

def abnormal_fraction(values, low=None, high=None):
    values = np.array(values)
    if len(values) == 0:
        return np.nan
    mask = np.ones_like(values, dtype=bool)
    if low is not None:
        mask &= values < low
    if high is not None:
        mask &= values > high
    return mask.sum() / len(values)

def engineer_group(values, prefix, label):
    slope_val = compute_slope(values)
    lagged_delta = pseudo_lagged_delta(values)
    feats = {
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_iqr": iqr(values) if len(values) > 1 else 0,
        f"{prefix}_netchange": values[-1] - values[0],
        f"{prefix}_slope": slope_val,
        f"{prefix}_trend": np.sign(slope_val) if not np.isnan(slope_val) else 0,
        f"{prefix}_lagged_delta": lagged_delta
    }
    # Threshold-based abnormal fractions
    if "SpO2" in label:
        feats[f"{prefix}_time_below_90"] = abnormal_fraction(values, low=90)
    if "Temperature" in label:
        feats[f"{prefix}_time_above_38_5"] = abnormal_fraction(values, high=38.5)
    if "Heart Rate" in label:
        feats[f"{prefix}_time_above_120"] = abnormal_fraction(values, high=120)
    if "Blood Pressure systolic" in label:
        feats[f"{prefix}_time_below_90"] = abnormal_fraction(values, low=90)
    if "Glucose" in label:
        feats[f"{prefix}_time_above_180"] = abnormal_fraction(values, high=180)
        feats[f"{prefix}_time_below_70"] = abnormal_fraction(values, low=70)
    return feats

# -------------------------
# 1. Load dictionaries
# -------------------------
d_items = pd.read_csv(d_items_path)
d_labitems = pd.read_csv(d_labitems_path)

cvd_vitals = {"Heart Rate","Non Invasive Blood Pressure systolic","Respiratory Rate","Temperature Celsius","SpO2"}
cvd_labs = {"Hemoglobin","Hematocrit","Platelet Count","WBC","Creatinine","Sodium","Potassium","Chloride","Magnesium","Glucose","Lactate","PT","INR","Troponin","BNP","NT-proBNP","BUN"}

vital_items = d_items[d_items["label"].isin(cvd_vitals)]
lab_items = d_labitems[d_labitems["label"].isin(cvd_labs)]

# -------------------------
# 2. Stream chartevents → write per chunk
# -------------------------
vital_out_path = os.path.join(output_dir,"temporal_vitals_features.csv")
first_write = True
chunk_id = 0

for chunk in pd.read_csv(chartevents_path, chunksize=500000):
    chunk_id += 1
    chunk = chunk[chunk["itemid"].isin(vital_items["itemid"])]
    if chunk.empty:
        print(f"Vitals chunk {chunk_id}: no rows")
        continue

    rows = []
    for (sid, hadm, itemid), group in chunk.groupby(["subject_id","hadm_id","itemid"]):
        values = group["valuenum"].dropna().tolist()
        if not values: continue
        label = vital_items.loc[vital_items["itemid"]==itemid,"label"].values[0]
        feats = engineer_group(values,"vital",label)
        feats.update({"subject_id":sid,"hadm_id":hadm,"label":label})
        rows.append(feats)

    if rows:
        pd.DataFrame(rows).to_csv(vital_out_path, mode="a", index=False, header=first_write)
        first_write = False
        print(f"Vitals chunk {chunk_id}: wrote {len(rows)} rows")

# -------------------------
# 3. Stream labevents → write per chunk
# -------------------------
lab_out_path = os.path.join(output_dir,"temporal_labs_features.csv")
first_write = True
chunk_id = 0

for chunk in pd.read_csv(labevents_path, chunksize=500000):
    chunk_id += 1
    chunk = chunk[chunk["itemid"].isin(lab_items["itemid"])]
    if chunk.empty:
        print(f"Labs chunk {chunk_id}: no rows")
        continue

    rows = []
    for (sid, hadm, itemid), group in chunk.groupby(["subject_id","hadm_id","itemid"]):
        values = group["valuenum"].dropna().tolist()
        if not values: continue
        label = lab_items.loc[lab_items["itemid"]==itemid,"label"].values[0]
        feats = engineer_group(values,"lab",label)
        feats.update({"subject_id":sid,"hadm_id":hadm,"label":label})
        rows.append(feats)

    if rows:
        pd.DataFrame(rows).to_csv(lab_out_path, mode="a", index=False, header=first_write)
        first_write = False
        print(f"Labs chunk {chunk_id}: wrote {len(rows)} rows")

print("✅ Finished streaming and writing labs + vitals features")