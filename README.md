# LUMINAE

**Latent‑Unified Multi‑Domain Integration for Cardio‑Networking Analysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete codebase for the paper:

> **A cost‑sensitive gradient boosting framework for multi‑label cardiovascular risk prediction using 12‑hour clinical data**  
> S. Refugio
> 
> *BMC Bioinformatics* (under review)

LUMINAE is a three‑module gradient‑boosting framework for cardiovascular risk prediction using the first 12 hours of admission data (the First Measured Physiological Vector, FMPV). It achieves state‑of‑the‑art diagnostic and prognostic accuracy while reducing carbon emissions by 4,000× compared to transformer‑based models.

---

## Repository Structure

> LUMINAE/ -> Data_Pipeline/:
> 
> Phase: I-III (Demographics, diagnoses, medication aggregation)
> 
> Phase: IV-V (Laboratory and vital sign integration)
> 
> Phase: VI-VIII (Data sharding for RiskNet, HematologyNet, ProgressionNet)
> 
> Phase: IX-XI (Temporal feature extraction (slopes, fractions, IQR)
> 
> Phase: XII (Final master dataset assembly)
> 
> LUMINAE/ -> models/:
> 
> RiskNetNEO.py: Pre‑diagnostic triage module
> 
> HematologyNet.py: Multi‑label diagnostic module
> 
> ProgressionNet.py: Prognostic module (HORIZON‑4, MORTALITY‑2)
> 
> requirements.txt
> 
> README.md
> 
> LICENSE

---
## Requirements

- Python 3.8+
- XGBoost 3.1.2
- LightGBM 4.6.0
- scikit‑learn 1.7.2
- pandas 2.3.3, numpy 2.3.5
- imbalanced‑learn 1.7.2
- matplotlib, seaborn
- shap 0.50.0
- joblib 1.5.2

Install all dependencies with:

```bash
pip install -r requirements.txt
```
---
## Data Access 

This project uses the MIMIC‑IV (v3.1) electronic health record database. You must obtain credentialed access from PhysioNet. The raw data cannot be redistributed.

All feature engineering scripts are provided in the data_pipeline/ directory. Trained models are available upon request.

---
## Reproducing Results 

To train a module, run: 

```bash
python models/HematologyNet.py --mode ensemble --sample 50000
```

Supported --mode values:

lightgbm – one‑vs‑rest LightGBM classifiers

xgboost_sota – one‑vs‑rest XGBoost with scale_pos_weight

ensemble – average of LGBM and XGBoost probabilities

Full experimental details are provided in the paper. The researcher shall grant access to the trained and evaluated model to credentialed MIMIC users.

---

## Citation

If you use this code or the LUMINAE framework in your research, please cite:

@article{refugio2026luminae,
  title={A cost-sensitive gradient boosting framework for multi-label cardiovascular risk prediction using 12-hour clinical data},
  author={Refugio, Seth},
  journal={BMC Bioinformatics},
  year={2026},
  note={under review}
}

---

## Contact 

Seth Refugio
Email: sethrefugio9@gmail.com
GitHub: @lumemix

For questions or collaboration, please open an issue on GitHub.


