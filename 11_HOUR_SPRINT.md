# 11-HOUR SPRINT — TPU v5e-8, 12h Session
# Kaggle Specs: TPU v5e-8 | 30 GiB RAM | 57.6 GiB Disk | 12h session
# Start: ~17:30 Thu 2026-07-16 | End: ~04:30 Fri 2026-07-17
# Buffer: 1 hour before session expires

---

## What Changes vs. 3-Hour Sprint

| Dimension | 3-Hour Sprint | 11-Hour Sprint |
|-----------|---------------|----------------|
| Baselines | RF, XGBoost only | + CGCNN, MEGNet (4 total) |
| Seeds | 3 | 5 |
| Datasets | MP + JARVIS | + QMOF (3 sources) |
| Ablations | 1 | 4 (fidelity, depth, heads, loss) |
| Uncertainty | Spearman ρ only | + ECE, NLL, PICP, reliability diagrams |
| Figures | 3-4 rough | 6 publication-quality |
| Paper | Scaffold | Complete draft (~4000 words) |
| Export | Manual | Auto-zip to filebin |

## Honest Assessment (Updated)

With 11 hours and strong baselines, this is no longer a proof-of-concept.
It's a **serious first experimental package** with real evidence.

The sprint answers: **"Do I have evidence that justifies investing several more weeks?"**
It does NOT predict acceptance — that depends on novelty verification, reviewer assignment, editorial priorities, and factors unknowable at this stage.

Remaining gap after sprint: baseline verification, literature depth, writing polish, error analysis.

---

## HOUR-BY-HOUR SCHEDULE

### Hour 0:00-0:30 — Setup & Data Pipeline (30 min)

```
Cell 1:  pip install + imports + TPU verification
Cell 2:  API key from Kaggle Secrets
Cell 3:  Download Materials Project data (~30-50k materials)
Cell 4:  Download JARVIS-DFT data
Cell 5:  Download QMOF dataset (metal-organic frameworks)
Cell 6:  Feature extraction (all 3 datasets)
Cell 7:  Dataset construction, train/val/test split (80/10/10),
         StandardScaler fit on train only
Cell 8:  GATE A: Data verification (print stats, distributions)
```

### Hour 0:30-1:30 — Train MFT Transformer, 5 Seeds (60 min)

```
Cell 9:  Transformer model definition (Flax/JAX)
Cell 10: Loss function (Gaussian NLL) + optimizer (AdamW + cosine)
Cell 11: Training loop — 5 seeds × ~10 min each on TPU
Cell 12: GATE C: Training verification (convergence, no NaN)
```

### Hour 1:30-3:30 — Train Baselines (120 min)

```
Cell 13: Random Forest (5 seeds) — ~5 min total
Cell 14: XGBoost (5 seeds) — ~10 min total
Cell 15: CGCNN (5 seeds) — ~50 min total on TPU
Cell 16: MEGNet (5 seeds) — ~50 min total on TPU
Cell 17: GATE C: All baselines verified
```

### Hour 3:30-5:00 — Ablation Studies (90 min)

```
Cell 18: A1 — Multi-task vs single-task (3 seeds)
Cell 19: A2 — Fidelity embedding removal (3 seeds)
Cell 20: A3 — Architecture depth: 1/2/3/4 layers (3 seeds each)
Cell 21: A4 — Loss function: MSE vs Gaussian NLL (3 seeds each)
Cell 22: Ablation summary table
```

### Hour 5:00-6:00 — Evaluation & Metrics (60 min)

```
Cell 23: Full metrics: MAE, RMSE, R² per model per seed (mean±std)
Cell 24: Statistical tests: Wilcoxon signed-rank, Cohen's d, CIs
Cell 25: Uncertainty: ECE, NLL, PICP, MPIW
Cell 26: Uncertainty: Spearman ρ, reliability diagrams
Cell 27: GATE D: Evaluation verified — all numbers from actual runs
```

### Hour 6:00-7:30 — Figures (90 min)

```
Cell 28: Figure 1 — Parity plots (MFT vs best baseline, 2 targets)
Cell 29: Figure 2 — Baseline comparison bar chart (mean±std, all models)
Cell 30: Figure 3 — Uncertainty calibration (reliability diagrams)
Cell 31: Figure 4 — Ablation results (4 subplots)
Cell 32: Figure 5 — Spearman correlation (predicted σ vs actual error)
Cell 33: Figure 6 — Dataset overview (distribution + fidelity split)
```

### Hour 7:30-10:00 — Paper Writing (150 min)

```
Use paper_draft_template.md, fill in actual numbers.

Sections:
  Abstract         (200 words) — 15 min
  Introduction     (800 words) — 30 min
  Methods          (1000 words) — 25 min
  Results & Disc.  (1500 words) — 40 min
  Limitations      (200 words) — 10 min
  Future Persp.    (200 words) — 10 min [REQUIRED by Materials Futures]
  Conclusion       (300 words) — 10 min
  References       (verified) — 15 min
  Data statement + AI disclosure — 5 min
```

### Hour 10:00-10:30 — Consistency Check (30 min)

```
Cell 34: Re-run entire pipeline end-to-end (verify reproducibility)
Cell 35: Cross-check all paper numbers against actual outputs
Cell 36: GATE F: Paper consistency verified
```

### Hour 10:30-11:00 — Export & Upload (30 min)

```
Cell 37: Package everything into zip
         - figures/ (all PNG, 300 DPI)
         - results/ (CSV tables, JSON summary)
         - paper/ (markdown + LaTeX if converted)
         - code/ (notebook .ipynb export)
         - data/ (processed features, splits)
         - checkpoints/ (trained model weights)
         - README.md

Cell 38: Upload to filebin.net (or alternative)
         - Generate download link
         - Print link to notebook output
```

---

## CODE ARCHITECTURE

### Dependencies (Cell 1)
```python
!pip install -q mp-api jarvis-tools einops flax optax scikit-learn xgboost torch torch-geometric

import jax, jax.numpy as jnp, flax.linen as nn, optax
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb, time, json, os, zipfile
from scipy import stats
```

### Model Definitions Needed
1. **MultiFidelityTransformer** (JAX/Flax) — your primary model
2. **CGCNN** (PyTorch Geometric) — standard crystal GNN baseline
3. **MEGNet** (PyTorch or TF) — Materials Project group baseline
4. **RandomForestRegressor** (sklearn) — fast classical baseline
5. **XGBRegressor** (xgboost) — fast gradient boosting baseline

### Export Package Structure
```
materials_futures_sprint/
├── README.md                    # Project overview, how to reproduce
├── notebook.ipynb               # Full Kaggle notebook
├── figures/
│   ├── fig1_parity.png          # 300 DPI
│   ├── fig2_comparison.png
│   ├── fig3_calibration.png
│   ├── fig4_ablation.png
│   ├── fig5_spearman.png
│   └── fig6_dataset.png
├── results/
│   ├── main_results.csv         # All models × all seeds
│   ├── ablation_results.csv     # All ablations
│   ├── uncertainty_metrics.csv  # ECE, NLL, PICP, MPIW
│   └── statistical_tests.csv    # Wilcoxon, Cohen's d
├── paper/
│   └── draft.md                 # Complete paper draft
├── data/
│   ├── train_features.npy       # Processed features
│   ├── val_features.npy
│   ├── test_features.npy
│   ├── train_targets.npy
│   ├── val_targets.npy
│   ├── test_targets.npy
│   └── scaler.pkl               # Fitted StandardScaler
├── checkpoints/
│   ├── mft_seed42.pkl           # Best transformer checkpoint
│   ├── mft_seed123.pkl
│   ├── mft_seed456.pkl
│   ├── mft_seed789.pkl
│   ├── mft_seed1024.pkl
│   └── cgcnn_seed42.pkl         # Best CGCNN checkpoint
└── requirements.txt             # Pinned versions
```

### Upload Script (Cell 38)
```python
import zipfile, subprocess, os

# Create zip
zip_path = 'materials_futures_sprint.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('materials_futures_sprint'):
        for file in files:
            filepath = os.path.join(root, file)
            zf.write(filepath)

print(f"Package size: {os.path.getsize(zip_path) / 1e6:.1f} MB")

# Upload to filebin
result = subprocess.run(
    ['curl', '-s', '-T', zip_path, 'https://filebin.net/materials_sprint_2026/'],
    capture_output=True, text=True
)

# Alternative: if filebin fails, try transfer.sh
if result.returncode != 0:
    result = subprocess.run(
        ['curl', '--upload-file', zip_path, 'https://transfer.sh/materials_futures_sprint.zip'],
        capture_output=True, text=True
    )

print(f"Download URL: {result.stdout.strip()}")
print("Save this URL — the notebook session will expire!")
```

---

## RISK MITIGATION

| Risk | Probability | Mitigation |
|------|------------|-----------|
| TPU OOM with CGCNN | Medium | Use smaller batch size (64), gradient checkpointing |
| CGCNN code incompatible | Medium | Use pure-JAX fallback: GraphNet from scratch |
| MEGNet install fails | Medium | Skip, use 3 baselines (RF, XGBoost, CGCNN) |
| MP API rate limit | Low | Cache data, batch queries |
| Session timeout at 11h | Low | Save checkpoints every 30 min, 1h buffer |
| Disk full (57.6 GiB) | Low | Clean intermediate files, compress checkpoints |

## FALLBACK PLAN

If CGCNN/MEGNet installation fails on Kaggle:
1. Implement a simple crystal graph network in JAX/Flax (2-3 hours)
2. Or use MatGL's pretrained models if available
3. Or use SOAP descriptors + kernel ridge regression as a strong baseline
4. Paper still viable with 3 baselines (RF, XGBoost, simple-GNN)

---

## GO/NO-GO DECISION POINTS

### After Hour 1:30 (MFT trained)
Check: Does MFT converge? Is MAE reasonable?
- If NaN losses → debug architecture, check features
- If MAE > 2× RF → simplify model, check data pipeline

### After Hour 3:30 (Baselines trained)
Check: Does MFT beat any baseline?
- If MFT < all baselines → GO (strong result)
- If MFT ≈ baselines → GO (uncertainty is the differentiator)
- If MFT > baselines by >30% → investigate (possible bug)

### After Hour 6:00 (Evaluation done)
Check: Are results paper-worthy?
- If uncertainty correlates with error (ρ > 0.3) → GO
- If ablation shows multi-fidelity helps → GO
- If nothing works → write it up as negative result

---

## FINAL OUTPUT CHECKLIST

Before uploading, verify:

- [ ] All figures are 300 DPI, labeled, publication-quality
- [ ] All tables have mean ± std (not point estimates)
- [ ] Paper has Future Perspectives section (200 words, REQUIRED)
- [ ] Paper has AI-Contribution Disclosure
- [ ] Paper has Data Availability Statement
- [ ] No "first" claims in title or abstract
- [ ] Every number in paper traces to an actual run
- [ ] README.md explains how to reproduce
- [ ] requirements.txt has pinned versions
- [ ] Zip file is complete and under 500 MB
- [ ] Download URL is valid and tested
