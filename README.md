# Materials Futures Research Sprint

## Multi-Fidelity Transformer for Crystal Property Prediction with Predictive Uncertainty

### Quick Stats
- **Compute:** Kaggle TPU v5e-8
- **Models:** MFT, Random Forest, XGBoost, CGCNN
- **Seeds:** 5 (42, 123, 456, 789, 1024)

### Reproduction

```bash
# 1. Verify file integrity
python reproduce.py

# 2. Or full re-run: open the Kaggle notebook and run all cells
```

### Package Contents

```
materials_futures_sprint/          # Deliverable package
├── README.md
├── figures/                       # 6 publication-quality figures (300 DPI)
├── results/                       # CSV tables + JSON summary
├── paper/                         # Complete draft
├── data/                          # Processed features + splits
├── checkpoints/                   # Model weights (MFT)
└── requirements.txt

experiments/v1/                    # Frozen experiment snapshot
├── FREEZE_MANIFEST.json           # SHA-256 hashes + timestamp
├── reproduce.py                   # Load, verify, regenerate
├── configs/
├── checkpoints/                   # All models (MFT, RF, XGBoost, CGCNN)
│   ├── mft_seed{42,123,456,789,1024}.pkl    # JAX params + metadata
│   ├── rf_seed{42,123,456,789,1024}.pkl     # sklearn models
│   ├── xgb_fe_seed{42,...}.json             # XGBoost formation energy
│   ├── xgb_bg_seed{42,...}.json             # XGBoost band gap
│   └── cgcnn_seed{42,...}.pkl              # CGCNN JAX params
├── predictions/                   # Per-run predictions (enables re-analysis)
│   ├── test_predictions_exp_*_mft_seed42.csv
│   ├── val_predictions_exp_*_mft_seed42.csv
│   └── ...
├── logs/
├── metrics/                       # All metrics + configs + checklist
│   ├── results_summary.json
│   ├── run_configs.json           # Per-run hyperparams + results
│   ├── reproducibility_checklist.json
│   └── uncertainty_metrics.csv
├── figures/                       # 6 PNG (300 DPI)
├── figure_data/                   # CSV data underlying each figure
│   ├── fig1_parity_data.csv
│   ├── fig2_comparison_data.csv
│   ├── fig3_calibration_data.csv
│   ├── fig4_ablation_data.csv
│   ├── fig5_spearman_data.csv
│   └── fig6_dataset_data.csv
├── tables/                        # LaTeX + Markdown + CSV tables
│   ├── results_table.tex
│   ├── results_table.md
│   └── main_results.csv
├── paper/
└── metadata/                      # Environment + git + data provenance
    ├── environment.txt            # pip freeze + platform info
    ├── requirements.txt           # Pinned deps only
    ├── git_commit.txt             # Code version
    └── data_manifest.json         # Dataset hashes + split definitions
```

### Model Loading

```python
# Transformer
import pickle
with open('experiments/v1/checkpoints/mft_seed42.pkl', 'rb') as f:
    ckpt = pickle.load(f)
params = ckpt['params']
# ckpt also contains: exp_id, model_architecture, param_count,
#                     jax_version, dtype, seed, train_time_s

# XGBoost
import xgboost as xgb
xgb_fe = xgb.XGBRegressor()
xgb_fe.load_model('experiments/v1/checkpoints/xgb_fe_seed42.json')

# Random Forest
with open('experiments/v1/checkpoints/rf_seed42.pkl', 'rb') as f:
    rf = pickle.load(f)
pred = rf['fe_model'].predict(X_new)

# Scaler (for new data)
with open('materials_futures_sprint/data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X_new)
```

### Requirements
```
mp-api>=0.40.0
jarvis-tools>=2024.3.20
flax>=0.8.0
optax>=0.2.0
einops>=0.7.0
scikit-learn>=1.3.0
xgboost>=2.0.0
jax>=0.4.20
scipy>=1.11.0
```
