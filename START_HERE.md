# 🚀 START HERE — 11-Hour Research Sprint

## Your Kaggle Specs
- **TPU v5e-8** | 30 GiB RAM | 57.6 GiB Disk | **12h session**
- **Buffer:** 1 hour before expiry

## What You Get After 11 Hours
- ✅ Complete working code on Kaggle TPU (31 cells)
- ✅ 4 models trained × 5 seeds = 20 runs
- ✅ 4 ablation studies
- ✅ 6 publication-quality figures (300 DPI)
- ✅ Full uncertainty evaluation (ECE, NLL, PICP, Spearman)
- ✅ Complete paper draft (~4000 words) with **actual numbers**
- ✅ All models saved for re-use (MFT, RF, XGBoost, CGCNN)
- ✅ Results freeze with file hashes
- ✅ Everything zipped and uploaded

## Artifacts Produced

| Artifact | Location | Purpose |
|----------|----------|--------|
| `run_configs.json` | experiments/v1/metrics/ | Per-run hyperparams + results |
| `reproducibility_checklist.json` | experiments/v1/metrics/ | Explicit rigor verification |
| `results_summary.json` | experiments/v1/metrics/ | Aggregate results |
| `main_results.csv` | experiments/v1/tables/ | All models × seeds, machine-readable |
| `ablation_results.csv` | experiments/v1/tables/ | All ablation results |
| `uncertainty_metrics.csv` | experiments/v1/tables/ | ECE, NLL, PICP, MPIW, Spearman |
| `results_table.tex` | experiments/v1/tables/ | LaTeX table for paper |
| `results_table.md` | experiments/v1/tables/ | Markdown table for draft |
| `environment.txt` | experiments/v1/metadata/ | pip freeze |
| `requirements.txt` | experiments/v1/metadata/ | Pinned deps |
| `git_commit.txt` | experiments/v1/metadata/ | Code version |
| `data_manifest.json` | experiments/v1/metadata/ | Dataset hashes + splits |
| `FREEZE_MANIFEST.json` | experiments/v1/ | Results freeze timestamp + hashes |
| `mft_seed*.pkl` | experiments/v1/checkpoints/ | Transformer weights (5 seeds) |
| `rf_seed*.pkl` | experiments/v1/checkpoints/ | Random Forest models (5 seeds) |
| `xgb_*_seed*.json` | experiments/v1/checkpoints/ | XGBoost models (5 seeds × 2 targets) |
| `cgcnn_seed*.pkl` | experiments/v1/checkpoints/ | CGCNN models (5 seeds) |

## Files I've Prepared

| File | What It Is |
|------|-----------|
| `START_HERE.md` | This file — read first |
| `REVISED_MASTER_PLAN.md` | Full 4-week plan (post-sprint) |
| `11_HOUR_SPRINT.md` | Hour-by-hour schedule |
| `kaggle_11h_cells.py` | **Complete executable code — 30 cells** |
| `paper_draft_template.md` | Paper scaffold (for reference) |

## Step-by-Step

### Minute 0-5: CRITICAL — Revoke Exposed Token
1. Go to https://www.kaggle.com/settings → API → Delete exposed token
2. Generate new token
3. Get Materials Project API key from https://materialsproject.org

### Minute 5-10: Kaggle Setup
1. Open: https://www.kaggle.com/code/aldrinmanon/notebook73e75c8cf6
2. Settings: **TPU v5e-8**, Python, Internet ON
3. Add-ons → Secrets → Add `MP_API_KEY`

### Hour 0:00-0:30: Data Pipeline (Cells 1-8)
- Install deps, download MP + JARVIS + QMOF
- Feature extraction, train/val/test split
- Gate A verification

### Hour 0:30-1:30: Train Transformer (Cells 9-11)
- Multi-Fidelity Transformer, 5 seeds
- Checkpoints saved after each seed

### Hour 1:30-3:30: Train Baselines (Cells 12-14)
- Random Forest (5 seeds) — fast
- XGBoost (5 seeds) — fast
- CGCNN (5 seeds) — ~50 min on TPU

### Hour 3:30-5:00: Ablations (Cells 19-22)
- A1: Multi-task vs single-task
- A2: Fidelity embedding removal
- A3: Architecture depth (1/2/3/4 layers)
- A4: Loss function (NLL vs MSE)

### Hour 5:00-6:00: Evaluation (Cells 15, 25)
- All metrics (MAE, RMSE, R²)
- Uncertainty (ECE, NLL, PICP, Spearman)
- Statistical tests (Wilcoxon)

### Hour 6:00-7:30: Figures (Cells 8, 16-18, 23-24)
- 6 figures, all 300 DPI

### Hour 7:30-10:00: Paper (Cell 27)
- Auto-generated with actual numbers
- Review and edit

### Hour 10:00-10:30: Consistency Check
- Verify all paper numbers match actual runs

### Hour 10:30-11:00: Export & Upload (Cell 30)
- Zip everything
- Upload to filebin/transfer.sh/0x0.st
- **Save the download URL!**

## Go/No-Go Decision (After Hour 1:30)

| Signal | Action |
|--------|--------|
| MFT converges, MAE within 2× of RF | ✅ GO |
| MFT beats RF on at least one metric | ✅ GO (strong) |
| NaN losses, no convergence | 🔧 Debug architecture |
| MAE > 3× RF | 🔧 Check data pipeline |

## What This Won't Be

- ❌ Camera-ready final (needs human review + 2-3 more weeks)
- ❌ Exhaustive literature (needs 15+ more verified refs)
- ❌ Strongest possible baselines (needs CGCNN via PyG, MEGNet, SchNet)
- ❌ A prediction of whether the paper will be accepted

## The Real Question

The sprint answers:

> **"Do I have experimental evidence that justifies investing several more weeks?"**

If YES → proceed to Phase 2 (strengthen evidence + writing)
If NO → saved weeks of work on a dead end

## After the Sprint

See `REVISED_MASTER_PLAN.md` for the 4-week plan.

## Mandatory Checkpoints (Before Writing the Paper)

| Checkpoint | Why It Matters |
|-----------|---------------|
| Baseline verification | JAX CGCNN must reproduce published behavior on a known dataset |
| Dataset integrity | No leakage between train/val/test, especially across MP+JARVIS+QMOF |
| Compute accounting | Record training times, parameter counts, inference speed |
| Statistical reporting | Mean ± std across seeds, never highlight only best run |
| Reproducibility | Save hyperparams, dataset versions, random seeds for every run |
