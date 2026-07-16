# Materials Futures — Multi-Fidelity Transformer

A reproducible ML research pipeline for crystal property prediction with predictive uncertainty, targeting [Materials Futures](https://iopscience.iop.org/journal/2752-5724) (IOP Publishing, Scopus-indexed, free APC).

## Quick Start

### Colab (recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aldrin7/materials-futures/blob/main/materials-futures-colab.ipynb)

1. Click the badge above
2. Runtime → **T4 GPU** or **v5e-1 TPU**
3. Run all cells
4. Enter your [Materials Project](https://materialsproject.org) API key when prompted

### Kaggle
[Open on Kaggle](https://www.kaggle.com/code/aldrinmanon/materials-futures)

1. Settings → Accelerator → **TPU v5e-8**
2. Add-ons → Secrets → Add `MP_API_KEY`
3. Run all cells

## What This Does

| Component | Details |
|-----------|---------|
| **Model** | Multi-Fidelity Transformer (JAX/Flax) |
| **Baselines** | Random Forest, XGBoost, CGCNN |
| **Data** | Materials Project + JARVIS-DFT + QMOF |
| **Targets** | Formation energy (eV/atom), Band gap (eV) |
| **Seeds** | 5 per model (42, 123, 456, 789, 1024) |
| **Ablations** | 4 (multi-task, fidelity, depth, loss) |
| **Uncertainty** | ECE, NLL, PICP, MPIW, Spearman ρ |
| **Figures** | 6 publication-quality (300 DPI) |

## Repository Structure

```
├── materials-futures-colab.ipynb    # Colab notebook (open this)
├── materials-futures-sprint.ipynb   # Kaggle notebook
├── kaggle_11h_cells.py             # Source code (34 cells)
├── smoke_test.py                   # 10-min pipeline validation
├── reproduce.py                    # Reproduce script
├── HYPOTHESES.md                   # Pre-registered hypotheses
├── REVISED_MASTER_PLAN.md          # 4-week post-sprint plan
├── 11_HOUR_SPRINT.md               # Hour-by-hour schedule
├── experiments/
│   └── v1/
│       ├── FREEZE_MANIFEST.json    # Results snapshot (SHA-256)
│       ├── reproduce.py            # Load, verify, regenerate
│       ├── CITATION.cff            # GitHub citation
│       ├── configs/                # Per-run configurations
│       ├── checkpoints/            # Model weights
│       ├── predictions/            # Per-run predictions
│       ├── metrics/                # JSON + CSV metrics
│       ├── figures/                # Publication figures
│       ├── figure_data/            # CSV data for figures
│       ├── tables/                 # LaTeX + Markdown tables
│       └── metadata/               # Environment, git, data manifest
└── materials_futures_sprint/
    ├── data/                       # Processed features
    ├── figures/                    # Generated figures
    ├── results/                    # Results tables
    └── paper/                      # Paper draft
```

## Reproducibility

```bash
python experiments/v1/reproduce.py --exp-version v1
```

## Pre-Registered Hypotheses

| Hypothesis | Criterion |
|------------|-----------|
| H1 (primary) | MFT beats baselines on at least one target, p < 0.05 |
| H2 | Fidelity embedding improves accuracy |
| H3 | Multi-task beats single-task |
| H4 | Uncertainty correlates with error (ρ > 0.3) |
| H5 | Low-uncertainty subset has 20%+ lower MAE |

## Target Journal

**Materials Futures** (IOP Publishing) — Scopus + ESCI, free APC, fast turnaround.

## License

MIT
