# HYPOTHESES.md — Pre-Registered Research Hypotheses

**Written before any experiments run.** This distinguishes confirmatory
from exploratory analysis and keeps the manuscript honest.

**Date:** 2026-07-16
**Experiment:** v1

---

## Primary Hypothesis

**H1:** A transformer-based model that explicitly encodes fidelity level
will achieve lower MAE for crystal property prediction (formation energy,
band gap) than classical ML baselines (Random Forest, XGBoost) trained
on the same features, when evaluated across multiple random seeds.

**Success criterion:** MFT MAE is lower than RF and XGBoost MAE for at
least one target property (formation energy OR band gap), with the
difference significant at p < 0.05 (Wilcoxon signed-rank test across 5
seeds).

**If H1 fails:** The transformer architecture does not provide a meaningful
advantage for this feature representation. Consider whether the issue is
(a) features too simple for attention to exploit, (b) dataset too small,
or (c) architecture not suited to tabular crystal data.

---

## Secondary Hypotheses

**H2 (Multi-fidelity):** Incorporating fidelity-level information via a
learned embedding improves prediction accuracy compared to ignoring
fidelity levels.

**Success criterion:** MFT with fidelity embedding achieves lower MAE
than the same architecture without fidelity embedding (ablation A2).

**H3 (Multi-task):** Jointly predicting formation energy and band gap
yields better per-task accuracy than training separate single-task models.

**Success criterion:** Multi-task MFT achieves lower MAE than single-task
MFT for at least one target (ablation A1).

**H4 (Uncertainty calibration):** Predicted uncertainty from the Gaussian
NLL loss is monotonically correlated with actual prediction error.

**Success criterion:** Spearman ρ > 0.3 between predicted σ and |error|
for at least one target, with p < 0.05.

**H5 (Uncertainty utility):** Predicted uncertainty enables identification
of a high-confidence subset with meaningfully lower error than the full
test set.

**Success criterion:** The bottom quartile of predictions (by uncertainty)
has MAE at least 20% lower than the full test set MAE.

---

## Exploratory Analyses (not pre-registered)

The following are exploratory — findings will be reported but not treated
as confirmatory:

- Architecture depth ablation (1/2/3/4 layers)
- Loss function comparison (MSE vs Gaussian NLL)
- Performance on QMOF (different chemistry space)
- Error analysis by composition/structure type
- Feature importance analysis

---

## Planned Reporting

- All results: mean ± std across 5 seeds
- Statistical tests: Wilcoxon signed-rank (non-parametric)
- Effect sizes: Cohen's d where applicable
- Figures: parity plots, calibration, ablation, Spearman scatter
- Every number in the paper traces to an actual run
- Limitations section will honestly report what didn't work
