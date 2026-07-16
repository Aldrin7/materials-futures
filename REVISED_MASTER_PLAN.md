# Revised Master Plan — Materials Futures Research Paper
# Incorporating pragmatic phasing and honest scope assessment

---

## Phase 1: Proof of Concept (11 Hours — TODAY)

### Objective
Answer ONE question: **Do I have experimental evidence that justifies
investing several more weeks into this research?**

If NO → saved weeks of work on a dead end.
If YES → proceed to Phase 2 (strengthen evidence + writing).

Note: This sprint does NOT predict acceptance. It predicts whether the
idea has enough promise to warrant further investment.

### Scope (Honest)
- 4 baselines (RF, XGBoost, CGCNN, simplified CGCNN)
- 5 seeds — enough for preliminary variance
- 4 ablations (multi-task, fidelity, depth, loss)
- Full uncertainty evaluation (ECE, NLL, PICP, Spearman)
- Complete draft with actual numbers

### Deliverables
- Working Kaggle notebook (all code runs clean)
- Results table (all models × all seeds)
- 6 publication-quality figures
- Complete paper draft (~4000 words)
- Everything zipped and uploaded

### Mandatory Checkpoints
Before writing the paper, verify:
1. **Baseline verification:** CGCNN reproduces published behavior on known data
2. **Dataset integrity:** No leakage between splits across MP+JARVIS+QMOF
3. **Compute accounting:** Training times, param counts, inference speed logged
4. **Statistical reporting:** Mean ± std, never highlight best run only
5. **Reproducibility:** Hyperparams, dataset versions, seeds all saved

---

## Phase 2: Build Reviewer Confidence (1-2 Weeks)

Only proceed if Phase 1 shows promising results.

### Priority Order (by impact on acceptance probability):

1. **Standard Benchmarks** (~1 day)
   - Run on MatBench (formation energy + band gap tasks)
   - Compare against MatBench leaderboard
   - This single step eliminates the #1 rejection risk

2. **Strong Baselines** (~2-3 days)
   - CGCNN (standard crystal GNN, ~2000 citations)
   - MEGNet (Materials Project group's method)
   - SchNet or DimeNet (geometric deep learning)
   - Same train/val/test splits, same seeds as your MFT

3. **Statistical Rigor** (~0.5 day)
   - 5 random seeds (42, 123, 456, 789, 1024)
   - Mean ± std on all metrics
   - Wilcoxon signed-rank test (non-parametric)
   - Cohen's d effect sizes
   - 95% confidence intervals

4. **Full Ablation Suite** (~2 days)
   - A1: Multi-task vs single-task (done in Phase 1)
   - A2: Fidelity embedding removal
   - A3: Architecture depth (1/2/3/4 layers)
   - A4: Attention heads (1/2/4/8)
   - A5: Loss function (MSE vs Gaussian NLL)
   - A6: Feature group importance

5. **Proper Uncertainty Evaluation** (~1 day)
   - ECE (Expected Calibration Error)
   - NLL (Negative Log-Likelihood)
   - Reliability diagrams
   - PICP (Prediction Interval Coverage Probability)
   - MPIW (Mean Prediction Interval Width)

6. **Error Analysis** (~1 day)
   - 10 worst predictions: why did they fail?
   - 10 best predictions: what makes them predictable?
   - Error vs. structure type / composition
   - Case studies with uncertainty context

---

## Phase 3: Paper Quality (1 Week)

### Writing
- Strong literature review (15-20 verified references)
- Clear positioning against prior work (no "first" claims)
- Honest Limitations section
- Future Perspectives (200 words, required by Materials Futures)

### Reproducibility
- GitHub repo with pinned requirements
- Zenodo archive of data + checkpoints
- Public Kaggle notebook

### Figures (Final)
- Figure 1: Parity plots (publication quality)
- Figure 2: Uncertainty calibration (reliability diagrams)
- Figure 3: Baseline comparison (mean ± std, all methods)
- Figure 4: Ablation results
- Figure 5: Error analysis / case studies

### Final Checklist
- [ ] All numbers from actual runs (Rule 1)
- [ ] All citations verified (Rule 2)
- [ ] Scope honestly declared (Rule 3)
- [ ] End-to-end re-run clean (Rule 4)
- [ ] Multi-seed variance reported (Rule 5)
- [ ] Dataset provenance stated (Rule 6)
- [ ] Anomalies diagnosed (Rule 7)
- [ ] AI-contribution disclosure (Rule 8)
- [ ] Cross-section consistency (Rule 9)

---

## Realistic Timeline

| When | What | Outcome |
|------|------|---------|
| **Today (3 hrs)** | Phase 1 | Go/no-go decision |
| **Week 1** | Strong baselines + MatBench + 5 seeds | Core results solid |
| **Week 2** | Ablations + uncertainty eval + error analysis | Evidence complete |
| **Week 3** | Writing + figures + reproducibility | Draft complete |
| **Week 4** | Internal review + polish + submit | Submission |

---

## Revised Claim Language

### ❌ Don't write:
"First transformer that jointly handles multi-fidelity crystal data
with calibrated uncertainty."

### ✅ Do write:
"We propose a transformer-based framework that jointly models
multi-fidelity crystal property prediction and predictive uncertainty."

Then let the literature review establish how the work differs from
prior methods. Reviewers position the novelty; you don't declare it.

---

## Decision Criteria (After Phase 1)

### GO (proceed to Phase 2) if:
- MFT converges without instability
- MFT is competitive with at least one baseline (within ~30%)
- Uncertainty shows any monotonic relationship with error (ρ > 0.2)
- No fundamental data leakage or pipeline bugs

### PIVOT if:
- MFT significantly worse than all baselines
- Training is unstable (NaN losses, no convergence)
- Consider: simpler architecture, different features, different task

### ABANDON if:
- Fundamental data issues (too few samples, too noisy)
- CGCNN baseline can't be verified against published numbers
- Idea has been done thoroughly (check literature first!)
