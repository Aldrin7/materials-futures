# Post-Sprint Roadmap: 20-40% → 60-80% Acceptance Probability
## Materials Futures — Letter Format

---

## Priority 1: Stronger Baselines (Highest Impact, ~3 days)

Your current baselines (RF, XGBoost) are too weak. Reviewers will ask
"why not compare against methods designed for crystals?"

### Add These 3 Baselines:

| Baseline | What It Is | Why Reviewers Expect It | How Long |
|----------|-----------|------------------------|----------|
| **CGCNN** [1] | Crystal Graph CNN — the standard | Most cited crystal property predictor (~2000 citations) | 4-6 hours on Kaggle |
| **MEGNet** [2] | MatErials Graph Network | Standard benchmark from Materials Project group | 4-6 hours |
| **SchNet** [3] | Continuous-filter CNN for molecules/crystals | Established geometric deep learning baseline | 4-6 hours |

### Implementation Plan:
```
Day 1: CGCNN (PyTorch Geometric or original TF implementation)
       - Use their official code: https://github.com/txie-93/cgcnn
       - Train on same train/val/test split as your MFT
       - 3 seeds minimum

Day 2: MEGNet + SchNet
       - MEGNet: https://github.com/materialsvirtuallab/megnet
       - SchNet: https://github.com/atomistic-machine-learning/SchNet
       - Same splits, same seeds

Day 3: Run all baselines, verify metrics, generate comparison tables
```

### Why This Matters:
A paper comparing only against RF/XGBoost looks like you chose weak
baselines to make your method look good. CGCNN/MEGNet/SchNet are the
methods reviewers will benchmark against in their heads. If you don't
include them, they'll assume your method would lose to them.

---

## Priority 2: Multiple Datasets (~2 days)

Currently: Materials Project + JARVIS (essentially one merged source)

### Add 1-2 Independent Datasets:

| Dataset | What It Contains | Why It Helps |
|---------|-----------------|-------------|
| **QMOF** [4] | Metal-organic framework properties | Different chemistry space, tests generalization |
| **MatBench** [5] | Standardized materials ML benchmark | Enables direct comparison with published results |

### Implementation:
```
Day 4: Download QMOF dataset, extract features, run full pipeline
       - If MFT performs well on MOFs → strong generalization claim
       - If it doesn't → honest limitation, still publishable

Day 5: Run MatBench tasks (formation energy + band gap)
       - Compare against MatBench leaderboard entries
       - This is the single strongest benchmarking move
```

### Why This Matters:
Single-dataset papers are the #1 reason for desk rejection at
materials ML venues. MatBench exists specifically to prevent this
problem. Using it signals rigor.

---

## Priority 3: Full Ablation Suite (~2 days)

Currently: 1 ablation (multi-task vs single-task)

### Required Ablations:

| Ablation | What It Tests | How |
|----------|--------------|-----|
| **A1** ✅ Multi-task vs single-task | Does joint learning help? | DONE |
| **A2** Fidelity embedding removal | Does multi-fidelity encoding matter? | Remove fidelity embedding, retrain |
| **A3** Architecture depth (1/2/3/4 layers) | Is 3 layers optimal? | Train with N_layers = 1, 2, 3, 4 |
| **A4** Attention heads (1/2/4/8) | Does more heads help? | Train with num_heads = 1, 2, 4, 8 |
| **A5** Loss function (MSE vs Gaussian NLL) | Does uncertainty loss help accuracy? | Replace NLL with MSE, compare |
| **A6** Feature importance | Which features matter most? | Ablate feature groups one at a time |

### Implementation:
```
Day 6: A2 + A3 (fidelity ablation + architecture depth)
Day 7: A4 + A5 + A6 (heads, loss, features)
```

### Minimum for submission: A1 + A2 + A3 + A5
### Ideal: All 6

---

## Priority 4: Proper Uncertainty Evaluation (~1 day)

Currently: Only Spearman ρ

### Required Metrics:

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **ECE** (Expected Calibration Error) | Binned calibration | 10 bins, |accuracy - confidence| per bin |
| **NLL** (Negative Log-Likelihood) | Probabilistic quality | Mean Gaussian NLL on test set |
| **Reliability diagram** | Visual calibration | Plot predicted σ vs actual error per bin |
| **PICP** (Prediction Interval Coverage Probability) | Coverage of 95% CI | % of true values within μ ± 1.96σ |
| **MPIW** (Mean Prediction Interval Width) | Sharpness | Average width of 95% CI |

### Additional Analysis:
```
Day 8: Compute all metrics
       - Reliability diagram (Figure 2 upgrade)
       - ECE table across all models
       - PICP/MPIW analysis
       - Case studies: show 5 high-uncertainty predictions
         and 5 low-uncertainty predictions with actual errors
```

### Why This Matters:
The uncertainty claim is your paper's main novelty. A single Spearman
ρ is insufficient. Reviewers want to see that uncertainty is
*calibrated* (not just correlated), *sharp* (not just conservative),
and *practically useful* (case studies).

---

## Priority 5: Statistical Rigor (~0.5 day)

Currently: 3 seeds, basic t-test

### Upgrade to:
- **5 seeds minimum** (42, 123, 456, 789, 1024)
- **Wilcoxon signed-rank test** (non-parametric, safer than t-test)
- **Effect size** (Cohen's d) alongside p-values
- **Confidence intervals** on all reported metrics

### Implementation:
```
Day 8 afternoon: Rerun all models with 5 seeds
       - This alone takes ~2-3 hours on Kaggle
       - Saves directly to the results table
```

---

## Priority 6: Error Analysis & Case Studies (~1 day)

Currently: None

### What to Add:
```
Day 9:
1. Identify the 10 worst predictions (largest errors)
   - What materials are they? (composition, structure type)
   - Why did the model fail? (out-of-distribution? unusual bonding?)
   - Is the uncertainty high for these? (if yes → model knows it's uncertain)

2. Identify the 10 best predictions with lowest uncertainty
   - What makes these predictable?
   - Are they from well-represented structure types?

3. Error distribution analysis
   - Plot error vs. each input feature
   - Are there systematic biases? (e.g., overpredicts for oxides?)

4. Figure 5: Error analysis visualization
   - 2D scatter: error vs predicted uncertainty, colored by structure type
   - Shows practical utility of uncertainty
```

---

## Priority 7: Literature & Writing Polish (~2 days)

Currently: Template with placeholder references

### Literature Review Expansion:
```
Day 10-11:
1. Add 15-20 verified references (each confirmed via search)
   - 5 on crystal property prediction (CGCNN, MEGNet, SchNet, etc.)
   - 5 on multi-fidelity methods in materials science
   - 5 on uncertainty quantification in ML for science

2. Rewrite Introduction to position against specific prior work
   - "Unlike CGCNN [1], which treats all DFT data as equivalent..."
   - "Compared to the multi-fidelity approach of [X], our method..."

3. Strengthen Discussion
   - Compare your results directly to published numbers on same datasets
   - Address each limitation honestly but constructively

4. Future Perspectives (200 words, REQUIRED by Materials Futures)
   - Already in template, but refine based on actual results
```

---

## Priority 8: Reproducibility Package (~0.5 day)

### Create:
```
Day 12:
1. GitHub repository with:
   - requirements.txt (pinned versions)
   - README.md with setup instructions
   - run_all.sh (single command to reproduce)
   - configs/ (all hyperparameters)
   - Data download scripts

2. Zenodo archive of:
   - Processed dataset (features + splits)
   - Trained model checkpoints
   - Results tables (CSV)

3. Kaggle notebook set to public
```

---

## COMPLETE TIMELINE

| Day | Work | Impact |
|-----|------|--------|
| **1-3** | Strong baselines (CGCNN, MEGNet, SchNet) | HIGH — eliminates #1 rejection risk |
| **4-5** | Additional datasets (QMOF or MatBench) | HIGH — eliminates #2 rejection risk |
| **6-7** | Full ablation suite (A2-A6) | MEDIUM-HIGH — addresses #3 risk |
| **8** | Uncertainty metrics (ECE, NLL, PICP) + 5 seeds | HIGH — strengthens main novelty claim |
| **9** | Error analysis & case studies | MEDIUM — adds depth |
| **10-11** | Literature + writing polish | MEDIUM — professional presentation |
| **12** | Reproducibility package | LOW-MEDIUM — but expected by reviewers |
| **13-14** | Buffer for reruns, fixes, final proofread | SAFETY NET |

---

## REVISED ESTIMATES

| Version | Acceptance Probability |
|---------|----------------------|
| 3-hour sprint only | 20-40% |
| + Strong baselines | 35-55% |
| + Multiple datasets | 45-65% |
| + Full ablations | 55-70% |
| + Proper uncertainty eval | 60-75% |
| + All of the above + polished writing | 65-80% |
| + Genuine SOTA performance with statistical evidence | 70-90% |

---

## THE SINGLE MOST IMPACTFUL THING

If you can only do ONE thing after the sprint:

**→ Run MatBench benchmarks and compare against the leaderboard.**

This single action:
- Proves your method works on a standardized benchmark
- Enables direct comparison with 10+ published methods
- Shows you're serious about fair evaluation
- Takes ~1 day of work

It's the difference between "we compared against Random Forest"
and "we benchmarked against the MatBench state-of-the-art."

---

## What I Can Help With

After the 3-hour sprint, I can:
1. Write the CGCNN/MEGNet/SchNet training code for Kaggle
2. Implement all uncertainty metrics (ECE, NLL, PICP, MPIW)
3. Generate the ablation code for A2-A6
4. Draft the expanded literature review with verified references
5. Write the error analysis code
6. Format the final paper

Just ask after you complete the sprint.
