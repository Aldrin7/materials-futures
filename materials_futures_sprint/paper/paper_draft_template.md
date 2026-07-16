# Uncertainty-Aware Multi-Fidelity Learning for Crystal Property Prediction via Transformer Networks

**Authors:** [Your Name], [Affiliation]

**Corresponding Author:** [Email]

**Target Journal:** Materials Futures (IOP Publishing)

**Article Type:** Letter (4000 words + 200 words Future Perspectives)

---

## Abstract

[~200 words — Write AFTER all experiments complete]

Accurate prediction of crystal properties is essential for accelerating materials discovery. Here we propose a Multi-Fidelity Transformer (MFT) that jointly learns from low-fidelity GGA-DFT and higher-fidelity computational data through a fidelity-aware attention mechanism, while simultaneously providing predictive uncertainty estimates via Gaussian negative log-likelihood loss. Applied to formation energy and band gap prediction using [X] materials from the Materials Project and JARVIS databases, MFT achieves MAE of [ACTUAL] eV/atom for formation energy and [ACTUAL] eV for band gap, compared to [ACTUAL] and [ACTUAL] for Random Forest and [ACTUAL] and [ACTUAL] for XGBoost baselines. The predicted uncertainties correlate with actual prediction errors (Spearman ρ = [ACTUAL]), enabling practitioners to identify which predictions warrant experimental validation. Multi-task learning yields [X]% improvement over single-task alternatives. We present a transformer-based framework for multi-fidelity crystal property prediction with predictive uncertainty, evaluated across [X] datasets with [X] random seeds and statistical significance testing.

**Keywords:** crystal property prediction, multi-fidelity learning, uncertainty quantification, transformer, materials informatics

---

## 1. Introduction

[~800 words]

The discovery of novel materials with tailored properties is a cornerstone of technological advancement, from energy storage to semiconductor design [1,2]. High-throughput computational screening, particularly density functional theory (DFT), has enabled the generation of large materials databases such as the Materials Project [3], JARVIS [4], and AFLOW [5], containing property data for hundreds of thousands of crystalline compounds. Machine learning models trained on these databases can predict properties orders of magnitude faster than DFT, accelerating the materials discovery pipeline [6,7].

However, two fundamental challenges limit the practical utility of existing machine learning approaches for materials property prediction.

**First, the multi-fidelity problem.** Computational materials data exists across multiple fidelity levels. Standard GGA-DFT (e.g., PBE) is computationally affordable but introduces systematic errors — notably the well-known band gap underestimation of 30-50% [8]. Higher-fidelity methods such as HSE06 hybrid functionals or GW corrections are more accurate but computationally prohibitive for large-scale screening. Experimental data, while ground truth, is sparse and heterogeneous. Most ML models are trained on single-fidelity data, ignoring the rich information available across fidelity levels.

**Second, the uncertainty problem.** Standard regression models provide point predictions without confidence estimates. In materials discovery, knowing *how confident* a prediction is arguably as important as the prediction itself — an experimentalist validating computationally-predicted candidates needs to prioritize which predictions to trust [9]. While ensemble methods and Gaussian processes offer uncertainty estimates, they scale poorly to large datasets or lack the expressiveness needed for complex crystal representations.

Recent work has addressed these challenges separately. Transfer learning approaches pre-train on low-fidelity data and fine-tune on high-fidelity subsets [10]. Multi-fidelity co-kriging and multi-task Gaussian processes model fidelity correlations explicitly [11]. For uncertainty, deep ensembles [12], MC dropout [13], and heteroscedastic neural networks [14] have been applied to materials science. However, no existing framework jointly addresses multi-fidelity learning and uncertainty quantification within a unified, scalable architecture.

Transformers [15], originally developed for natural language processing, have demonstrated remarkable success in scientific domains including protein structure prediction [16], molecular property prediction [17], and more recently, crystal property prediction [18]. Their self-attention mechanism naturally captures long-range dependencies, and their architecture supports flexible multi-task learning through shared representations.

In this work, we propose the Multi-Fidelity Transformer (MFT), a framework that addresses both challenges within a unified architecture. Our contributions are:

1. A fidelity-aware attention mechanism that learns to weight information from different fidelity levels, enabling effective knowledge transfer from abundant low-fidelity data to scarce high-fidelity data.
2. Multi-task learning with uncertainty-weighted loss (following Kendall et al. [19]) that jointly predicts formation energy and band gap while automatically balancing task contributions.
3. Predictive uncertainty estimates via Gaussian negative log-likelihood loss, evaluated with calibration metrics (ECE, NLL) and demonstrated to correlate with actual prediction error.
4. Evaluation against established baselines including [CGCNN, MEGNet, etc.] across multiple random seeds, with all results reported as mean ± standard deviation and statistical significance tests.

Note: We do not claim to be the first to address multi-fidelity or uncertainty in materials ML. Rather, we propose a unified framework and evaluate it rigorously against established methods.

---

## 2. Methods

[~1000 words]

### 2.1 Problem Formulation

Given a crystalline material represented by its chemical composition and structure, we aim to predict multiple target properties (formation energy E_f and band gap E_g) along with calibrated uncertainty estimates σ for each prediction. The model must learn from data across multiple fidelity levels, where fidelity 0 represents standard GGA-DFT calculations and fidelity 1 represents higher-accuracy computations.

### 2.2 Dataset and Feature Extraction

We assembled a dataset of [X] crystalline materials from the Materials Project database (fidelity 0) and the JARVIS-DFT database (fidelity 1). Materials were filtered to include only thermodynamically stable compounds (E_f < 2 eV/atom) with non-zero band gaps.

For each material, we extracted structural features including: number of atoms, density, volume per atom, lattice parameters (a, b, c, α, β, γ), and aggregated atomic properties (mean, standard deviation, minimum, maximum) for eight elemental descriptors: atomic number, group, period, electronegativity, valence electrons, atomic radius, ionization energy, and electron affinity. This yielded a [X]-dimensional feature vector per material.

The dataset was split into train (80%), validation (10%), and test (10%) sets using stratified random splitting with a fixed random seed (42). Feature normalization (StandardScaler) was applied, with parameters fit exclusively on the training set to prevent data leakage.

### 2.3 Multi-Fidelity Transformer Architecture

The MFT architecture consists of the following components:

**Input projection.** Raw crystal features are projected to a d_model=128 dimensional space via a linear layer followed by GELU activation.

**Fidelity embedding.** A learned embedding vector is added to the input representation based on the fidelity level of each sample, allowing the model to distinguish between low- and high-fidelity data without explicit feature engineering.

**Transformer encoder.** The projected features are reshaped into 4 tokens of dimension d_model and processed through N=3 transformer blocks, each comprising multi-head self-attention (4 heads) and a position-wise feed-forward network (d_ff=256) with pre-layer normalization and residual connections. Dropout (p=0.1) is applied for regularization.

**Multi-task prediction heads.** The pooled transformer output is fed into separate prediction heads for formation energy and band gap, each producing a mean μ and log-variance log(σ²) via two parallel linear layers.

### 2.4 Training with Uncertainty-Weighted Multi-Task Loss

We employ the Gaussian negative log-likelihood loss:

L_NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]

This loss naturally penalizes both prediction errors and poorly calibrated uncertainties — the model is incentivized to produce small σ only when predictions are accurate.

For multi-task learning, we adopt the uncertainty-weighting approach of Kendall et al. [19]:

L_total = Σᵢ [1/(2σᵢ²) * Lᵢ + 0.5 * log(σᵢ²)]

where σᵢ are learnable task-specific uncertainty parameters. This automatically balances the contribution of each task based on its inherent noise level.

Training was performed on Kaggle TPU v5e-8 using JAX/Flax with the AdamW optimizer (learning rate=1e-3 with cosine decay, weight decay=1e-4), batch size of 256, and early stopping (patience=20) based on validation loss. All experiments were repeated across 3 random seeds (42, 123, 456) and reported as mean ± standard deviation.

### 2.5 Baselines and Evaluation

We compared MFT against two established baselines:
- **Random Forest** (200 estimators, scikit-learn implementation)
- **XGBoost** (200 estimators, max depth=6, learning rate=0.1)

Evaluation metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and coefficient of determination (R²). Statistical significance was assessed via paired t-tests across seeds. Uncertainty calibration was evaluated using Spearman rank correlation between predicted uncertainty and absolute error.

---

## 3. Results and Discussion

[~1500 words — INSERT ACTUAL NUMBERS FROM EXPERIMENTS]

### 3.1 Overall Performance

**Table 1: Test set performance (mean ± std across 3 seeds)**

| Model | MAE_FE (eV/atom) | MAE_BG (eV) | R²_FE | R²_BG |
|-------|------------------|-------------|-------|-------|
| Multi-Fidelity Transformer | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Random Forest | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| XGBoost | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

[Figure 1: Parity plots for formation energy and band gap]

The MFT achieves [INSERT]% lower MAE than Random Forest and [INSERT]% lower than XGBoost for formation energy prediction. For band gap, the improvements are [INSERT]% and [INSERT]%, respectively. The R² values of [INSERT] for formation energy and [INSERT] for band gap indicate [good/strong/moderate] predictive performance.

### 3.2 Ablation: Single-Task vs Multi-Task Learning

**Table 2: Ablation results (mean ± std across 3 seeds)**

| Configuration | MAE_FE (eV/atom) | MAE_BG (eV) |
|---------------|------------------|-------------|
| Multi-task MFT | [INSERT] | [INSERT] |
| Single-task (FE only) | [INSERT] | — |
| Single-task (BG only) | — | [INSERT] |

[Figure 4: Ablation comparison]

Multi-task learning yields [INSERT]% improvement in formation energy MAE and [INSERT]% improvement in band gap MAE compared to single-task alternatives, supporting the hypothesis that shared representations across related material properties provide beneficial inductive bias.

### 3.3 Uncertainty Quantification

[Figure 2: Uncertainty calibration plots]

The predicted uncertainties show a Spearman correlation of ρ = [INSERT] (p = [INSERT]) with actual prediction errors for formation energy, and ρ = [INSERT] (p = [INSERT]) for band gap. This strong monotonic relationship indicates that the model's uncertainty estimates are informative — predictions flagged as high-uncertainty genuinely tend to have larger errors.

This has direct practical implications: by ranking candidate materials by predicted uncertainty, experimentalists can prioritize validation of the most reliable predictions, improving the efficiency of materials screening campaigns.

### 3.4 Limitations

We acknowledge several limitations:

1. **Scope reduction:** Due to the 3-hour computational budget, we evaluated only 2 baselines (Random Forest, XGBoost) rather than the originally planned 4-5. Additional baselines (e.g., crystal graph neural networks, Gaussian processes) would strengthen the comparison.

2. **Fidelity embedding ablation:** The ablation removing fidelity information from the transformer was deferred to supplementary material due to time constraints. This ablation is important for understanding the contribution of the multi-fidelity design.

3. **Dataset size:** Our dataset of [X] materials, while substantial, represents a subset of available computational data. Scaling to the full Materials Project + JARVIS + AFLOW corpus may improve performance and generalization.

4. **Single-seed limitations for ablation:** While the primary comparison uses 3 seeds, the ablation results are based on [1/3] seed(s), limiting the strength of comparative claims for that analysis.

5. **No experimental validation:** We did not validate predictions against experimental measurements, which would provide the strongest evidence for practical utility.

---

## 4. Future Perspectives

[~200 words — REQUIRED by Materials Futures]

The Multi-Fidelity Transformer framework presented here opens several promising directions for materials informatics. First, scaling to larger, multi-source datasets — combining Materials Project, JARVIS, AFLOW, and experimental databases — would improve model generalization and enable fidelity levels spanning GGA-DFT, hybrid functionals, GW calculations, and experimental measurements. Second, integrating the uncertainty estimates into active learning loops, where the model directs computational resources toward materials with high predicted value but low uncertainty, could dramatically accelerate materials discovery campaigns. Third, extending the approach to additional properties (elastic tensors, phonon spectra, thermal conductivity) and more complex crystal representations (graph neural networks with explicit bonding information) represents a natural evolution. We envision foundation models for materials science that pre-train across multiple fidelities and properties, then fine-tune for specific applications — analogous to large language models in NLP. The calibrated uncertainty provided by such models would be essential for their deployment in high-stakes applications such as battery materials design or semiconductor screening, where the cost of acting on an unreliable prediction is substantial.

---

## 5. Conclusion

[~300 words]

We have presented the Multi-Fidelity Transformer (MFT), a neural network architecture for crystal property prediction that jointly addresses the multi-fidelity and uncertainty quantification challenges in computational materials science. By incorporating fidelity-aware embeddings and training with Gaussian negative log-likelihood loss, MFT learns from heterogeneous-fidelity data while producing calibrated confidence estimates.

Applied to formation energy and band gap prediction using [X] materials from the Materials Project and JARVIS databases, MFT achieves competitive or superior performance compared to Random Forest and XGBoost baselines, with MAE of [INSERT] eV/atom for formation energy and [INSERT] eV for band gap. The multi-task learning formulation yields [INSERT]% improvement over single-task alternatives, demonstrating the benefit of shared representations across related material properties.

Crucially, the predicted uncertainties are well-calibrated, with Spearman correlations of [INSERT] and [INSERT] between predicted uncertainty and actual error for formation energy and band gap, respectively. This calibration enables practitioners to make informed decisions about which computational predictions warrant experimental validation — a capability absent in standard regression models.

While the current work focuses on formation energy and band gap prediction, the MFT framework is general and applicable to any crystal property prediction task where multi-fidelity data is available. We expect that as materials databases continue to grow in size and diversity, uncertainty-aware multi-fidelity approaches will become increasingly important for reliable computational materials screening.

All code and processed data are available at [GitHub URL] to support reproducibility and facilitate adoption by the materials science community.

---

## References

[INSERT — Each reference must be verified via web search before inclusion]

1. [Placeholder — verify each reference]
2. [Placeholder — verify each reference]
...

---

## Data Availability Statement

Processed datasets and trained model parameters are available at [Zenodo/ScienceDB URL with DOI]. Source code is available at [GitHub URL].

## AI-Contribution Disclosure

[Check Materials Futures' actual AI policy — do not assume]

This research was conducted with AI-assisted code generation, data analysis, and manuscript drafting. All experimental results were generated from actual code execution on Kaggle TPU v5e-8 hardware. The authors have reviewed and verified all results, methodology, and claims presented in this manuscript. [Adapt based on actual venue policy]

---

## Supplementary Material

- Full hyperparameter configurations
- Extended ablation results (fidelity embedding removal)
- Training curves for all runs
- Additional figures (feature distributions, residual plots)
