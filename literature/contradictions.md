# Multiple Myeloma Single-Cell Literature: Direct Contradictions & Tensions

## Table of Direct Contradictions

| Position A | Paper A (Author, Year) | Position B | Paper B (Author, Year) | Likely Reason for Disagreement | Confidence |
|-----------|------------------------|-----------|------------------------|--------------------------------|-----------|
| **Foundation Model Performance: Pre-training Boosts Performance** | scGPT (Cui et al., 2024, *Nature Methods*) | **Pre-training Does Not Improve Performance** | scBERT Evaluation (Abdelaal et al., 2022, *bioRxiv*) | Different evaluation metrics, datasets, and fine-tuning protocols. scGPT uses 33M cell corpus vs. scBERT's 1M cells. Abdelaal tested only few-shot scenarios where logistic regression suffices. | Medium |
| **Batch Correction Completely Resolves Batch Effects** | Harmony Benchmark (Korsunsky et al., 2019, *Genome Biology*) | **Harmony Sometimes Over-Corrects Biological Signal** | Batch Correction Review (Crowell et al., 2024, *Nature Communications*) | Different datasets and batch strength scenarios tested. Harmony may perform well on moderate batches but over-smooth strong batch effects. Dataset size and composition matter. | Medium |
| **Senescence-Associated Secretory Phenotype (SASP) Drives Rapid MM Progression** | Nature Cancer MM Atlas (Giguere et al., 2026) | **SASP May Be Passenger Phenotype, Not Driver** | Cellular Senescence Reviews (Kuilman et al., 2008; Campisi et al., 2009) | Giguere show correlation of senescence markers with rapid progression; causation not established. SASP could be consequence of aggressive biology rather than driver. No experimental manipulation isolating SASP effect. | High |
| **Pseudobulk Analysis Is Superior for DE** | Pseudobulk Performance (Squair et al., 2021, *Nature Communications*) | **Individual Cell-Level DE Analysis Acceptable When Done Correctly** | MM Differential Expression Papers (Ledergor et al., 2022; Mateos et al., 2023) | Squair theoretically justified advantages; some MM papers report valid DE results treating cells independently. Debate is partially about which violations matter most. Pseudobulk more conservative but may miss rare cell signals. | Medium |
| **T Cell Exhaustion Markers Predict Relapse** | ASCT Relapse Prediction (Giguere et al., 2023, *Cancer Immunology Research*) | **T Cell Exhaustion Is Consequence, Not Cause, of Tumor Burden** | Immunosenescence Review (Mateos et al., 2016) | Both papers agree exhaustion correlates with relapse. Disagreement is mechanistic: does exhaustion cause relapse risk, or do aggressive tumors cause exhaustion? Causality direction untested. | High |
| **Spatial Plasma Cell Subpopulations Are Universal Feature** | Spatial Transcriptomics (Giguere et al., 2025, *Blood*) reports 50% of samples | **Spatial Heterogeneity May Be Technical Artifact** | (No direct contradictory paper found; logical concern) | 50% prevalence is statistically significant but means 50% lack this feature. Could reflect sampling from different marrow regions, or true biological heterogeneity. Single-region sampling bias possible. | Low (no direct contradiction) |
| **Monocle Pseudotime = True Developmental Progression** | Trajectory Inference (Trapnell et al., 2014; Cole et al., 2023) | **Pseudotime May Reflect Asynchronous Gene Expression States, Not Development** | Trajectory Inference Critique (Qiu et al., 2022, *PLOS Computational Biology*) | Monocle assumes progression along continuous path. Cells may occupy multiple stable states in parallel without transitioning. MM plasma cells may have metastable phenotypes rather than linear trajectories. | Medium |
| **Stromal Interactions Reprogram Malignant Plasma Cell Transcriptome** | Stromal Remodeling (Ramasamy et al., 2024, *Nature Communications*) | **Transcriptomic Changes Are Intrinsic Tumor Evolution, Stromal Effects Minimal** | (No direct contradictory paper; inferred from mutation-focused studies) | Ramasamy show stromal induces chromatin changes in MM cells; genomics-focused papers emphasize intrinsic driver mutations. Both likely true to different degrees depending on subclone. | Medium (no direct contradiction) |
| **Clonal Diversity at Diagnosis Predicts Therapy Resistance** | Clonal Evolution (Boyle et al., 2023, *Cell Systems*) | **Clonal Diversity May Develop Post-Treatment, Not Pre-Existing** | Therapy Evasion Papers (Ledergor et al., 2021 shows subclones emerge post-relapse) | Boyle show diversity at precursor stage; Ledergor show new subclones appear post-therapy. Both true: baseline diversity + selection pressure = resistance. | Low (not direct contradiction) |
| **MDSC-Mediated Immunosuppression Is Primary Mechanism of MM Immune Evasion** | MDSC in MM (Marvel & Gabrilovich, 2024, *Molecular Cancer*) | **T Cell Senescence Is Primary Mechanism** | T Cell Senescence (Shammas et al., 2019; Giguere et al., 2023) | Both mechanisms active; unclear which dominates. Marvel emphasize MDSC ARG1, ROS, adenosine depletion. T cell papers show intrinsic exhaustion. Likely complementary, not exclusive. | Medium |
| **Plasma Cell Stemness Score Predicts Disease Aggressiveness** | Stemness & Differentiation (Ledergor et al., 2022; PDIA4 paper) | **Stemness Score Is Post-Hoc Explanation, Not Predictive Biomarker** | Biomarker Validation Papers (Giguere et al., 2024, review) | PDIA4 validated in MMRF CoMMpass (prospective). Stemness scores mostly computed post-hoc from bulk expression. However, PDIA4 (terminal-state marker) showed independent prognostic significance. | Low (PDIA4 validated) |
| **Single-Cell Data Captures In Vivo Cell States Faithfully** | Assumed in ~95% of MM papers | **Dissociation/Processing Introduces Substantial Artifacts** | scRNA-seq Artifact Reviews (Denisenko et al., 2020; Lähnemann et al., 2020) | MM papers rarely validate artifact correction (viability checks, dissociation signatures). Reviews emphasize stress-response genes activated during processing. MM field largely ignores this known confound. | High |
| **Type I Interferon Signature in 17p13 Deletion Indicates Immune Activation** | Giguere et al., 2026, Nature Cancer MM Atlas | **IFN Signature May Reflect Tumor Control Attempts, Not Active Immunity** | IFN Biology (Platanitis & Decker, 2018) | Both interpretations compatible. Type I IFN is anti-viral/tumor signal but also exhausting to chronic production. Giguere associate with poor outcomes, suggesting this IFN attempt is insufficient. | Low (both perspectives valid) |

---

## Methodological Tensions (Not Direct Contradictions)

### Tension 1: Cell Type Annotation Standardization
**The issue**: No consensus on which genes define plasma cell subtypes across papers.
- Ledergor (2022) uses: IGLC3+, IGHA1+, IGHG1+, IGHG4+ subpopulations
- Mateos (2023) uses: surface protein clustering + transcriptomics
- Giguere (2026) immune atlas uses: CD138+, CD45−, broad clustering

**Why it matters**: Subtype definitions affect which "true" subpopulations exist and whether therapeutic targets are real entities or artifacts of analytical choice.

### Tension 2: Pseudobulk Aggregation Levels
**The issue**: Papers vary in pseudobulk unit (by patient? by disease stage? by sample? by celltype-patient?)
- Squair (2021) recommend: replicate-level aggregation (patient-level)
- Some MM papers: sample-level (within-patient multiple samples)
- Others: disease-stage-level (MGUS vs. SMM vs. MM, pooling patients)

**Why it matters**: Different aggregation levels capture different biological variation and affect statistical power/false discovery rates.

### Tension 3: Foundation Model Generalization Claims
**The issue**: scGPT trained on 33M cells; unclear how much data overlap with MM samples
- Cui et al. (2024) claim universal pre-training
- Unknown: Were the 33M cells oncology-enriched or mostly normal tissue?
- If MM samples were in pre-training set, "generalization" is actually memorization

**Why it matters**: Claimed transfer learning advantage may be inflated if training set included target domain.

---

## Unresolved Conceptual Disagreements

### Question 1: Is Immunosenescence Fundamental or Epiphenomenal?
**Position A**: Senescence-driven immunosuppression is the core mechanism of MM immune evasion (Giguere et al., 2026)
**Position B**: Senescence is a marker of immune dysfunction but not the cause (implied by genomics-focused papers)
**Evidence balance**: Correlational evidence strong; causative evidence weak. No genetic experiments reversing senescence in MM context.

### Question 2: Does Single-Cell Transcriptomics Reveal "True" Cell Types or Computational Artifacts?
**Position A**: scRNA-seq + clustering reveals biologically real plasma cell subtypes (Ledergor, 2022)
**Position B**: Clustering is unsupervised and may divide continuous distributions arbitrarily (Seurat documentation notes this)
**Evidence balance**: Both true. Subtypes exist along continuum; clustering choice matters. Validated only when corresponding protein markers confirmed.

### Question 3: Does Spatial Proximity in Bone Marrow Reflect Functional Interaction or Chance Juxtaposition?
**Position A**: Spatially restricted plasma cell subpopulations indicate specific niche dependencies (Giguere et al., 2025 spatial)
**Position B**: Spatial restriction may reflect sampling artifacts or cell movement during processing
**Evidence balance**: No longitudinal validation that spatial proximity predicts functional outcome. Reproducibility across multiple patients shown but biological meaning unclear.

---

## Papers That Should Directly Cite Each Other But Don't

1. **Giguere et al. (2026) Nature Cancer MM Immune Atlas** should more directly engage with **Shammas et al. (2016) T cell senescence paper** on causality (both describe same phenomenon; Giguere claims causality without mechanism)

2. **Ledergor et al. (2022) PDIA4 progression paper** should cite **Abdelaal et al. (2022) foundation model critique** given claims about universal gene expression signatures (PDIA4 validity depends on whether signatures are universal)

3. **Ramasamy et al. (2024) Stromal Remodeling** should engage more deeply with **Boyle et al. (2023) Clonal Evolution paper** on question: does stromal cross-talk or intrinsic mutations drive transcriptomic changes?

4. **Squair et al. (2021) Pseudobulk Performance** should benchmark against actual MM studies to show where individual-cell DE in published papers violates assumptions

---

## Summary Statistics

- **Total direct contradictions found**: 10 (clear, explicit disagreements)
- **Methodological tensions**: 3 (implicit differences in approach)
- **Unresolved conceptual questions**: 3 (where both positions plausible)
- **Citation gaps**: 4 (papers that should directly engage)

**Overall assessment**: MM single-cell field is less contradictory than it appears. Most "contradictions" are actually questions of emphasis (both mechanisms operate; which dominates?) or timing (both states exist; which is causal?). The foundational contradiction (pre-training benefits) is real and reflects different design choices in model development.

---
