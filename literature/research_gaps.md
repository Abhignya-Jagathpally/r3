# Multiple Myeloma Single-Cell Literature: Five Unanswered Research Questions

## Gap 1: Does Immunosenescence Cause Disease Progression, or Is It a Marker of Aggressive Biology?

### The Question
Is T cell senescence a **driver** of rapid MM progression (such that reversing it would improve outcomes), or is it a **biomarker** of underlying aggressive tumor biology that causes senescence as a consequence?

### Why the Gap Exists
- **Correlational evidence is strong**: Giguere et al. (2026) show senescence-associated secretory phenotype (SASP) T cells present at diagnosis in rapidly progressing patients; Giguere et al. (2023) show exhausted T cells 3 months post-ASCT predict clinical relapse before detection
- **Causative evidence is absent**: No experimental studies in MM context that:
  - Genetically reverse senescence markers (e.g., knockout p16/p21 in MM-resident T cells)
  - Use senolytic drugs (dasatinib + quercetin) to eliminate senescent cells and measure progression impact
  - Sort senescent vs. non-senescent T cells from MM patients and test anti-tumor capacity ex vivo
  - Perform longitudinal tracking of individual senescent cells to determine their functional contribution

### Which Paper Came Closest
**Giguere et al., 2023, *Cancer Immunology Research*** - "T-cell Exhaustion in Multiple Myeloma Relapse after Autotransplant"
- Temporal resolution: Shows T cell exhaustion state at +3 months post-ASCT predicts relapse at months 12-24
- This temporal precedence suggests causality but does not prove it (reverse causality possible: aggressive clones cause T cell exhaustion)
- No functional reversal experiment

### Methodology to Close the Gap

**Experiment 1: Senolytic Drug Trial**
- Patient-derived T cells from MM patients + MM cells co-cultured
- Treatment with senolytic drugs (dasatinib 5µM + quercetin 10µM) vs. vehicle
- Readouts:
  - Flow cytometry: reduction of p16+CD8+ T cells
  - RNA-seq: SASP gene expression (IL-6, IL-8, TNF-α, IL-10) changes
  - Functional: plate-based cytotoxicity assay (51Cr release or real-time cell analysis)
  - Clonogenic: MM cell colony formation
- **Prediction**: If senescence is causal, senolytic treatment → reduced SASP → enhanced T cell cytotoxicity → reduced MM growth

**Experiment 2: Genetic Reversal in Humanized Mouse Model**
- Lentiviral delivery of p16/p21 knockout constructs to T cells from MM-bearing humanized NSG mice
- Compare p16−/p21− T cells vs. WT T cells in anti-MM function
- Measure: MM burden, T cell persistence, T cell phenotype over 8 weeks

**Experiment 3: Single-Cell Functional Screen**
- Isolate senescent (p16+/p21+) vs. non-senescent CD8+ T cells from MM patient bone marrow using FACS
- Perform single-cell RNA-seq to confirm senescence signature
- In separate experiment, co-culture sorted populations with autologous MM cells
- Measure killing capacity per cell and secretome by multiplex cytokine array
- **Null prediction**: If senescence is merely marker, sorting does not change per-cell function

---

## Gap 2: What Determines Plasma Cell Spatial Localization in Bone Marrow, and Is It Therapeutically Targetable?

### The Question
Do plasma cells actively home to specific bone marrow niches with matching stromal composition (active homing), or does the niche select for specific plasma cell phenotypes already present (passive selection)? Can spatial organization be disrupted therapeutically?

### Why the Gap Exists
- **Spatial heterogeneity demonstrated**: Ledergor et al. (2025) and Giguere et al. (2025) show spatially restricted plasma cell subpopulations in ~50% of newly diagnosed MM patients with zone-specific immune/stromal composition
- **Mechanism unknown**: No papers address whether:
  - Plasma cells express homing receptors (VLA-4, CXCR4, etc.) that guide them to specific zones
  - Stromal niches produce gradients (SDF-1, VCAM-1) that attract specific plasma cell states
  - Plasma cell arrival precedes or follows stromal remodeling
  - Spatial zones are dynamic or stable over disease course

### Which Paper Came Closest
**Ramasamy et al., 2024, *Nature Communications*** - "Bone marrow stromal cells induce chromatin remodeling in multiple myeloma cells"
- Shows stromal-plasma cell communication is bidirectional and mechanistic (chromatin remodeling via JAK/STAT)
- Does NOT address spatial homing or localization specificity
- Does NOT test whether blocking stromal signals alters spatial distribution

### Methodology to Close the Gap

**Experiment 1: Homing Receptor & Ligand Profiling**
- Apply single-cell RNA-seq to plasma cells from spatially restricted niches vs. diffuse infiltration patients (from Visium data)
- Generate RNA-seq of stromal fibroblasts from same niches
- Search for expression of:
  - Adhesion molecules on plasma cells: integrin α4β1 (VLA-4), integrin α5β1, L-selectin
  - Homing receptors: CXCR4, CCR7, CCR10
  - Cognate ligands on stromal fibroblasts: VCAM-1, FN, SDF-1, CCL21
- **Prediction**: Spatially restricted plasma cells express homing receptors; zones express matching ligands

**Experiment 2: Functional Homing Assay**
- Isolate CD138+ plasma cells from patients with spatial restriction
- Perform transwell migration assay toward:
  - Stromal fibroblasts isolated from same niche vs. distant niche
  - Conditioned media from niche-specific stromal cells
  - Recombinant SDF-1, VCAMs
- Blocking experiments: anti-VLA-4, anti-CXCR4 neutralizing antibodies
- **Prediction**: Spatially restricted plasma cells preferentially migrate toward autologous niche-derived stromal signals

**Experiment 3: In Vivo Spatial Disruption**
- NSG mice engrafted with MM + autologous patient-derived stromal fibroblasts
- Treat with:
  - Anti-VLA-4 (natalizumab analog)
  - CXCR4 antagonists (plerixafor)
  - Anti-SDF-1
- Read out via spatial transcriptomics at weeks 2, 4, 8:
  - Does plasma cell spatial organization break down?
  - Do plasma cells redistribute (more diffuse)?
  - Does MM burden change?
  - Do T cells infiltrate more effectively into previously "protected" zones?

---

## Gap 3: Can Single-Cell Foundation Models (scGPT, scBERT) Improve MM Risk Stratification Beyond Clinical + Genetic Features?

### The Question
Do pre-trained transformer models that learn universal cell representations add independent prognostic information over traditional approaches (clinical staging + genetic testing + immune phenotyping)?

### Why the Gap Exists
- **Foundation model promise is theoretical**: Cui et al. (2024, scGPT) and Tian et al. (2022, scBERT) claim pre-training on millions of cells enables transfer learning and robust inference
- **But clinical validation absent**: No papers apply foundation models to MM patient cohorts with survival outcomes
- **Competing evidence exists**: Abdelaal et al. (2022) show logistic regression matches scBERT on few-shot tasks, questioning value of pre-training
- **No head-to-head comparison**: Is foundation model annotation more predictive than Seurat/Harmony clustering? Unknown

### Which Paper Came Closest
**Giguere et al., 2026, *Nature Cancer*** - MM Immune Atlas
- Used traditional clustering (Seurat) and achieves strong prognostic stratification
- Did NOT use scGPT or scBERT
- Claims about immune senescence phenotype are descriptive, not foundation-model-driven

### Methodology to Close the Gap

**Experiment 1: Prospective Validation Cohort**
- Enroll 300 newly diagnosed MM patients
- Collect bone marrow at diagnosis for:
  - Full scRNA-seq (10x Genomics) + full clinical/genetic workup (cytogenetics, iFISH, LDH, ISS, CRAB criteria)
  - Flow cytometry immune phenotyping
- Pre-process with scGPT (using model from Cui et al., 2024)
- Extract:
  - scGPT cell embeddings (384-dim vector per cell)
  - scGPT predicted cell types
  - Cell-level quality scores from model confidence
- Patient-level summary: mean embedding, cell type distribution, quality quartile
- Co-variable modeling:
  - **Model A (Clinical + Genetic)**: ISS, cytogenetics (del17p, t(4;14), t(14;16)), LDH → logistic regression for 2-year relapse-free survival
  - **Model B (scGPT only)**: scGPT embeddings (384 dims) → dimension reduction (PCA to 10 PCs) → logistic regression
  - **Model C (Combined)**: Model A features + scGPT PCs → logistic regression
  - **Model D (Immune phenotyping only)**: Flow cytometry immune subset percentages → logistic regression
  - **Model E (Seurat traditional)**: Seurat clustering + per-cluster gene expression modules → logistic regression
- **Comparison**: AUC/C-index across models for 2-year RFS and OS
- **Prediction**: Model C (combined) superior; scGPT alone (Model B) non-inferior to Model D (immune phenotyping)

**Experiment 2: Feature Importance & Interpretability**
- If Model C shows improvement, identify which scGPT-derived features (cell types, embeddings, genes) drive prediction
- Compare to Model A features (are scGPT features redundant with genetic features, or independent?)
- Apply SHAP (SHapley Additive exPlanations) to decompose model predictions
- Test: Are scGPT features clinically interpretable or black boxes?

**Experiment 3: Robustness to Batch & Cohort Variation**
- Test scGPT model trained on Giguere et al. (2026) MM data on independent cohort (e.g., MMRF, external validation)
- Does prognostic performance transfer? Or is overfitting to training batch?

---

## Gap 4: What Are the Molecular Triggers of Extramedullary Progression, and Can They Be Predicted from Bone Marrow Transcriptomics?

### The Question
Extramedullary myeloma (EMM) is rare but highly aggressive; can early detection of extramedullary-initiating cell (EMIC) states in bone marrow predict which patients will progress to EMM, and what mechanisms drive the transition?

### Why the Gap Exists
- **EMM pathophysiology partially described**: Li et al. (2023, *Blood Advances*) identified extramedullary-initiating cells (EMICs) from pleural effusion scRNA-seq with upregulated p53 signaling and proliferation
- **But bone marrow predictors absent**: No prospective study shows whether EMIC signatures can be detected in bone marrow samples from patients who later progress to EMM
- **Trigger mechanisms unclear**: What causes transition from medullary to extramedullary disease? Genetic event? Microenvironmental selection? Immune escape?
- **No early detection tool**: EMM is diagnosed clinically (imaging); could be prevented if predicted early from bone marrow

### Which Paper Came Closest
**Li et al., 2023, *Blood Advances*** - "Identification of evolutionary mechanisms of myelomatous effusion"
- Excellent characterization of EMICs: upregulated p53, increased proliferation, distinct transcriptome from bone marrow plasma cells
- **Limitation**: Retrospective comparison of pleural effusion to contemporary bone marrow; no matched longitudinal samples predating EMM
- Does NOT have access to bone marrow samples from patients before EMM diagnosis

### Methodology to Close the Gap

**Experiment 1: Retrospective Cohort with Stored Samples**
- Identify 50 MM patients in your institution/consortium who eventually progressed to EMM (within 3 years of diagnosis)
- Retrieve stored bone marrow scRNA-seq from diagnosis (or earliest timepoint before EMM)
- Retrieve matched EMM sample (biopsy/effusion/CSF) at progression
- Apply Li et al. (2023) EMIC signature (upregulated p53, proliferation genes, immune evasion) to diagnostic bone marrow
- **Question**: Are EMIC-like cells detectable in bone marrow pre-EMM? At what frequency?
- **Comparison**: Control MM patients without EMM progression matched by stage/risk
- **Prediction**: EMIC signature enrichment in pre-EMM bone marrow vs. controls; prevalence correlates with time-to-EMM

**Experiment 2: Mechanistic Investigation**
- If EMIC precursors detected in bone marrow:
  - Perform trajectory inference (Monocle) to infer differentiation path from typical plasma cell → EMIC
  - Identify genes/pathways enriched in early trajectory stages
  - Do bone marrow stromal cells suppress EMIC trajectory? Test via co-culture with stromal fibroblasts
  - Is EMIC state selected by high-dose chemotherapy, or pre-existing at diagnosis?
- If EMIC precursors NOT detected:
  - Investigate genetic vs. transcriptomic features distinguishing EMM patients at diagnosis
  - Do EMM-progression patients have higher genomic complexity?
  - Profile immune/stromal composition at diagnosis: are EMM patients already immunologically distinct?

**Experiment 3: Prospective Validation & Clinical Trial Design**
- Moving forward, enroll newly diagnosed MM patients into cohort
- Perform scRNA-seq at diagnosis; score each patient for EMIC signature prevalence
- Prospectively track: which high-EMIC-signature patients progress to EMM? Timeline?
- If EMIC signature predicts EMM:
  - Design intervention trial: high-EMIC patients → experimental therapy (e.g., p53 pathway inhibition, aggressive early consolidation) vs. standard
  - Endpoint: EMM-free survival at 3 years

---

## Gap 5: Is Single-Cell Transcriptomics Faithfully Capturing In Vivo Plasma Cell States, or Are Dissociation & Processing Artifacts Obscuring True Phenotypes?

### The Question
How much of the transcriptional diversity observed in scRNA-seq is biological vs. technical artifact from tissue dissociation, enzymatic digestion, and single-cell isolation stress responses?

### Why the Gap Exists
- **Widespread acknowledgment of artifact risk**: Review papers (Denisenko et al., 2020; Lähnemann et al., 2020) extensively document dissociation-induced stress signatures (heat shock proteins, apoptosis markers, immediate-early genes)
- **But MM field largely ignores it**: Of 50+ MM scRNA-seq papers reviewed, only 2–3 mention dissociation artifact; none systematically validate or correct for it
- **Artifact risk is NOT uniform**: Immune cells (especially T cells, monocytes) are highly stress-sensitive; plasma cells may be more robust
- **No gold standard correction in MM context**: Denisenko et al. propose dissociation score; not applied to MM

### Which Paper Came Closest
**Denisenko et al., 2020, *Genome Biology*** - "Dissociation induces gene expression changes in tissues unrelated to dissociation protocol"
- General framework for quantifying dissociation artifacts across tissues
- Not applied to MM specifically
- No MM samples in test set

### Methodology to Close the Gap

**Experiment 1: Comparative Sampling (Fresh vs. Dissociated)**
- Collect bone marrow from MM patients (n=10)
- Split sample into two:
  - **Route A (Fresh)**: Immediately disaggregate by mechanical disruption only (pipetting), no enzymatic digestion; perform scRNA-seq (10x Genomics) within 15 minutes
  - **Route B (Standard)**: Enzymatic digestion (collagenase, DNase) + incubation, then scRNA-seq as per standard protocol (30-60 min start to finish)
- Comparison:
  - Perform quality control: viability, multiplet rate, transcripts per cell
  - RNA-seq: identify genes differentially expressed between Route A and Route B
  - Hypothesis: Route B shows upregulation of stress genes (HSP70, HSP90, JUN, FOS, JUNB, JUND) and apoptosis markers (FAS, BAX, PMAIP1)
  - Quantify effect size: what fraction of differentially expressed genes in standard comparisons (e.g., MGUS vs. MM) are dissociation artifacts?

**Experiment 2: Dissociation Artifact Scoring & Correction**
- Apply dissociation artifact scoring (Denisenko et al. approach) to published MM datasets
  - Giguere et al. (2026) 1.4M cells: what % are high-artifact?
  - Ledergor et al. (2022) GSE271107: same analysis
- Build correction model:
  - Identify consensus dissociation signature (genes consistently upregulated in Route B above)
  - Apply linear regression to remove artifact from expression matrix (similar to DoubletFinder approach)
  - Re-analyze: Do clusters change? Do key plasma cell subtypes persist? Does PDIA4 signature still associate with progression?
- **Concern**: Aggressive correction might also remove real biology
- Use fresh-sample data (Route A) as ground truth to validate correction is valid

**Experiment 3: Cell-Type-Specific Sensitivity**
- Separate analysis by cell type (plasma cells vs. T cells vs. monocytes vs. stromal cells)
- Hypothesis: Plasma cells relatively insensitive to dissociation (stable transcriptome); T cells highly sensitive (artifact score high)
- Consequence: If T cell signatures (exhaustion, senescence) are artifact-driven, major conclusions (Shammas 2016, Giguere 2023) are compromised
- Mitigation: Validate T cell signatures using:
  - Sorted fresh T cells from MM bone marrow
  - Surface protein flow cytometry on same samples (compare protein vs. RNA exhaustion markers)

**Experiment 4: Protocol Optimization**
- If dissociation artifacts are significant, test optimizations:
  - Rapid dissociation (mechanical + brief enzymatic, <10 min)
  - Cold incubation (slows stress response)
  - Anti-oxidant addition (reduce ROS)
  - Immediate permeabilization/fixation (arrest active responses)
- Compare artifact scores across protocols
- Recommend lowest-artifact protocol for future MM studies

---

## Summary: Research Gap Priorities

| Gap | Mechanism Impact | Clinical Relevance | Tractability | Priority |
|-----|------------------|-------------------|--------------|----------|
| **Gap 1: Senescence causation** | Fundamental (drives progression or marks it?) | HIGH (guides immunotherapy design) | Medium (requires in vivo models) | **Highest** |
| **Gap 2: Spatial homing** | High (may explain niches; therapy resistance) | MEDIUM (spatial targeting speculative) | Medium (new technology needed) | **High** |
| **Gap 3: Foundation models clinical utility** | Medium (improves stratification?) | MEDIUM (if improves risk prediction) | High (data available) | **Medium-High** |
| **Gap 4: EMM prediction** | High (early detection urgent unmet need) | HIGHEST (could prevent aggressive disease) | Medium (rare disease, large cohort needed) | **High** |
| **Gap 5: Dissociation artifacts** | Very High (may invalidate key findings) | MEDIUM (affects interpretation) | High (method available) | **Highest** |

---
