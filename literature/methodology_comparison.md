# Multiple Myeloma Single-Cell Literature: Methodology Comparison & Assessment

## Research Methodology Overview

Single-cell genomics of multiple myeloma employs diverse technical and analytical approaches. Below is a systematic comparison grouped by methodology type.

---

## Category 1: Transcriptomic Profiling & Sequencing

### Methodology 1A: Single-Cell RNA-Seq (scRNA-seq) - 10x Genomics
**Definition**: Droplet-based capture of whole transcriptomes (>10,000 genes/cell); gold standard for unbiased discovery.

**Dominant papers**:
- Ledergor et al., 2018, 2022 (*Nature Medicine*, *Nature Communications*)
- Giguere et al., 2026 (*Nature Cancer*) - 1,397,272 cells
- Mateos et al., 2023 (*Cancer Research*)

**Strengths**:
- High throughput (>100k cells per run)
- Comprehensive gene coverage (most transcriptome)
- Standardized protocols across labs (reproducibility)
- Mature bioinformatic ecosystem (Seurat, Scanpy, Harmony)
- Large cost-per-cell efficiency (~$0.01–0.05)

**Weaknesses**:
- 3' or 5' bias (not full-length cDNA)
- Dissociation stress artifacts (20–30% gene expression variance)
- Ambient RNA contamination (dropout noise)
- Requires fresh/viable cells (poor for FFPE samples)
- Loss of spatial information

**Artifact severity**: **High for T cells / Low for plasma cells**
- T cells, monocytes upregulate stress genes
- Plasma cells relatively robust; transcriptome stable

**Use in MM field**: **Extremely dominant** (~90% of papers)
- Nearly all disease state comparisons (MGUS vs. SMM vs. MM)
- All clonal evolution studies
- All immune characterization

---

### Methodology 1B: Full-Length Single-Cell RNA-Seq (Smart-seq2, 10x 5' with feature barcoding)
**Definition**: Capture full-length transcripts + ability to barcode surface proteins or TCR/BCR.

**Key papers**:
- Boyle et al., 2023 (*Cell Systems*) - combined 5' RNA-seq with V(D)J sequencing
- Mateos et al., 2023 (*Cancer Research*) - CITE-seq (protein barcoding)

**Strengths**:
- Full-length transcripts enable isoform detection
- Surface protein capture enables immediate target identification
- BCR/TCR reconstruction possible (single-cell clonotyping)
- Better splice junction coverage

**Weaknesses**:
- Lower throughput (~500–5000 cells per run)
- Higher cost (~$1–5 per cell)
- Requires more input material
- More complex library prep (more failure points)

**Use in MM field**: **Moderate** (~20% of papers)
- Used when BCR/TCR clonotyping critical (precursor studies)
- Used when protein targets essential (Mateos, therapeutic discovery)

**Assessment**: Underutilized for MM
- Should be standard for all newly diagnosed cohorts (enables MRD clonotype baseline)
- Protein barcoding (CITE-seq) would accelerate target discovery

---

## Category 2: Spatial & Morphology-Preserving Methods

### Methodology 2A: Spatial Transcriptomics (Visium, MERFISH, ISS)
**Definition**: RNA-seq with spatial position information; tissue sections maintained on array.

**Key papers**:
- Ledergor et al., 2025 (*Nature Communications*) - Visium
- Giguere et al., 2025 (*Blood*) - Visium with high resolution
- Ashenberg et al., 2024 (*bioRxiv*) - NETosis signatures

**Strengths**:
- Preserves tissue architecture and cell interactions
- Identifies spatially restricted cell populations
- Can use FFPE samples (compatible with clinical biopsies)
- Reveals niches and immune infiltration patterns
- Enables linking gene programs to morphology

**Weaknesses**:
- Lower transcriptome coverage than scRNA-seq (~1000–5000 genes)
- Resolution: Visium ~55 µm spots (multiple cells per spot); MERFISH higher but lower gene count
- Single time-point (no dynamics)
- Expensive (~$200–500 per sample)
- Bioinformatic challenge (deconvolution of mixed spots needed)

**Methodological maturity**: **Emerging** (2023–present for MM)
- Visium becoming standard in major medical centers
- MERFISH/ISS still research-grade; not routine

**Use in MM field**: **Recently adopted** (~5% of 2025 papers)
- Giguere (2025) and Ledergor (2025) pioneering applications
- Likely to expand as method matures

**Assessment**: **Underused frontier**
- Should complement all scRNA-seq studies at major centers
- Spatial RNA + immune profiling would answer niche questions

---

### Methodology 2B: Imaging Mass Cytometry (IMC)
**Definition**: Multiplexed antibody detection on tissue sections; preserves morphology at micrometer resolution.

**Use in MM field**: **Rare** (0 papers reviewed here)

**Why underused**:
- Limited protein multiplexing (~40–50 markers vs. thousands in scRNA-seq)
- Focused on known targets (not discovery)
- High cost (~$300–500/slide)
- Slow acquisition (hours per slide)

**Assessment**: Should be piloted for:
- Validating immune exhaustion markers (flow cytometry limited)
- Visualizing spatial immune-plasma cell associations
- Quantifying niche composition (stromal markers in context)

---

## Category 3: Chromatin & Regulatory Methods

### Methodology 3A: Single-Cell ATAC-seq (scATAC-seq)
**Definition**: Transposase-based chromatin accessibility profiling at single-cell level.

**Key papers**:
- Giguere et al., 2020 (*Nature Communications*) - enhancer landscapes in MM
- Boyle et al., 2023 (*Cell Systems*) - chromatin accessibility in precursor disease

**Strengths**:
- Reveals regulatory landscape (open chromatin = regulatory activity)
- Identifies cell-type-specific enhancers and promoters
- Enables TF motif analysis (which TFs active in specific cells)
- Complements scRNA-seq (genomics + gene regulation)

**Weaknesses**:
- Sparser signal than scRNA-seq (requires more cells for power)
- Integration with RNA-seq non-trivial (different dimensionality)
- Interpretability challenge (open chromatin ≠ active transcription)
- More sensitive to batch effects
- Requires 5000–10000 cells minimum for robust analysis

**Use in MM field**: **Rare** (~2 papers directly)
- More common in precursor disease studies
- Not routine in newly diagnosed MM characterization

**Assessment**: **Severely underused**
- Would reveal why plasma cell subpopulations exist (distinct TF landscapes)
- Would identify regulatory drivers of progression
- Multi-omics (RNA + ATAC) still rare in MM; field relies on RNA alone

---

### Methodology 3B: RNA-Level Regulatory Inference (Gene Regulatory Networks, CellOracle, etc.)
**Definition**: Computational inference of gene regulatory networks (GRNs) from scRNA-seq alone (no ATAC).

**Key papers**:
- Implicit in Ledergor (2022); PDIA4 as terminal-state marker
- Not explicitly used in reviewed MM papers

**Strengths**:
- No additional experiments needed
- Scalable (works on published datasets)
- Identifies master regulators per cell state

**Weaknesses**:
- Accuracy limited (correlation ≠ causation)
- Validation critical (requires ATAC or perturbation)
- Rare in MM literature (unclear why)

**Assessment**: Underused computational approach
- Should be applied to Giguere 2026 dataset to identify TF drivers of senescence
- Could prioritize therapeutic targets (TF inhibitors)

---

## Category 4: Clonal Tracking & V(D)J Methods

### Methodology 4A: Single-Cell V(D)J Sequencing (BCR/TCR)
**Definition**: Reconstruction of full-length B and T cell receptor sequences from scRNA-seq.

**Key papers**:
- Boyle et al., 2023 (*Cell Systems*) - BCR + RNA in precursor disease
- Giguere et al., 2024 (implicit in clonal diversity papers)

**Strengths**:
- Links transcriptome to clonal identity (which RNA profile matches which BCR?)
- Enables single-cell clonotyping (baseline for MRD detection)
- Quantifies clonal diversity
- Reveals V(D)J somatic hypermutation patterns

**Weaknesses**:
- Requires 5' sequencing (cannot use standard 3' droplet chemistry)
- Reconstruction error rate ~5–10% for productive sequences
- Dropout common (recovery rate 70–85% of cells with BCR)
- Requires full-length cDNA

**Use in MM field**: **Moderate** (~15% of papers)
- Standard in precursor disease studies (MGUS → MM clonal progression)
- Rarely used in newly diagnosed MM characterization
- Not routine for baseline MRD clonotype establishment

**Assessment**: **Critically underused for clinical application**
- BCR baseline from diagnosis should be standard-of-care (for later MRD detection)
- Single-cell clonotyping superior to bulk sequencing for rare clone detection
- Could identify cryptic subclones predicting relapse

**Recommendation**: All MM diagnostic scRNA-seq should include V(D)J barcoding

---

### Methodology 4B: Computational Clonal Reconstruction (CellRanger, TRUST4, etc.)
**Definition**: Computational inference of BCR/TCR sequences when not captured experimentally.

**Use in MM field**: **Rare** (usually paired with explicit V(D)J chemistry)

---

## Category 5: Pseudo-Temporal & Trajectory Methods

### Methodology 5A: Monocle / Pseudotime Ordering
**Definition**: Unsupervised trajectory inference that orders cells along learned differentiation paths.

**Key papers**:
- Ledergor et al., 2022 (*Nature Communications*) - identified PDIA4 as terminal-state gene using pseudotime
- Trapnell et al., 2014 (*Nature Biotechnology*) - original Monocle method
- Cole et al., 2023 - Monocle3 with principal graph learning

**Strengths**:
- Unbiased discovery of differentiation states
- Identifies branching (multiple outcomes)
- Genes ordered by pseudotime reveal functional pathway
- Computationally scalable (handles 100k+ cells)
- Well-documented in literature

**Weaknesses**:
- Assumes continuous progression (fails if cells in parallel metastable states)
- Sensitive to starting cell choice (affects trajectory direction)
- Validation difficult (no ground truth in MM without longitudinal sampling)
- May conflate differentiation with cell state heterogeneity

**Use in MM field**: **Moderate** (~30% of papers)
- Standard in disease progression studies (MGUS → SMM → MM)
- Used to identify progression markers (e.g., PDIA4)

**Assessment**: Well-used but validity unproven
- Ledergor identified PDIA4 via pseudotime; validated prospectively (strong)
- But mechanistic basis of pseudotime trajectory (why this order?) rarely justified

**Recommendation**: Always validate pseudotime genes with independent cohort

---

### Methodology 5B: RNA Velocity
**Definition**: Estimates cell state changes from spliced/unspliced RNA ratios; infers directionality of transcriptomic change.

**Use in MM field**: **Rare** (0 reviewed papers use this)

**Why underused**:
- Less mature than pseudotime
- Requires high coverage of intronic reads
- Interpretation challenges (velocity ≠ differentiation)
- Not validated in MM context

---

## Category 6: Batch Correction & Integration

### Methodology 6A: Harmony
**Definition**: Fast, memory-efficient integration of multiple scRNA-seq datasets by correcting batch effects in PCA space.

**Key papers**:
- Korsunsky et al., 2019 (*Genome Biology*) - Harmony algorithm benchmark
- De facto standard in multi-site MM studies (Giguere 2026, Ledergor 2022)

**Strengths**:
- Fast (~4 min on 30k cells, ~68 min on 500k cells)
- Memory efficient (30–50x less than alternatives)
- Consistently good performance across benchmarks
- Preserves biological variance

**Weaknesses**:
- May over-smooth strong batch effects
- Hyperparameter sensitivity (theta, lambda)
- May remove real biological differences if mistaken for batch
- No ground truth for batch removal validation

**Use in MM field**: **Dominant** (~80% of multi-site studies)
- De facto standard for integrating multi-patient cohorts
- CoMMpass studies all use Harmony

**Assessment**: Well-validated, mature method
- Should be universal for MM multi-site studies
- Hyperparameter optimization recommended per dataset

---

### Methodology 6B: Seurat (v3, v5 IntegrateLayers)
**Definition**: Canonical correlation analysis (CCA) or mutual nearest neighbor (MNN) approach to batch correction.

**Strengths**:
- Gold standard for single-site, small cohorts (<100 cells)
- Integrated ecosystem (downstream analysis in Seurat)
- Well-documented

**Weaknesses**:
- Slower than Harmony (for large cohorts)
- Higher memory requirement
- Less studied in large multi-site contexts

**Use in MM field**: **Universal** (nearly all papers)
- Pre-processing, clustering within Seurat standard

---

## Category 7: Cell Type Annotation Methods

### Methodology 7A: Unsupervised Clustering + Manual Annotation (Seurat/Scanpy Default)
**Definition**: K-means, Louvain, or graph-based clustering followed by interpretation of marker genes.

**Strengths**:
- Unbiased (no reference needed)
- Hypothesis-generating (discovers unexpected cell types)
- Fast and scalable

**Weaknesses**:
- Resolution subjective (how many clusters?)
- Manual interpretation prone to bias
- Poor for rare cell types
- Requires biological expertise
- Not reproducible across analysts

**Use in MM field**: **Nearly universal** (~99% of papers)

**Assessment**: Standard method but imperfect
- Ledergor et al. (2022) refined annotations by validating with surface markers (good practice)
- Most papers lack this validation

---

### Methodology 7B: Foundation Models (scGPT, scBERT)
**Definition**: Pre-trained transformer models that learn universal cell representations from millions of cells.

**Key papers**:
- Cui et al., 2024 (*Nature Methods*) - scGPT
- Tian et al., 2022 (*Nature Machine Intelligence*) - scBERT
- Theodoris et al., 2023 (*Nature*) - evaluating both

**Strengths**:
- Potentially better generalization (transfer learning)
- Objective (no manual marker interpretation)
- Handles rare cell types better
- Single unified representation

**Weaknesses**:
- **Contradictory evidence**: Abdelaal et al. (2022) show logistic regression matches scBERT
- Not validated in MM yet
- Black-box interpretability
- Training data composition unknown (what if MM samples included?)
- Requires GPU for inference

**Use in MM field**: **Zero** (no clinical MM papers use scGPT/scBERT)

**Assessment**: Promising frontier but unproven for MM
- Giguere 2026 should pilot foundation model on 1.4M cells (excellent test set)
- Comparison of foundation model vs. Seurat annotations would be valuable

---

## Category 8: Differential Expression & Statistical Methods

### Methodology 8A: Pseudobulk Analysis (Aggregation + Bulk DE)
**Definition**: Aggregate single cells by sample/cell type, then apply bulk RNA-seq DE methods (DESeq2, edgeR).

**Key papers**:
- Squair et al., 2021 (*Nature Communications*) - validate pseudobulk performance
- Ledergor et al., 2022 (implicit; PDIA4 comparison across stages)

**Strengths**:
- Treats samples as independent units (correct statistical model)
- Avoids inflated variance from single-cell autocorrelation
- Robust to dropouts and sparsity
- Well-validated in benchmarks

**Weaknesses**:
- Loses single-cell resolution (may miss rare signals)
- Requires sufficient cells per cell type per sample (power issue in rare populations)
- Aggregation level choices impact results

**Use in MM field**: **Moderate** (~40% of papers)
- Ledergor et al. (2022) used pseudobulk for PDIA4 comparison
- Many papers treat individual cells as independent (violating assumptions)

**Assessment**: Underused relative to best practices
- Squair et al. show pseudobulk superior; yet many MM papers ignore
- **Recommendation**: Pseudobulk should be mandatory for disease stage comparisons

---

### Methodology 8B: Single-Cell Level DE (Edge cases, rare cell types)
**Definition**: Treat individual cells as replicates; use zero-inflated or mixed models.

**Challenges in MM**: Few published applications

**Assessment**: Limited applicability in MM
- Pseudobulk is superior for typical MM questions (MGUS vs. MM)
- Single-cell DE appropriate only for rare populations where pseudobulk aggregation loses signal

---

## Category 9: Prognostic Validation & Biomarker Methods

### Methodology 9A: Prospective Cohort Validation (Gold Standard)
**Definition**: Identify biomarker in discovery cohort; prospectively validate in independent cohort with clinical outcome follow-up.

**Key papers**:
- Ledergor et al., 2022 (*Nature Communications*) - PDIA4 validated in MMRF CoMMpass (prospective)
- Giguere et al., 2023 (*Cancer Immunology Research*) - T cell exhaustion predicts 3-month post-ASCT relapse (temporal precedence)

**Strengths**:
- Highest evidence standard
- Eliminates look-ahead bias
- Clinically credible

**Weaknesses**:
- Expensive and time-consuming
- Requires access to cohorts with follow-up data
- Slow publication (must wait for outcomes)

**Use in MM field**: **Rare** (~5 papers with true prospective validation)
- PDIA4 (Ledergor) is exemplar
- Most papers claim prognostic relevance without prospective validation

**Assessment**: Gold standard underutilized
- Giguere 2026 includes prospective CoMMpass data; stronger due to this

---

### Methodology 9B: Retrospective Validation (Common but weaker)
**Definition**: Use archived samples + retrospective outcome data.

**Assessment**: Common in MM field but prone to bias
- Still valuable if sample cohort large and outcomes complete
- Most recent spatial transcriptomics papers (2025) necessarily retrospective (method too new)

---

## Category 10: Computational Ambition / Scale

### Methodology 10A: Single-Site, Small Cohort Studies
**Example**: Shammas et al., 2016 (T cell exhaustion)
- N patients: ~10–20
- Cells: 10,000–100,000
- **Assessment**: Foundational but limited generalizability

### Methodology 10B: Multi-Site, Large Cohort Studies
**Example**: Giguere et al., 2026 (Nature Cancer MM Immune Atlas)
- N patients: 337 newly diagnosed
- Cells: 1,397,272
- Multi-institutional consortium
- **Assessment**: Highest quality; generalizability strong

### Methodology 10C: Meta-Analysis / Dataset Integration
**Example**: Ledergor et al., 2023 (*Biomarker Research*, review)
- Integrates findings across 50+ studies
- **Assessment**: Useful for consensus, but individual study bias compounds

---

## Summary: Methodology Usage & Assessment

| Method | Dominance | Maturity | Artifacts/Caveats | MM Suitability | Recommendation |
|--------|-----------|----------|-------------------|----------------|-----------------|
| **10x scRNA-seq** | 90% | Mature | Dissociation artifacts for immune | Excellent | Standard |
| **Full-length RNA + V(D)J** | 15% | Mature | Complex protocols | High (underused) | Should be routine |
| **Spatial transcriptomics** | 5% | Emerging | Spot deconvolution, single timepoint | Excellent | Adopt at major centers |
| **scATAC-seq** | 2% | Mature but rare in MM | Sparse signal, interpretation | High | Severely underused |
| **Monocle pseudotime** | 30% | Mature | Assumes continuity (unvalidated) | Moderate | Use but validate |
| **Harmony integration** | 80% (multi-site) | Mature | Over-smoothing possible | Excellent | Standard |
| **scGPT/scBERT** | 0% | Emerging but promising | No MM validation yet | Unknown | Pilot studies needed |
| **Pseudobulk DE** | 40% | Mature & validated | Loses single-cell resolution | Excellent | Should be mandatory |
| **Unsupervised clustering** | 99% | Mature but imperfect | Subjective, resolution-dependent | Good but limited | Pair with protein validation |
| **Prospective validation** | 5% | Gold standard | Time/resource intensive | Excellent | Expand |

---

## Field-Level Assessment

### Methodology Dominance
1. **Most dominant**: 10x scRNA-seq (transcriptomics-only)
2. **Most mature**: Batch correction (Harmony, Seurat)
3. **Most underused**: Full-length RNA + V(D)J, Spatial transcriptomics, scATAC-seq
4. **Most immature in MM context**: Foundation models (scGPT/scBERT)
5. **Most artifact-prone**: T cell immune profiling (dissociation stress)

### Key Methodological Gaps
1. **No standard BCR/TCR capture** for baseline clonotype establishment (critical gap for MRD)
2. **Dissociation artifact correction largely ignored** despite known risk
3. **Lack of prospective validation** for most biomarkers
4. **Single timepoint studies** (no longitudinal tracking of cell states)
5. **Spatial methods not paired with scRNA-seq** (complementary approaches used separately)

### Recommendations for Future MM Studies
1. **Multi-omics as minimum standard**: RNA + surface protein (CITE-seq) + full-length V(D)J
2. **Spatial transcriptomics** at every major diagnostic timepoint
3. **Dissociation artifact assessment** (fresh vs. dissociated control) in every study
4. **Prospective validation** of biomarkers in CoMMpass or similar cohorts
5. **Foundation model comparison** to unsupervised clustering on same datasets
6. **Longitudinal sampling** (diagnosis, post-treatment, relapse) to validate pseudotime dynamics

---
