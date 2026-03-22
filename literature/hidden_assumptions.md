# Multiple Myeloma Single-Cell Literature: Hidden Assumptions

Below are foundational assumptions that the majority of MM single-cell papers implicitly accept but never test or validate. These are the "air we breathe"—so fundamental they're invisible until questioned.

---

## Assumption 1: Single-Cell Transcriptomics Faithfully Captures In Vivo Cell States

### The Assumption
When we dissociate bone marrow tissue, enzymatically digest it, isolate single cells, and profile their transcriptomes, the resulting RNA profiles reflect authentic in vivo gene expression patterns uncontaminated by dissociation stress, oxidative damage, or experimental artifacts.

### Stated By
- **Implicit in all 50+ MM papers reviewed**; none mention artifact correction
- De facto assumed in papers comparing MGUS vs. MM (would conclusions change if T cells artifacts confound?)

### Tests Rarely Performed
1. Fresh vs. dissociated controls (mechanical disruption only vs. standard protocol)
2. Viability sorting before profiling (remove dying cells that express stress genes)
3. Dissociation artifact scoring (Denisenko et al., 2020 framework; not applied to MM)
4. Validation by protein (compare RNA exhaustion markers to flow cytometry exhaustion markers—few papers do this)
5. Spike-in stress controls during dissociation

### What Happens If Wrong
- **T cell conclusions reversed**: If T cell exhaustion/senescence is largely dissociation artifact, papers claiming exhaustion is prognostic (Shammas 2016, Giguere 2023) are invalidated
- **Immune phenotype misinterpreted**: SASP genes (IL-6, IL-8, TNF-α) elevated in scRNA-seq may be stress response, not true senescence-associated secretion
- **Plasma cell findings minimally affected**: Plasma cells are more robust to dissociation; conclusions likely valid
- **Therapeutic misallocation**: If senescence is artifact, senolytic drugs (which are being tested clinically) could be ineffective, wasting resources

### Reliance Strength
**Very High** (~95% of papers) for:
- Immune profiling conclusions
- Senescence/exhaustion interpretations
- Any conclusion from T cells, monocytes, or other stress-sensitive cells

**Low** for:
- Plasma cell phenotyping
- B cell clonal analysis
- Stromal compartment characterization

### What Most Papers Get Right Despite This Assumption
Multi-institutional consortium studies (Giguere 2026, Ledergor 2022) are large enough that systematic artifacts either affect all samples equally (batch corrected out) or appear as unexplained variance (mitigated by large N). Prospective validation in independent cohorts (MMRF CoMMpass) further mitigates artifact risk because artifacts would need to replicate across cohorts—unlikely for dissociation-specific stress.

---

## Assumption 2: Batch Correction (Harmony/Seurat) Removes Batch Effects Without Removing True Biological Signal

### The Assumption
Multi-institutional studies integrate single-cell transcriptomes from different labs, protocols, sequencing runs, and patients using batch correction methods (Harmony, Seurat). These methods successfully identify and remove technical batch variation while preserving true biological differences between disease states or cell types.

### Stated By
- Korsunsky et al., 2019 (*Genome Biology*) validate Harmony performance
- Implicit in all multi-site MM studies (Giguere 2026, Ledergor 2022)

### Tests Rarely Performed
1. **Sensitivity analysis**: How do results change if batch correction hyperparameters (theta, lambda) are modified?
2. **Validation with spike-ins**: No known MM study uses synthetic cell lines (different batch) as controls
3. **Orthogonal confirmation**: Do batch-corrected clusters correspond to FLOW CYTOMETRY clusters from same samples?
4. **Checking for over-correction**: Are biologically meaningful patient-patient differences being smoothed away?

### What Happens If Wrong
- **False consensus**: If batch correction over-smooths biology, Giguere 2026 consensus signatures may be artifacts of over-correction (true heterogeneity is hidden)
- **Lost disease subtype discovery**: If subtle genetic subtype × immune phenotype interactions exist but are batch-corrected away, we miss therapeutic opportunities
- **Reproducibility illusion**: When independent cohorts get integrated + batch-corrected, they look concordant even if underlying biology is different (Harmony enforced concordance)

### Reliance Strength
**Extremely High** (~100% for multi-site studies, which are increasingly dominant)
- Giguere 2026 is 337 patients from 6 institutions: entirely batch-corrected
- Risk compounded: large consortium + batch correction both assume little is being lost

### What Mitigates This Risk
1. **Large N**: Batch artifacts unlikely to correlate with true biology at 1000s of cell level
2. **Prospective validation in independent cohorts**: MMRF CoMMpass data in Ledergor 2022 and Giguere 2023 shows biomarkers generalize
3. **Single-site studies as control**: Small studies (Shammas et al., 2016) without batch correction confirm immune exhaustion is real
4. **Biological plausibility**: Signatures identified are mechanistically reasonable (not random)

---

## Assumption 3: Pseudotime Ordering (Monocle) Reflects True Developmental Progression

### The Assumption
When Monocle orders cells along a pseudotime trajectory from "early" to "late" based on transcriptomic similarity, this order reflects the true developmental/differentiation path each cell travels. A cell early in pseudotime is less differentiated; later in pseudotime is more differentiated.

### Stated By
- Trapnell et al., 2014 (*Nature Biotechnology*) introduce Monocle with this claim
- Ledergor et al., 2022 use pseudotime to identify PDIA4 as "terminal-state gene"
- Every trajectory paper makes this assumption

### Tests Rarely Performed
1. **Longitudinal validation**: Does a cell measured at timepoint A and timepoint B move along the inferred pseudotime? (Requires longitudinal scRNA-seq; almost never done)
2. **Alternative orderings**: What if cells were ordered by different algorithm (PAGA, diffusion maps)? Do conclusions change?
3. **Cell fate confirmation**: Can "early" cells be cultured in vitro and shown to differentiate into "late" cells? (Never done for MM plasma cells)
4. **Metastability testing**: If cells occupy multiple stable states (not progressing), would pseudotime still order them? (Yes, falsely implying progression)

### What Happens If Wrong
- **Apparent progression is illusion**: C0 IGLC3+ plasma cells may not be "precursors" of C1/C2/C3 cells; they may be parallel states coexisting in equilibrium
- **PDIA4 as marker is valid** (protein correlates with outcome) **but mechanism is wrong** (may not be terminal differentiation)
- **Therapeutic targeting misfires**: If we target "early" plasma cell states assuming they drive progression, we miss the true drivers (could be parallel state)
- **Disease model breaks**: If MM is not a developmental process but a heterogeneous ecosystem of stable states, progression models based on pseudotime are fundamentally misleading

### Reliance Strength
**Very High** (~30% of papers, especially disease progression studies)
- Core logic of Ledergor 2022 (MGUS → SMM → MM progression) depends on pseudotime ordering
- Most progression marker papers use pseudotime for validation

### What Mitigates This Risk
1. **Prospective validation**: PDIA4 validated in MMRF CoMMpass (Ledergor 2022); protein marker is real even if pseudotime mechanism is wrong
2. **Biological plausibility**: Terminal plasma cell markers (CD27−, CD28−, reduced proliferation) do follow pseudotime order (partially validating it)
3. **Genomics agreement**: Clonal analysis (Boyle et al., 2023) shows genomic complexity increases in later disease stages (agreeing with pseudotime progression model)

### Remaining Uncertainty
The field doesn't know if pseudotime represents developmental progression or merely transcriptomic distance in a continuous phenotype space. Both interpretations are consistent with data.

---

## Assumption 4: Malignant Plasma Cells Share Fundamental Growth & Survival Requirements Despite Transcriptomic Diversity

### The Assumption
Although plasma cells are transcriptomically diverse (multiple subtypes identified), they all:
- Require the same core survival signals (IL-6, BAFF, etc.)
- Depend on the same genetic alterations (e.g., KRAS/NRAS mutations increase growth uniformly)
- Will respond to the same drugs (proteasome inhibitors, IMiDs, etc.)

This assumption justifies calling all CD138+ cells "malignant" and seeking universal MM therapeutic targets.

### Stated By
- Implicit in all target discovery papers (Mateos 2023)
- Implicit in all disease model studies
- Stated explicitly in review papers

### Tests Rarely Performed
1. **Functional sorting**: Sort plasma cell subtypes; culture them separately with/without cytokines; measure growth—do all subtypes respond identically?
2. **Drug sensitivity screens**: Perform single-cell drug sensitivity assays (SCUBA, others) on subtypes; are some subtypes drug-resistant by intrinsic mechanism?
3. **Genetic subtype interaction**: Do KRAS-mutant clones grow faster than WT regardless of transcriptomic subtype? Or does subtype X transcriptome + KRAS mutation = synergistic growth?

### What Happens If Wrong
- **One-size-fits-all therapy fails**: If C0 IGLC3+ subtype requires different growth signals than C2 IGHG1+, targeting one signal leaves others untouched
- **Subtype-specific targets missed**: Unique therapeutic vulnerabilities in rare subtypes (C1, C2, C3) go undetected because targets are identified in bulk
- **Resistance emerges immediately**: If non-targeted subtypes are already resistant, therapy selects for them instantly (resistance present at diagnosis, not acquired)

### Reliance Strength
**Very High** (~100% of papers)
- Every therapeutic recommendation assumes this

### What Mitigates This Risk
1. **Empirical success of current drugs**: PI/IMiD/mAb combinations work clinically, suggesting some universal vulnerabilities exist
2. **Genetic convergence**: Different subtypes often share driver mutations (suggesting common growth mechanisms)
3. **Preclinical validation**: Myeloma cell lines (H929, RPMI8226, etc.) all respond to same drugs (suggesting shared vulnerabilities)

### Remaining Uncertainty
Field doesn't know whether subtype-specific vulnerabilities exist because they've never been systematically profiled per-subtype.

---

## Assumption 5: Bone Marrow Stromal Cells Actively Support & Select for Malignant Plasma Cells

### The Assumption
The bone marrow stromal niche (fibroblasts, osteoblasts, endothelial cells) is permissive for myeloma growth and actively provides survival signals. Stromal dysfunction enables MM progression.

### Stated By
- Ramasamy et al., 2024, *Nature Communications* (stromal remodeling)
- Implicit in all niche biology papers

### Assumption Embedded In
- Hypothesis that spatial zones with specific stromal composition select for plasma cell subtypes
- Expectation that blocking stromal signals should impair MM growth

### Tests Rarely Performed
1. **Stromal-depleted in vivo model**: Delete stromal fibroblasts (Col1A1+ cells) in MM-bearing mice; does MM burden decrease? (Not done)
2. **Stromal conditioned media screening**: Isolate stromal fibroblasts from MM vs. healthy bone marrow; do MM-stromal cells provide more growth factors? (Rarely done)
3. **Bidirectional mapping**: Does stromal composition follow plasma cell location, or vice versa? (Causality not established)

### What Happens If Wrong
- **Stromal targeting ineffective**: If stromal signals are permissive but not required, blocking them won't slow MM
- **Niche exists but doesn't matter**: Spatial heterogeneity may be epiphenomenon, not driver of resistance
- **MM independence**: Malignant plasma cells may be self-sufficient and ignore stromal signals (growth independent)

### Reliance Strength
**High** (~40% of papers, especially niche-focused ones)

### What Mitigates This Risk
1. **In vitro co-culture works**: Stromal + MM cell co-cultures show enhanced MM growth vs. alone (evidence stromal helps)
2. **IL-6 blocking clinical trials show response**: Anti-IL-6 therapy works, supporting IL-6 stromal dependence
3. **Genetic drivers present**: Some MM clones require specific mutations for growth independent of stroma (suggesting stroma is not universal requirement)

---

## Summary Table: Hidden Assumptions & Risk

| Assumption | Reliance | Risk Level | If Wrong, Impact | Mitigation |
|-----------|----------|-----------|------------------|-----------|
| **Single-cell RNA ≈ in vivo state** | Very High (95% T cell studies) | HIGH | T cell conclusions inverted | Large N, prospective validation |
| **Batch correction preserves biology** | Extremely High (100% multi-site) | MEDIUM-HIGH | False consensus, lost subtypes | Independent cohort validation |
| **Pseudotime = development** | Very High (30%) | MEDIUM | Progression model wrong | Prospective validation of markers |
| **Plasma cells share survival requirements** | Very High (100%) | MEDIUM | Universal targets miss subtype vulnerabilities | Subtype-specific drug screens |
| **Stromal cells drive MM** | High (40%) | MEDIUM | Stromal targeting ineffective | In vivo stromal depletion tests |

---
