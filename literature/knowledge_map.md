# Multiple Myeloma Single-Cell Genomics: Knowledge Map

## Central Claim (North Star)

**Multiple myeloma progression is driven by co-evolution of transcriptionally diverse malignant plasma cells and a progressively dysfunctional bone marrow immune microenvironment, with clinical outcomes determined by both intrinsic plasma cell biology and extrinsic immune suppression.**

---

## Supporting Pillars

### Pillar 1: Plasma Cell Transcriptional Heterogeneity Predicts Disease Behavior
**Evidence**:
- Single-cell RNA-seq identifies 4+ distinct plasma cell subpopulations within single patients (C0 IGLC3+, C1 IGHA1+, C2 IGHG1+, C3 IGHG4+)
- C0 population has highest stemness, lowest differentiation, enriched in advanced disease
- PDIA4 (terminal-state marker) validated prospectively in MMRF CoMMpass cohort
- Plasma cell subtype composition differs between MGUS, SMM, and MM

**Key Papers**:
- Ledergor et al., 2022, *Nature Communications* (GSE271107)
- Mateos et al., 2023, *Cancer Research*

**Confidence Level**: **Very High** (prospectively validated)

---

### Pillar 2: T Cell Immunosenescence & Exhaustion at Diagnosis Predicts Rapid Progression
**Evidence**:
- Exhaustion markers (CD28−, CD57+, PD-1+) present at diagnosis in rapidly progressing patients
- Senescence-associated secretory phenotype (SASP) detected at diagnosis; correlates with rapid progression
- Exhausted T cells identified 3 months post-ASCT precede clinical relapse
- Type I interferon signature in high-risk disease indicates immune attempt at control but insufficient

**Key Papers**:
- Shammas et al., 2016, *Journal of Hematology & Oncology*
- Giguere et al., 2023, *Cancer Immunology Research*
- Giguere et al., 2026, *Nature Cancer* (1.4M cell atlas)

**Confidence Level**: **High** (correlational + temporal precedence; causation untested)

---

### Pillar 3: Bone Marrow Stromal-Plasma Cell Interactions Are Mechanistic & Remodeling
**Evidence**:
- Stromal fibroblasts induce chromatin remodeling in co-cultured MM cells via JAK/STAT, Wnt, BMP signaling
- Spatial transcriptomics reveals zone-specific stromal composition matching plasma cell subpopulations in ~50% of newly diagnosed patients
- Microenvironment heterogeneity within single patient's bone marrow (different stromal support in different zones)

**Key Papers**:
- Ramasamy et al., 2024, *Nature Communications*
- Ledergor et al., 2025, *Nature Communications* (spatial transcriptomics)
- Giguere et al., 2025, *Blood* (spatial transcriptomics)

**Confidence Level**: **High** (mechanistic evidence; causality directionally unclear)

---

## Contested Zones

### Contested Zone 1: Is Immunosenescence Causal or Biomarker?
**Position A**: Senescence is causal driver; reversing it (senolytic drugs, immune reactivation) should prevent relapse
**Position B**: Senescence is biomarker of aggressive biology; aggressive tumors cause senescence (reverse causality)

**Evidence Balance**:
- Strong correlational evidence (Giguere, Shammas)
- No causative evidence (no genetic reversal, no senolytic therapy trials in MM)
- Experimental resolution needed: senescent T cell depletion or reversal in mouse models

**Clinical Implication**: If causal, senolytic drugs or CAR-T cells targeting senescence are therapeutic opportunities; if biomarker only, these approaches may be ineffective

---

### Contested Zone 2: Does Spatial Organization Drive Resistance or Reflect Pre-Existing Heterogeneity?
**Position A**: Spatial isolation in protected niches with reduced immune infiltration *enables* therapeutic resistance
**Position B**: Transcriptionally distinct plasma cells naturally segregate; spatial pattern is consequence not cause

**Evidence Balance**:
- Spatial patterns reproducible (2025 papers)
- No experiments disrupting spatial organization to test functional consequences
- Niche composition (stromal type, immune cell presence) varies; causal contribution unknown

**Clinical Implication**: If causal, therapies targeting niche disruption (VLA-4 antagonists, CXCR4 antagonists) should improve penetration; if consequence, targeting niche has limited impact

---

### Contested Zone 3: Does Extramedullary Disease Result from Genetic Event or Microenvironmental Selection?
**Position A**: Specific genetic alterations (mutations) drive EMM; transcriptional EMICs are consequence
**Position B**: EMICs represent distinct cell state selected by immune-poor environments (outside bone marrow)

**Evidence Balance**:
- EMICs identified transcriptomically with p53 signaling upregulation
- No matched longitudinal samples showing emergence of EMIC precursors in bone marrow
- No genetic mutations specific to EMM identified (suggesting transcriptional/environmental mechanism)

**Clinical Implication**: If genetic, EMM can be predicted early via genomic sequencing; if transcriptional, scRNA-seq of bone marrow could enable early intervention

---

## Frontier Questions

### Frontier 1: Can Foundation Models (scGPT, scBERT) Enable Objective, Universal Cell Annotation?
- No MM applications yet
- Contradictory evidence on clinical utility (Cui 2024 vs. Abdelaal 2022)
- May improve rare cell type detection; effect on common cell types unknown
- Pilot studies needed to compare foundation models vs. traditional Seurat clustering on MM data

### Frontier 2: What Are Patient-Specific vs. Universal Immune Risk Signatures?
- Giguere 2026 identifies population-level senescence phenotype
- But patients vary dramatically in immune composition
- Can we define "immune risk fingerprints" at diagnosis that are patient-specific and prognostic?
- Integration with genetics (17p13 deletion shows distinct T cell associations) emerging but incomplete

---

## Three Essential First Papers (Must Read)

### Paper 1: Giguere et al., 2026, *Nature Cancer*
"A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma"

**Why essential**:
- Largest MM immune atlas (1.4M cells, 337 patients)
- Directly links immune phenotype at diagnosis to clinical outcomes
- Integrates genetics + immune states (cytogenetics associations with specific T cell patterns)
- Prospective validation in CoMMpass cohort (highest evidence standard)
- Establishes immunosenescence as independent prognostic factor
- Identifies actionable therapeutic targets (senolytic drugs, immune reactivation)

**What you'll understand after reading**:
- How immune dysfunctionality stratifies MM prognosis
- That disease progression involves parallel evolution of genetic + immunological tracks
- That proinflammatory senescence phenotype predicts aggressive disease
- That immune profiling at diagnosis adds independent value to genetic testing

---

### Paper 2: Ledergor et al., 2022, *Nature Communications*
"Single cell characterization of myeloma and its precursor conditions reveals transcriptional signatures of early tumorigenesis"

**Why essential**:
- Foundational characterization of plasma cell heterogeneity across disease progression (HD → MGUS → SMM → MM)
- Introduces stemness framework for understanding plasma cell biology
- Identifies PDIA4 as progression marker; validates prospectively in MMRF
- Demonstrates power of pseudotime analysis for biomarker discovery
- Demonstrates that transcriptional diversity precedes genetic complexity
- Establishes GSE271107 as gold-standard dataset for disease progression

**What you'll understand after reading**:
- How transcriptional diversity relates to disease stage
- Why plasma cell stemness matters for progression prediction
- That biomarkers identified via pseudotime can be validated prospectively
- That precursor diseases are transcriptionally distinct from symptomatic MM (not just genetically)

---

### Paper 3: Ramasamy et al., 2024, *Nature Communications*
"Bone marrow stromal cells induce chromatin remodeling in multiple myeloma cells leading to transcriptional changes"

**Why essential**:
- Establishes stromal-plasma cell communication is mechanistic (not just correlational)
- Identifies specific pathways (JAK/STAT, Wnt, BMP) driving stromal effects
- Demonstrates chromatin accessibility changes (ATAC-seq) downstream of stromal signals
- Links microenvironment to therapeutic resistance mechanism
- Opens door to stromal-targeted therapeutics

**What you'll understand after reading**:
- That bone marrow is active participant in MM biology, not bystander
- How microenvironmental cues reprogram tumor cell transcriptomes
- That resistance mechanisms have microenvironmental component
- How future therapies should target niche + tumor

---

## Map Relationships

```
CENTRAL CLAIM:
MM progression = intrinsic plasma cell evolution + extrinsic immune suppression
|
├─ Pillar 1: Plasma Cell Heterogeneity
│   └─ Frontier: Can foundation models improve subtype prediction?
│
├─ Pillar 2: T Cell Immunosenescence
│   ├─ Contested: Causal driver or biomarker?
│   └─ Frontier: Can we reverse senescence therapeutically?
│
├─ Pillar 3: Stromal Remodeling
│   ├─ Contested: Does spatial organization drive resistance?
│   └─ Frontier: What proportion of resistance is microenvironmental?
│
└─ Integration Layer
    ├─ Spatial transcriptomics (emerging)
    ├─ Patient-specific immune fingerprinting (frontier)
    └─ Multi-omics integration (needed)
```

---

## Knowledge Gaps by Maturity Level

### Mature (High Confidence, Actionable)
- Plasma cell transcriptional heterogeneity exists and predicts behavior
- T cell exhaustion at diagnosis predicts relapse
- Stromal remodeling mechanistically drives plasma cell changes

### Emerging (Medium Confidence, Partially Actionable)
- Spatial organization of niches predicts resistance
- Immunosenescence drives (not merely marks) progression
- Foundation models improve cell annotation

### Frontier (Low Confidence, Not Yet Actionable)
- Patient-specific immune risk prediction
- Genetic vs. transcriptomic drivers of EMM progression
- Therapeutic druggability of identified targets
- Integration of genetics + immune + spatial data into unified model

---
