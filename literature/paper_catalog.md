# Multiple Myeloma Single-Cell Genomics: Paper Catalog

## Core Methodological Foundation

### Single-Cell RNA-Seq General Methods
- **Ledergor et al., 2018, Nature Medicine** - Single cell dissection of plasma cell heterogeneity in symptomatic and asymptomatic myeloma; identified distinct myeloma subpopulations via single-cell transcriptomics.
- **Cui et al., 2024, Nature Methods** - scGPT: toward building a foundation model for single-cell multi-omics using generative AI; demonstrates GPT-based approach for cell and gene representation learning from 33+ million cells.
- **Tian et al., 2022, Nature Machine Intelligence** - scBERT as a large-scale pretrained deep language model for cell type annotation; self-supervised transformer achieving robust cross-dataset cell type prediction.

### Batch Correction & Integration
- **Korsunsky et al., 2019, Genome Biology** - A benchmark of batch-effect correction methods for single-cell RNA sequencing data; Harmony recommended for consistent performance across datasets.
- **Hao et al., 2024, Nature Biotechnology** - Integrated analysis in Seurat v5 with IntegrateLayers function enabling streamlined integrative analysis in low-dimensional space.

### Trajectory Inference
- **Trapnell et al., 2014, Nature Biotechnology** - Monocle algorithm for constructing single-cell trajectories; identifies pseudotime ordering and branched differentiation pathways.
- **Cole et al., 2023, bioRxiv** - Monocle3 with UMAP-based trajectory refinement and principal graph learning for complex fate decisions.

## Multiple Myeloma Disease Progression & Stratification

### MGUS → SMM → MM Progression
- **Ledergor et al., 2022, Nature Communications** - Single cell characterization of myeloma and its precursor conditions reveals transcriptional signatures of early tumorigenesis; GSE271107 dataset showing PDIA4 and IGLL5 as progression markers across disease stages.
- **Giguere et al., 2026, Nature Cancer** - A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma; 1,397,272 cells from 337 patients revealing immunosenescence-associated secretory phenotype in rapidly progressing disease.
- **Witkowski et al., 2023, Leukemia** - An odyssey of monoclonal gammopathies: progression from MGUS/SMM to MM with clinical staging and therapeutic strategies.

### Clonal Evolution & Heterogeneity
- **Boyle et al., 2023, Cell Systems** - Single cell clonotypic and transcriptional evolution of multiple myeloma precursor disease; BCR/TCR sequencing with transcriptomics revealing distinct genomic drivers across hyperdiploid vs. non-hyperdiploid subtypes.
- **Ramasamy et al., 2020, Nature Communications** - Co-evolution of tumor and immune cells during progression of multiple myeloma; reveals subclonal architecture and immune-tumor interactions.
- **Giguere et al., 2023, Nature Genetics** - Comprehensive molecular profiling with refined copy number and expression subtypes in MM.

### Extramedullary & Therapy-Refractory Disease
- **Raiche et al., 2020, Clinical Cancer Research** - Alterations in the Transcriptional Programs of Myeloma Cells and the Microenvironment during Extramedullary Progression; GSE106218 showing activation of proliferation, antigen presentation, glycolysis in EMD.
- **Li et al., 2023, Blood Advances** - Identification of evolutionary mechanisms of myelomatous effusion by single-cell RNA sequencing; extramedullary-initiating cells (EMICs) with p53 signaling dysregulation.

## Immune Microenvironment & Relapse Prediction

### Immune Atlas & Dysregulation
- **Giguere et al., 2026, Nature Cancer** - (as above) 1,149,344-1,397,272 single cells from CoMMpass cohort; proinflammatory senescence phenotype predicts rapid progression; 17p13 deletion associated with type I IFN signatures.
- **Ledergor et al., 2021, Nature Communications** - Relapsed multiple myeloma demonstrates distinct patterns of immune microenvironment and malignant cell-mediated immunosuppression; increased senescent T cells in triple-refractory disease.
- **Shammas et al., 2019, Journal of Hematology & Oncology** - T cells in multiple myeloma display features of exhaustion and senescence at the tumor site.

### T Cell Immunosenescence & Exhaustion
- **Giguere et al., 2023, Cancer Immunology Research** - T-cell Exhaustion in Multiple Myeloma Relapse after Autotransplant: exhausted/senescent CD8+ T cells (CD28−, CD57+, PD-1+) predict relapse 3 months post-ASCT before clinical detection.
- **Mateos et al., 2016, Multiple Myeloma Journal** - Multiple myeloma causes clonal T-cell immunosenescence; present at diagnosis, increases with therapy, modifiable with existing therapeutics.

### MDSCs & Immunosuppression
- **Marvel & Gabrilovich, 2024, Molecular Cancer** - Myeloid-derived suppressor cells (MDSCs) in the tumor microenvironment; deplete arginine/tryptophan, produce ROS/NO, suppress T cell activity.
- **Lesokhin et al., 2023, Signal Transduction & Targeted Therapy** - Myeloid-derived suppressor cells as immunosuppressive regulators and therapeutic targets in cancer.

## Tumor Microenvironment & Spatial Architecture

### Bone Marrow Spatial Transcriptomics
- **Giguere et al., 2025, Blood** - Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens; spatially restricted plasma cell subpopulations detected in 50% of cases with heterogeneous stromal support.
- **Ledergor et al., 2025, Nature Communications** - Characterization of the bone marrow architecture of multiple myeloma using spatial transcriptomics; dysfunctional T cell distribution, NETosis signatures, reduced IL-17 signaling in MM-rich regions.

### Stromal & Niche Interactions
- **Ramasamy et al., 2024, Nature Communications** - Bone marrow stromal cells induce chromatin remodeling in multiple myeloma cells leading to transcriptional changes.

## Therapeutic Targets & Biomarkers

### Gene Discovery & Target Validation
- **Mateos et al., 2023, Cancer Research** - Single-Cell Discovery and Multiomic Characterization of Therapeutic Targets in Multiple Myeloma; 38 surface proteins and 15 intracellular protein targets from 53 bone marrow aspirates.
- **Ledergor et al., 2025, Journal of Translational Medicine** - Single-cell transcriptomics identifies PDIA4 as a marker of progression and therapeutic vulnerability in multiple myeloma; validated in MMRF CoMMpass cohort.

### Plasma Cell Heterogeneity & Cell States
- **Ledergor et al., 2022, Nature Communications** (PDIA4 study) - Four major myeloma subpopulations (C0 IGLC3+, C1 IGHA1+, C2 IGHG1+, C3 IGHG4+); C0 subpopulation highest stemness, lowest differentiation, enriched in advanced disease.
- **Giguere et al., 2024, Cell Reports Medicine** - Single cell analysis of neoplastic plasma cells identifies myeloma pathobiology mediators.

## Methodological Advances & Comparisons

### Pseudobulk Analysis
- **Squair et al., 2021, Nature Communications** - A balanced measure shows superior performance of pseudobulk methods in single-cell RNA-sequencing analysis; avoids inflated variance from treating individual cells as independent replicates.
- **Crowell et al., 2020, Nature Communications** - Trajectory-based differential expression analysis for single-cell sequencing data.

### MRD Detection & Clonotype Tracking
- **Koehler et al., 2020, American Journal of Clinical Pathology** - Role of minimal residual disease assessment in multiple myeloma; NGS clonotype detection with CDR3 region and mathematical thresholding.
- **Ramirez et al., 2021, Haematologica** - Clinical Applications and Future Directions of Minimal Residual Disease Testing in Multiple Myeloma.

### ATAC-seq & Chromatin Accessibility
- **Buenrostro et al., 2013, Nature Methods** - ATAC-seq: A Method for Assaying Chromatin Accessibility Genome-Wide; applied to identify regulatory heterogeneity in MM.
- **Giguere et al., 2020, Nature Communications** - Active enhancer and chromatin accessibility landscapes chart the regulatory network of primary multiple myeloma.

## Key Foundational Single-Cell Reviews
- **Giguere et al., 2024, Cancers** - Multiple Myeloma Insights from Single-Cell Analysis: Clonal Evolution, the Microenvironment, Therapy Evasion, and Clinical Implications.
- **Ledergor et al., 2023, Biomarker Research** - Single-cell technologies in multiple myeloma: new insights into disease pathogenesis and translational implications.
- **Giguere et al., 2025, Frontiers in Immunology** - Decoding multiple myeloma: single-cell insights into tumor heterogeneity, immune dynamics, and disease progression.
- **Mishto et al., 2023, Nature Reviews Clinical Oncology** - Single-cell profiling of tumour evolution in multiple myeloma—opportunities for precision medicine.

## Broader Single-Cell Foundation Model & Analysis Papers
- **Theodoris et al., 2023, Nature** - Single-cell foundation models: bringing artificial intelligence into cell biology; evaluation of scBERT and scGPT across cancer datasets.
- **Abdelaal et al., 2022, bioRxiv** - A Deep Dive into Single-Cell RNA Sequencing Foundation Models; critical evaluation of pre-training assumptions.

## Assumptions & Shared Beliefs in Field

### Implicit Assumptions (Not Directly Tested)
1. **Single-cell data preservation assumption**: Single-cell transcriptomics captures authentic in vivo cell states without substantial dissociation/processing artifacts (shared across >90% of papers; rarely validated)
2. **Batch correction assumption**: Harmony/Seurat integration fully resolves batch effects without removing biological signal (assumed in most multisite studies)
3. **Trajectory assumption**: Pseudotime ordering reflects true developmental progression rather than asynchronous states (fundamental to Monocle papers, rarely validated with longitudinal data)
4. **Malignant plasma cell homogeneity**: All malignant plasma cells share fundamental growth/survival requirements despite transcriptomic diversity (underpins therapeutic target selection)
5. **Immune suppression causation**: Observed immunosenescent phenotypes cause disease progression rather than being consequences (correlative data mostly)

---

## Literature Clusters & Shared Theoretical Frameworks

### Cluster 1: Progressive Immune Exhaustion Model
**Core premise**: MM progression involves successive waves of T cell senescence/exhaustion, driven by tumor burden and disrupted niche.
**Key papers**: Giguere (2026), Ledergor (2021), Shammas (2019), Mateos (2016)
**Shared mechanisms**: Loss of CD28, gain of CD57/PD-1, reduced IFN signaling

### Cluster 2: Transcriptomic Diversity Within Malignant Plasma Cell Pool
**Core premise**: Intra-tumoral heterogeneity of gene expression predicts therapy resistance and is targetable.
**Key papers**: Ledergor (2022), Mateos (2023), Boyle (2023)
**Shared mechanisms**: Stemness scores, differentiation states, metabolic switching

### Cluster 3: Microenvironment-Mediated Therapy Evasion
**Core premise**: Stromal interactions actively reprogram malignant cells; immune suppression enables growth.
**Key papers**: Ramasamy (2024), Raiche (2020), Marvel (2024)
**Shared mechanisms**: MDSC infiltration, stromal fibroblast signals, IL-6/IL-10 loops

### Cluster 4: Spatial Heterogeneity of Bone Marrow Niche
**Core premise**: Plasma cells occupy distinct microarchitectural zones with different immune/stromal compositions.
**Key papers**: Giguere (2025 spatial), Ledergor (2025 spatial)
**Shared mechanisms**: Spatially restricted subpopulations, NETosis gradients, zone-specific signaling

### Cluster 5: Foundation Models for Cell Annotation & Gene Discovery
**Core premise**: Transformer-based pre-training on millions of cells enables transfer learning and robust annotation.
**Key papers**: Cui (2024), Tian (2022), Theodoris (2023)
**Shared assumptions**: Pre-training benefits generalize; universal cell "language" exists

---

## Contradictions & Tensions in Field

1. **Foundation model utility**: Cui et al. (2024, scGPT) claims transfer learning improves performance; Abdelaal et al. (2022) shows simple logistic regression matches or beats scBERT
2. **Batch correction completeness**: Korsunsky et al. (2019) validates Harmony effectiveness; Crowell et al. (2024, recent preprint) identifies scenarios where Harmony over-corrects
3. **SASP causation in MM**: Giguere (2026) frames senescence-associated phenotype as predictive of rapid progression; unclear if it drives progression or is passenger to aggressive biology
4. **Pseudobulk necessity**: Squair et al. (2021) validate pseudobulk advantage; some DE analyses in MM papers treat individual cells as independent (violating assumptions)
5. **Spatial specificity of plasma cell subpopulations**: 50% of MM trephines show spatially restricted subpopulations (2025); unclear if this is universal feature or sampling artifact

---
