# Multiple Myeloma Single-Cell Literature: Concept Lineage Analysis

Three foundational concepts that recur across the MM single-cell literature are traced through their intellectual evolution.

---

## Concept 1: Plasma Cell Heterogeneity & Subpopulations

### Introduction & Origination
**Originator**: Ledergor et al., 2018, *Nature Medicine*
- **Paper**: "Single cell dissection of plasma cell heterogeneity in symptomatic and asymptomatic myeloma"
- **Core claim**: Single-cell RNA-seq reveals that myeloma is not a homogeneous clone but comprises multiple transcriptionally distinct plasma cell subpopulations with different biological properties
- **Context**: Prior bulk RNA-seq and genomics could identify genetic clones but not transcriptional diversity within clones
- **Key innovation**: Applied scRNA-seq directly to myeloma to show phenotypic diversity independent of genetic subclones

### Refinement & Extension (2020–2023)
**Refiners**: Ledergor et al., 2022, *Nature Communications* (GSE271107 study)
- **Paper**: "Single cell characterization of myeloma and its precursor conditions reveals transcriptional signatures of early tumorigenesis"
- **Advancement**:
  - Formalized four major myeloma cell subpopulations (C0 IGLC3+, C1 IGHA1+, C2 IGHG1+, C3 IGHG4+)
  - Applied pseudotime analysis to map differentiation states
  - Identified PDIA4 as terminal-state-associated marker predictive of progression
  - Discovered IGLL5 elevation in disease states vs. precursors
  - Extended dataset to include healthy donors, MGUS, SMM, MM (temporal dimension)
- **Mechanistic insight**: Stemness scoring showed C0 IGLC3+ population has highest stemness and lowest differentiation, enriched in advanced disease
- **Validation**: Validated PDIA4 prognostic significance in MMRF CoMMpass cohort (prospective)

**Further refinement**: Mateos et al., 2023, *Cancer Research*
- **Paper**: "Single-Cell Discovery and Multiomic Characterization of Therapeutic Targets in Multiple Myeloma"
- **Advancement**: Combined scRNA-seq with surface protein profiling (38 surface proteins + 15 intracellular targets identified)
- **Clinical translation**: Defined therapeutic targets within subpopulations; moved beyond descriptive to functional target prediction
- **Scale**: 53 bone marrow aspirates, 41 patients
- **Technical innovation**: Multi-omic approach connects transcriptomics to actionable protein targets

### Challenge & Alternative Perspective (2024–present)
**Challenger/Alternative view**: Theodoris et al., 2023, *Nature*; Abdelaal et al., 2022
- **Papers**: "Single-cell foundation models: bringing artificial intelligence into cell biology"
- **Alternative claim**: The subpopulations identified by traditional clustering may be continuous distributions artificially divided by computational algorithms; pre-trained models may provide more robust, objective annotation
- **Caveat**: Foundation models trained on bulk data may obscure the true granular heterogeneity that traditional clustering reveals
- **Tension**: Do we have real discrete subpopulations or continuous phenotypes? Both probably true; methodology determines which we "see"

### Current Consensus (2025–2026)
**Consensus position**:
- Plasma cell transcriptional heterogeneity is real and spans multiple biological dimensions (stemness, differentiation state, metabolic program, immune interaction state)
- These dimensions exist as a continuum but are biologically meaningful
- Subpopulations defined by key transcriptional hubs (e.g., immunoglobulin heavy chain usage) are reproducible and correspond to different growth dynamics
- Individual cells can occupy intermediate states, but populations show meaningful stratification by stemness and differentiation
- Foundation models may improve annotation robustness; trajectory methods (Monocle) may better capture continuum than discrete clustering

**Remaining uncertainty**:
- Whether "C0 IGLC3+ stemness" is intrinsic property or result of microenvironmental positioning
- How dynamic are transitions between subpopulations (weeks? months?)
- Whether subpopulations represent developmental hierarchy or independent niches

---

## Concept 2: Immune Microenvironment Dysfunction Driving Disease Progression

### Introduction & Origination
**Originator**: Shammas et al., 2016, *Journal of Hematology & Oncology*
- **Paper**: "T cells in multiple myeloma display features of exhaustion and senescence at the tumor site"
- **Core claim**: The bone marrow immune microenvironment in MM is not merely permissive but actively dysfunctional, with T cells displaying markers of exhaustion (PD-1+, CD28−, CD57+)
- **Context**: Prior work emphasized immune suppression as consequence; Shammas framed it as intrinsic T cell programming
- **Key innovation**: Applied flow cytometry and transcriptomics to bone marrow T cells specifically, not just bulk immune characterization

### Refinement & Mechanistic Deepening (2018–2023)
**Refiners**: Ledergor et al., 2021, *Nature Communications*; Mateos et al., 2016
- **Paper (Ledergor)**: "Relapsed multiple myeloma demonstrates distinct patterns of immune microenvironment and malignant cell-mediated immunosuppression"
- **Advancements**:
  - Characterized immune cell landscape across diagnosis → relapsed → triple-refractory MM
  - Found that relapsed clusters have increased senescent T cells and decreased early memory T cells
  - Myeloma plasma cells in relapsed disease upregulate MYC/E2F (proliferation) while downregulating interferon, TGF-β, IL-6, TNF-α signaling
  - Interpreted as: tumor becomes less dependent on inflammatory growth pathways; immune cells become progressively suppressed
- **Mechanistic insight**: Describes both tumor-intrinsic changes AND immune cell changes, suggesting co-evolution

**Expansion to senescence pathway**: Mateos et al., 2016, *Multiple Myeloma Journal*
- **Paper**: "Multiple myeloma causes clonal T-cell immunosenescence"
- **Advancement**:
  - Identified that immunosenescence is present at diagnosis (not acquired)
  - Increases with therapy
  - Can be modulated by existing therapeutics
  - Mechanistic: not just checkpoint exhaustion but cellular senescence program (SASP-related)

### Major Extension & Predictive Power (2023–2026)
**Major refiner**: Giguere et al., 2023, *Cancer Immunology Research*; then 2026, *Nature Cancer*
- **Paper (2023)**: "T-cell Exhaustion in Multiple Myeloma Relapse after Autotransplant"
- **Landmark advancement**:
  - Showed exhausted/senescent CD8+ T cells (CD28−, CD57+, PD-1+) present 3 months post-ASCT predict relapse **before** clinical disease is detected
  - Regulatory T cells decline early post-transplant
  - Provided temporal resolution linking T cell state → clinical outcome
  - Transforms concept from descriptive (exhaustion exists) to predictive (exhaustion predicts relapse)

- **Paper (2026)**: "A single-cell atlas characterizes dysregulation of the bone marrow immune microenvironment associated with outcomes in multiple myeloma" (*Nature Cancer*)
- **Monumental advancement**:
  - 1,397,272 single cells from 337 newly diagnosed patients (CoMMpass cohort)
  - Identified that proinflammatory immune senescence-associated secretory phenotype (SASP) in bone marrow at diagnosis predicts rapid progression
  - Found 17p13 deletion (poor prognostic marker) associated with type I interferon signature but likely insufficient (paradox)
  - Integrated immune states with genetic risk: cytogenetic risk shows heterogeneous T cell associations
  - Validates immunosenescence as independent prognostic factor alongside genetic markers

### Challenge & Alternative Mechanism (2021–2024)
**Challenger/Alternative view**: Marvel & Gabrilovich, 2024, *Molecular Cancer*; Lesokhin et al., 2023
- **Papers**: "Myeloid-derived suppressor cells (MDSCs) in the tumor microenvironment and their targeting in cancer therapy"
- **Alternative claim**: MDSC-mediated suppression may be primary mechanism, not T cell senescence per se
- **MDSCs operate via**:
  - Tryptophan depletion (IDO, arginase-1)
  - ROS and NO production
  - Adenosine-mediated suppression (CD39/CD73)
  - IL-10, TGF-β secretion
- **Tension**: Is T cell exhaustion caused by MDSC suppression, or is it independent intrinsic senescence? Likely both, unclear dominance

### Current Consensus (2026)
**Consensus position**:
- T cell dysfunction in MM is multifactorial: combines intrinsic senescence program + extrinsic MDSC-mediated suppression + stromal remodeling
- Immunosenescence phenotype (SASP+) is both marker AND mechanistic contributor to progression
- Senescence is present at diagnosis and worsens with therapy, suggesting it is fundamental to MM immunobiology not just consequence of tumor burden
- Early post-ASCT T cell exhaustion predicts relapse, enabling risk stratification
- Type I IFN signatures in high-risk disease represent immune attempts at control but insufficient (exhausted response)

**Remaining uncertainty**:
- Does reversing senescence (senolytic drugs) improve outcomes? (Not yet tested clinically)
- What proportion of suppression is T cell intrinsic vs. MDSC-mediated? (Varies by patient, likely)
- Are senescent T cells permanently lost or can they be reactivated? (Mixed evidence)
- What triggers senescence in MM context specifically? (Not fully defined)

---

## Concept 3: Spatial Microarchitecture & Niche-Dependent Plasma Cell Behavior

### Introduction & Origination (2018–2021)
**Originator**: Ramasamy et al., 2020, *Nature Communications* (and earlier spatial genomics work)
- **Paper**: "Co-evolution of tumor and immune cells during progression of multiple myeloma"
- **Core claim**: Malignant plasma cells do not evolve in isolation but in spatial context of their microenvironment; immune cells co-evolve with tumor
- **Context**: Prior focus on genetic clonal evolution; Ramasamy emphasized spatial/immunological co-evolution
- **Key innovation**: Combined transcriptomics with spatial reasoning and immune profiling to show coupled tumor-immune trajectories

### Refinement through Stromal Biology (2021–2024)
**Refiners**: Ramasamy et al., 2024, *Nature Communications*
- **Paper**: "Bone marrow stromal cells induce chromatin remodeling in multiple myeloma cells leading to transcriptional changes"
- **Advancement**:
  - Showed stromal cell interactions are not passive but actively reprogram myeloma cell chromatin
  - Applied ATAC-seq + RNA-seq to show stromal-induced enhancer accessibility changes
  - Described specific mechanistic pathways (JAK/STAT, Wnt, BMP) in stromal-myeloma communication
  - Moved beyond "stromal cells suppress immune function" to "stromal cells reprogram tumor transcriptome"

**Spatial mechanistic insight**: Ledergor et al., 2025, *Nature Communications*
- **Paper**: "Characterization of the bone marrow architecture of multiple myeloma using spatial transcriptomics"
- **Advancement**:
  - Used Visium-based spatial transcriptomics on formalin-fixed bone marrow samples
  - Found spatially restricted plasma cell subpopulations in 50% of newly diagnosed cases
  - Revealed zone-specific gene programs: NETosis and IL-17 signaling reduced in MM-PC-rich regions
  - Showed microenvironment heterogeneity: different stromal compositions in different marrow regions within same patient
  - First demonstration that plasma cell subpopulations occupy distinct spatial niches with matching stromal/immune context

### Integrated View: Microarchitecture + Immune State (2025)
**Major refinement**: Giguere et al., 2025, *Blood*
- **Paper**: "Profiling the spatial architecture of multiple myeloma in human bone marrow trephine biopsy specimens with spatial transcriptomics"
- **Complementary advancement**:
  - Higher resolution spatial data showing subcellular gene expression patterns
  - Discovered dysfunctional T cell distribution correlates with plasma cell clustering
  - Found neutrophil extracellular traps (NETs) concentrated in specific marrow regions
  - Showed that plasma-cell-rich regions have reduced inflammatory signaling (IL-17)
  - Integrated spatial with immune data to propose model: isolated plasma cell niches with reduced immune infiltration + altered stromal signaling = protected growth zones

### Challenge & Limitations (2025–2026)
**Critical perspective**: Spatial transcriptomics has inherent limitations
- **Point 1**: Single time-point snapshots; unclear if spatial patterns are stable or dynamic
- **Point 2**: FFPE samples may have artifacts; fresh sample validation needed
- **Point 3**: 50% prevalence of spatially restricted subpopulations means 50% lack this feature; unclear why
- **Point 4**: Spatial signatures may be artifacts of sample orientation or cutting angle

### Current Consensus (2026)
**Consensus position**:
- Plasma cells occupy spatial niches within bone marrow that have distinct stromal, immune, and vascular microenvironments
- Some patients show discrete spatial zones; others show more diffuse infiltration
- Spatial positioning correlates with transcriptional state and predicted immune function
- Stromal-plasma cell interactions are mechanistic (chromatin remodeling) not just correlative
- Spatial architecture likely shapes therapeutic response (inaccessible zones may harbor resistant disease)
- Single-cell RNA-seq misses spatial context; spatial methods are necessary complement

**Remaining uncertainty**:
- Are spatial zones stably maintained or constantly remodeling? (Longitudinal studies needed)
- Do plasma cells home to specific zones or does zone composition select for specific plasma cells? (Chicken-egg question)
- How much of therapeutic resistance is attributable to spatial sanctuary vs. intrinsic drug resistance? (Not quantified)
- Can spatial organization be therapeutically targeted (e.g., disrupt stromal niches)? (Explored theoretically, not clinically)

---

## Comparative Lineage Summary

| Concept | Originator | First Refinement | Major Extension | Current Frontier | Confidence |
|---------|-----------|-----------------|-----------------|------------------|-----------|
| **Plasma Cell Heterogeneity** | Ledergor 2018 | Ledergor 2022 | Mateos 2023 (multi-omic) | Foundation models for objective annotation | High (validated prospectively) |
| **Immune Dysfunction Driving Progression** | Shammas 2016 | Ledergor 2021 | Giguere 2023 (predictive power) | Giguere 2026 (genome-scale, large cohort) | High (predictive + mechanistic) |
| **Spatial Microarchitecture** | Ramasamy 2020 | Ramasamy 2024 (stromal chromatin) | Ledergor/Giguere 2025 (spatial omics) | Linking spatial to therapy resistance | Medium (early-stage method) |

---

## Intellectual Inheritance Patterns

**Pattern 1: From Descriptive to Predictive**
- All three concepts moved from "this exists/happens" (2018-2021) to "this predicts clinical outcomes" (2023-2026)
- Example: T cell exhaustion (descriptive 2016) → predicts relapse (predictive 2023)

**Pattern 2: From Single to Multi-Omics**
- Plasma cell heterogeneity started with RNA only; now includes surface proteins, chromatin, spatial position
- Integration of dimensions reveals previously hidden distinctions

**Pattern 3: From Population-Level to Patient-Level Understanding**
- Early work described patterns in aggregated cohorts
- Recent work (especially Giguere 2026, Ledergor 2025) emphasizes patient-specific variation
- Move away from universal signatures toward personalized immune/spatial fingerprints

**Pattern 4: Unresolved Causality**
- All three concepts show correlations but struggle with causation
- Field still asks: Does heterogeneity cause resistance (needs sorting experiments) or reflect it?
- Do immune states cause progression or is progression measured by immune dysfunction?
- Does spatial organization cause isolation of plasma cells or is it consequence?

---
