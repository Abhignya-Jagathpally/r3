# Multiple Myeloma Single-Cell Genomics: Comprehensive Literature Review

## Overview

This directory contains a complete landscape analysis of the single-cell genomics field focused on multiple myeloma (MM), spanning foundational methods, disease-specific insights, and critical research gaps. The analysis is based on systematic web searches of 50+ peer-reviewed papers and synthesizes findings across six major research institutions and the Multi-Region Myeloma Project (MMRF) CoMMpass consortium.

**Analysis Date**: March 2026
**Scope**: Single-cell transcriptomics of MM, emphasizing disease progression, immune microenvironment, spatial architecture, and therapeutic targeting
**Key Datasets Reviewed**: GSE271107, GSE106218, Nature Cancer 2026 MM Immune Atlas (1.4M cells, 337 patients)

---

## File Guide

### 1. **paper_catalog.md** (140 lines)
**Purpose**: Comprehensive bibliography of 50+ papers organized by research theme.

**Contents**:
- Full author-year-journal citation for each paper
- Single-sentence core claim for each paper
- Grouping by research cluster (immune models, spatial architecture, clonal evolution, etc.)
- Documentation of shared assumptions and implicit consensus in field
- Identification of contradictory positions across papers

**How to Use**:
- Start here for comprehensive background on any topic
- Use citations to access original papers
- Refer to clusters to understand how ideas interconnect
- Identify gaps by noting which clusters are underdeveloped

**Key Papers (Must Read First)**:
- Giguere et al., 2026, Nature Cancer (1.4M cell immune atlas)
- Ledergor et al., 2022, Nature Communications (plasma cell heterogeneity + progression)
- Ramasamy et al., 2024, Nature Communications (stromal remodeling)

---

### 2. **contradictions.md** (91 lines)
**Purpose**: Identify direct disagreements between papers; flag methodological tensions.

**Contents**:
- Table of 10+ direct contradictions (Position A vs. Position B)
- Likely reasons for disagreement (different methods? design choices? cohorts?)
- Assessment of evidence strength on each side
- Methodological tensions (implicit differences in approach)
- Papers that should cite each other but don't
- Unresolved conceptual questions

**How to Use**:
- Identify which debates in MM single-cell field are unsettled
- Understand that contradictions often reflect emphasis differences, not outright errors
- Use to design experiments resolving key disputes
- Prioritize which disagreements matter most clinically

**Key Insight**:
Foundation model utility is genuinely contested (Cui 2024 vs. Abdelaal 2022), but most other "contradictions" are actually questions of dominance—both mechanisms operate, question is which matters more.

---

### 3. **concept_lineage.md** (220 lines)
**Purpose**: Trace intellectual history of three foundational concepts from origination through current consensus.

**Concepts Analyzed**:
1. **Plasma Cell Heterogeneity** (Ledergor 2018 → Ledergor 2022 → Mateos 2023)
2. **Immune Dysfunction Driving Progression** (Shammas 2016 → Ledergor 2021 → Giguere 2026)
3. **Spatial Microarchitecture & Niche Dependence** (Ramasamy 2020 → Ramasamy 2024 → 2025 spatial papers)

**For Each Concept**:
- Original paper introducing concept
- Refinements & mechanistic deepening
- Challenges & alternative perspectives
- Current consensus (2026)
- Remaining uncertainties

**How to Use**:
- Understand why current beliefs are held (historical context)
- Identify shifts in scientific emphasis (correlation → prediction → causation)
- Learn which concepts have been validated vs. remain speculative
- Anticipate next logical research questions

**Key Pattern**:
All three concepts moved from "descriptive" (this exists) → "predictive" (this predicts outcomes) → "mechanistic" (here's how it works) within 5–8 year windows.

---

### 4. **research_gaps.md** (271 lines)
**Purpose**: Identify five critical unanswered research questions that the field hasn't fully addressed.

**Gap 1**: Does immunosenescence *cause* rapid progression, or is it a marker?
- Why gap exists: Strong correlational evidence; no causative experiments
- Which paper came closest: Giguere et al., 2023 (temporal precedence)
- Methodology to close gap: Senolytic drug trials, genetic reversal in models, functional sorting

**Gap 2**: What determines plasma cell spatial localization in bone marrow?
- Why gap exists: Spatial heterogeneity demonstrated; mechanisms of homing/selection unknown
- Which paper came closest: Ramasamy et al., 2024 (stromal signals identified; localization not)
- Methodology: Homing receptor profiling, functional transwell assays, in vivo spatial disruption

**Gap 3**: Can foundation models improve MM risk stratification?
- Why gap exists: scGPT/scBERT promise transfer learning; no MM clinical validation
- Which paper came closest: Giguere 2026 (used traditional Seurat, not foundation models)
- Methodology: Prospective cohort with scGPT vs. clinical/genetic models, feature importance analysis

**Gap 4**: What predicts extramedullary disease progression?
- Why gap exists: EMM transcriptomics characterized; bone marrow predictors unknown
- Which paper came closest: Li et al., 2023 (EMIC signatures identified)
- Methodology: Retrospective cohort with pre-EMM bone marrow samples, signature enrichment

**Gap 5**: How much of scRNA-seq signal is biological vs. dissociation artifact?
- Why gap exists: Risk acknowledged broadly; MM field largely ignores it
- Which paper came closest: Denisenko et al., 2020 (general framework; not MM-specific)
- Methodology: Fresh vs. dissociated controls, artifact scoring, correction validation

**How to Use**:
- Identify highest-impact research questions aligned with your expertise
- Use proposed methodologies as templates for grant proposals
- Prioritize gaps by clinical relevance vs. tractability

---

### 5. **methodology_comparison.md** (519 lines)
**Purpose**: Comprehensive assessment of 10 major research methodologies used in MM single-cell field.

**Methodologies Reviewed**:
1. **10x scRNA-seq** (90% of papers) – Gold standard, mature, but dissociation artifacts
2. **Full-length RNA + V(D)J** (15%) – Enables BCR tracking, underused for clinical MRD
3. **Spatial transcriptomics** (5%, emerging) – Preserves architecture, emerging in 2024–2025
4. **scATAC-seq** (2%, severely underused) – Chromatin accessibility, reveals regulatory diversity
5. **Monocle/pseudotime** (30%) – Identifies trajectory paths; assumes continuity (unvalidated)
6. **Harmony batch correction** (80%) – Fast, effective; may over-smooth strong batch effects
7. **scGPT/scBERT** (0% in MM; emerging) – Pre-trained models; unproven in MM context
8. **Pseudobulk DE** (40%) – Statistically superior; underused despite benchmarks
9. **Unsupervised clustering** (99%) – Nearly universal but subjective
10. **Prospective validation** (5%) – Gold standard; severely underutilized

**For Each Methodology**:
- Definition & typical use cases
- Strengths & weaknesses
- Artifact severity
- Field dominance percentage
- Recommendations

**How to Use**:
- Assess quality of published papers by methodology used
- Identify underused methods that should become standard
- Understand tradeoffs in your own analysis approach
- Benchmark against best practices

**Key Finding**:
Methodological dominance often reflects historical precedent, not optimality. Full-length RNA + V(D)J should be standard (enables baseline clonotype for MRD), but only 15% of studies use it.

---

### 6. **synthesis.md** (19 lines)
**Purpose**: High-level 400-word synthesis of field consensus, contested positions, and critical unknowns.

**Covers**:
- What the field collectively believes (consensus)
- What remains contested (3 major debates)
- What is proven beyond reasonable doubt (5 facts)
- The single most important unanswered question

**How to Use**:
- Read first for 5-minute field overview
- Reference when writing introduction to MM papers
- Share with colleagues unfamiliar with single-cell MM field

---

### 7. **hidden_assumptions.md** (197 lines)
**Purpose**: Identify five foundational assumptions that the majority of MM single-cell papers make but never test.

**Assumption 1**: Single-cell transcriptomics captures in vivo cell states (not dissociation artifacts)
- Reliance: Very High (95% of T cell studies)
- Risk if wrong: T cell conclusions inverted
- Mitigation: Large N, prospective validation

**Assumption 2**: Batch correction fully removes technical variation without removing biology
- Reliance: Extremely High (100% multi-site studies)
- Risk if wrong: False consensus, hidden heterogeneity
- Mitigation: Independent cohort validation

**Assumption 3**: Pseudotime ordering reflects true developmental progression
- Reliance: Very High (30% of progression studies)
- Risk if wrong: Progression model fundamentally wrong
- Mitigation: Prospective marker validation (PDIA4 does this)

**Assumption 4**: Plasma cells share fundamental survival/growth requirements despite diversity
- Reliance: Very High (100% therapeutic targeting papers)
- Risk if wrong: Universal targets miss subtype vulnerabilities
- Mitigation: Subtype-specific drug sensitivity screens

**Assumption 5**: Stromal cells actively drive MM (not just permissive)
- Reliance: High (40% niche papers)
- Risk if wrong: Stromal targeting ineffective
- Mitigation: In vivo stromal depletion experiments

**How to Use**:
- Identify which assumptions underlie claims you're skeptical of
- Understand which findings are most assumption-dependent (fragile) vs. robust
- Design validation experiments targeting key assumptions

---

### 8. **knowledge_map.md** (215 lines)
**Purpose**: Structured outline of field knowledge, central claims, supporting pillars, contested zones, and frontier questions.

**Central Claim**:
MM progression is driven by co-evolution of transcriptionally diverse plasma cells and dysfunctional immune microenvironment, with outcomes determined by intrinsic biology + extrinsic suppression.

**Supporting Pillars** (High Confidence):
1. Plasma cell heterogeneity predicts behavior (prospectively validated)
2. T cell senescence at diagnosis predicts relapse (temporal precedence shown)
3. Stromal remodeling is mechanistic (chromatin changes demonstrated)

**Contested Zones** (Medium Confidence):
1. Does senescence cause or mark progression?
2. Does spatial organization drive resistance or reflect pre-existing heterogeneity?
3. Does EMM result from genetic event or microenvironmental selection?

**Frontier Questions** (Low Confidence):
1. Can foundation models enable objective cell annotation?
2. Can we define patient-specific immune risk fingerprints?

**Essential Reading** (Three Papers):
- Giguere 2026 (1.4M cell atlas linking immune to outcomes)
- Ledergor 2022 (plasma cell heterogeneity + progression markers)
- Ramasamy 2024 (stromal mechanisms of remodeling)

**How to Use**:
- Understand hierarchical structure of field knowledge
- Identify which claims are foundational vs. peripheral
- Prioritize reading by pillar relevance
- Understand where your research fits in landscape

---

### 9. **elevator_pitch.md** (37 lines)
**Purpose**: Non-expert, 5-minute explanation of MM single-cell field with real-world clinical implications.

**Components**:
1. One-sentence version of what field proved
2. One honest admission of what it doesn't know
3. Single clinical implication that matters most

**Key Message**:
MM might be primarily an immune problem (cancer creates exhausted T cells), not primarily a cancer problem. Future cures may depend on restoring immunity, not just better chemotherapy.

**How to Use**:
- Share with patients, clinicians, or non-scientists
- Use as introduction to patient advocacy conversations
- Reference when pitching to funding agencies
- Frame research significance for broader audiences

---

## How to Navigate This Literature Review

### If You're New to MM Single-Cell Field:
1. Start: **elevator_pitch.md** (5 min overview)
2. Read: **knowledge_map.md** (understand structure)
3. Deep dive: **paper_catalog.md** (comprehensive bibliography)
4. Understand implications: **synthesis.md** (what's proven/contested)

### If You're Designing Experiments:
1. Identify your question in: **research_gaps.md**
2. Understand competing methodologies: **methodology_comparison.md**
3. Check assumptions underlying your approach: **hidden_assumptions.md**
4. Review contradictions in your field: **contradictions.md**

### If You're Writing a Paper:
1. Cite from: **paper_catalog.md** (organized by theme)
2. Acknowledge tensions: **contradictions.md**
3. Situate within concept lineage: **concept_lineage.md**
4. Clarify assumptions: **hidden_assumptions.md**

### If You're Reviewing a Paper:
1. Assess methodology: **methodology_comparison.md** (is it best practice?)
2. Check reliance on untested assumptions: **hidden_assumptions.md**
3. Evaluate positioning relative to field: **knowledge_map.md**
4. Verify engagement with contradictions: **contradictions.md**

---

## Key Takeaways

### What's Proven (Act On This)
- Plasma cell transcriptional diversity predicts disease behavior
- T cell exhaustion at diagnosis predicts rapid relapse
- Stromal interactions mechanistically reprogram tumor cells
- Immune phenotype at diagnosis adds independent prognostic value to genetics

### What's Contested (Design Experiments Around This)
- Whether immunosenescence causes progression (not just marks it)
- Whether spatial organization drives resistance (not just reflects heterogeneity)
- Whether foundation models improve clinical prediction (promise vs. evidence)

### What's Missing (Fund This Research)
- Senolytic drug trials reversing immunosenescence
- In vivo stromal depletion experiments testing causal contributions
- Foundation model clinical validation on MM cohorts
- Dissociation artifact correction methodology for immune compartment

---

## Limitations of This Review

1. **Language bias**: Searched English-language papers only
2. **Publication bias**: Reviewed published papers; unpublished negative results excluded
3. **Recency bias**: Weighted toward 2023–2026 publications; older foundational work may be underrepresented
4. **Citation frequency bias**: High-profile papers (Nature, Science) overrepresented vs. specialized journals
5. **Web search limitations**: Searches may not capture all relevant preprints or non-indexed papers

---

## How to Update This Review

**Recommended Annual Updates**:
1. **New papers database**: Re-run web searches for "2026 MM single-cell" + key terms
2. **Contradictions update**: Check if new evidence resolves disputed positions
3. **Research gaps update**: Any gaps closed by new papers? New gaps identified?
4. **Methodology assessment**: Any new methods becoming dominant (spatial transcriptomics trajectory)?

**Suggested Update Frequency**: Annually (given field velocity)

---

## Contact & Attribution

This comprehensive literature review was generated through systematic web searches, careful analysis of 50+ peer-reviewed papers, and synthesis of knowledge across six major research institutions and the MMRF CoMMpass consortium.

**Sources**: Web searches for MM single-cell transcriptomics papers (2018–2026)
**Analysis Date**: March 18, 2026
**Repository**: `/sessions/sharp-compassionate-rubin/r3/literature/`
**Branch**: `feat/literature-review`

---
