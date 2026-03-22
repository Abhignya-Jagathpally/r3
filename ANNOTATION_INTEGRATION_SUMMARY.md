# Annotation and Integration Module Summary

## Overview
Completed production-quality annotation and integration modules for the Multiple Myeloma (MM) single-cell transcriptomics pipeline. All code includes type hints, Google-style docstrings, comprehensive error handling, and logging.

## Integration Module (`src/integration/`)

### 1. HarmonyIntegrator (`harmony.py`)
Harmony-based batch effect correction using PCA embeddings.

**Methods:**
- `compute_pca(adata, n_comps=50)` - Compute PCA embeddings
- `compute_neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15)` - KNN graph
- `compute_umap(adata, min_dist=0.1, spread=1.0)` - UMAP reduction
- `integrate(adata, batch_key='batch', n_comps=50, n_neighbors=15)` - Full pipeline

**Outputs:**
- `adata.obsm['X_pca']` - PCA embeddings
- `adata.obsm['X_pca_harmony']` - Harmony-corrected embeddings
- `adata.obsm['X_umap']` - UMAP embeddings
- `adata.obsp['connectivities']` - Neighbor graph

### 2. ScVIIntegrator (`scvi_integration.py`)
VAE-based integration using scVI-tools for learning latent representations.

**Methods:**
- `setup_anndata(adata, batch_key, layer='counts')` - Register data with scVI
- `train_scvi(adata, n_latent=30, n_epochs=100)` - Train VAE model
- `get_latent(adata, model)` - Extract latent representations
- `integrate(adata, batch_key='batch', layer='counts', ...)` - Full pipeline

**Outputs:**
- `adata.obsm['X_scVI']` - Latent embeddings (n_cells × n_latent)
- `adata.uns['scvi']` - Model metadata

### 3. ScANVIIntegrator (`scanvi_integration.py`)
Semi-supervised integration combining scVI with cell type labels for joint batch correction and annotation.

**Methods:**
- `setup_anndata(adata, batch_key, labels_key, layer='counts')` - Register data
- `train_scanvi(adata, labels_key, n_latent=30, n_epochs=50)` - Train with labels
- `predict_labels(adata, model)` - Predict cell types
- `get_latent(adata, model)` - Extract latent
- `integrate(adata, batch_key, labels_key, ...)` - Full pipeline

**Outputs:**
- `adata.obsm['X_scANVI']` - Latent embeddings
- `adata.obs['scanvi_pred']` - Predicted cell types
- `adata.uns['scanvi']` - Model metadata

## Annotation Module (`src/annotation/`)

### 1. MarkerAnnotator (`marker_based.py`)
Hardcoded marker gene-based annotation tailored for MM bone marrow.

**Marker Sets (9 cell types):**
- **Plasma cells**: SDC1, CD138, TNFRSF17, BCMA, XBP1, IRF4, PRDM1
- **T cells**: CD3D, CD3E, CD4, CD8A, CD8B
- **NK cells**: NCAM1, NKG7, GNLY, KLRD1
- **Monocytes**: CD14, LYZ, CST3, FCGR3A
- **B cells**: CD79A, MS4A1, CD20, CD19
- **Erythroid**: HBA1, HBB, GYPA
- **HSC/progenitors**: CD34, KIT, THY1
- **Osteoclasts**: ACP5, CTSK, MMP9
- **Mast cells**: KIT, CPA3, TPSAB1

**Methods:**
- `score_markers(adata)` - Score genes per cell type
- `annotate(adata, threshold=0.5)` - Assign by max score
- `get_marker_dict()` - Get marker gene sets

**Outputs:**
- `adata.obs['cell_type_marker']` - Assigned cell types
- `adata.obs['marker_score_{cell_type}']` - Per-type scores
- `adata.obs['marker_score_max']` - Maximum score per cell

### 2. CellTypistAnnotator (`celltypist_annotator.py`)
Deep learning-based automated annotation using pre-trained CellTypist models.

**Methods:**
- `annotate(adata, model_name='Immune_All_Low.pkl')` - Predict cell types
- `annotate_majority_voting(adata, model_name)` - Robust predictions
- `map_to_cell_ontology(labels)` - Map to Cell Ontology IDs

**Outputs:**
- `adata.obs['cell_type_celltypist']` - CellTypist predictions
- `adata.obs['cell_type_celltypist_prob']` - Prediction confidence

### 3. CellOntologyMapper (`cell_ontology.py`)
Standardized cell type mapping to Cell Ontology (CL) IDs for interoperability.

**Mappings:**
- 80+ cell type labels → CL IDs
- Common bone marrow and immune cell types
- Comprehensive reverse mappings

**Methods:**
- `map_labels(labels, case_sensitive=False)` - Map to CL IDs
- `get_label_name(cl_id)` - Reverse lookup
- `validate_labels(labels)` - Check validity
- `get_all_mappings()` - Complete dictionary

**Example:**
```
"B cell" → "CL:0000236"
"T cell" → "CL:0000084"
"Plasma cell" → "CL:0000786"
```

### 4. ConsensusAnnotator (`consensus.py`)
Multi-method consensus annotation using majority voting.

**Methods:**
- `build_consensus(adata, methods=['marker', 'celltypist', 'scanvi'])` - Combine methods
- `get_uncertain_cells(adata, confidence_threshold=0.5)` - Low confidence cells
- `get_disagreement_cells(adata)` - Method disagreement

**Outputs:**
- `adata.obs['cell_type_consensus']` - Final consensus annotations
- `adata.obs['annotation_confidence']` - Confidence (0-1)
- `adata.obs['n_methods_agree']` - Number of methods agreeing

### 5. PseudobulkAggregator (`pseudobulk.py`) — CRITICAL
Patient-level aggregation implementing X_{p,c,g}^{pseudo} = Σ x_{j,g} formula.

**Methods:**
- `aggregate(adata, patient_key='patient_id', celltype_key='cell_type_consensus')` 
  - Sums counts per patient × cell type
  - Returns new AnnData with (patient, cell_type) rows
  
- `aggregate_by_compartment(adata, patient_key, celltype_key, compartment_key)`
  - Three-level grouping: patient × compartment × cell type
  
- `compute_cell_fractions(adata, patient_key, celltype_key)`
  - Returns DataFrame with cell type composition per patient
  
- `to_parquet(pseudobulk_adata, output_path)`
  - Export as parquet for bulk analysis

**Outputs:**
- Pseudobulk AnnData: (n_patients × n_celltypes) × n_genes
- obs: patient_id, cell_type, n_cells
- Parquet export for downstream bulk analysis

**Example:**
```python
agg = PseudobulkAggregator()
pb = agg.aggregate(adata, patient_key='patient_id', 
                   celltype_key='cell_type_consensus')
# pb.shape = (54, 20000)  # 18 patients × 3 cell types × 20K genes
# pb.obs contains n_cells per group
```

## Testing (`tests/`)

### test_annotation.py (11 test cases)
- `TestMarkerAnnotator` (3 tests)
  - Marker dictionary retrieval
  - Marker scoring on synthetic data
  - Cell type annotation

- `TestCellOntologyMapper` (5 tests)
  - Label → CL ID mapping
  - Case-insensitive mapping
  - Reverse mapping
  - Label validation
  - Complete mappings

- `TestConsensusAnnotator` (3 tests)
  - Consensus building
  - Uncertain cell identification
  - Disagreement detection

- `TestPseudobulkAggregator` (5 tests)
  - Count preservation across aggregation
  - Dimension correctness
  - Compartment-based aggregation
  - Cell fractions computation
  - Cell count verification

- `TestCellTypistAnnotator` (1 test)
  - Cell Ontology mapping

### test_integration.py (12 test cases)
- `TestHarmonyIntegrator` (4 tests)
  - PCA computation
  - Parameter validation
  - Neighbor computation
  - UMAP computation requirements

- `TestScVIIntegrator` (3 tests)
  - AnnData setup validation
  - Model training parameter validation
  
- `TestScANVIIntegrator` (3 tests)
  - Setup validation
  - Training parameter validation

- `TestIntegrationConsistency` (3 tests)
  - Output structure verification

## Code Quality

### Type Hints
- Full function parameter and return type annotations
- Optional types properly marked
- Generic types (Dict, List, Tuple) imported from typing

### Docstrings
- Google-style format for all classes and methods
- Args, Returns, Raises, Examples sections
- Clear descriptions of parameters and outputs
- Usage examples in docstrings

### Error Handling
- Input validation with informative ValueError messages
- Missing key checks with available alternatives listed
- Shape validation for array inputs
- Layer/column existence checks

### Logging
- INFO level for major operations (integration, annotation steps)
- WARNING level for missing genes, skipped operations
- Descriptive messages with counts/metrics
- All classes instantiate logger: `self.logger = logger`

## Key Designs

### Pseudobulk Aggregation (CRITICAL)
Implements the mathematical formula:
```
X_{p,c,g}^{pseudo} = Σ_{j ∈ cells(p,c)} x_{j,g}
```
where:
- p = patient ID
- c = cell type
- g = gene
- j = individual cell
- x_{j,g} = raw count for cell j, gene g

Returns (n_patients × n_celltypes) × n_genes matrix for bulk analysis.

### Consensus Annotation
Combines three complementary methods:
1. **Marker-based**: Fast, interpretable, hardcoded MM markers
2. **CellTypist**: Deep learning, broad model availability
3. **scANVI**: Semi-supervised, leverages labeled data

Uses majority voting with confidence scoring and uncertainty flagging.

### Integration Pipeline
Three complementary approaches:
1. **Harmony**: Fast, preserves biology, works on PCA
2. **scVI**: Generative model, handles complex batch effects
3. **scANVI**: Semi-supervised, joint integration + annotation

## Usage Example

```python
from src.integration import HarmonyIntegrator, ScVIIntegrator
from src.annotation import (
    MarkerAnnotator, CellTypistAnnotator, ConsensusAnnotator,
    PseudobulkAggregator
)

# Integration
harmony = HarmonyIntegrator()
adata = harmony.integrate(adata, batch_key='batch')  # → X_pca_harmony, X_umap

# Annotation
marker_ann = MarkerAnnotator()
adata = marker_ann.annotate(adata)  # → cell_type_marker

celltypist = CellTypistAnnotator()
adata = celltypist.annotate(adata)  # → cell_type_celltypist

# Consensus
consensus = ConsensusAnnotator()
adata = consensus.build_consensus(adata)  # → cell_type_consensus

# Pseudobulk
agg = PseudobulkAggregator()
pseudobulk = agg.aggregate(adata)  # → (n_patients × n_celltypes) × n_genes
agg.to_parquet(pseudobulk, 'pseudobulk.parquet')
```

## Files Created

**Integration Module:**
- `/src/integration/harmony.py` - 235 lines
- `/src/integration/scvi_integration.py` - 255 lines
- `/src/integration/scanvi_integration.py` - 316 lines
- `/src/integration/__init__.py` - Updated with exports

**Annotation Module:**
- `/src/annotation/marker_based.py` - 191 lines
- `/src/annotation/celltypist_annotator.py` - 223 lines
- `/src/annotation/cell_ontology.py` - 260 lines
- `/src/annotation/consensus.py` - 244 lines
- `/src/annotation/pseudobulk.py` - 349 lines
- `/src/annotation/__init__.py` - Updated with exports

**Tests:**
- `/tests/test_annotation.py` - 373 lines (11 test cases)
- `/tests/test_integration.py` - 309 lines (12 test cases)

**Total: 2,796 lines of production code + tests**

## Branch and Commit

**Branch:** `feat/annotation-integration`
**Commit:** `73aed53` - "feat: implement complete annotation and integration modules for MM pipeline"

All code ready for integration into the main MM pipeline.
