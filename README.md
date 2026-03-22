# R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology

A comprehensive, production-quality pipeline for analyzing single-cell RNA-seq data from Multiple Myeloma (MM) patient samples. This pipeline integrates state-of-the-art computational methods for data preprocessing, integration, cell type annotation, and downstream analyses.

## Overview

The R3-MM pipeline implements a complete workflow for single-cell RNA-seq analysis with emphasis on reproducibility, scalability, and agentic optimization. The pipeline is built on modern bioinformatics tools and frameworks, supporting both local execution and high-performance computing (HPC) environments.

### Key Features

- **Multi-Stage Data Management**: Enforced staging layers (raw → standardized → analysis_ready)
- **Batch Effect Correction**: Integration methods including Harmony and scVI
- **Cell Type Annotation**: Multiple annotation methods (CellTypist, scGPT)
- **Agentic Tuning**: Automatic hyperparameter optimization with configurable search space
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Containerization**: Docker and Apptainer support for reproducible execution
- **Workflow Orchestration**: Both Nextflow and Snakemake pipelines
- **HPC Support**: SLURM integration for cluster execution

## Project Structure

```
r3/
├── README.md                       # Project documentation
├── pyproject.toml                  # Python packaging configuration
├── Dockerfile                      # Docker container definition
├── Apptainer.def                   # Apptainer/Singularity definition
├── configs/
│   ├── pipeline_config.yaml        # Master configuration
│   ├── mlflow_config.yaml          # MLflow settings
│   ├── wandb_config.yaml           # Weights & Biases settings
│   ├── dvc.yaml                    # DVC pipeline stages
│   └── nextflow.config             # Nextflow configuration
├── data/
│   ├── raw/                        # Original downloaded data
│   ├── standardized/               # QC-filtered data
│   └── analysis_ready/             # Integrated and annotated data
├── src/
│   ├── config.py                   # Configuration management
│   ├── data/
│   │   ├── download.py             # GEO data download
│   │   └── storage.py              # Storage management
│   ├── preprocessing/              # QC and preprocessing
│   ├── annotation/                 # Cell type annotation
│   ├── integration/                # Batch correction
│   ├── models/                     # Machine learning models
│   ├── evaluation/                 # Metrics and analysis
│   └── agentic/                    # Agentic tuning
├── workflows/
│   ├── main.nf                     # Nextflow workflow
│   ├── Snakefile                   # Snakemake workflow
│   └── modules/                    # Workflow modules
├── tests/
│   ├── test_config.py              # Configuration tests
│   └── ...
├── notebooks/                      # Analysis notebooks
├── benchmarks/                     # Benchmarking scripts
└── literature/                     # Reference materials
```

## Installation

### Requirements

- Python ≥ 3.11
- 16 GB RAM minimum (64 GB recommended for full pipeline)
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abhignya-j/r3-mm-pipeline.git
   cd r3
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install package**:
   ```bash
   pip install -e .
   ```

4. **Install optional dependencies**:
   ```bash
   # For development
   pip install -e ".[dev]"

   # For testing
   pip install -e ".[test]"

   # For documentation
   pip install -e ".[docs]"
   ```

### Docker Installation

```bash
# Build Docker image
docker build -t r3-mm-pipeline:latest .

# Run container
docker run --rm -v $(pwd):/app r3-mm-pipeline:latest python -m src.data.download
```

### Apptainer/Singularity Installation

```bash
# Build Apptainer image
apptainer build r3-mm-pipeline.sif Apptainer.def

# Run container
apptainer run r3-mm-pipeline.sif python -m src.data.download
```

## Configuration

### Main Configuration File

Edit `configs/pipeline_config.yaml` to customize pipeline parameters:

```yaml
pipeline:
  name: "r3-mm-pipeline"
  version: "0.1.0"

data_sources:
  datasets:
    - accession: "GSE271107"
      name: "Multiple Myeloma Longitudinal"
      # ...

qc:
  min_genes: 200
  max_genes: 5000
  max_mito_pct: 20

preprocessing:
  normalization:
    method: "log_normalize"
  hvg_selection:
    n_top_genes: 5000

integration:
  methods:
    - name: "harmony"
      theta: 2.0

annotation:
  methods:
    - name: "celltypist"
    - name: "scgpt"

agentic:
  enabled: true
  search_budget: 100
  editable_surface:
    - "qc.max_mito_pct"
    - "preprocessing.hvg_selection.n_top_genes"
```

## Usage

### Quick Start

1. **Download data**:
   ```bash
   python -m src.data.download
   ```

2. **Run full pipeline** (Nextflow):
   ```bash
   nextflow run workflows/main.nf --config configs/pipeline_config.yaml
   ```

3. **Alternative: Snakemake**:
   ```bash
   snakemake --cores all --config config_file=configs/pipeline_config.yaml
   ```

### Command-Line Examples

```bash
# Download specific datasets
python -c "from src.data.download import download_gse_data; download_gse_data(['GSE271107'])"

# Run QC on raw data
python -m src.preprocessing.qc --config configs/pipeline_config.yaml

# Integrate datasets
python -m src.integration.integrate --config configs/pipeline_config.yaml

# Annotate cell types
python -m src.annotation.annotate --config configs/pipeline_config.yaml

# Run evaluation metrics
python -m src.evaluation.evaluate --config configs/pipeline_config.yaml
```

### Python API Usage

```python
from src.config import load_config
from src.data.download import download_gse_data
from src.data.storage import StorageManager

# Load configuration
config = load_config("configs/pipeline_config.yaml")

# Download data
results = download_gse_data(["GSE271107", "GSE106218"])

# Initialize storage manager
sm = StorageManager(root_dir="./data")

# List available datasets
datasets = sm.list_available("raw")
print(f"Available datasets: {datasets}")

# Read data
adata = sm.read_raw("GSE271107")
print(f"Shape: {adata.shape}")
```

## Data Sources

The pipeline includes data from the following GEO datasets:

1. **GSE271107**: Multiple Myeloma Longitudinal
   - Single-cell transcriptomics across treatment timepoints
   - ~150,000 cells
   - 10x Genomics platform

2. **GSE106218**: Multiple Myeloma Atlas
   - Comprehensive MM single-cell atlas
   - ~100,000 cells
   - 10x Genomics platform

## Pipeline Stages

### 1. Data Download
- Retrieve data from GEO using GEOparse
- Convert to AnnData format
- Store in raw layer

### 2. Quality Control
- Filter cells by gene count, UMI count, mitochondrial percentage
- Remove low-quality cells
- Calculate QC metrics

### 3. Preprocessing
- Log normalization
- Highly variable gene selection
- PCA dimensionality reduction
- Store in standardized layer

### 4. Integration
- Harmony-based batch effect correction
- scVI deep learning integration
- Store in analysis_ready layer

### 5. Annotation
- CellTypist automated annotation
- scGPT foundation model annotation
- Manual annotation support

### 6. Clustering
- Leiden community detection
- Resolution parameter optimization
- UMAP visualization

### 7. Evaluation
- Clustering quality metrics (silhouette, Davies-Bouldin)
- Batch correction assessment (kBET, LISI)
- Annotation accuracy metrics

### 8. Pseudobulk Analysis
- Aggregate cells by cell type and batch
- Prepare for downstream DE analysis

## Agentic Tuning

The pipeline includes agentic optimization for automatic hyperparameter tuning:

```yaml
agentic:
  enabled: true
  search_budget: 100  # Total configurations to evaluate
  editable_surface:   # Parameters to optimize
    - "qc.max_mito_pct"
    - "preprocessing.hvg_selection.n_top_genes"
    - "clustering.leiden.resolution"
  frozen_modules:     # Fixed/core modules
    - "download"
    - "storage"
  optimization_metric: "silhouette_score"
  optimization_direction: "maximize"
```

The agentic system automatically:
- Samples parameter configurations from editable_surface
- Runs pipeline with different configurations
- Tracks metrics across runs
- Recommends optimal parameter set

## Experiment Tracking

### MLflow

```bash
# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

Automatically logs:
- Pipeline parameters
- Quality metrics
- Cell counts and statistics
- Clustering metrics
- Annotation results

### Weights & Biases

Enable in `configs/pipeline_config.yaml`:
```yaml
wandb:
  enabled: true
  project: "r3-mm-pipeline"
  entity: "your-username"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py -v

# With coverage report
pytest --cov=src --cov-report=html
```

## HPC Execution

### SLURM Cluster

```bash
# Submit Nextflow job to SLURM
nextflow run workflows/main.nf \
  -profile hpc \
  --config configs/pipeline_config.yaml \
  -with-trace execution_trace.txt
```

Configuration in `configs/nextflow.config`:
```groovy
process {
    executor = 'slurm'
    queue = 'normal'
    cpus = 16
    memory = '64 GB'
}
```

### Distributed Computing with Ray

```python
import ray
from src.agentic import AgenticTuner

ray.init()
tuner = AgenticTuner(config)
results = tuner.optimize()
ray.shutdown()
```

## Performance Benchmarks

Typical runtimes on modern hardware:

| Stage | Time | Cores | RAM |
|-------|------|-------|-----|
| Download | 5-10 min | 4 | 8 GB |
| QC | 10-20 min | 8 | 16 GB |
| Integration | 30-60 min | 16 | 32 GB |
| Annotation | 20-40 min | 16 | 32 GB |
| Evaluation | 10-20 min | 8 | 16 GB |
| **Total** | **1.5-3 hrs** | - | - |

## Troubleshooting

### Memory Issues
- Reduce batch size in config: `compute.batch_size: 16`
- Use distributed computing with Dask/Ray
- Process datasets separately

### CUDA Out of Memory
- Disable GPU: Set `compute.gpu_enabled: false`
- Reduce batch size
- Use CPU-only libraries

### Download Failures
- Check NCBI API key in environment: `export NCBI_API_KEY=your_key`
- Verify internet connection
- Increase retry attempts: `data_sources.download.retry_attempts: 5`

## Citation

If you use this pipeline, please cite:

```bibtex
@software{r3_mm_pipeline_2026,
  title={R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology},
  author={Jagathpally, Abhignya},
  year={2026},
  url={https://github.com/abhignya-j/r3-mm-pipeline}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## Contact

- Author: Abhignya Jagathpally
- Email: abhignya.j@gmail.com
- GitHub: [@abhignya-j](https://github.com/abhignya-j)

## References

Key papers and tools used in this pipeline:

- **Scanpy**: Wolf et al. (2018) - Single-cell analysis in Python
- **Harmony**: Korsunsky et al. (2019) - Fast, sensitive, and accurate integration
- **scVI**: Lopez et al. (2018) - Deep generative modeling of scRNA-seq data
- **CellTypist**: Domínguez Conde et al. (2022) - Automated cell type annotation
- **scGPT**: Hao et al. (2023) - Generative pre-training for single-cell data

## Acknowledgments

This pipeline builds upon the excellent work of the computational biology and bioinformatics communities, including the developers of Scanpy, AnnData, and the numerous single-cell analysis tools it integrates.

---

Last updated: March 2026
