"""
R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology Pipeline

A comprehensive pipeline for analyzing single-cell RNA-seq data from multiple myeloma patients,
including data preprocessing, integration, cell type annotation, and downstream analysis.

Modules:
    - config: Configuration management using Pydantic
    - data: Data download, loading, and storage management
    - preprocessing: Quality control and preprocessing
    - annotation: Cell type annotation
    - integration: Batch effect correction and integration
    - models: Machine learning models
    - evaluation: Evaluation metrics and analyses
    - agentic: Agentic tuning and optimization

Example:
    >>> from src.config import load_config
    >>> config = load_config("configs/pipeline_config.yaml")
    >>> print(config.pipeline.name)
"""

__version__ = "0.1.0"
__author__ = "Abhignya Jagathpally"
__email__ = "abhignya.j@gmail.com"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
