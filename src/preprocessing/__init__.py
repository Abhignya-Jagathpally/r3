"""
Preprocessing module for the R3-MM pipeline.

This module provides quality control and preprocessing functionality for
single-cell RNA-seq data, including QC filtering, doublet detection, ambient RNA
correction, normalization, HVG selection, and batch annotation.
"""

from src.preprocessing.ambient_rna import AmbientRNACorrector
from src.preprocessing.batch_info import BatchAnnotator
from src.preprocessing.doublet_removal import DoubletDetector
from src.preprocessing.normalization import Normalizer
from src.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    PreprocessingReport,
)
from src.preprocessing.qc import QCFilter, QCMetrics

__all__ = [
    "QCFilter",
    "QCMetrics",
    "DoubletDetector",
    "AmbientRNACorrector",
    "Normalizer",
    "BatchAnnotator",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "PreprocessingReport",
]
