"""
Annotation module for the R3-MM pipeline.

This module provides comprehensive cell type annotation using multiple methods:
- Marker-based annotation with hardcoded MM markers
- CellTypist automated annotation
- Semi-supervised scANVI annotation
- Consensus annotation across methods
- Pseudobulk aggregation for bulk-level analysis

Classes:
    MarkerAnnotator: Marker gene-based annotation.
    CellTypistAnnotator: Deep learning-based automated annotation.
    ConsensusAnnotator: Consensus across annotation methods.
    CellOntologyMapper: Standardized cell type mapping.
    PseudobulkAggregator: Patient × cell type aggregation.
"""

from .cell_ontology import CellOntologyMapper
from .celltypist_annotator import CellTypistAnnotator
from .consensus import ConsensusAnnotator
from .marker_based import MarkerAnnotator
from .pseudobulk import PseudobulkAggregator

__all__ = [
    "MarkerAnnotator",
    "CellTypistAnnotator",
    "ConsensusAnnotator",
    "CellOntologyMapper",
    "PseudobulkAggregator",
]
