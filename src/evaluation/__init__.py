"""
Evaluation module for the R3-MM pipeline.

This module provides functionality for:
- MM-specific benchmark metrics (ARI, NMI, rare cell recall, etc.)
- Patient-level train-test splitting to prevent data leakage
- Experiment tracking with MLflow and W&B backends
"""

from src.evaluation.experiment_tracker import (
    ExperimentTracker,
    MLflowTracker,
    WandBTracker,
)
from src.evaluation.metrics import (
    BenchmarkSuite,
    compute_ari,
    compute_batch_asw,
    compute_bio_conservation,
    compute_graph_connectivity,
    compute_nmi,
    compute_rare_cell_recall,
    compute_transfer_score,
)
from src.evaluation.splits import (
    CrossValidator,
    PatientLevelSplitter,
    TimeAwareSplitter,
    ensure_no_patient_overlap,
)

__all__ = [
    # Metrics
    "compute_ari",
    "compute_nmi",
    "compute_batch_asw",
    "compute_graph_connectivity",
    "compute_rare_cell_recall",
    "compute_bio_conservation",
    "compute_transfer_score",
    "BenchmarkSuite",
    # Splits
    "PatientLevelSplitter",
    "TimeAwareSplitter",
    "CrossValidator",
    "ensure_no_patient_overlap",
    # Tracking
    "ExperimentTracker",
    "MLflowTracker",
    "WandBTracker",
]
