"""
Agentic tuning configuration for the R3-MM pipeline.

This module provides Pydantic models for configuring the agentic hyperparameter
search layer, following Karpathy's autoresearch pattern with fixed search budgets
and constrained editable surfaces.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AgenticConfig(BaseModel):
    """Configuration for agentic hyperparameter tuning.

    This config enforces Karpathy's autoresearch principles:
    - One primary metric for optimization
    - Fixed search budget (wall-clock and trial count)
    - Constrained editable surface (only training/model config)
    - Frozen preprocessing (immutable data contract)

    Attributes:
        primary_metric: The single metric to optimize (e.g., "bio_conservation")
        search_budget: Maximum number of experiments (fixed wall-clock trials)
        max_wallclock_hours: Hard time limit for search in hours
        editable_surface: List of parameters that agents can modify
        frozen_modules: List of module patterns that cannot be edited
        preprocessing_contract_path: Path to frozen preprocessing contract JSON
        experiment_log_dir: Directory for storing experiment logs and results
    """

    optimization_metric: str = Field(
        default="silhouette_score",
        description="Single metric to optimize"
    )

    # Alias for backward compatibility
    @property
    def primary_metric(self):
        return self.optimization_metric

    optimization_direction: str = Field(
        default="maximize",
        description="Direction: 'maximize' or 'minimize'"
    )

    search_budget: int = Field(
        default=100,
        ge=1,
        description="Maximum number of experiments in search"
    )

    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        description="Trials without improvement before stopping"
    )

    trial_timeout_seconds: int = Field(
        default=3600,
        description="Max seconds per trial"
    )

    max_wallclock_hours: float = Field(
        default=24.0,
        gt=0,
        description="Hard time limit for hyperparameter search in hours"
    )

    editable_surface: List[str] = Field(
        default=[
            "qc.max_mito_pct",
            "clustering.leiden.resolution",
            "integration.methods.harmony.theta",
            "preprocessing.hvg_selection.n_top_genes",
            "annotation.methods.celltypist.over_clustering",
        ],
        description="Parameters that agents can modify during search"
    )

    frozen_modules: List[str] = Field(
        default=["download", "storage", "logging"],
        description="Module patterns that are frozen and cannot be edited"
    )

    preprocessing_contract_path: Optional[str] = Field(
        default=None,
        description="Path to frozen preprocessing contract JSON"
    )

    experiment_log_dir: str = Field(
        default="experiments/",
        description="Directory for experiment logs and results"
    )

    class Config:
        """Pydantic config."""
        frozen = False


class SearchSpaceConfig(BaseModel):
    """Configuration for hyperparameter search space definition.

    Attributes:
        strategy: Search strategy ('random', 'bayesian', 'grid')
        n_random_init: Number of random trials before Bayesian optimization
        use_gpu: Whether to use GPU for training
        n_cpus_per_trial: CPU cores per trial
        n_gpus_per_trial: GPU count per trial
    """

    strategy: str = Field(
        default="bayesian",
        description="Search strategy: 'random', 'bayesian', or 'grid'"
    )

    n_random_init: int = Field(
        default=10,
        ge=1,
        description="Number of random initialization trials before Bayesian optimization"
    )

    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU for model training"
    )

    n_cpus_per_trial: int = Field(
        default=4,
        ge=1,
        description="CPU cores allocated per trial"
    )

    n_gpus_per_trial: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="GPU fraction allocated per trial (0-1)"
    )

    class Config:
        """Pydantic config."""
        frozen = False


class TunerConfig(BaseModel):
    """Configuration for distributed tuning backend.

    Attributes:
        backend: Tuning backend ('ray', 'dask', 'sequential')
        n_workers: Number of parallel workers
        memory_limit_per_worker: Memory limit per worker in GB
        checkpoint_dir: Directory for trial checkpoints
    """

    backend: str = Field(
        default="ray",
        description="Tuning backend: 'ray', 'dask', or 'sequential'"
    )

    n_workers: int = Field(
        default=4,
        ge=1,
        description="Number of parallel workers for distributed search"
    )

    memory_limit_per_worker: float = Field(
        default=16.0,
        gt=0,
        description="Memory limit per worker in GB"
    )

    checkpoint_dir: str = Field(
        default="checkpoints/",
        description="Directory for storing trial checkpoints"
    )

    class Config:
        """Pydantic config."""
        frozen = False
