"""
Configuration management for the R3-MM pipeline.

This module provides Pydantic-based configuration models for loading and validating
the pipeline configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    accession: str = Field(..., description="GEO accession ID")
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    organism: str = Field(default="Homo sapiens", description="Organism")
    tissue: str = Field(default="Bone Marrow", description="Tissue type")
    cell_count_expected: int = Field(..., description="Expected number of cells")
    platform: str = Field(..., description="Sequencing platform")

    class Config:
        frozen = False


class DataDownloadConfig(BaseModel):
    """Configuration for data downloading."""

    method: str = Field(default="GEO", description="Download method")
    ncbi_api_key: Optional[str] = Field(None, description="NCBI API key")
    timeout_seconds: int = Field(default=300, description="Timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    batch_size: int = Field(default=32, description="Batch size for downloads")

    class Config:
        frozen = False


class DataSourcesConfig(BaseModel):
    """Configuration for data sources."""

    datasets: List[DatasetConfig] = Field(..., description="List of datasets")
    download: DataDownloadConfig = Field(default_factory=DataDownloadConfig)

    class Config:
        frozen = False


class QCConfig(BaseModel):
    """Quality control parameters."""

    min_genes: int = Field(default=200, description="Minimum genes per cell")
    max_genes: int = Field(default=5000, description="Maximum genes per cell")
    min_cells: int = Field(default=3, description="Minimum cells per gene")
    max_mito_pct: float = Field(default=20, description="Maximum mitochondrial percentage")
    max_ribo_pct: float = Field(default=50, description="Maximum ribosomal percentage")
    min_genes_per_cell: int = Field(default=200, description="Minimum genes per cell")
    min_umis_per_cell: int = Field(default=500, description="Minimum UMIs per cell")
    outlier_detection: str = Field(default="mad", description="Outlier detection method")
    mad_threshold: float = Field(default=3.0, description="MAD threshold")

    class Config:
        frozen = False

    @validator("outlier_detection")
    def validate_outlier_detection(cls, v):
        if v not in ["mad", "iqr"]:
            raise ValueError("outlier_detection must be 'mad' or 'iqr'")
        return v


class PreprocessingConfig(BaseModel):
    """Preprocessing parameters."""

    normalization: Dict[str, Any] = Field(
        default={"method": "log_normalize", "target_sum": 1e4}
    )
    hvg_selection: Dict[str, Any] = Field(
        default={"n_top_genes": 5000, "flavor": "seurat_v3", "span": 0.3}
    )
    scale: Dict[str, Any] = Field(
        default={"zero_center": True, "max_value": None, "with_mean": True, "with_std": True}
    )

    class Config:
        frozen = False


class IntegrationMethodConfig(BaseModel):
    """Configuration for a single integration method."""

    name: str = Field(..., description="Method name")
    batch_key: Optional[str] = Field(None, description="Batch key")

    class Config:
        frozen = False
        extra = "allow"


class IntegrationConfig(BaseModel):
    """Integration parameters."""

    enabled: bool = Field(default=True, description="Enable integration")
    methods: List[IntegrationMethodConfig] = Field(..., description="Integration methods")

    class Config:
        frozen = False


class AnnotationMethodConfig(BaseModel):
    """Configuration for a single annotation method."""

    name: str = Field(..., description="Method name")

    class Config:
        frozen = False
        extra = "allow"


class AnnotationConfig(BaseModel):
    """Annotation parameters."""

    methods: List[AnnotationMethodConfig] = Field(..., description="Annotation methods")
    cell_type_markers: Dict[str, List[str]] = Field(
        default={}, description="Cell type marker genes"
    )

    class Config:
        frozen = False


class ClusteringConfig(BaseModel):
    """Clustering parameters."""

    pca: Dict[str, Any] = Field(default={"n_comps": 50, "svd_solver": "auto"})
    umap: Dict[str, Any] = Field(
        default={"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
    )
    leiden: Dict[str, Any] = Field(default={"resolution": [0.5, 1.0, 1.5], "random_state": 42})

    class Config:
        frozen = False


class EvaluationConfig(BaseModel):
    """Evaluation parameters."""

    metrics: List[str] = Field(..., description="Evaluation metrics")
    batch_correction_metrics: List[str] = Field(
        default=[], description="Batch correction metrics"
    )
    annotation_metrics: List[str] = Field(default=[], description="Annotation metrics")

    class Config:
        frozen = False


class PseudobulkConfig(BaseModel):
    """Pseudobulk analysis parameters."""

    grouping_variables: List[str] = Field(..., description="Grouping variables")
    min_cells_per_group: int = Field(default=10, description="Minimum cells per group")
    assay: str = Field(default="counts", description="Assay type")
    aggregation_method: str = Field(default="sum", description="Aggregation method")

    class Config:
        frozen = False


class DifferentialExpressionConfig(BaseModel):
    """Differential expression parameters."""

    methods: List[Dict[str, Any]] = Field(..., description="DE methods")

    class Config:
        frozen = False


class AgenticConfig(BaseModel):
    """Agentic tuning parameters."""

    enabled: bool = Field(default=True, description="Enable agentic tuning")
    search_budget: int = Field(default=100, description="Total configurations to search")
    editable_surface: List[str] = Field(..., description="Editable parameters")
    frozen_modules: List[str] = Field(..., description="Frozen modules")
    optimization_metric: str = Field(default="silhouette_score")
    optimization_direction: str = Field(default="maximize")
    early_stopping_patience: int = Field(default=10)
    trial_timeout_seconds: int = Field(default=3600)

    class Config:
        frozen = False

    @validator("optimization_direction")
    def validate_optimization_direction(cls, v):
        if v not in ["maximize", "minimize"]:
            raise ValueError("optimization_direction must be 'maximize' or 'minimize'")
        return v


class MLFlowConfig(BaseModel):
    """MLflow configuration."""

    enabled: bool = Field(default=True)
    tracking_uri: str = Field(default="file:./mlruns")
    experiment_name: str = Field(default="r3-mm-pipeline")
    run_name: Optional[str] = Field(None)
    tags: Dict[str, str] = Field(default={})
    params_log_frequency: int = Field(default=10)

    class Config:
        frozen = False


class WandBConfig(BaseModel):
    """Weights and Biases configuration."""

    enabled: bool = Field(default=False)
    project: str = Field(default="r3-mm-pipeline")
    entity: Optional[str] = Field(None)
    tags: List[str] = Field(default=[])
    notes: str = Field(default="")

    class Config:
        frozen = False


class DVCConfig(BaseModel):
    """DVC configuration."""

    enabled: bool = Field(default=True)
    remote_storage: str = Field(default="./dvc-storage")
    auto_push: bool = Field(default=False)

    class Config:
        frozen = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="logs/pipeline.log")
    console_output: bool = Field(default=True)

    class Config:
        frozen = False


class ComputeConfig(BaseModel):
    """Compute resource configuration."""

    n_jobs: int = Field(default=-1)
    gpu_enabled: bool = Field(default=True)
    gpu_device: int = Field(default=0)
    batch_size: int = Field(default=32)
    num_workers: int = Field(default=4)
    pin_memory: bool = Field(default=True)

    class Config:
        frozen = False


class PathsConfig(BaseModel):
    """File path configuration."""

    data_root: str = Field(default="./data")
    raw_data: str = Field(default="./data/raw")
    standardized_data: str = Field(default="./data/standardized")
    analysis_ready_data: str = Field(default="./data/analysis_ready")
    results: str = Field(default="./results")
    logs: str = Field(default="./logs")
    checkpoints: str = Field(default="./checkpoints")
    configs: str = Field(default="./configs")

    class Config:
        frozen = False


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: List[str] = Field(default=["h5ad", "parquet", "zarr"])
    compression: str = Field(default="gzip")
    level: int = Field(default=9)
    include_metadata: bool = Field(default=True)
    save_raw_counts: bool = Field(default=True)
    save_logs: bool = Field(default=True)

    class Config:
        frozen = False


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    name: str = Field(default="r3-mm-pipeline")
    version: str = Field(default="0.1.0")
    description: str = Field(default="")
    environment: str = Field(default="production")
    random_seed: int = Field(default=42)

    class Config:
        frozen = False


class Config(BaseSettings):
    """Main configuration class for the entire pipeline."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data_sources: DataSourcesConfig
    qc: QCConfig = Field(default_factory=QCConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pseudobulk: PseudobulkConfig = Field(default_factory=PseudobulkConfig)
    differential_expression: DifferentialExpressionConfig = Field(
        default_factory=DifferentialExpressionConfig
    )
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    dvc: DVCConfig = Field(default_factory=DVCConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config = SettingsConfigDict(
        yaml_file="configs/pipeline_config.yaml",
        yaml_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
    )


def load_config(config_path: str) -> Config:
    """
    Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Loaded and validated Config object.

    Raises:
        FileNotFoundError: If configuration file is not found.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If configuration validation fails.

    Example:
        >>> config = load_config("configs/pipeline_config.yaml")
        >>> print(config.pipeline.name)
        r3-mm-pipeline
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    return Config(**config_dict)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration object to save.
        output_path: Path where to save the configuration.

    Example:
        >>> config = load_config("configs/pipeline_config.yaml")
        >>> save_config(config, "configs/pipeline_config_backup.yaml")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump(exclude_none=False)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    config = load_config("configs/pipeline_config.yaml")
    print(f"Loaded configuration: {config.pipeline.name}")
