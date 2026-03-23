"""
Checkpoint management for the R3-MM pipeline.

Provides save/load functionality for pipeline state at each stage,
enabling reproducibility, fault tolerance, and training traceability.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class StageMetadata:
    """Metadata for a single pipeline stage checkpoint."""

    stage_name: str
    status: str  # "started", "completed", "failed"
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    n_cells_in: int = 0
    n_cells_out: int = 0
    n_genes_in: int = 0
    n_genes_out: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: str = ""
    data_hash: str = ""


@dataclass
class PipelineManifest:
    """Full manifest tracking all pipeline stages and lineage."""

    pipeline_name: str
    pipeline_version: str
    run_id: str
    random_seed: int
    start_time: str = ""
    end_time: str = ""
    status: str = "initialized"
    stages: List[StageMetadata] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages pipeline checkpoints for traceability and fault tolerance.

    Each pipeline run creates a directory under checkpoints_dir with:
    - A manifest.yaml tracking all stages
    - Per-stage .h5ad snapshots of the data
    - Per-stage metadata JSON files
    - A config snapshot for full reproducibility

    Attributes:
        checkpoints_dir: Root directory for all checkpoints
        run_id: Unique identifier for the current pipeline run
        run_dir: Directory for the current run's checkpoints
        manifest: Pipeline manifest tracking all stages
    """

    def __init__(
        self,
        checkpoints_dir: str = "./checkpoints",
        run_id: Optional[str] = None,
        pipeline_name: str = "r3-mm-pipeline",
        pipeline_version: str = "0.1.0",
        random_seed: int = 42,
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.checkpoints_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = PipelineManifest(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            run_id=self.run_id,
            random_seed=random_seed,
            start_time=datetime.now(timezone.utc).isoformat(),
            status="running",
        )

        logger.info(f"CheckpointManager initialized: run_id={self.run_id}, dir={self.run_dir}")

    def _generate_run_id(self) -> str:
        """Generate a unique run ID from timestamp + random hash."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]
        return f"run_{ts}_{h}"

    def _compute_data_hash(self, adata: ad.AnnData) -> str:
        """Compute a lightweight hash of the AnnData shape and a sample of values."""
        shape_str = f"{adata.n_obs}x{adata.n_vars}"
        if adata.X is not None:
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            flat = np.asarray(X).ravel()
            sample = flat[:: max(1, len(flat) // 1000)]
            val_hash = hashlib.md5(sample.tobytes()).hexdigest()[:12]
        else:
            val_hash = "no_X"
        return f"{shape_str}_{val_hash}"

    def save_config_snapshot(self, config_dict: Dict[str, Any]) -> Path:
        """Save the full pipeline configuration for reproducibility."""
        self.manifest.config_snapshot = config_dict
        path = self.run_dir / "config_snapshot.yaml"
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config snapshot saved: {path}")
        return path

    def save_environment_info(self) -> Dict[str, str]:
        """Capture and save environment information."""
        import platform
        import sys

        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Capture key package versions
        for pkg in ["scanpy", "anndata", "scvi", "torch", "numpy", "pandas"]:
            try:
                mod = __import__(pkg)
                env_info[f"{pkg}_version"] = str(getattr(mod, "__version__", "unknown"))
            except ImportError:
                env_info[f"{pkg}_version"] = "not_installed"

        self.manifest.environment = env_info

        path = self.run_dir / "environment.json"
        with open(path, "w") as f:
            json.dump(env_info, f, indent=2)

        logger.info(f"Environment info saved: {path}")
        return env_info

    def start_stage(
        self,
        stage_name: str,
        adata: ad.AnnData,
        params: Optional[Dict[str, Any]] = None,
    ) -> StageMetadata:
        """
        Record the start of a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage
            adata: Input AnnData at stage entry
            params: Parameters used for this stage

        Returns:
            StageMetadata for tracking
        """
        meta = StageMetadata(
            stage_name=stage_name,
            status="started",
            start_time=datetime.now(timezone.utc).isoformat(),
            n_cells_in=adata.n_obs,
            n_genes_in=adata.n_vars,
            params=params or {},
        )

        logger.info(
            f"Stage '{stage_name}' started: {adata.n_obs} cells, {adata.n_vars} genes"
        )
        return meta

    def complete_stage(
        self,
        meta: StageMetadata,
        adata: ad.AnnData,
        metrics: Optional[Dict[str, float]] = None,
        save_data: bool = True,
    ) -> Path:
        """
        Record completion of a pipeline stage and save checkpoint.

        Args:
            meta: StageMetadata from start_stage()
            adata: Output AnnData after stage processing
            metrics: Any metrics computed during the stage
            save_data: Whether to save the AnnData snapshot

        Returns:
            Path to the saved checkpoint file
        """
        meta.status = "completed"
        meta.end_time = datetime.now(timezone.utc).isoformat()
        meta.n_cells_out = adata.n_obs
        meta.n_genes_out = adata.n_vars
        meta.metrics = metrics or {}
        meta.data_hash = self._compute_data_hash(adata)

        # Compute duration
        start = datetime.fromisoformat(meta.start_time)
        end = datetime.fromisoformat(meta.end_time)
        meta.duration_seconds = (end - start).total_seconds()

        # Save data checkpoint
        checkpoint_path = self.run_dir / f"{meta.stage_name}.h5ad"
        if save_data:
            # Ensure all object-dtype obs columns are proper strings for h5ad
            for col in adata.obs.columns:
                if adata.obs[col].dtype == object:
                    adata.obs[col] = adata.obs[col].fillna("").astype(str)
            adata.write_h5ad(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save stage metadata
        meta_path = self.run_dir / f"{meta.stage_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2, default=str)

        # Append to manifest
        self.manifest.stages.append(meta)
        self._save_manifest()

        logger.info(
            f"Stage '{meta.stage_name}' completed in {meta.duration_seconds:.1f}s: "
            f"{meta.n_cells_in}→{meta.n_cells_out} cells, "
            f"{meta.n_genes_in}→{meta.n_genes_out} genes"
        )

        return checkpoint_path

    def fail_stage(
        self,
        meta: StageMetadata,
        error: Exception,
    ) -> None:
        """Record a failed pipeline stage."""
        meta.status = "failed"
        meta.end_time = datetime.now(timezone.utc).isoformat()
        meta.error_message = str(error)

        start = datetime.fromisoformat(meta.start_time)
        end = datetime.fromisoformat(meta.end_time)
        meta.duration_seconds = (end - start).total_seconds()

        self.manifest.stages.append(meta)
        self.manifest.status = "failed"
        self._save_manifest()

        logger.error(f"Stage '{meta.stage_name}' FAILED after {meta.duration_seconds:.1f}s: {error}")

    def load_checkpoint(self, stage_name: str) -> Optional[ad.AnnData]:
        """
        Load a previously saved checkpoint.

        Args:
            stage_name: Name of the stage to load

        Returns:
            AnnData object if checkpoint exists, None otherwise
        """
        path = self.run_dir / f"{stage_name}.h5ad"
        if path.exists():
            adata = ad.read_h5ad(path)
            logger.info(f"Loaded checkpoint '{stage_name}': {adata.shape}")
            return adata
        logger.warning(f"No checkpoint found for stage '{stage_name}'")
        return None

    def resume_from(self, stage_name: str) -> Optional[ad.AnnData]:
        """
        Resume pipeline from a specific stage checkpoint.

        Args:
            stage_name: Stage to resume from

        Returns:
            AnnData from that stage's checkpoint
        """
        logger.info(f"Attempting to resume from stage '{stage_name}'...")
        return self.load_checkpoint(stage_name)

    def get_completed_stages(self) -> List[str]:
        """Return list of successfully completed stage names."""
        return [s.stage_name for s in self.manifest.stages if s.status == "completed"]

    def get_stage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary dict of all stage metadata."""
        return {
            s.stage_name: {
                "status": s.status,
                "duration_seconds": s.duration_seconds,
                "cells": f"{s.n_cells_in}→{s.n_cells_out}",
                "genes": f"{s.n_genes_in}→{s.n_genes_out}",
                "metrics": s.metrics,
            }
            for s in self.manifest.stages
        }

    def finalize(self) -> Path:
        """
        Finalize the pipeline run and save the complete manifest.

        Returns:
            Path to the final manifest file
        """
        self.manifest.end_time = datetime.now(timezone.utc).isoformat()
        if self.manifest.status == "running":
            failed = any(s.status == "failed" for s in self.manifest.stages)
            self.manifest.status = "failed" if failed else "completed"

        manifest_path = self._save_manifest()

        # Print summary
        total_time = sum(s.duration_seconds for s in self.manifest.stages)
        completed = sum(1 for s in self.manifest.stages if s.status == "completed")
        failed = sum(1 for s in self.manifest.stages if s.status == "failed")

        logger.info(
            f"Pipeline run {self.run_id} finalized: "
            f"{completed} completed, {failed} failed, "
            f"total time: {total_time:.1f}s"
        )

        return manifest_path

    def _save_manifest(self) -> Path:
        """Save the current manifest to disk."""
        path = self.run_dir / "manifest.yaml"
        manifest_dict = asdict(self.manifest)
        with open(path, "w") as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def list_runs(cls, checkpoints_dir: str = "./checkpoints") -> List[Dict[str, Any]]:
        """
        List all pipeline runs in the checkpoints directory.

        Returns:
            List of run summary dicts with run_id, status, start_time
        """
        root = Path(checkpoints_dir)
        runs = []
        if not root.exists():
            return runs

        for run_dir in sorted(root.iterdir()):
            manifest_path = run_dir / "manifest.yaml"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f)
                runs.append({
                    "run_id": manifest.get("run_id", run_dir.name),
                    "status": manifest.get("status", "unknown"),
                    "start_time": manifest.get("start_time", ""),
                    "n_stages": len(manifest.get("stages", [])),
                })

        return runs

    @classmethod
    def load_run(cls, run_id: str, checkpoints_dir: str = "./checkpoints") -> "CheckpointManager":
        """Load an existing run's CheckpointManager for inspection or resumption."""
        run_dir = Path(checkpoints_dir) / run_id
        manifest_path = run_dir / "manifest.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found for run: {run_id}")

        with open(manifest_path) as f:
            manifest_dict = yaml.safe_load(f)

        mgr = cls.__new__(cls)
        mgr.checkpoints_dir = Path(checkpoints_dir)
        mgr.run_id = run_id
        mgr.run_dir = run_dir

        stages = []
        for s in manifest_dict.pop("stages", []):
            stages.append(StageMetadata(**s))
        mgr.manifest = PipelineManifest(**manifest_dict, stages=stages)

        logger.info(f"Loaded existing run: {run_id}")
        return mgr
