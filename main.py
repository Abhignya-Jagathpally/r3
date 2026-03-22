#!/usr/bin/env python3
"""
R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology Pipeline

End-to-end orchestrator for single-cell RNA-seq analysis of multiple myeloma,
including data download, QC, integration, annotation, modeling, evaluation,
and agentic hyperparameter tuning.

Usage:
    python main.py                                  # Full pipeline
    python main.py --config configs/pipeline_config.yaml
    python main.py --resume <run_id>                # Resume from checkpoint
    python main.py --stage preprocessing            # Run single stage
    python main.py --list-runs                      # List previous runs

Author: Abhignya Jagathpally
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Pipeline stage registry
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "download",
    "preprocessing",
    "integration",
    "clustering",
    "annotation",
    "modeling",
    "evaluation",
    "agentic_tuning",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
class TeeStream:
    """Write to both a file and a stream (tee-like behavior for stdout/stderr)."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    @property
    def encoding(self):
        return getattr(self.stream, 'encoding', 'utf-8')

    def fileno(self):
        return self.stream.fileno()


def setup_logging(log_config: Dict[str, Any], tee_logfile: str = "pipeline_run.log") -> None:
    """
    Configure pipeline-wide logging with tee-style transparency.

    All output goes to both the console and a log file, ensuring full
    traceability of every pipeline run.
    """
    log_dir = Path(log_config.get("file", "logs/pipeline.log")).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup tee: mirror stdout/stderr to a run-level log file
    tee_path = log_dir / tee_logfile
    tee_fh = open(tee_path, "a")
    sys.stdout = TeeStream(sys.__stdout__, tee_fh)
    sys.stderr = TeeStream(sys.__stderr__, tee_fh)

    handlers = [
        logging.FileHandler(log_config.get("file", "logs/pipeline.log")),
    ]
    if log_config.get("console_output", True):
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=handlers,
        force=True,
    )

    logging.getLogger("r3mm.main").info(f"Tee log active: {tee_path}")


logger = logging.getLogger("r3mm.main")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def stage_download(config, checkpoint_mgr, storage_mgr):
    """
    Stage 1: Download raw scRNA-seq data from GEO.

    Downloads each dataset listed in config.data_sources.datasets,
    saves to the raw storage layer, and creates a merged AnnData.
    """
    from src.data.download import GEODataDownloader

    meta = checkpoint_mgr.start_stage(
        "download",
        ad.AnnData(np.zeros((1, 1))),  # Placeholder for pre-download
        params={"datasets": [d.accession for d in config.data_sources.datasets]},
    )

    try:
        downloader = GEODataDownloader(
            ncbi_api_key=config.data_sources.download.ncbi_api_key,
            timeout=config.data_sources.download.timeout_seconds,
            retry_attempts=config.data_sources.download.retry_attempts,
        )

        all_adatas = []
        download_metrics = {}

        for dataset_cfg in config.data_sources.datasets:
            accession = dataset_cfg.accession
            logger.info(f"Downloading {accession}: {dataset_cfg.name}")

            # Check if already downloaded
            raw_path = Path(config.paths.raw_data) / f"{accession}.h5ad"
            if raw_path.exists():
                logger.info(f"Found existing download: {raw_path}")
                adata = ad.read_h5ad(raw_path)
            else:
                output_path, dl_metadata = downloader.download_gse(
                    accession=accession,
                    output_dir=Path(config.paths.raw_data),
                    name=dataset_cfg.name,
                )
                adata = ad.read_h5ad(output_path)
                download_metrics[accession] = dl_metadata

            adata.obs["dataset"] = accession
            adata.obs["dataset_name"] = dataset_cfg.name
            all_adatas.append(adata)
            logger.info(f"  {accession}: {adata.n_obs} cells, {adata.n_vars} genes")

        # Merge all datasets
        if len(all_adatas) == 1:
            merged = all_adatas[0]
        else:
            # Find common genes for concatenation
            merged = ad.concat(all_adatas, join="outer", label="dataset_source")
            logger.info(f"Merged {len(all_adatas)} datasets: {merged.shape}")

        # Store raw counts
        merged.layers["raw_counts"] = merged.X.copy()

        # Save to raw storage
        storage_mgr.write_raw(merged, "merged_raw")

        checkpoint_mgr.complete_stage(
            meta, merged,
            metrics={"n_datasets": len(all_adatas), "total_cells": merged.n_obs},
        )
        return merged

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_preprocessing(config, checkpoint_mgr, adata):
    """
    Stage 2: Quality control, filtering, normalization, and HVG selection.

    Applies QC filters, removes doublets, corrects ambient RNA,
    normalizes, selects highly variable genes, and scales.
    """
    from src.preprocessing.pipeline import PreprocessingPipeline

    meta = checkpoint_mgr.start_stage(
        "preprocessing", adata,
        params={
            "qc": config.qc.model_dump(),
            "preprocessing": config.preprocessing.model_dump(),
        },
    )

    try:
        pipeline = PreprocessingPipeline()

        adata, report = pipeline.run(
            adata,
            qc_config={
                "min_genes": config.qc.min_genes,
                "max_genes": config.qc.max_genes,
                "min_cells": config.qc.min_cells,
                "max_mito_pct": config.qc.max_mito_pct,
                "max_ribo_pct": config.qc.max_ribo_pct,
                "min_umis_per_cell": config.qc.min_umis_per_cell,
                "outlier_detection": config.qc.outlier_detection,
                "mad_threshold": config.qc.mad_threshold,
            },
            norm_config=config.preprocessing.normalization,
            hvg_config=config.preprocessing.hvg_selection,
            scale_config=config.preprocessing.scale,
        )

        metrics = {
            "cells_before": report.n_cells_before,
            "cells_after": report.n_cells_after,
            "genes_before": report.n_genes_before,
            "genes_after": report.n_genes_after,
            "pct_cells_retained": (
                report.n_cells_after / report.n_cells_before * 100
                if report.n_cells_before > 0 else 0
            ),
        }

        checkpoint_mgr.complete_stage(meta, adata, metrics=metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_integration(config, checkpoint_mgr, adata):
    """
    Stage 3: Batch effect correction using Harmony and/or scVI.

    Integrates data across datasets/batches, producing corrected
    embeddings in adata.obsm.
    """
    meta = checkpoint_mgr.start_stage(
        "integration", adata,
        params={"methods": [m.name for m in config.integration.methods]},
    )

    try:
        if not config.integration.enabled:
            logger.info("Integration disabled in config, skipping")
            checkpoint_mgr.complete_stage(meta, adata, metrics={"skipped": 1})
            return adata

        integration_metrics = {}

        for method_cfg in config.integration.methods:
            method_name = method_cfg.name
            batch_key = method_cfg.batch_key or "dataset"
            logger.info(f"Running integration: {method_name}")

            if method_name == "harmony":
                from src.integration.harmony import HarmonyIntegrator

                integrator = HarmonyIntegrator()
                extra_params = {}
                for k in ["theta", "npcs", "max_iter_harmony"]:
                    v = getattr(method_cfg, k, None)
                    if v is not None:
                        # Map config key to harmony param key
                        param_key = "max_iter" if k == "max_iter_harmony" else k
                        extra_params[param_key] = v

                adata = integrator.integrate(adata, batch_key=batch_key, **extra_params)
                integration_metrics["harmony_completed"] = 1.0

            elif method_name == "scvi":
                from src.integration.scvi_integration import ScVIIntegrator

                integrator = ScVIIntegrator(
                    n_latent=getattr(method_cfg, "n_latent", 32),
                    dispersion=getattr(method_cfg, "dispersion", "gene-batch"),
                    gene_likelihood=getattr(method_cfg, "gene_likelihood", "nb"),
                )
                integrator.setup(adata, batch_key=batch_key)
                integrator.train()
                adata = integrator.integrate(adata)

                # Save model checkpoint
                model_dir = Path(config.paths.checkpoints) / "scvi_model"
                integrator.save(str(model_dir))
                integration_metrics["scvi_completed"] = 1.0

            elif method_name == "scanvi":
                from src.integration.scanvi_integration import ScANVIIntegrator

                integrator = ScANVIIntegrator(
                    n_latent=getattr(method_cfg, "n_latent", 32),
                )
                integrator.integrate(adata, labels_key="cell_type", batch_key=batch_key)
                integration_metrics["scanvi_completed"] = 1.0

            else:
                logger.warning(f"Unknown integration method: {method_name}")

        checkpoint_mgr.complete_stage(meta, adata, metrics=integration_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_clustering(config, checkpoint_mgr, adata):
    """
    Stage 4: Dimensionality reduction and clustering.

    Runs PCA, computes neighbor graph, UMAP, and Leiden clustering
    at multiple resolutions.
    """
    import scanpy as sc

    meta = checkpoint_mgr.start_stage(
        "clustering", adata,
        params=config.clustering.model_dump(),
    )

    try:
        pca_cfg = config.clustering.pca
        umap_cfg = config.clustering.umap
        leiden_cfg = config.clustering.leiden

        # PCA
        n_comps = min(pca_cfg.get("n_comps", 50), adata.n_vars - 1, adata.n_obs - 1)
        logger.info(f"Computing PCA with {n_comps} components")
        sc.tl.pca(adata, n_comps=n_comps, svd_solver=pca_cfg.get("svd_solver", "auto"))

        # Use integrated embedding if available
        use_rep = "X_pca"
        if "X_harmony" in adata.obsm:
            use_rep = "X_harmony"
            logger.info("Using Harmony embeddings for neighbor graph")
        elif "X_scVI" in adata.obsm:
            use_rep = "X_scVI"
            logger.info("Using scVI embeddings for neighbor graph")

        # Neighbor graph
        sc.pp.neighbors(
            adata,
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            use_rep=use_rep,
            metric=umap_cfg.get("metric", "euclidean"),
        )

        # UMAP
        sc.tl.umap(adata, min_dist=umap_cfg.get("min_dist", 0.1))

        # Leiden clustering at multiple resolutions
        resolutions = leiden_cfg.get("resolution", [1.0])
        if isinstance(resolutions, (int, float)):
            resolutions = [resolutions]

        clustering_metrics = {}
        for res in resolutions:
            key = f"leiden_{res}"
            sc.tl.leiden(
                adata,
                resolution=res,
                random_state=leiden_cfg.get("random_state", 42),
                key_added=key,
            )
            n_clusters = adata.obs[key].nunique()
            clustering_metrics[f"n_clusters_res{res}"] = n_clusters
            logger.info(f"Leiden resolution {res}: {n_clusters} clusters")

        # Set default clustering to resolution 1.0 if available
        default_res = 1.0 if 1.0 in resolutions else resolutions[0]
        adata.obs["leiden"] = adata.obs[f"leiden_{default_res}"]

        checkpoint_mgr.complete_stage(meta, adata, metrics=clustering_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_annotation(config, checkpoint_mgr, adata):
    """
    Stage 5: Cell type annotation using marker genes, CellTypist, and consensus.

    Applies multiple annotation methods and builds a consensus label.
    """
    meta = checkpoint_mgr.start_stage(
        "annotation", adata,
        params={"methods": [m.name for m in config.annotation.methods]},
    )

    try:
        annotation_metrics = {}

        # Marker-based annotation
        if config.annotation.cell_type_markers:
            from src.annotation.marker_based import MarkerAnnotator

            marker_annotator = MarkerAnnotator(markers=config.annotation.cell_type_markers)
            adata = marker_annotator.annotate(adata)
            annotation_metrics["marker_based_completed"] = 1.0
            n_types = adata.obs["marker_annotation"].nunique()
            annotation_metrics["marker_n_cell_types"] = n_types
            logger.info(f"Marker annotation: {n_types} cell types")

        # Method-specific annotations
        annotation_keys = []
        if "marker_annotation" in adata.obs.columns:
            annotation_keys.append("marker_annotation")

        for method_cfg in config.annotation.methods:
            method_name = method_cfg.name

            if method_name == "celltypist":
                from src.annotation.celltypist_annotator import CellTypistAnnotator

                annotator = CellTypistAnnotator()
                adata = annotator.annotate(
                    adata,
                    model_name=getattr(method_cfg, "model", "Immune_All_Low.pkl"),
                    majority_voting=getattr(method_cfg, "majority_voting", True),
                    over_clustering=getattr(method_cfg, "over_clustering", True),
                )
                annotation_keys.append("celltypist_label")
                annotation_metrics["celltypist_completed"] = 1.0

            elif method_name == "scgpt":
                try:
                    from src.models.scgpt_wrapper import ScGPTModel

                    model_path = getattr(method_cfg, "model_path", "scGPT_human")
                    model = ScGPTModel(
                        model_dir=model_path,
                        n_hvg=getattr(method_cfg, "n_output_genes", 2000),
                    )
                    adata = model.predict(adata)
                    annotation_keys.append("scgpt_label")
                    annotation_metrics["scgpt_completed"] = 1.0
                except (ImportError, FileNotFoundError) as e:
                    logger.warning(f"scGPT skipped: {e}")

            else:
                logger.warning(f"Unknown annotation method: {method_name}")

        # Consensus annotation
        if len(annotation_keys) > 1:
            from src.annotation.consensus import ConsensusAnnotator

            consensus = ConsensusAnnotator()
            adata = consensus.annotate(adata, annotation_keys=annotation_keys)
            annotation_metrics["consensus_completed"] = 1.0
            logger.info("Consensus annotation computed")
        elif len(annotation_keys) == 1:
            adata.obs["cell_type"] = adata.obs[annotation_keys[0]]

        # Cell ontology mapping
        if "cell_type" in adata.obs.columns:
            from src.annotation.cell_ontology import CellOntologyMapper

            mapper = CellOntologyMapper()
            adata.obs["cell_type_ontology"] = mapper.map(adata.obs["cell_type"].tolist())

        checkpoint_mgr.complete_stage(meta, adata, metrics=annotation_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_modeling(config, checkpoint_mgr, adata):
    """
    Stage 6: Train classical and foundation models for cell type classification.

    Trains baseline classifiers (LogReg, RF, SVM, Ensemble) and optionally
    scGPT, evaluating on patient-level splits to prevent data leakage.
    """
    from src.evaluation.splits import PatientLevelSplitter
    from src.models.classical_baselines import ClassicalEnsemble

    meta = checkpoint_mgr.start_stage("modeling", adata, params={"models": "classical_ensemble"})

    try:
        model_metrics = {}

        # Determine label key
        label_key = "cell_type"
        if label_key not in adata.obs.columns:
            for fallback in ["consensus_label", "celltypist_label", "marker_annotation", "leiden"]:
                if fallback in adata.obs.columns:
                    label_key = fallback
                    break

        logger.info(f"Training models with label key: {label_key}")

        # Get embedding
        embed_key = None
        for key in ["X_harmony", "X_scVI", "X_pca"]:
            if key in adata.obsm:
                embed_key = key
                break

        if embed_key is None:
            logger.warning("No embedding found, computing PCA")
            import scanpy as sc
            sc.tl.pca(adata, n_comps=50)
            embed_key = "X_pca"

        X = adata.obsm[embed_key]
        y = adata.obs[label_key].values

        # Patient-level split
        patient_key = "dataset"
        if patient_key in adata.obs.columns and adata.obs[patient_key].nunique() > 1:
            splitter = PatientLevelSplitter()
            train_idx, test_idx = splitter.split(
                adata, patient_key=patient_key, test_size=0.2,
            )
        else:
            # Fallback to random split
            from sklearn.model_selection import train_test_split
            indices = np.arange(adata.n_obs)
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=config.pipeline.random_seed,
                stratify=y,
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"Train: {len(train_idx)} cells, Test: {len(test_idx)} cells")

        # Train classical ensemble
        ensemble = ClassicalEnsemble()
        ensemble.fit(X_train, y_train)

        # Evaluate
        y_pred = ensemble.predict(X_test)

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
        )

        model_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        model_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        model_metrics["f1_weighted"] = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )

        logger.info(
            f"Ensemble performance: acc={model_metrics['accuracy']:.3f}, "
            f"bal_acc={model_metrics['balanced_accuracy']:.3f}, "
            f"f1={model_metrics['f1_weighted']:.3f}"
        )

        # Save model checkpoint
        model_dir = Path(config.paths.checkpoints) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(model_dir / "classical_ensemble"))

        # Store predictions in adata
        adata.obs["ensemble_pred"] = ""
        adata.obs.loc[adata.obs.index[test_idx], "ensemble_pred"] = y_pred

        checkpoint_mgr.complete_stage(meta, adata, metrics=model_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_evaluation(config, checkpoint_mgr, adata):
    """
    Stage 7: Comprehensive evaluation with benchmark metrics and experiment tracking.

    Computes clustering quality, batch correction, annotation accuracy,
    and biological conservation metrics. Logs everything to MLflow/W&B.
    """
    meta = checkpoint_mgr.start_stage(
        "evaluation", adata,
        params={"metrics": config.evaluation.metrics},
    )

    try:
        from src.evaluation.metrics import BenchmarkSuite

        suite = BenchmarkSuite()
        eval_metrics = {}

        # Determine keys
        label_key = "cell_type"
        for fallback in ["consensus_label", "celltypist_label", "marker_annotation", "leiden"]:
            if fallback in adata.obs.columns:
                label_key = fallback
                break

        embed_key = None
        for key in ["X_harmony", "X_scVI", "X_pca"]:
            if key in adata.obsm:
                embed_key = key
                break

        # Run benchmark suite
        results = suite.run_all(
            adata,
            labels_true_key=label_key,
            labels_pred_key="ensemble_pred" if "ensemble_pred" in adata.obs.columns else label_key,
            batch_key="dataset",
            embed_key=embed_key or "X_pca",
        )
        eval_metrics.update(results)

        logger.info("Evaluation metrics:")
        for k, v in eval_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

        # Experiment tracking
        if config.mlflow.enabled:
            try:
                from src.evaluation.experiment_tracker import MLflowTracker

                tracker = MLflowTracker(
                    tracking_uri=config.mlflow.tracking_uri,
                    experiment_name=config.mlflow.experiment_name,
                )
                run_name = config.mlflow.run_name or checkpoint_mgr.run_id
                tracker.start_run(run_name)
                tracker.log_params({"run_id": checkpoint_mgr.run_id})
                tracker.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
                tracker.end_run()
                logger.info("Metrics logged to MLflow")
            except Exception as mlflow_err:
                logger.warning(f"MLflow logging failed: {mlflow_err}")

        if config.wandb.enabled:
            try:
                from src.evaluation.experiment_tracker import WandBTracker

                tracker = WandBTracker(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                )
                tracker.start_run(checkpoint_mgr.run_id)
                tracker.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
                tracker.end_run()
                logger.info("Metrics logged to W&B")
            except Exception as wandb_err:
                logger.warning(f"W&B logging failed: {wandb_err}")

        # Save evaluation results
        results_dir = Path(config.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        eval_path = results_dir / f"evaluation_{checkpoint_mgr.run_id}.yaml"
        with open(eval_path, "w") as f:
            yaml.dump(
                {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in eval_metrics.items()},
                f, default_flow_style=False,
            )
        logger.info(f"Evaluation results saved: {eval_path}")

        checkpoint_mgr.complete_stage(meta, adata, metrics=eval_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


def stage_agentic_tuning(config, checkpoint_mgr, adata):
    """
    Stage 8: Agentic hyperparameter tuning.

    Uses the AutoResearchAgent to search over the editable parameter surface,
    running pipeline trials with different configurations to optimize the
    target metric (e.g., silhouette_score).
    """
    meta = checkpoint_mgr.start_stage(
        "agentic_tuning", adata,
        params={
            "search_budget": config.agentic.search_budget,
            "optimization_metric": config.agentic.optimization_metric,
            "editable_surface": config.agentic.editable_surface,
        },
    )

    try:
        if not config.agentic.enabled:
            logger.info("Agentic tuning disabled in config, skipping")
            checkpoint_mgr.complete_stage(meta, adata, metrics={"skipped": 1})
            return adata

        from src.agentic.autoresearch_agent import AutoResearchAgent
        from src.agentic.config import AgenticConfig as AgenticTunerConfig

        tuner_config = AgenticTunerConfig(
            search_budget=config.agentic.search_budget,
            editable_surface=config.agentic.editable_surface,
            frozen_modules=config.agentic.frozen_modules,
            optimization_metric=config.agentic.optimization_metric,
            optimization_direction=config.agentic.optimization_direction,
            early_stopping_patience=config.agentic.early_stopping_patience,
            trial_timeout_seconds=config.agentic.trial_timeout_seconds,
        )

        agent = AutoResearchAgent(config=tuner_config)
        result = agent.run(adata)

        tuning_metrics = {
            "best_metric_value": result.best_score,
            "total_trials": result.total_trials,
            "successful_trials": result.successful_trials,
        }

        # Generate report
        from src.agentic.report_generator import ReportGenerator

        reporter = ReportGenerator()
        report_md = reporter.generate(result)
        report_path = Path(config.paths.results) / f"agentic_report_{checkpoint_mgr.run_id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info(f"Agentic tuning report saved: {report_path}")

        # Log best params
        logger.info(f"Best metric ({config.agentic.optimization_metric}): {result.best_score:.4f}")
        logger.info(f"Best params: {result.best_params}")

        checkpoint_mgr.complete_stage(meta, adata, metrics=tuning_metrics)
        return adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


# ---------------------------------------------------------------------------
# Pseudobulk analysis (post-pipeline)
# ---------------------------------------------------------------------------
def run_pseudobulk_analysis(config, adata, checkpoint_mgr):
    """
    Post-pipeline pseudobulk aggregation and differential expression.

    Aggregates single-cell data to pseudobulk profiles for downstream
    statistical analysis.
    """
    meta = checkpoint_mgr.start_stage(
        "pseudobulk", adata,
        params=config.pseudobulk.model_dump(),
    )

    try:
        from src.annotation.pseudobulk import PseudobulkAggregator

        aggregator = PseudobulkAggregator()

        groupby = config.pseudobulk.grouping_variables
        # Filter to columns that exist
        groupby = [g for g in groupby if g in adata.obs.columns]
        if not groupby:
            groupby = ["leiden"]

        pb_adata = aggregator.aggregate(
            adata,
            groupby=groupby,
            min_cells=config.pseudobulk.min_cells_per_group,
        )

        # Save pseudobulk data
        from src.data.storage import StorageManager

        storage = StorageManager(root_dir=config.paths.data_root)
        storage.write_analysis_ready(pb_adata, "pseudobulk")

        checkpoint_mgr.complete_stage(
            meta, pb_adata,
            metrics={"n_pseudobulk_samples": pb_adata.n_obs},
        )
        return pb_adata

    except Exception as e:
        checkpoint_mgr.fail_stage(meta, e)
        raise


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    config_path: str = "configs/pipeline_config.yaml",
    resume_run_id: Optional[str] = None,
    start_stage: Optional[str] = None,
    end_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full R3-MM pipeline end-to-end.

    Args:
        config_path: Path to pipeline configuration YAML
        resume_run_id: Run ID to resume from (uses checkpoints)
        start_stage: Stage to start from (skips earlier stages)
        end_stage: Stage to stop after

    Returns:
        Dictionary with run summary, metrics, and paths
    """
    # Load config
    from src.config import load_config

    config = load_config(config_path)

    # Setup logging
    setup_logging(config.logging.model_dump())
    logger.info("=" * 80)
    logger.info(f"R3-MM Pipeline v{config.pipeline.version}")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 80)

    # Set random seed
    np.random.seed(config.pipeline.random_seed)

    # Initialize checkpoint manager
    from src.checkpoints import CheckpointManager

    if resume_run_id:
        checkpoint_mgr = CheckpointManager.load_run(resume_run_id, config.paths.checkpoints)
        logger.info(f"Resuming run: {resume_run_id}")
    else:
        checkpoint_mgr = CheckpointManager(
            checkpoints_dir=config.paths.checkpoints,
            pipeline_name=config.pipeline.name,
            pipeline_version=config.pipeline.version,
            random_seed=config.pipeline.random_seed,
        )

    # Save config and environment
    checkpoint_mgr.save_config_snapshot(config.model_dump())
    checkpoint_mgr.save_environment_info()

    # Initialize storage
    from src.data.storage import StorageManager

    storage_mgr = StorageManager(root_dir=config.paths.data_root)

    # Determine which stages to run
    completed = set(checkpoint_mgr.get_completed_stages()) if resume_run_id else set()
    stages_to_run = list(PIPELINE_STAGES)

    if start_stage:
        idx = stages_to_run.index(start_stage)
        stages_to_run = stages_to_run[idx:]
    if end_stage:
        idx = stages_to_run.index(end_stage)
        stages_to_run = stages_to_run[: idx + 1]

    logger.info(f"Stages to run: {stages_to_run}")
    logger.info(f"Already completed: {completed}")

    # Execute pipeline stages
    adata = None

    # --- Stage 1: Download ---
    if "download" in stages_to_run:
        if "download" in completed:
            logger.info("Resuming from download checkpoint")
            adata = checkpoint_mgr.resume_from("download")
        else:
            adata = stage_download(config, checkpoint_mgr, storage_mgr)
    else:
        # Load from checkpoint or raw storage
        adata = checkpoint_mgr.resume_from("download")
        if adata is None:
            adata = storage_mgr.read_raw("merged_raw")

    # --- Stage 2: Preprocessing ---
    if "preprocessing" in stages_to_run:
        if "preprocessing" in completed:
            logger.info("Resuming from preprocessing checkpoint")
            adata = checkpoint_mgr.resume_from("preprocessing")
        else:
            adata = stage_preprocessing(config, checkpoint_mgr, adata)

    # --- Stage 3: Integration ---
    if "integration" in stages_to_run:
        if "integration" in completed:
            adata = checkpoint_mgr.resume_from("integration")
        else:
            adata = stage_integration(config, checkpoint_mgr, adata)

    # --- Stage 4: Clustering ---
    if "clustering" in stages_to_run:
        if "clustering" in completed:
            adata = checkpoint_mgr.resume_from("clustering")
        else:
            adata = stage_clustering(config, checkpoint_mgr, adata)

    # --- Stage 5: Annotation ---
    if "annotation" in stages_to_run:
        if "annotation" in completed:
            adata = checkpoint_mgr.resume_from("annotation")
        else:
            adata = stage_annotation(config, checkpoint_mgr, adata)

    # --- Stage 6: Modeling ---
    if "modeling" in stages_to_run:
        if "modeling" in completed:
            adata = checkpoint_mgr.resume_from("modeling")
        else:
            adata = stage_modeling(config, checkpoint_mgr, adata)

    # --- Stage 7: Evaluation ---
    if "evaluation" in stages_to_run:
        if "evaluation" in completed:
            adata = checkpoint_mgr.resume_from("evaluation")
        else:
            adata = stage_evaluation(config, checkpoint_mgr, adata)

    # --- Stage 8: Agentic Tuning ---
    if "agentic_tuning" in stages_to_run:
        if "agentic_tuning" in completed:
            adata = checkpoint_mgr.resume_from("agentic_tuning")
        else:
            adata = stage_agentic_tuning(config, checkpoint_mgr, adata)

    # --- Post-pipeline: Pseudobulk ---
    if "pseudobulk" not in completed and adata is not None:
        try:
            run_pseudobulk_analysis(config, adata, checkpoint_mgr)
        except Exception as e:
            logger.warning(f"Pseudobulk analysis failed (non-critical): {e}")

    # Save final analysis-ready data
    if adata is not None:
        storage_mgr.write_analysis_ready(adata, "final_annotated")
        logger.info(f"Final dataset: {adata.n_obs} cells, {adata.n_vars} genes")

    # Finalize
    manifest_path = checkpoint_mgr.finalize()
    summary = checkpoint_mgr.get_stage_summary()

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Run ID: {checkpoint_mgr.run_id}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info("Stage summary:")
    for stage, info in summary.items():
        logger.info(f"  {stage}: {info['status']} ({info['duration_seconds']:.1f}s) {info['cells']}")
    logger.info("=" * 80)

    return {
        "run_id": checkpoint_mgr.run_id,
        "status": checkpoint_mgr.manifest.status,
        "manifest_path": str(manifest_path),
        "summary": summary,
        "final_shape": (adata.n_obs, adata.n_vars) if adata is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """CLI entry point for the R3-MM pipeline."""
    parser = argparse.ArgumentParser(
        description="R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Full pipeline run
  python main.py --config configs/pipeline_config.yaml
  python main.py --resume run_20260320_143000_abc123
  python main.py --stage preprocessing              # Single stage
  python main.py --start integration --end evaluation
  python main.py --list-runs                        # Show previous runs
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--resume", "-r",
        default=None,
        help="Resume from a previous run ID (loads checkpoints)",
    )
    parser.add_argument(
        "--stage", "-s",
        default=None,
        choices=PIPELINE_STAGES,
        help="Run only a single stage",
    )
    parser.add_argument(
        "--start",
        default=None,
        choices=PIPELINE_STAGES,
        help="Start from this stage (inclusive)",
    )
    parser.add_argument(
        "--end",
        default=None,
        choices=PIPELINE_STAGES,
        help="Stop after this stage (inclusive)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all previous pipeline runs",
    )
    parser.add_argument(
        "--run-info",
        default=None,
        help="Show detailed info for a specific run ID",
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_runs:
        from src.checkpoints import CheckpointManager

        runs = CheckpointManager.list_runs()
        if not runs:
            print("No previous runs found.")
        else:
            print(f"{'Run ID':<45} {'Status':<12} {'Stages':<8} {'Started'}")
            print("-" * 90)
            for run in runs:
                print(
                    f"{run['run_id']:<45} {run['status']:<12} "
                    f"{run['n_stages']:<8} {run['start_time']}"
                )
        return

    if args.run_info:
        from src.checkpoints import CheckpointManager

        mgr = CheckpointManager.load_run(args.run_info)
        summary = mgr.get_stage_summary()
        print(f"\nRun: {mgr.run_id}")
        print(f"Status: {mgr.manifest.status}")
        print(f"Started: {mgr.manifest.start_time}")
        print(f"\nStages:")
        for stage, info in summary.items():
            print(f"  {stage}: {info['status']} ({info['duration_seconds']:.1f}s) {info['cells']}")
            if info["metrics"]:
                for k, v in info["metrics"].items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
        return

    # Run pipeline
    start_stage = args.stage or args.start
    end_stage = args.stage or args.end

    result = run_pipeline(
        config_path=args.config,
        resume_run_id=args.resume,
        start_stage=start_stage,
        end_stage=end_stage,
    )

    # Exit with appropriate code
    sys.exit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
