"""
Complete preprocessing pipeline for single-cell RNA-seq data.

This module provides the PreprocessingPipeline class that chains all preprocessing
steps (QC, doublet removal, ambient RNA correction, normalization, HVG selection,
and batch annotation) with comprehensive logging and reporting.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata
import pandas as pd
from pydantic import BaseModel, Field

from src.preprocessing.ambient_rna import AmbientRNACorrector
from src.preprocessing.batch_info import BatchAnnotator
from src.preprocessing.doublet_removal import DoubletDetector
from src.preprocessing.normalization import Normalizer
from src.preprocessing.qc import QCFilter

logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    # QC parameters
    min_genes: int = Field(default=200, description="Minimum genes per cell")
    max_genes: int = Field(default=5000, description="Maximum genes per cell")
    min_cells: int = Field(default=3, description="Minimum cells per gene")
    max_mito_pct: float = Field(default=20.0, description="Max mitochondrial percentage")
    max_ribo_pct: float = Field(default=50.0, description="Max ribosomal percentage")
    min_umi: int = Field(default=500, description="Minimum UMI count per cell")

    # Doublet detection
    detect_doublets: bool = Field(default=True, description="Enable doublet detection")
    doublet_method: str = Field(
        default="scrublet", description="Doublet detection method"
    )
    expected_doublet_rate: float = Field(default=0.06, description="Expected doublet rate")

    # Ambient RNA correction
    correct_ambient_rna: bool = Field(default=True, description="Enable ambient RNA correction")
    ambient_rna_method: str = Field(
        default="soupx", description="Ambient RNA correction method"
    )

    # Normalization
    normalization_method: str = Field(
        default="scanpy", description="Normalization method"
    )
    target_sum: float = Field(default=1e4, description="Target sum for normalization")

    # HVG selection
    select_hvgs: bool = Field(default=True, description="Enable HVG selection")
    n_hvgs: int = Field(default=2000, description="Number of HVGs to select")
    hvg_flavor: str = Field(default="seurat_v3", description="HVG selection flavor")

    # Batch annotation
    annotate_batch: bool = Field(default=True, description="Enable batch annotation")

    class Config:
        frozen = False


@dataclass
class PreprocessingReport:
    """Report from preprocessing pipeline."""

    n_cells_before: int
    n_cells_after: int
    n_genes_before: int
    n_genes_after: int
    qc_report: Dict
    doublet_stats: Optional[Dict] = None
    ambient_rna_stats: Optional[Dict] = None
    normalization_stats: Optional[Dict] = None
    hvg_stats: Optional[Dict] = None
    steps_completed: list = None

    # Aliases for backward compatibility
    @property
    def n_cells_initial(self):
        return self.n_cells_before

    @property
    def n_cells_final(self):
        return self.n_cells_after

    @property
    def n_genes_initial(self):
        return self.n_genes_before

    @property
    def n_genes_final(self):
        return self.n_genes_after

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        data = asdict(self)
        if self.steps_completed is None:
            data["steps_completed"] = []
        return data

    def to_json(self, output_path: str) -> None:
        """Save report as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved preprocessing report to {output_path}")


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for single-cell RNA-seq data.

    Chains all preprocessing steps in order:
    1. QC filtering
    2. Doublet removal
    3. Ambient RNA correction
    4. Normalization
    5. HVG selection
    6. Batch annotation

    Provides configurable parameters and comprehensive logging/reporting.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.

        Args:
            config: PreprocessingConfig object. Uses defaults if not provided.
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Initialized PreprocessingPipeline with config: {self.config}")

        # Initialize component classes
        self.qc_filter = QCFilter(
            min_genes=self.config.min_genes,
            max_genes=self.config.max_genes,
            min_cells=self.config.min_cells,
            max_mito_pct=self.config.max_mito_pct,
            max_ribo_pct=self.config.max_ribo_pct,
            min_umi=self.config.min_umi,
        )
        self.doublet_detector = DoubletDetector()
        self.ambient_rna_corrector = AmbientRNACorrector()
        self.normalizer = Normalizer()
        self.batch_annotator = BatchAnnotator()

    def run(
        self,
        adata: anndata.AnnData,
        raw_adata: Optional[anndata.AnnData] = None,
        qc_config: Optional[Dict] = None,
        norm_config: Optional[Dict] = None,
        hvg_config: Optional[Dict] = None,
        scale_config: Optional[Dict] = None,
    ) -> Tuple[anndata.AnnData, PreprocessingReport]:
        """
        Run complete preprocessing pipeline.

        Executes all configured preprocessing steps in sequence.

        Args:
            adata: Input annotated data matrix.
            raw_adata: Optional raw AnnData with unfiltered counts for ambient RNA correction.

        Returns:
            Tuple of:
                - Preprocessed AnnData object
                - PreprocessingReport with all statistics

        Raises:
            ValueError: If preprocessing fails at any step.
        """
        logger.info("=" * 80)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 80)

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot preprocess")

        # Override config from kwargs if provided
        if qc_config:
            for k, v in qc_config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.qc_filter = QCFilter(
                min_genes=self.config.min_genes,
                max_genes=self.config.max_genes,
                min_cells=self.config.min_cells,
                max_mito_pct=self.config.max_mito_pct,
                max_ribo_pct=self.config.max_ribo_pct,
                min_umi=self.config.min_umi,
            )
        if norm_config:
            if "target_sum" in norm_config:
                self.config.target_sum = float(norm_config["target_sum"])
            if "method" in norm_config:
                method_map = {"log_normalize": "scanpy"}
                self.config.normalization_method = method_map.get(
                    norm_config["method"], norm_config["method"]
                )
        if hvg_config:
            if "n_top_genes" in hvg_config:
                self.config.n_hvgs = hvg_config["n_top_genes"]
            if "flavor" in hvg_config:
                self.config.hvg_flavor = hvg_config["flavor"]

        n_cells_initial = adata.n_obs
        n_genes_initial = adata.n_vars
        steps_completed = []

        report_data = {
            "n_cells_initial": n_cells_initial,
            "n_genes_initial": n_genes_initial,
            "qc_report": {},
            "doublet_stats": None,
            "ambient_rna_stats": None,
            "normalization_stats": None,
            "hvg_stats": None,
        }

        # Step 1: QC Filtering
        logger.info("STEP 1: Quality Control Filtering")
        logger.info("-" * 40)
        try:
            adata, qc_report = self.qc_filter.run(adata)
            report_data["qc_report"] = qc_report
            steps_completed.append("qc_filtering")
            logger.info("✓ QC filtering completed")
        except Exception as e:
            logger.error(f"QC filtering failed: {e}")
            raise

        # Step 2: Doublet Detection
        if self.config.detect_doublets:
            logger.info("STEP 2: Doublet Detection")
            logger.info("-" * 40)
            try:
                if self.config.doublet_method == "scrublet":
                    adata = self.doublet_detector.detect_scrublet(
                        adata, expected_doublet_rate=self.config.expected_doublet_rate
                    )
                    adata, doublet_stats = self.doublet_detector.consensus_filter(adata)
                    report_data["doublet_stats"] = doublet_stats
                    steps_completed.append("doublet_detection")
                    logger.info("✓ Doublet detection completed")
            except Exception as e:
                logger.warning(f"Doublet detection failed (non-fatal): {e}")
        else:
            logger.info("Doublet detection disabled")

        # Step 3: Ambient RNA Correction
        if self.config.correct_ambient_rna:
            logger.info("STEP 3: Ambient RNA Correction")
            logger.info("-" * 40)
            try:
                adata, ambient_stats = self.ambient_rna_corrector.run(
                    adata, method=self.config.ambient_rna_method, raw_adata=raw_adata
                )
                report_data["ambient_rna_stats"] = ambient_stats
                steps_completed.append("ambient_rna_correction")
                logger.info("✓ Ambient RNA correction completed")
            except Exception as e:
                logger.warning(f"Ambient RNA correction failed (non-fatal): {e}")
        else:
            logger.info("Ambient RNA correction disabled")

        # Step 4: Normalization
        logger.info("STEP 4: Normalization")
        logger.info("-" * 40)
        try:
            if self.config.normalization_method == "scanpy":
                adata, norm_stats = self.normalizer.scanpy_normalize(
                    adata, target_sum=self.config.target_sum
                )
            elif self.config.normalization_method == "scran":
                adata, norm_stats = self.normalizer.scran_normalize(adata)
            elif self.config.normalization_method == "pearson_residuals":
                adata, norm_stats = self.normalizer.pearson_residuals(adata)
            else:
                raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")

            report_data["normalization_stats"] = norm_stats
            steps_completed.append("normalization")
            logger.info("✓ Normalization completed")
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            raise

        # Step 5: HVG Selection
        if self.config.select_hvgs:
            logger.info("STEP 5: Highly Variable Gene Selection")
            logger.info("-" * 40)
            try:
                adata, hvg_stats = self.normalizer.select_hvgs(
                    adata, n_top_genes=self.config.n_hvgs, flavor=self.config.hvg_flavor
                )
                report_data["hvg_stats"] = hvg_stats
                steps_completed.append("hvg_selection")
                logger.info("✓ HVG selection completed")
            except Exception as e:
                logger.warning(f"HVG selection failed (non-fatal): {e}")
        else:
            logger.info("HVG selection disabled")

        # Step 6: Batch Annotation
        if self.config.annotate_batch:
            logger.info("STEP 6: Batch Annotation")
            logger.info("-" * 40)
            try:
                adata = self.batch_annotator.run(adata)
                steps_completed.append("batch_annotation")
                logger.info("✓ Batch annotation completed")
            except Exception as e:
                logger.warning(f"Batch annotation failed (non-fatal): {e}")
        else:
            logger.info("Batch annotation disabled")

        # Create final report
        report_data["n_cells_after"] = adata.n_obs
        report_data["n_genes_after"] = adata.n_vars
        report_data["steps_completed"] = steps_completed

        # Rename initial → before for constructor
        report_data["n_cells_before"] = report_data.pop("n_cells_initial")
        report_data["n_genes_before"] = report_data.pop("n_genes_initial")

        report = PreprocessingReport(**report_data)

        logger.info("=" * 80)
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Summary: {n_cells_initial} cells -> {adata.n_obs} cells")
        logger.info(f"Summary: {n_genes_initial} genes -> {adata.n_vars} genes")
        logger.info("=" * 80)

        return adata, report

    def freeze_contract(self, output_path: str) -> None:
        """
        Save preprocessing configuration as a frozen contract.

        Creates a JSON file with all preprocessing parameters and configuration.
        This contract ensures reproducibility and serves as a record of the
        preprocessing parameters used.

        Args:
            output_path: Path where to save the frozen contract.

        Raises:
            IOError: If file cannot be written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create contract
        contract = {
            "preprocessing_version": "0.1.0",
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": self.config.model_dump(exclude_none=False),
            "frozen": True,
        }

        try:
            with open(output_path, "w") as f:
                json.dump(contract, f, indent=2)

            logger.info(f"Saved frozen preprocessing contract to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save contract: {e}")
            raise
