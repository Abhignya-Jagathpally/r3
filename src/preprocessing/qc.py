"""
Quality control (QC) filtering for single-cell RNA-seq data.

This module provides QC metrics calculation and filtering for single-cell datasets,
including removal of low-quality cells based on gene counts, UMI counts, and
mitochondrial/ribosomal content.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class QCMetrics:
    """Container for QC metrics."""

    n_cells_total: int
    n_cells_after_filtering: int
    n_genes_total: int
    n_genes_after_filtering: int
    pct_cells_removed: float
    pct_genes_removed: float
    mean_genes_per_cell: float
    mean_umis_per_cell: float
    pct_mito: float
    pct_ribo: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of QC metrics.
        """
        return asdict(self)


class QCFilter:
    """
    Quality control filter for single-cell RNA-seq data.

    Provides methods to calculate QC metrics, filter cells and genes based on
    configurable thresholds for gene counts, UMI counts, mitochondrial percentage,
    and ribosomal percentage.

    Attributes:
        min_genes: Minimum number of genes per cell.
        max_genes: Maximum number of genes per cell.
        min_cells: Minimum number of cells expressing a gene.
        max_mito_pct: Maximum mitochondrial percentage per cell.
        max_ribo_pct: Maximum ribosomal percentage per cell.
        min_umi: Minimum UMI count per cell.
    """

    def __init__(
        self,
        min_genes: int = 200,
        max_genes: int = 5000,
        min_cells: int = 3,
        max_mito_pct: float = 20.0,
        max_ribo_pct: float = 50.0,
        min_umi: int = 500,
    ):
        """
        Initialize QC filter.

        Args:
            min_genes: Minimum number of genes per cell. Default: 200.
            max_genes: Maximum number of genes per cell. Default: 5000.
            min_cells: Minimum number of cells expressing a gene. Default: 3.
            max_mito_pct: Maximum mitochondrial percentage per cell. Default: 20.0.
            max_ribo_pct: Maximum ribosomal percentage per cell. Default: 50.0.
            min_umi: Minimum UMI count per cell. Default: 500.
        """
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.min_cells = min_cells
        self.max_mito_pct = max_mito_pct
        self.max_ribo_pct = max_ribo_pct
        self.min_umi = min_umi

        logger.info(
            f"Initialized QCFilter with parameters: "
            f"min_genes={min_genes}, max_genes={max_genes}, "
            f"min_cells={min_cells}, max_mito_pct={max_mito_pct}, "
            f"max_ribo_pct={max_ribo_pct}, min_umi={min_umi}"
        )

    def calculate_qc_metrics(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Calculate quality control metrics.

        Uses scanpy's calculate_qc_metrics to compute gene counts, UMI counts,
        and percentage of mitochondrial/ribosomal genes for each cell.

        Args:
            adata: Annotated data matrix.

        Returns:
            AnnData object with QC metrics added to adata.obs and adata.var.

        Raises:
            ValueError: If the data matrix is empty or malformed.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot calculate QC metrics")

        logger.info(f"Calculating QC metrics for {adata.n_obs} cells and {adata.n_vars} genes")

        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)

        # Identify mitochondrial genes (MT-)
        adata.var["mito"] = adata.var_names.str.startswith("MT-")
        # Identify ribosomal genes (RP[SL]-)
        adata.var["ribo"] = adata.var_names.str.contains(r"^RP[SL]", regex=True)

        # Calculate mitochondrial and ribosomal percentages
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mito", "ribo"], inplace=True, log1p=False, percent_top=None
        )

        logger.info("QC metrics calculated successfully")
        return adata

    def filter_cells(self, adata: anndata.AnnData) -> Tuple[anndata.AnnData, pd.DataFrame]:
        """
        Filter cells based on QC thresholds.

        Removes cells that do not meet the following criteria:
        - n_genes >= min_genes
        - n_genes <= max_genes
        - total_counts >= min_umi
        - pct_counts_mito <= max_mito_pct
        - pct_counts_ribo <= max_ribo_pct

        Args:
            adata: Annotated data matrix with QC metrics in adata.obs.

        Returns:
            Tuple of:
                - Filtered AnnData object
                - DataFrame with filtering statistics

        Raises:
            ValueError: If required QC columns are missing.
        """
        required_cols = ["n_genes_by_counts", "total_counts", "pct_counts_mito", "pct_counts_ribo"]
        missing_cols = [col for col in required_cols if col not in adata.obs.columns]

        if missing_cols:
            raise ValueError(f"Missing required QC columns: {missing_cols}")

        n_cells_before = adata.n_obs

        # Create boolean mask for filtering
        cell_mask = (
            (adata.obs["n_genes_by_counts"] >= self.min_genes)
            & (adata.obs["n_genes_by_counts"] <= self.max_genes)
            & (adata.obs["total_counts"] >= self.min_umi)
            & (adata.obs["pct_counts_mito"] <= self.max_mito_pct)
            & (adata.obs["pct_counts_ribo"] <= self.max_ribo_pct)
        )

        # Apply filter
        adata = adata[cell_mask].copy()
        n_cells_after = adata.n_obs
        n_removed = n_cells_before - n_cells_after
        pct_removed = (n_removed / n_cells_before * 100) if n_cells_before > 0 else 0.0

        logger.info(
            f"Cell filtering: {n_cells_before} cells -> {n_cells_after} cells "
            f"({n_removed} removed, {pct_removed:.2f}%)"
        )

        # Create statistics DataFrame
        stats = pd.DataFrame(
            {
                "n_cells_before": [n_cells_before],
                "n_cells_after": [n_cells_after],
                "n_removed": [n_removed],
                "pct_removed": [pct_removed],
            }
        )

        return adata, stats

    def filter_genes(self, adata: anndata.AnnData) -> Tuple[anndata.AnnData, pd.DataFrame]:
        """
        Filter genes based on minimum cell count threshold.

        Removes genes expressed in fewer than min_cells cells.

        Args:
            adata: Annotated data matrix.

        Returns:
            Tuple of:
                - Filtered AnnData object
                - DataFrame with filtering statistics

        Raises:
            ValueError: If required columns are missing.
        """
        if "n_cells_by_counts" not in adata.var.columns:
            raise ValueError("Missing 'n_cells_by_counts' in adata.var")

        n_genes_before = adata.n_vars

        # Filter genes
        gene_mask = adata.var["n_cells_by_counts"] >= self.min_cells
        adata = adata[:, gene_mask].copy()

        n_genes_after = adata.n_vars
        n_removed = n_genes_before - n_genes_after
        pct_removed = (n_removed / n_genes_before * 100) if n_genes_before > 0 else 0.0

        logger.info(
            f"Gene filtering: {n_genes_before} genes -> {n_genes_after} genes "
            f"({n_removed} removed, {pct_removed:.2f}%)"
        )

        # Create statistics DataFrame
        stats = pd.DataFrame(
            {
                "n_genes_before": [n_genes_before],
                "n_genes_after": [n_genes_after],
                "n_removed": [n_removed],
                "pct_removed": [pct_removed],
            }
        )

        return adata, stats

    def run(self, adata: anndata.AnnData) -> Tuple[anndata.AnnData, Dict]:
        """
        Run full QC filtering pipeline.

        Performs the complete QC workflow:
        1. Calculate QC metrics
        2. Filter cells
        3. Filter genes
        4. Return filtered data and report

        Args:
            adata: Input annotated data matrix.

        Returns:
            Tuple of:
                - Filtered AnnData object
                - QC report dictionary with metrics and statistics

        Raises:
            ValueError: If input data is invalid or empty.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Cannot run QC on empty AnnData object")

        logger.info(f"Starting QC pipeline on data with {adata.n_obs} cells and {adata.n_vars} genes")

        # Step 1: Calculate metrics
        adata = self.calculate_qc_metrics(adata)

        # Step 2: Filter cells
        adata, cell_stats = self.filter_cells(adata)

        # Step 3: Filter genes
        adata, gene_stats = self.filter_genes(adata)

        # Create comprehensive report
        report = {
            "qc_metrics": {
                "n_cells_total": cell_stats["n_cells_before"].values[0],
                "n_cells_after_filtering": cell_stats["n_cells_after"].values[0],
                "n_genes_total": gene_stats["n_genes_before"].values[0],
                "n_genes_after_filtering": gene_stats["n_genes_after"].values[0],
                "pct_cells_removed": float(cell_stats["pct_removed"].values[0]),
                "pct_genes_removed": float(gene_stats["pct_removed"].values[0]),
                "mean_genes_per_cell": float(adata.obs["n_genes_by_counts"].mean()),
                "mean_umis_per_cell": float(adata.obs["total_counts"].mean()),
                "pct_mito": float(adata.obs["pct_counts_mito"].mean()),
                "pct_ribo": float(adata.obs["pct_counts_ribo"].mean()),
            },
            "cell_stats": cell_stats.to_dict(orient="records")[0],
            "gene_stats": gene_stats.to_dict(orient="records")[0],
        }

        logger.info(f"QC pipeline completed: {adata.n_obs} cells and {adata.n_vars} genes remaining")

        return adata, report
