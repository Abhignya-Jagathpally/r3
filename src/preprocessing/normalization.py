"""
Normalization and highly variable gene selection for single-cell RNA-seq data.

This module provides multiple normalization strategies including scanpy-based
log-normalization, SCRAN-based size factor normalization, and analytic Pearson
residuals. Also includes HVG selection using various methods.
"""

import logging
from typing import Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Normalization for single-cell RNA-seq data.

    Provides multiple normalization strategies:
    - scanpy_normalize: Standard log-normalization
    - scran_normalize: Size factor-based normalization
    - pearson_residuals: Analytic Pearson residuals

    Attributes:
        Normalization parameters are passed to individual methods.
    """

    def __init__(self):
        """Initialize Normalizer."""
        logger.info("Initialized Normalizer")

    def scran_normalize(
        self, adata: anndata.AnnData, chunks: int = 5000
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Normalize using SCRAN (size factor) approach.

        Calculates size factors using a pooling approach that is more robust
        for sparse data than simple library size normalization.

        Args:
            adata: Input annotated data matrix.
            chunks: Number of chunks for computing pooling statistics. Default: 5000.

        Returns:
            Tuple of:
                - Normalized AnnData object with log-transformed counts
                - Dictionary with normalization statistics

        Raises:
            ImportError: If scran (via python-scran) is not available.
            ValueError: If data is invalid.

        Note:
            Stores raw counts in adata.layers['counts'] before transformation.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot normalize")

        logger.info(f"Applying SCRAN normalization to {adata.n_obs} cells and {adata.n_vars} genes")

        # Store raw counts
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
            logger.info("Stored raw counts in adata.layers['counts']")

        try:
            from scipy.sparse import issparse, csr_matrix
            import scanpy as sc
        except ImportError:
            raise ImportError("scipy and scanpy are required for SCRAN normalization")

        # Convert to dense if sparse for easier processing
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)

        # Compute library sizes
        lib_sizes = X.sum(axis=1)

        # Compute size factors using simple approach
        # For each cell, use geometric mean of gene expression for cells with similar library size
        size_factors = np.zeros(adata.n_obs)

        for i in range(adata.n_obs):
            # Find cells with similar library size (within 2-fold)
            similar_mask = (lib_sizes > lib_sizes[i] / 2) & (lib_sizes < lib_sizes[i] * 2)
            similar_indices = np.where(similar_mask)[0]

            if len(similar_indices) > 0:
                # Compute geometric mean of gene expression in similar cells
                similar_X = X[similar_indices, :]
                # Use median expression for robustness
                median_expr = np.median(similar_X, axis=0)
                # Size factor as ratio to median
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = X[i, :] / (median_expr + 1)
                    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                    if len(ratios) > 0:
                        size_factors[i] = np.exp(np.median(np.log(ratios)))

        # Set minimum size factor to avoid division by zero
        size_factors = np.maximum(size_factors, 0.5)

        # Normalize by size factors and log-transform
        normalized_X = (X.T / size_factors).T
        adata.X = np.log1p(normalized_X)

        stats = {
            "method": "scran",
            "mean_size_factor": float(np.mean(size_factors)),
            "median_size_factor": float(np.median(size_factors)),
            "library_size_before": float(lib_sizes.mean()),
            "library_size_after": float(adata.X.sum(axis=1).mean()),
        }

        logger.info(f"SCRAN normalization completed. Mean size factor: {stats['mean_size_factor']:.4f}")
        return adata, stats

    def scanpy_normalize(
        self, adata: anndata.AnnData, target_sum: float = 1e4
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Normalize using scanpy's standard log-normalization.

        Applies library size normalization followed by log-transformation.

        Args:
            adata: Input annotated data matrix.
            target_sum: Target sum for normalization. Default: 1e4.

        Returns:
            Tuple of:
                - Normalized AnnData object
                - Dictionary with normalization statistics

        Raises:
            ValueError: If data is invalid.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot normalize")

        logger.info(
            f"Applying scanpy log-normalization to {adata.n_obs} cells and {adata.n_vars} genes "
            f"(target_sum={target_sum})"
        )

        # Store raw counts
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
            logger.info("Stored raw counts in adata.layers['counts']")

        # Calculate library size before normalization
        lib_size_before = adata.X.sum(axis=1).mean()

        # Ensure float dtype for normalization
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)

        # Apply normalization
        sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
        sc.pp.log1p(adata, base=np.e)

        lib_size_after = adata.X.sum(axis=1).mean()

        stats = {
            "method": "scanpy_log_normalize",
            "target_sum": float(target_sum),
            "library_size_before": float(lib_size_before),
            "library_size_after": float(lib_size_after),
        }

        logger.info("Scanpy log-normalization completed")
        return adata, stats

    def pearson_residuals(self, adata: anndata.AnnData) -> Tuple[anndata.AnnData, Dict]:
        """
        Normalize using analytic Pearson residuals.

        Implements the Pearson residuals approach from Lause et al. (2021),
        which stabilizes variance and performs variance-preserving transformation.

        Args:
            adata: Input annotated data matrix.

        Returns:
            Tuple of:
                - Normalized AnnData object
                - Dictionary with normalization statistics

        Raises:
            ValueError: If data is invalid.

        Reference:
            Lause et al. (2021). Analytic Pearson residuals for normalization
            of single-cell RNA-seq UMI data. bioRxiv.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot compute Pearson residuals")

        logger.info(
            f"Computing Pearson residuals for {adata.n_obs} cells and {adata.n_vars} genes"
        )

        # Store raw counts
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
            logger.info("Stored raw counts in adata.layers['counts']")

        try:
            from scipy.sparse import issparse
        except ImportError:
            raise ImportError("scipy is required for Pearson residuals computation")

        # Convert to dense if sparse
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)

        # Compute Pearson residuals
        # 1. Compute size factors (library size)
        size_factors = X.sum(axis=1)
        size_factors = size_factors / size_factors.mean()

        # 2. Compute gene mean expression
        gene_means = X.mean(axis=0)

        # 3. Compute expected values (Poisson with overdispersion)
        # Using negative binomial approximation
        expected = np.outer(size_factors, gene_means)

        # 4. Compute variance (with overdispersion)
        # Var = mu + mu^2 / theta, where theta is NB dispersion parameter
        # Estimate gene-wise dispersion via method of moments
        gene_vars = np.asarray(X.var(axis=0)).flatten()
        gene_means_flat = np.asarray(gene_means).flatten()
        theta = gene_means_flat**2 / np.maximum(gene_vars - gene_means_flat, 1e-6)
        theta = np.clip(theta, 1e-6, 1e6)
        variance = expected + (expected ** 2) / theta[np.newaxis, :]
        variance = np.maximum(variance, 0.1)  # Prevent division by zero

        # 5. Compute residuals
        residuals = (X - expected) / np.sqrt(variance)

        # 6. Store as normalized matrix
        adata.X = residuals

        stats = {
            "method": "pearson_residuals",
            "mean_gene_expression": float(gene_means.mean()),
            "mean_size_factor": float(size_factors.mean()),
            "residuals_mean": float(residuals.mean()),
            "residuals_std": float(residuals.std()),
        }

        logger.info("Pearson residuals computation completed")
        return adata, stats

    def select_hvgs(
        self,
        adata: anndata.AnnData,
        n_top_genes: int = 2000,
        flavor: str = "seurat_v3",
        **kwargs
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Select highly variable genes (HVGs).

        Args:
            adata: Input annotated data matrix.
            n_top_genes: Number of top genes to select. Default: 2000.
            flavor: Selection method ('seurat_v3', 'seurat', 'cell_ranger').
                Default: 'seurat_v3'.
            **kwargs: Additional arguments passed to sc.pp.highly_variable_genes.

        Returns:
            Tuple of:
                - AnnData with HVG selection in adata.var['highly_variable']
                - Dictionary with selection statistics

        Raises:
            ValueError: If data or parameters are invalid.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot select HVGs")

        if n_top_genes > adata.n_vars:
            logger.warning(
                f"n_top_genes ({n_top_genes}) exceeds number of genes ({adata.n_vars}). "
                f"Using all genes."
            )
            n_top_genes = adata.n_vars

        logger.info(
            f"Selecting {n_top_genes} HVGs using {flavor} method from {adata.n_vars} genes"
        )

        # Select HVGs
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor=flavor, inplace=True, **kwargs
        )

        n_hvgs = adata.var["highly_variable"].sum()
        logger.info(f"Selected {n_hvgs} highly variable genes")

        stats = {
            "n_hvgs_selected": int(n_hvgs),
            "n_genes_total": adata.n_vars,
            "pct_hvgs": float(n_hvgs / adata.n_vars * 100),
            "flavor": flavor,
        }

        return adata, stats
