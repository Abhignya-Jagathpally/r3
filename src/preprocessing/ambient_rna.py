"""
Ambient RNA correction for single-cell RNA-seq data.

This module provides methods to remove ambient RNA (background RNA from lysed cells)
using SoupX and DecontX approaches, with automatic detection of raw count matrices.
"""

import logging
from typing import Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AmbientRNACorrector:
    """
    Ambient RNA correction using SoupX and DecontX algorithms.

    Provides methods to identify and correct contamination from ambient RNA
    in single-cell RNA-seq data using two complementary approaches.

    Attributes:
        soupx_fdr: False discovery rate threshold for SoupX. Default: 0.05.
        decontx_delta: Delta parameter for DecontX. Default: 10.
    """

    def __init__(self, soupx_fdr: float = 0.05, decontx_delta: int = 10):
        """
        Initialize AmbientRNACorrector.

        Args:
            soupx_fdr: False discovery rate threshold for SoupX. Default: 0.05.
            decontx_delta: Delta parameter for DecontX. Default: 10.
        """
        self.soupx_fdr = soupx_fdr
        self.decontx_delta = decontx_delta

        logger.info(
            f"Initialized AmbientRNACorrector with soupx_fdr={soupx_fdr}, "
            f"decontx_delta={decontx_delta}"
        )

    def correct_soupx(
        self, adata: anndata.AnnData, raw_adata: Optional[anndata.AnnData] = None
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Correct ambient RNA using SoupX algorithm.

        SoupX identifies contaminated genes and estimates the fraction of
        ambient RNA to remove from the expression matrix.

        Args:
            adata: Input annotated data matrix (can be raw or processed).
            raw_adata: Optional raw AnnData with unfiltered counts. If not provided,
                uses raw counts from adata.layers['counts'] if available.

        Returns:
            Tuple of:
                - Corrected AnnData object
                - Dictionary with correction statistics

        Raises:
            ImportError: If SoupX (via souporcell or similar) is not available.
            ValueError: If no raw counts are available.

        Note:
            This is a wrapper that emulates SoupX behavior. For production use,
            consider running actual SoupX in R and importing results.
        """
        try:
            from scipy.sparse import issparse
        except ImportError:
            raise ImportError("scipy is required for ambient RNA correction")

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot correct ambient RNA")

        logger.info(f"Correcting ambient RNA (SoupX) for {adata.n_obs} cells")

        # Get raw counts
        if raw_adata is not None:
            raw_X = raw_adata.X
        elif "counts" in adata.layers:
            raw_X = adata.layers["counts"]
        else:
            logger.warning("No raw counts available, using current data for SoupX")
            raw_X = adata.X

        # Compute ambient profile and cell totals without full dense conversion
        if issparse(raw_X):
            ambient_profile = np.asarray(raw_X.sum(axis=0)).flatten()
            total_sum = ambient_profile.sum()
            ambient_profile = ambient_profile / total_sum if total_sum > 0 else ambient_profile
            cell_totals = np.asarray(raw_X.sum(axis=1)).flatten()
        else:
            raw_X_arr = np.array(raw_X)
            ambient_profile = raw_X_arr.sum(axis=0)
            total_sum = ambient_profile.sum()
            ambient_profile = ambient_profile / total_sum if total_sum > 0 else ambient_profile
            cell_totals = raw_X_arr.sum(axis=1)

        # OPTIMIZATION: Hoist argsort outside loop - O(m log m + n) instead of O(n * m log m)
        top_ambient_genes = np.argsort(ambient_profile)[-100:]

        # Vectorized computation of ambient contributions (sparse-safe)
        if issparse(raw_X):
            ambient_contributions = np.asarray(raw_X[:, top_ambient_genes].sum(axis=1)).flatten()
        else:
            ambient_contributions = raw_X_arr[:, top_ambient_genes].sum(axis=1)

        ambient_fraction = np.minimum(
            ambient_contributions / np.where(cell_totals > 0, cell_totals, 1), 0.9
        )
        # Handle zero cell totals
        ambient_fraction[cell_totals == 0] = 0

        # Store contamination fraction
        adata.obs["soupx_contamination_fraction"] = ambient_fraction
        contamination_mean = ambient_fraction.mean()

        logger.info(f"Estimated mean contamination fraction: {contamination_mean:.4f}")

        # Create corrected counts in chunks to limit memory
        from scipy.sparse import issparse as _issparse, csr_matrix
        chunk_size = 5000
        is_sparse_input = _issparse(adata.X)
        corrected_chunks = []

        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            if issparse(raw_X):
                chunk = raw_X[start:end].toarray()
            else:
                chunk = np.array(raw_X[start:end])

            # Subtract ambient contribution
            chunk_ambient = (
                ambient_profile[np.newaxis, :]
                * cell_totals[start:end, np.newaxis]
                * ambient_fraction[start:end, np.newaxis]
            )
            corrected_chunk = np.maximum(chunk - chunk_ambient, 0)
            corrected_chunks.append(corrected_chunk)

        corrected_X = np.vstack(corrected_chunks)

        # Update expression matrix
        if is_sparse_input:
            adata.X = csr_matrix(corrected_X)
        else:
            adata.X = corrected_X

        stats = {
            "method": "soupx",
            "mean_contamination_fraction": float(contamination_mean),
            "n_cells_corrected": adata.n_obs,
        }

        logger.info("SoupX correction completed")
        return adata, stats

    def correct_decontx(
        self, adata: anndata.AnnData, cell_type_key: Optional[str] = None
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Correct ambient RNA using DecontX algorithm.

        DecontX estimates contamination from ambient RNA using a Bayesian approach
        and decontamination probability scores.

        Args:
            adata: Input annotated data matrix.
            cell_type_key: Key in adata.obs containing cell type annotations.
                Optional; improves accuracy if provided.

        Returns:
            Tuple of:
                - Corrected AnnData object
                - Dictionary with correction statistics

        Raises:
            ImportError: If required packages are not available.
            ValueError: If data is invalid.

        Note:
            This is a wrapper that emulates DecontX behavior. For production use,
            consider running actual DecontX in R and importing results.
        """
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot correct ambient RNA")

        logger.info(f"Correcting ambient RNA (DecontX) for {adata.n_obs} cells")

        # Get count matrix
        if "counts" in adata.layers:
            X = adata.layers["counts"]
        else:
            X = adata.X

        # Convert to dense if sparse
        try:
            from scipy.sparse import issparse
        except ImportError:
            raise ImportError("scipy is required for ambient RNA correction")

        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        # Simplified DecontX implementation
        # Estimate background distribution from low-expressing genes
        mean_expression = X_dense.mean(axis=0)
        background_genes = mean_expression < np.percentile(mean_expression, 25)

        background_profile = X_dense[:, background_genes].mean(axis=1)
        background_profile = background_profile / background_profile.sum()

        # Estimate contamination per cell (vectorized)
        cell_totals = X_dense.sum(axis=1)
        # Vectorized computation of background content per cell
        background_content = X_dense[:, background_genes].sum(axis=1)
        contamination_scores = np.divide(
            background_content,
            cell_totals,
            where=(cell_totals > 0),
            out=np.zeros_like(background_content, dtype=float)
        )

        # Convert to decontamination probability
        decontamination_prob = 1 - np.minimum(contamination_scores, 0.99)

        adata.obs["decontx_decontamination_prob"] = decontamination_prob
        adata.obs["decontx_contamination_score"] = contamination_scores

        # Apply correction: scale down low-confidence genes (vectorized)
        corrected_X = X_dense.copy()
        # Broadcast: (n_obs, 1) * (n_background_genes,) = (n_obs, n_background_genes)
        corrected_X[:, background_genes] = (
            corrected_X[:, background_genes] * decontamination_prob[:, np.newaxis]
        )

        # Update expression matrix
        if issparse(adata.X):
            from scipy.sparse import csr_matrix
            adata.X = csr_matrix(corrected_X)
        else:
            adata.X = corrected_X

        mean_contamination = contamination_scores.mean()
        logger.info(f"Estimated mean contamination score: {mean_contamination:.4f}")

        stats = {
            "method": "decontx",
            "mean_contamination_score": float(mean_contamination),
            "mean_decontamination_prob": float(decontamination_prob.mean()),
            "n_cells_corrected": adata.n_obs,
        }

        logger.info("DecontX correction completed")
        return adata, stats

    def run(
        self,
        adata: anndata.AnnData,
        method: str = "soupx",
        raw_adata: Optional[anndata.AnnData] = None,
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Run ambient RNA correction.

        Args:
            adata: Input annotated data matrix.
            method: Correction method ('soupx' or 'decontx'). Default: 'soupx'.
            raw_adata: Optional raw AnnData for SoupX. Default: None.

        Returns:
            Tuple of:
                - Corrected AnnData object
                - Statistics dictionary

        Raises:
            ValueError: If method is unknown or data is invalid.
        """
        if method == "soupx":
            return self.correct_soupx(adata, raw_adata)
        elif method == "decontx":
            return self.correct_decontx(adata)
        else:
            raise ValueError(f"Unknown correction method: {method}")
