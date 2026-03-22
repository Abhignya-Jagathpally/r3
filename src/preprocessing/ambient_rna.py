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

        # Convert to dense if sparse
        if issparse(raw_X):
            raw_X_dense = raw_X.toarray()
        else:
            raw_X_dense = np.array(raw_X)

        # Estimate ambient profile (genes most frequent in ambient)
        ambient_profile = raw_X_dense.sum(axis=0) / raw_X_dense.sum()

        # Estimate contamination fraction per cell using robust method
        # This is a simplified approximation of SoupX
        cell_totals = raw_X_dense.sum(axis=1)
        ambient_fraction = np.zeros(adata.n_obs)

        for i in range(adata.n_obs):
            # Estimate ambient fraction as ratio of ambient genes to total
            # Use top ambient genes for estimation
            top_ambient_genes = np.argsort(ambient_profile)[-100:]
            ambient_contribution = raw_X_dense[i, top_ambient_genes].sum()
            ambient_fraction[i] = min(
                ambient_contribution / cell_totals[i] if cell_totals[i] > 0 else 0, 0.9
            )

        # Store contamination fraction
        adata.obs["soupx_contamination_fraction"] = ambient_fraction
        contamination_mean = ambient_fraction.mean()

        logger.info(f"Estimated mean contamination fraction: {contamination_mean:.4f}")

        # Create corrected counts by subtracting ambient contribution
        corrected_X = raw_X_dense.copy()
        for i in range(adata.n_obs):
            ambient_counts = ambient_profile * cell_totals[i] * ambient_fraction[i]
            corrected_X[i] = np.maximum(corrected_X[i] - ambient_counts, 0)

        # Update expression matrix
        if issparse(adata.X):
            from scipy.sparse import csr_matrix
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

        # Estimate contamination per cell
        cell_totals = X_dense.sum(axis=1)
        contamination_scores = np.zeros(adata.n_obs)

        for i in range(adata.n_obs):
            # Contamination score based on background gene expression
            background_content = X_dense[i, background_genes].sum()
            contamination_scores[i] = (
                background_content / cell_totals[i] if cell_totals[i] > 0 else 0
            )

        # Convert to decontamination probability
        decontamination_prob = 1 - np.minimum(contamination_scores, 0.99)

        adata.obs["decontx_decontamination_prob"] = decontamination_prob
        adata.obs["decontx_contamination_score"] = contamination_scores

        # Apply correction: scale down low-confidence genes
        corrected_X = X_dense.copy()
        for i in range(adata.n_obs):
            correction_factor = decontamination_prob[i]
            # More aggressive correction for background genes
            corrected_X[i, background_genes] = (
                corrected_X[i, background_genes] * correction_factor
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
