"""
Doublet detection and removal for single-cell RNA-seq data.

This module provides methods to identify and remove doublets (two cells captured
together) using both Scrublet and DoubletFinder approaches, with options for
consensus-based filtering.
"""

import logging
from typing import Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DoubletDetector:
    """
    Doublet detection using Scrublet and DoubletFinder approaches.

    Provides methods to identify doublets in single-cell RNA-seq data using
    two complementary algorithms: Scrublet (simulated doublets) and DoubletFinder
    (nearest neighbor-based).

    Attributes:
        scrublet_threshold: Score threshold for Scrublet doublet classification.
        doubletfinder_threshold: Score threshold for DoubletFinder classification.
    """

    def __init__(
        self, scrublet_threshold: float = 0.5, doubletfinder_threshold: float = 0.5
    ):
        """
        Initialize DoubletDetector.

        Args:
            scrublet_threshold: Classification threshold for Scrublet scores.
                Default: 0.5.
            doubletfinder_threshold: Classification threshold for DoubletFinder scores.
                Default: 0.5.
        """
        self.scrublet_threshold = scrublet_threshold
        self.doubletfinder_threshold = doubletfinder_threshold

        logger.info(
            f"Initialized DoubletDetector with scrublet_threshold={scrublet_threshold}, "
            f"doubletfinder_threshold={doubletfinder_threshold}"
        )

    def detect_scrublet(
        self, adata: anndata.AnnData, expected_doublet_rate: float = 0.06
    ) -> anndata.AnnData:
        """
        Detect doublets using Scrublet algorithm.

        Uses simulated doublets to identify true doublets in the data.

        Args:
            adata: Input annotated data matrix.
            expected_doublet_rate: Expected doublet rate (default: 0.06 for 6%).

        Returns:
            AnnData object with Scrublet scores in adata.obs['scrublet_score']
            and doublet predictions in adata.obs['scrublet_doublet'].

        Raises:
            ValueError: If scrublet is not installed or data is invalid.
        """
        try:
            import scrublet as scr
        except ImportError:
            logger.warning(
                "Scrublet not installed. Install with: pip install scrublet"
            )
            raise ImportError("Scrublet is required for Scrublet doublet detection")

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot detect doublets")

        logger.info(f"Running Scrublet on {adata.n_obs} cells with expected rate {expected_doublet_rate}")

        # Initialize Scrublet
        scrub = scr.Scrublet(
            adata.X, expected_doublet_rate=expected_doublet_rate, random_state=42
        )

        # Detect doublets
        doublet_scores, predicted_doublets = scrub.predict_doublets(min_counts=2, min_cells=2)

        # Store results
        adata.obs["scrublet_score"] = doublet_scores
        adata.obs["scrublet_doublet"] = predicted_doublets

        n_doublets = predicted_doublets.sum()
        logger.info(
            f"Scrublet detected {n_doublets} doublets "
            f"({n_doublets/adata.n_obs*100:.2f}% of cells)"
        )

        return adata

    def detect_doubletfinder(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Detect doublets using DoubletFinder algorithm.

        Uses k-nearest neighbor approach to identify doublets based on
        transcriptional similarity to simulated doublets.

        Args:
            adata: Input annotated data matrix with PCA or UMAP reduction.

        Returns:
            AnnData object with DoubletFinder scores and predictions.
                - adata.obs['doubletfinder_score']
                - adata.obs['doubletfinder_doublet']

        Raises:
            ImportError: If DoubletFinder-py is not installed.
            ValueError: If required reductions are missing.

        Note:
            Requires PCA to be computed. Will compute if not present.
        """
        try:
            import doubletfinder_py as dbf
        except ImportError:
            logger.warning(
                "DoubletFinder-py not installed. Install with: pip install doubletfinder"
            )
            raise ImportError("DoubletFinder is required for DoubletFinder doublet detection")

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Empty AnnData object: cannot detect doublets")

        logger.info(f"Running DoubletFinder on {adata.n_obs} cells")

        # Compute PCA if not present
        if "X_pca" not in adata.obsm:
            logger.info("PCA not found, computing PCA")
            try:
                import scanpy as sc
                sc.pp.pca(adata, n_comps=50)
            except (ValueError, KeyError, AttributeError) as e:
                logger.error(f"Failed to compute PCA: {e}")
                raise ValueError("Cannot compute PCA for DoubletFinder")

        try:
            # Run DoubletFinder
            dbf.doubletfinder_py(
                adata,
                pca="X_pca",
                n_neighbors=15,
                n_top_genes=2000,
                expected_doublet_rate=0.06,
                random_state=42,
            )

            # Find the column name added by DoubletFinder (usually 'DF_classifications')
            df_cols = [col for col in adata.obs.columns if "DF_" in col or "doublet" in col]

            if df_cols:
                # Store the main result
                df_col = df_cols[0]
                adata.obs["doubletfinder_doublet"] = adata.obs[df_col] == "Doublet"

                # If there's a score column, extract it
                score_cols = [col for col in adata.obs.columns if "score" in col.lower()]
                if score_cols:
                    adata.obs["doubletfinder_score"] = adata.obs[score_cols[0]]
                else:
                    # Use the classification as proxy for score
                    adata.obs["doubletfinder_score"] = (
                        adata.obs[df_col] == "Doublet"
                    ).astype(float)

            n_doublets = adata.obs["doubletfinder_doublet"].sum()
            logger.info(
                f"DoubletFinder detected {n_doublets} doublets "
                f"({n_doublets/adata.n_obs*100:.2f}% of cells)"
            )

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"DoubletFinder failed: {e}")
            raise ValueError(f"DoubletFinder detection failed: {e}")

        return adata

    def consensus_filter(
        self, adata: anndata.AnnData, require_both: bool = False
    ) -> Tuple[anndata.AnnData, Dict]:
        """
        Apply consensus doublet filtering.

        Removes cells flagged as doublets by one or both methods.

        Args:
            adata: Input annotated data matrix with doublet predictions.
            require_both: If True, only remove cells flagged by both methods.
                If False, remove cells flagged by either method. Default: False.

        Returns:
            Tuple of:
                - Filtered AnnData object
                - Dictionary with filtering statistics

        Raises:
            ValueError: If required doublet columns are missing.
        """
        has_scrublet = "scrublet_doublet" in adata.obs.columns
        has_doubletfinder = "doubletfinder_doublet" in adata.obs.columns

        if not has_scrublet and not has_doubletfinder:
            raise ValueError(
                "No doublet predictions found. Run detect_scrublet() or "
                "detect_doubletfinder() first."
            )

        n_cells_before = adata.n_obs

        # Create consensus mask
        if has_scrublet and has_doubletfinder:
            if require_both:
                # Remove only if flagged by BOTH methods
                doublet_mask = (
                    adata.obs["scrublet_doublet"].values
                    & adata.obs["doubletfinder_doublet"].values
                )
                method = "both"
            else:
                # Remove if flagged by EITHER method
                doublet_mask = (
                    adata.obs["scrublet_doublet"].values
                    | adata.obs["doubletfinder_doublet"].values
                )
                method = "either"
        elif has_scrublet:
            doublet_mask = adata.obs["scrublet_doublet"].values
            method = "scrublet_only"
        else:
            doublet_mask = adata.obs["doubletfinder_doublet"].values
            method = "doubletfinder_only"

        # Filter out doublets
        adata = adata[~doublet_mask].copy()
        n_cells_after = adata.n_obs
        n_removed = n_cells_before - n_cells_after
        pct_removed = (n_removed / n_cells_before * 100) if n_cells_before > 0 else 0.0

        logger.info(
            f"Consensus doublet filtering ({method}): {n_cells_before} cells -> {n_cells_after} cells "
            f"({n_removed} removed, {pct_removed:.2f}%)"
        )

        # Create statistics
        stats = {
            "n_cells_before": n_cells_before,
            "n_cells_after": n_cells_after,
            "n_doublets_removed": n_removed,
            "pct_doublets_removed": pct_removed,
            "method": method,
            "has_scrublet": has_scrublet,
            "has_doubletfinder": has_doubletfinder,
        }

        return adata, stats
