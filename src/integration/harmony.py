"""
Harmony integration for batch effect correction in the R3-MM pipeline.

This module implements the HarmonyIntegrator class which uses the Harmony algorithm
to correct batch effects in PCA embeddings while preserving biological variation.

References:
    Korsunsky et al. (2019). Fast, sensitive and accurate integration of single-cell
    data with Harmony. Nature Methods, 16(12), 1289-1296.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


class HarmonyIntegrator:
    """
    Harmony-based batch effect correction for scRNA-seq data.

    This integrator applies the Harmony algorithm to correct batch effects in PCA
    embeddings while preserving biological variation. It provides a complete pipeline
    from PCA to UMAP visualization.

    Attributes:
        None (stateless class)

    Example:
        >>> from src.integration import HarmonyIntegrator
        >>> integrator = HarmonyIntegrator()
        >>> adata = integrator.integrate(adata, batch_key='batch')
    """

    def __init__(self):
        """Initialize HarmonyIntegrator."""
        self.logger = logger

    def compute_pca(
        self,
        adata: ad.AnnData,
        n_comps: int = 50,
        use_rep: Optional[str] = None,
    ) -> ad.AnnData:
        """
        Compute PCA embeddings from preprocessed data.

        Args:
            adata: Annotated data matrix. Expected to have been preprocessed
                (log-normalized, HVG selected).
            n_comps: Number of principal components to compute. Default: 50.
            use_rep: Key in adata.obsm to use as input. If None, uses adata.X.
                Default: None.

        Returns:
            Modified adata with PCA embeddings in adata.obsm['X_pca'].

        Raises:
            ValueError: If n_comps < 1 or n_comps > min(n_obs, n_vars).
        """
        if n_comps < 1:
            raise ValueError(f"n_comps must be >= 1, got {n_comps}")
        if n_comps > min(adata.n_obs, adata.n_vars):
            raise ValueError(
                f"n_comps ({n_comps}) cannot exceed min(n_obs, n_vars) "
                f"({min(adata.n_obs, adata.n_vars)})"
            )

        self.logger.info(f"Computing PCA with {n_comps} components...")
        sc.pp.pca(adata, n_comps=n_comps)
        self.logger.info(
            f"PCA computed. Explained variance ratio: "
            f"{adata.uns['pca']['variance_ratio'][:5]}"
        )

        return adata

    def compute_neighbors(
        self,
        adata: ad.AnnData,
        use_rep: str = "X_pca_harmony",
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
    ) -> ad.AnnData:
        """
        Compute neighborhood graph from embeddings.

        Args:
            adata: Annotated data matrix with computed Harmony embeddings.
            use_rep: Key in adata.obsm to compute neighbors from. Default: 'X_pca_harmony'.
            n_neighbors: Number of neighbors per cell. Default: 15.
            n_pcs: Number of PCs to use if use_rep is a PC representation.
                Default: None (use all).

        Returns:
            Modified adata with neighborhood graph in adata.obsp and
            adata.obsp['distances'].

        Raises:
            ValueError: If use_rep not found in adata.obsm.
            ValueError: If n_neighbors < 1.
        """
        if use_rep not in adata.obsm:
            raise ValueError(
                f"use_rep '{use_rep}' not found in adata.obsm. "
                f"Available: {list(adata.obsm.keys())}"
            )
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.logger.info(
            f"Computing neighbors using {use_rep} with k={n_neighbors}..."
        )
        sc.pp.neighbors(
            adata, use_rep=use_rep, n_neighbors=n_neighbors, n_pcs=n_pcs
        )
        self.logger.info(f"Neighbor graph computed.")

        return adata

    def compute_umap(
        self,
        adata: ad.AnnData,
        min_dist: float = 0.1,
        spread: float = 1.0,
    ) -> ad.AnnData:
        """
        Compute UMAP dimensionality reduction.

        Args:
            adata: Annotated data matrix with computed neighborhood graph.
            min_dist: Minimum distance between embedded points. Default: 0.1.
            spread: Effective scale of embedded points. Default: 1.0.

        Returns:
            Modified adata with UMAP embeddings in adata.obsm['X_umap'].

        Raises:
            ValueError: If neighborhood graph not computed.
        """
        if "neighbors" not in adata.obsp:
            raise ValueError(
                "Neighborhood graph not found. Run compute_neighbors() first."
            )

        self.logger.info("Computing UMAP...")
        sc.tl.umap(adata, min_dist=min_dist, spread=spread)
        self.logger.info("UMAP computed.")

        return adata

    def integrate(
        self,
        adata: ad.AnnData,
        batch_key: str = "dataset",
        n_comps: int = 50,
        n_neighbors: int = 15,
        **kwargs,
    ) -> ad.AnnData:
        """
        Complete Harmony integration pipeline.

        Performs PCA -> Harmony correction -> neighbor graph -> UMAP.

        Args:
            adata: Annotated data matrix. Should be preprocessed (log-normalized,
                HVG selected).
            batch_key: Observation key containing batch labels. Default: 'batch'.
            n_comps: Number of PCA components. Default: 50.
            n_neighbors: Number of neighbors for graph. Default: 15.

        Returns:
            Modified adata with:
                - adata.obsm['X_pca']: PCA embeddings
                - adata.obsm['X_pca_harmony']: Harmony-corrected PCA embeddings
                - adata.obsm['X_umap']: UMAP embeddings
                - adata.obsp['connectivities']: Neighbor graph
                - adata.obsp['distances']: Neighbor distances

        Raises:
            ValueError: If batch_key not in adata.obs.
            ValueError: If preprocessing not done (e.g., HVG selection).
        """
        if batch_key not in adata.obs:
            raise ValueError(
                f"batch_key '{batch_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        if adata.n_vars == adata.var.index.size:
            self.logger.warning(
                "All genes present. Ensure HVG selection done if intended."
            )

        # Extract kwargs
        theta = kwargs.get("theta", 2.0)
        lambda_val = kwargs.get("lambda", 1.0)
        sigma = kwargs.get("sigma", 0.1)
        npcs = kwargs.get("npcs", n_comps)
        max_iter = kwargs.get("max_iter", 10)

        # Step 1: Compute PCA if not already present
        if "X_pca" not in adata.obsm:
            self.compute_pca(adata, n_comps=npcs)
        else:
            self.logger.info("PCA already computed, skipping")

        # Step 2: Apply Harmony correction
        self.logger.info(f"Running Harmony correction on batch '{batch_key}'...")
        try:
            import harmonypy
            ho = harmonypy.run_harmony(
                adata.obsm["X_pca"],
                adata.obs,
                batch_key,
                theta=theta,
                lamb=lambda_val,
                sigma=sigma,
                max_iter_harmony=max_iter,
            )
            harmony_embeddings = ho.Z_corr
        except ImportError:
            try:
                import harmony
                ho = harmony.Harmony(
                    adata.obsm["X_pca"],
                    adata.obs[[batch_key]].values.ravel(),
                    theta=[theta],
                    lambda_=[lambda_val],
                    sigma=sigma,
                    nclust=30,
                    max_iter_harmony=max_iter,
                    max_iter_clustering=30,
                    verbose=True,
                )
                harmony_embeddings = ho.Z_corr.T
            except ImportError:
                raise ImportError(
                    "Neither harmonypy nor harmony-pytorch installed. "
                    "Install with: pip install harmonypy"
                )

        adata.obsm["X_harmony"] = harmony_embeddings
        adata.uns["harmony_params"] = {
            "batch_key": batch_key,
            "theta": theta,
            "lambda": lambda_val,
            "sigma": sigma,
            "max_iter": max_iter,
        }
        self.logger.info("Harmony correction completed.")

        return adata
