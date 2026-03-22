"""
scVI integration for batch effect correction in the R3-MM pipeline.

This module implements the ScVIIntegrator class which uses scVI-tools to learn
a latent representation that corrects batch effects and enables integration
of multiple batches.

References:
    Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics.
    Nature Methods, 15(12), 1053-1058.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scvi

logger = logging.getLogger(__name__)


class ScVIIntegrator:
    """
    scVI-based batch effect correction for scRNA-seq data.

    This integrator uses a variational autoencoder (VAE) framework to learn
    a latent representation that corrects for batch effects while preserving
    biological variation.

    Attributes:
        None (methods are primarily wrappers around scvi-tools)

    Example:
        >>> from src.integration import ScVIIntegrator
        >>> integrator = ScVIIntegrator()
        >>> adata = integrator.integrate(adata, batch_key='batch')
    """

    def __init__(
        self,
        n_latent: int = 32,
        dispersion: str = "gene-batch",
        gene_likelihood: str = "nb",
    ):
        """Initialize ScVIIntegrator."""
        self.logger = logger
        self.n_latent = n_latent
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood
        self._model = None
        self._adata = None

    def setup(self, adata: ad.AnnData, batch_key: str = "dataset", layer: Optional[str] = None):
        """Setup AnnData for scVI training."""
        self._adata = adata
        use_layer = layer
        if use_layer is None:
            # Try common layer names
            for candidate in ["raw_counts", "counts"]:
                if candidate in adata.layers:
                    use_layer = candidate
                    break
        if use_layer and use_layer in adata.layers:
            scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer=use_layer)
        else:
            scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
        self._model = scvi.model.SCVI(
            adata, n_latent=self.n_latent,
            dispersion=self.dispersion,
            gene_likelihood=self.gene_likelihood,
        )
        self.logger.info(f"scVI model setup: n_latent={self.n_latent}, batch_key={batch_key}")

    def train(self, max_epochs: int = 100, early_stopping: bool = True):
        """Train the scVI model."""
        self._model.train(
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            plan_kwargs={"reduce_lr_on_plateau": True},
        )
        self.logger.info("scVI training complete")

    def save(self, path: str):
        """Save the trained scVI model."""
        self._model.save(path, overwrite=True)
        self.logger.info(f"scVI model saved to {path}")

    @classmethod
    def load(cls, path: str, adata: ad.AnnData) -> "ScVIIntegrator":
        """Load a saved scVI model."""
        obj = cls()
        obj._model = scvi.model.SCVI.load(path, adata=adata)
        return obj

    def setup_anndata(
        self,
        adata: ad.AnnData,
        batch_key: str,
        layer: str = "counts",
        categorical_covariate_keys: Optional[list] = None,
        continuous_covariate_keys: Optional[list] = None,
    ) -> ad.AnnData:
        """
        Set up AnnData object for scVI.

        This registers layers, batch keys, and covariates with scVI's registry.
        Must be called before model training.

        Args:
            adata: Annotated data matrix. Should have raw counts in specified layer.
            batch_key: Observation key containing batch labels.
            layer: Layer name containing count data. Default: 'counts'.
            categorical_covariate_keys: List of categorical covariate keys in obs.
                Default: None.
            continuous_covariate_keys: List of continuous covariate keys in obs.
                Default: None.

        Returns:
            Modified adata with scVI setup registered.

        Raises:
            ValueError: If layer not in adata.layers.
            ValueError: If batch_key not in adata.obs.
        """
        if layer not in adata.layers:
            raise ValueError(
                f"layer '{layer}' not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        if batch_key not in adata.obs:
            raise ValueError(
                f"batch_key '{batch_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        self.logger.info(
            f"Setting up AnnData for scVI with layer='{layer}', "
            f"batch_key='{batch_key}'..."
        )

        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            layer=layer,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )

        self.logger.info("scVI setup complete.")
        return adata

    def train_scvi(
        self,
        adata: ad.AnnData,
        n_latent: int = 30,
        n_epochs: int = 100,
        max_epochs: Optional[int] = None,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        use_gpu: bool = True,
    ) -> scvi.model.SCVI:
        """
        Train scVI model on AnnData.

        Args:
            adata: Setup AnnData object (must call setup_anndata first).
            n_latent: Dimension of latent space. Default: 30.
            n_epochs: Number of epochs to train. Default: 100.
            max_epochs: Maximum epochs (deprecated parameter). Default: None.
            batch_size: Training batch size. Default: 128.
            learning_rate: Learning rate for optimizer. Default: 1e-3.
            use_gpu: Whether to use GPU if available. Default: True.

        Returns:
            Trained SCVI model.

        Raises:
            RuntimeError: If AnnData not properly setup.
            ValueError: If n_latent < 1 or n_epochs < 1.
        """
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {n_latent}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")

        self.logger.info(
            f"Initializing scVI model with n_latent={n_latent}..."
        )
        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
            use_observed_lib_size=False,
        )

        self.logger.info(
            f"Training scVI for {n_epochs} epochs with lr={learning_rate}..."
        )
        model.train(
            max_epochs=max_epochs or n_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            use_gpu=use_gpu,
            plan_kwargs={"reduce_lr_on_plateau": True},
        )

        self.logger.info("scVI training complete.")
        return model

    def get_latent(
        self,
        adata: ad.AnnData,
        model: scvi.model.SCVI,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract latent representations from trained model.

        Args:
            adata: AnnData object used to train model.
            model: Trained scVI model.
            indices: Optional indices to extract subset of cells. If None,
                extracts all cells. Default: None.

        Returns:
            Latent embeddings of shape (n_cells, n_latent).

        Raises:
            ValueError: If indices out of bounds.
        """
        self.logger.info("Extracting latent representations...")
        latent = model.get_latent_representation(
            adata=adata,
            indices=indices,
        )
        self.logger.info(f"Extracted latent with shape {latent.shape}")
        return latent

    def integrate(
        self,
        adata: ad.AnnData,
        batch_key: str = "dataset",
        layer: str = "counts",
        n_latent: int = 32,
        n_epochs: int = 100,
        batch_size: int = 128,
    ) -> ad.AnnData:
        """
        Complete scVI integration pipeline.

        Performs setup -> train -> extract latent representation.

        Args:
            adata: Annotated data matrix with raw counts in specified layer.
            batch_key: Observation key containing batch labels. Default: 'batch'.
            layer: Layer with count data. Default: 'counts'.
            n_latent: Latent dimension. Default: 30.
            n_epochs: Training epochs. Default: 100.
            batch_size: Batch size for training. Default: 128.

        Returns:
            Modified adata with:
                - adata.obsm['X_scVI']: scVI latent embeddings
                - scVI model metadata in adata.uns['scvi']

        Raises:
            ValueError: If layer or batch_key not found.
        """
        # If setup/train already called (separate API), use stored model
        if self._model is not None:
            latent = self._model.get_latent_representation(adata=adata)
            adata.obsm["X_scVI"] = latent
            adata.uns["scvi"] = {
                "n_latent": self.n_latent,
                "batch_key": batch_key,
            }
            self.logger.info(f"scVI integration complete. Latent shape: {latent.shape}")
            return adata

        # All-in-one fallback
        if layer not in adata.layers:
            self.logger.warning(f"Layer '{layer}' not found, using X directly")
            scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
        else:
            scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer=layer)

        model = scvi.model.SCVI(adata, n_latent=n_latent)
        model.train(max_epochs=n_epochs, batch_size=batch_size)

        latent = model.get_latent_representation(adata=adata)
        adata.obsm["X_scVI"] = latent
        self._model = model

        adata.uns["scvi"] = {
            "n_latent": n_latent,
            "batch_key": batch_key,
        }

        self.logger.info(
            f"scVI integration complete. Latent shape: {latent.shape}"
        )

        return adata
