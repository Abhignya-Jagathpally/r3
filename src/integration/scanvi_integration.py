"""
scANVI integration with semi-supervised cell type annotation in the R3-MM pipeline.

This module implements the ScANVIIntegrator class which extends scVI with semi-supervised
learning to leverage labeled and unlabeled data for joint integration and cell type annotation.

References:
    Xu et al. (2021). Probabilistic harmonization and annotation of single-cell
    transcriptomics data with deep generative models. Nature Methods, 18(1), 76-82.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scvi

logger = logging.getLogger(__name__)


class ScANVIIntegrator:
    """
    scANVI-based semi-supervised integration and annotation for scRNA-seq data.

    Extends scVI with cell type labels to perform joint batch correction
    and cell type prediction using semi-supervised learning.

    Attributes:
        None (methods are primarily wrappers around scvi-tools)

    Example:
        >>> from src.integration import ScANVIIntegrator
        >>> integrator = ScANVIIntegrator()
        >>> adata = integrator.integrate(adata, batch_key='batch',
        ...                               labels_key='cell_type')
    """

    def __init__(self):
        """Initialize ScANVIIntegrator."""
        self.logger = logger

    def setup_anndata(
        self,
        adata: ad.AnnData,
        batch_key: str,
        labels_key: str,
        layer: str = "counts",
        categorical_covariate_keys: Optional[list] = None,
        continuous_covariate_keys: Optional[list] = None,
    ) -> ad.AnnData:
        """
        Set up AnnData object for scANVI.

        Registers layers, batch keys, labels, and covariates with scVI.

        Args:
            adata: Annotated data matrix with raw counts.
            batch_key: Observation key containing batch labels.
            labels_key: Observation key containing cell type labels.
            layer: Layer with count data. Default: 'counts'.
            categorical_covariate_keys: Additional categorical covariates. Default: None.
            continuous_covariate_keys: Additional continuous covariates. Default: None.

        Returns:
            Modified adata with scVI setup registered.

        Raises:
            ValueError: If layer, batch_key, or labels_key not found.
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
        if labels_key not in adata.obs:
            raise ValueError(
                f"labels_key '{labels_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        self.logger.info(
            f"Setting up AnnData for scANVI with layer='{layer}', "
            f"batch_key='{batch_key}', labels_key='{labels_key}'..."
        )

        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )

        self.logger.info("scANVI setup complete.")
        return adata

    def train_scanvi(
        self,
        adata: ad.AnnData,
        vae_model: Optional[scvi.model.SCVI] = None,
        labels_key: str = "cell_type",
        n_latent: int = 30,
        n_epochs: int = 50,
        unlabeled_category: str = "Unknown",
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        use_gpu: bool = True,
    ) -> scvi.model.SCANVI:
        """
        Train scANVI model for semi-supervised annotation.

        Can either initialize from scratch or from a pre-trained scVI model.

        Args:
            adata: Setup AnnData object.
            vae_model: Pre-trained scVI model to initialize from. If None,
                trains scVI first. Default: None.
            labels_key: Key in adata.obs with cell type labels. Default: 'cell_type'.
            n_latent: Latent dimension. Default: 30.
            n_epochs: Number of epochs to train. Default: 50.
            unlabeled_category: Category label for unlabeled cells. Default: 'Unknown'.
            batch_size: Training batch size. Default: 128.
            learning_rate: Optimizer learning rate. Default: 1e-3.
            use_gpu: Whether to use GPU if available. Default: True.

        Returns:
            Trained scANVI model.

        Raises:
            ValueError: If n_latent < 1 or n_epochs < 1.
        """
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {n_latent}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")

        self.logger.info(
            f"Initializing scANVI model with n_latent={n_latent}..."
        )

        if vae_model is None:
            self.logger.info(
                "Pre-trained scVI model not provided. "
                "Training scVI first (100 epochs)..."
            )
            vae_model = scvi.model.SCVI(
                adata,
                n_latent=n_latent,
                use_observed_lib_size=False,
            )
            vae_model.train(
                max_epochs=100,
                batch_size=batch_size,
                lr=learning_rate,
                use_gpu=use_gpu,
                plan_kwargs={"reduce_lr_on_plateau": True},
            )

        model = scvi.model.SCANVI.from_scvi_model(
            vae_model,
            unlabeled_category=unlabeled_category,
        )

        self.logger.info(
            f"Training scANVI for {n_epochs} epochs with lr={learning_rate}..."
        )
        model.train(
            max_epochs=n_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            use_gpu=use_gpu,
            plan_kwargs={"reduce_lr_on_plateau": True},
        )

        self.logger.info("scANVI training complete.")
        return model

    def predict_labels(
        self,
        adata: ad.AnnData,
        model: scvi.model.SCANVI,
        indices: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Predict cell type labels using trained scANVI.

        Args:
            adata: AnnData used to train model.
            model: Trained scANVI model.
            indices: Optional cell indices to predict. If None, predicts all.
                Default: None.

        Returns:
            Series with predicted cell types, indexed by cell barcodes.

        Raises:
            ValueError: If indices out of bounds.
        """
        self.logger.info("Predicting cell type labels...")
        predictions = model.predict(
            adata=adata,
            indices=indices,
        )
        self.logger.info(f"Predicted labels for {len(predictions)} cells")
        return predictions

    def get_latent(
        self,
        adata: ad.AnnData,
        model: scvi.model.SCANVI,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract latent representations from trained scANVI.

        Args:
            adata: AnnData used to train model.
            model: Trained scANVI model.
            indices: Optional indices to extract subset. Default: None.

        Returns:
            Latent embeddings of shape (n_cells, n_latent).
        """
        self.logger.info("Extracting scANVI latent representations...")
        latent = model.get_latent_representation(
            adata=adata,
            indices=indices,
        )
        self.logger.info(f"Extracted latent with shape {latent.shape}")
        return latent

    def integrate(
        self,
        adata: ad.AnnData,
        batch_key: str = "batch",
        labels_key: str = "cell_type",
        layer: str = "counts",
        n_latent: int = 30,
        n_epochs: int = 50,
        unlabeled_category: str = "Unknown",
        batch_size: int = 128,
    ) -> ad.AnnData:
        """
        Complete scANVI integration and annotation pipeline.

        Performs setup -> train -> extract latent and predictions.

        Args:
            adata: Annotated data with cell type labels (some can be unlabeled).
            batch_key: Key in adata.obs with batch labels. Default: 'batch'.
            labels_key: Key in adata.obs with cell type labels. Default: 'cell_type'.
            layer: Layer with count data. Default: 'counts'.
            n_latent: Latent dimension. Default: 30.
            n_epochs: Training epochs for scANVI. Default: 50.
            unlabeled_category: Category for unlabeled cells. Default: 'Unknown'.
            batch_size: Training batch size. Default: 128.

        Returns:
            Modified adata with:
                - adata.obsm['X_scANVI']: scANVI latent embeddings
                - adata.obs['scanvi_pred']: Predicted cell types
                - adata.uns['scanvi']: Model metadata

        Raises:
            ValueError: If required keys not found.
        """
        # Setup
        self.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
        )

        # Train (trains scVI internally first)
        model = self.train_scanvi(
            adata,
            labels_key=labels_key,
            n_latent=n_latent,
            n_epochs=n_epochs,
            unlabeled_category=unlabeled_category,
            batch_size=batch_size,
        )

        # Extract latent
        latent = self.get_latent(adata, model)
        adata.obsm["X_scANVI"] = latent

        # Predict labels
        predictions = self.predict_labels(adata, model)
        adata.obs["scanvi_pred"] = predictions

        # Store metadata
        adata.uns["scanvi"] = {
            "model_path": None,
            "n_latent": n_latent,
            "batch_key": batch_key,
            "labels_key": labels_key,
            "unlabeled_category": unlabeled_category,
        }

        self.logger.info(
            f"scANVI integration complete. "
            f"Latent shape: {latent.shape}. "
            f"Unique predictions: {predictions.nunique()}"
        )

        return adata
