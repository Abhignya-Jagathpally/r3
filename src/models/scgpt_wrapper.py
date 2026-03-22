"""
scGPT wrapper for Multiple Myeloma single-cell analysis.

scGPT is a foundation model pretrained on 33M+ cells from diverse tissues and conditions.
This wrapper provides convenient methods for preprocessing, encoding, fine-tuning, and
cell type prediction on MM datasets.

References:
    - scGPT: Towards Building Large-Scale Foundation Models for Single-Cell Transcriptomics
      URL: https://arxiv.org/abs/2402.16621
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from anndata import AnnData
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ScGPTConfig(BaseModel):
    """Configuration for scGPT model."""

    model_name: str = Field(default="scGPT", description="Model identifier")
    n_hvg: int = Field(default=3000, description="Number of highly variable genes")
    n_bins: int = Field(
        default=51, description="Number of bins for gene expression binning"
    )
    hidden_size: int = Field(default=512, description="Hidden dimension size")
    num_layers: int = Field(default=12, description="Number of transformer layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    vocab_size: int = Field(default=30000, description="Gene vocabulary size")
    max_seq_length: int = Field(
        default=1200, description="Maximum sequence length for input"
    )
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    @validator("n_hvg")
    def validate_n_hvg(cls, v):
        """Validate n_hvg is positive."""
        if v <= 0:
            raise ValueError("n_hvg must be positive")
        return v

    @validator("n_bins")
    def validate_n_bins(cls, v):
        """Validate n_bins is positive."""
        if v <= 0:
            raise ValueError("n_bins must be positive")
        return v


class ScGPTModel:
    """
    Wrapper for scGPT pretrained foundation model.

    This class provides high-level APIs for preprocessing, encoding, fine-tuning,
    and prediction on single-cell transcriptomics data, with special focus on
    Multiple Myeloma applications.

    Attributes:
        config: ScGPTConfig instance with model parameters.
        model_dir: Path to pretrained scGPT model weights.
        model: Underlying PyTorch model (lazy-loaded).
        tokenizer: Gene tokenizer for vocabulary mapping.
    """

    def __init__(
        self,
        model_dir: str,
        n_hvg: int = 3000,
        n_bins: int = 51,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize scGPT wrapper.

        Args:
            model_dir: Directory containing pretrained scGPT weights and config.
            n_hvg: Number of highly variable genes to use. Default: 3000.
            n_bins: Number of bins for binning gene expression. Default: 51.
            device: Device for inference ('cuda' or 'cpu'). Default: auto-detect.
            seed: Random seed. Default: 42.

        Raises:
            FileNotFoundError: If model_dir does not exist.
            ValueError: If config parameters are invalid.
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = ScGPTConfig(n_hvg=n_hvg, n_bins=n_bins, device=device, seed=seed)
        self.model_dir = model_path
        self.model = None
        self.tokenizer = None
        self._hvg_list = None
        self._fitted_scaler = None
        self._is_fine_tuned = False
        self.is_mock = False

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        logger.info(f"Initialized scGPT wrapper with config: {self.config}")

    def _load_pretrained_model(self) -> None:
        """
        Load pretrained scGPT model from disk.

        This is a lazy-load operation to avoid unnecessary memory usage.
        Attempts to load real checkpoint, falls back to mock if unavailable.
        """
        if self.model is not None:
            return

        logger.info(f"Loading pretrained model from {self.model_dir}")

        # Try to import scgpt and load real model
        try:
            import scgpt

            checkpoint_path = self.model_dir / "model.pt"
            if not checkpoint_path.exists():
                logger.warning(
                    f"scGPT checkpoint not found at {checkpoint_path}. "
                    "Falling back to mock model."
                )
                self._initialize_mock_model()
            else:
                # Load real scGPT model
                self.model = torch.load(checkpoint_path, map_location=self.config.device)
                self.is_mock = False
                logger.info(f"Successfully loaded pretrained scGPT model from {checkpoint_path}")

        except ImportError:
            logger.warning(
                "scGPT package is not installed. "
                "Install with: pip install scgpt. "
                "Falling back to mock model."
            )
            self._initialize_mock_model()
        except Exception as e:
            logger.warning(
                f"Failed to load scGPT checkpoint: {e}. "
                "Falling back to mock model."
            )
            self._initialize_mock_model()

    def _initialize_mock_model(self) -> None:
        """Initialize a mock transformer model for development/testing."""
        self.model = nn.Sequential(
            nn.Linear(self.config.n_hvg, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.model = self.model.to(self.config.device)
        self.is_mock = True
        logger.warning(
            "WARNING: Using mock model — results are not from pretrained scGPT"
        )

    def preprocess_for_scgpt(
        self, adata: AnnData, use_raw: bool = True
    ) -> AnnData:
        """
        Preprocess AnnData object for scGPT encoding.

        This includes:
        1. HVG selection (if not already done)
        2. Gene vocabulary mapping
        3. Expression binning into discrete values

        Args:
            adata: Input AnnData object with gene expression data.
            use_raw: Whether to use raw counts (default: True).

        Returns:
            Preprocessed AnnData object ready for encoding.

        Raises:
            ValueError: If adata lacks necessary counts or structure.
        """
        adata = adata.copy()

        # Use raw counts if available
        if use_raw and adata.raw is not None:
            adata = adata.raw.to_adata()

        if adata.n_vars == 0:
            raise ValueError("Input adata has no genes")

        if adata.n_obs == 0:
            raise ValueError("Input adata has no cells")

        # Select HVG
        if self._hvg_list is None:
            logger.info(f"Selecting {self.config.n_hvg} highly variable genes")
            sc.pp.highly_variable_genes(adata, n_top_genes=self.config.n_hvg)
            self._hvg_list = adata.var[adata.var["highly_variable"]].index.tolist()
            adata = adata[:, self._hvg_list].copy()
        else:
            # Use previously fitted HVG list
            common_genes = [g for g in self._hvg_list if g in adata.var_names]
            if len(common_genes) < len(self._hvg_list) * 0.8:
                logger.warning(
                    f"Only {len(common_genes)}/{len(self._hvg_list)} HVGs found in new data"
                )
            adata = adata[:, common_genes].copy()

        # Normalize counts to [0, 1]
        adata.X = np.log1p(adata.X)
        adata.X = (adata.X - adata.X.min()) / (adata.X.max() - adata.X.min() + 1e-8)

        # Bin expression into discrete values
        adata.X = np.digitize(adata.X, bins=np.linspace(0, 1, self.config.n_bins))
        adata.X = np.clip(adata.X, 0, self.config.n_bins - 1).astype(np.int32)

        logger.info(f"Preprocessed {adata.n_obs} cells x {adata.n_vars} genes")
        return adata

    def encode(self, adata: AnnData, batch_size: int = 512) -> np.ndarray:
        """
        Encode cells to latent representations using pretrained scGPT.

        Args:
            adata: Preprocessed AnnData object (from preprocess_for_scgpt).
            batch_size: Batch size for processing to avoid memory explosion. Default: 512.

        Returns:
            Cell embeddings of shape (n_cells, hidden_size).

        Raises:
            ValueError: If adata is not preprocessed correctly.
        """
        if self.is_mock:
            logger.warning("WARNING: Using mock model — results are not from pretrained scGPT")

        if adata.n_vars != self.config.n_hvg and self._hvg_list is None:
            logger.warning(
                f"Expected {self.config.n_hvg} genes, got {adata.n_vars}. "
                "Consider running preprocess_for_scgpt first."
            )

        self._load_pretrained_model()

        # Process in batches to avoid memory explosion
        embeddings = []
        for i in range(0, adata.n_obs, batch_size):
            batch = adata[i : i + batch_size]
            X = batch.X
            if hasattr(X, "toarray"):  # sparse matrix
                X = np.asarray(X.todense())

            X = torch.tensor(X, dtype=torch.float32).to(self.config.device)

            with torch.no_grad():
                batch_emb = self.model(X).cpu().numpy()
                embeddings.append(batch_emb)

        result = np.vstack(embeddings)
        logger.info(f"Encoded {result.shape[0]} cells to {result.shape[1]}-d")
        return result

    def fine_tune(
        self,
        adata: AnnData,
        task: str = "annotation",
        labels_key: str = "cell_type",
        n_epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 64,
        validation_split: float = 0.1,
    ) -> Dict[str, Union[List[float], float]]:
        """
        Fine-tune scGPT on downstream task.

        Args:
            adata: AnnData object with labels for fine-tuning.
            task: Task type ('annotation', 'prognosis', etc.). Default: 'annotation'.
            labels_key: Key in adata.obs for labels. Default: 'cell_type'.
            n_epochs: Number of training epochs. Default: 10.
            lr: Learning rate. Default: 1e-4.
            batch_size: Batch size. Default: 64.
            validation_split: Fraction of data for validation. Default: 0.1.

        Returns:
            Dictionary with training history (losses, accuracies).

        Raises:
            ValueError: If labels_key not in adata.obs.
            ValueError: If task is unsupported.
        """
        if self.is_mock:
            logger.warning("WARNING: Using mock model — results are not from pretrained scGPT")

        if labels_key not in adata.obs:
            raise ValueError(f"Label key '{labels_key}' not found in adata.obs")

        if task not in ["annotation", "prognosis", "proliferation"]:
            raise ValueError(f"Unsupported task: {task}")

        self._load_pretrained_model()

        # Get embeddings
        embeddings = self.encode(adata)

        # Get labels
        labels = adata.obs[labels_key]
        unique_labels = labels.unique()
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in labels])

        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset

        X_tensor = torch.tensor(embeddings, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Train-val split
        n_train = int(len(dataset) * (1 - validation_split))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train]
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Classifier head
        classifier = nn.Linear(self.config.hidden_size, len(unique_labels))
        classifier = classifier.to(self.config.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(n_epochs):
            # Training
            classifier.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                optimizer.zero_grad()
                logits = classifier(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.config.device)
                    y_batch = y_batch.to(self.config.device)

                    logits = classifier(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

        self.classifier = classifier
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self._is_fine_tuned = True

        logger.info(f"Fine-tuning complete. Best val_acc: {max(history['val_acc']):.4f}")
        return history

    def predict(self, adata: AnnData) -> pd.Series:
        """
        Predict cell types/labels after fine-tuning.

        Args:
            adata: AnnData object to predict on.

        Returns:
            Series with predicted labels for each cell.

        Raises:
            ValueError: If model has not been fine-tuned.
        """
        if not self._is_fine_tuned:
            raise ValueError("Model must be fine-tuned before prediction")

        embeddings = self.encode(adata)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.config.device)

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

        predictions = pd.Series(
            [self.idx_to_label[p] for p in preds],
            index=adata.obs_names,
            name="scgpt_prediction",
        )

        logger.info(f"Predicted labels for {len(predictions)} cells")
        return predictions

    def get_gene_embeddings(self) -> pd.DataFrame:
        """
        Get learned gene-level representations from scGPT.

        Returns:
            DataFrame with gene embeddings (n_genes, hidden_size).
        """
        if self._hvg_list is None:
            raise ValueError("No HVG list available. Run preprocess_for_scgpt first.")

        # Create dummy input with 1-hot encoding for each gene
        n_genes = len(self._hvg_list)
        gene_embeddings = np.zeros((n_genes, self.config.hidden_size))

        # Extract embeddings by forward pass through model
        self._load_pretrained_model()
        for i in range(n_genes):
            one_hot = np.zeros((1, self.config.n_hvg))
            one_hot[0, i] = 1.0
            one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32).to(
                self.config.device
            )

            with torch.no_grad():
                emb = self.model(one_hot_tensor).cpu().numpy()
                gene_embeddings[i] = emb[0]

        df = pd.DataFrame(
            gene_embeddings,
            index=self._hvg_list,
            columns=[f"dim_{j}" for j in range(self.config.hidden_size)],
        )

        logger.info(f"Extracted embeddings for {len(df)} genes")
        return df

    def batch_correct(self, adata: AnnData, batch_key: str = "batch") -> AnnData:
        """
        Use scGPT representations for batch correction via simple pooling.

        Args:
            adata: Input AnnData object.
            batch_key: Column in adata.obs identifying batch. Default: 'batch'.

        Returns:
            AnnData with batch-corrected embeddings in .obsm['X_scgpt_corrected'].

        Raises:
            ValueError: If batch_key not in adata.obs.
        """
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

        # Get raw embeddings
        embeddings = self.encode(adata)

        # Compute batch means
        batches = adata.obs[batch_key].unique()
        batch_means = {}
        for batch in batches:
            mask = adata.obs[batch_key] == batch
            batch_means[batch] = embeddings[mask].mean(axis=0)

        # Center embeddings to grand mean
        grand_mean = embeddings.mean(axis=0)
        corrected_embeddings = embeddings.copy()
        for batch in batches:
            mask = adata.obs[batch_key] == batch
            shift = grand_mean - batch_means[batch]
            corrected_embeddings[mask] += shift

        adata_corrected = adata.copy()
        adata_corrected.obsm["X_scgpt_corrected"] = corrected_embeddings

        logger.info(f"Batch-corrected embeddings for {len(batches)} batches")
        return adata_corrected
