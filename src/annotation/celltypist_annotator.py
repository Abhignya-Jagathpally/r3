"""
CellTypist-based cell type annotation for the R3-MM pipeline.

This module implements automated cell type annotation using the CellTypist tool,
which provides pre-trained deep learning models for cell type prediction.

References:
    Dominguez Conde et al. (2022). Cross-tissue immune cell analysis reveals
    tissue-specific adaptive immunity. Nature, 611, 440–446.
"""

import logging
from typing import Optional

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)


class CellTypistAnnotator:
    """
    CellTypist-based automated cell type annotation.

    Uses pre-trained CellTypist models to predict cell types. Supports both
    regular annotation and majority voting across multiple samplings.

    Attributes:
        None (stateless wrapper around celltypist library)

    Example:
        >>> from src.annotation import CellTypistAnnotator
        >>> annotator = CellTypistAnnotator()
        >>> adata = annotator.annotate(adata, model_name='Immune_All_Low.pkl')
    """

    # Mapping from CellTypist labels to Cell Ontology IDs
    # This is a subset for common immune/bone marrow cell types
    CELLTYPIST_TO_CL = {
        "B cell": "CL:0000236",
        "B cells": "CL:0000236",
        "T cell": "CL:0000084",
        "T cells": "CL:0000084",
        "CD4+ T cell": "CL:0000624",
        "CD8+ T cell": "CL:0000625",
        "NK cell": "CL:0000930",
        "NK cells": "CL:0000930",
        "Monocyte": "CL:0000576",
        "Monocytes": "CL:0000576",
        "Macrophage": "CL:0000235",
        "Macrophages": "CL:0000235",
        "Dendritic cell": "CL:0000451",
        "Dendritic cells": "CL:0000451",
        "Plasma cell": "CL:0000786",
        "Plasma cells": "CL:0000786",
        "Memory B cell": "CL:0000813",
        "Naive B cell": "CL:0000817",
        "Regulatory T cell": "CL:0000819",
        "Naive T cell": "CL:0000898",
        "Erythrocyte": "CL:0000232",
        "Megakaryocyte": "CL:0000557",
        "Neutrophil": "CL:0000775",
        "Eosinophil": "CL:0000771",
    }

    def __init__(self):
        """Initialize CellTypistAnnotator."""
        self.logger = logger
        self._celltypist_available = self._check_celltypist()

    def _check_celltypist(self) -> bool:
        """
        Check if celltypist is available.

        Returns:
            True if celltypist can be imported, False otherwise.
        """
        try:
            import celltypist
            return True
        except ImportError:
            self.logger.warning(
                "celltypist not available. Install with: "
                "pip install celltypist"
            )
            return False

    def annotate(
        self,
        adata: ad.AnnData,
        model_name: str = "Immune_All_Low.pkl",
        majority_voting: bool = True,
        over_clustering: bool = True,
    ) -> ad.AnnData:
        """
        Annotate cells using CellTypist pre-trained model.

        Args:
            adata: Annotated data matrix. Gene names must match model's gene set.
            model_name: CellTypist model name. Common options:
                - 'Immune_All_Low.pkl': Immune cell types (low resolution)
                - 'Immune_All_High.pkl': Immune cell types (high resolution)
                - 'Bone_Marrow_Atlas.pkl': Bone marrow specific
                Default: 'Immune_All_Low.pkl'.
            majority_voting: If True, use majority voting for more robust
                predictions. Default: False.

        Returns:
            Modified adata with:
                - adata.obs['cell_type_celltypist']: Predicted cell types
                - adata.obs['cell_type_celltypist_prob']: Prediction confidence
                (if available from model)

        Raises:
            ImportError: If celltypist not installed.
            ValueError: If model_name not found.
        """
        if not self._celltypist_available:
            raise ImportError(
                "celltypist not available. Install with: "
                "pip install celltypist"
            )

        import celltypist

        self.logger.info(
            f"Annotating with CellTypist model: {model_name} "
            f"(majority_voting={majority_voting})..."
        )

        try:
            # Download model if not available
            try:
                celltypist.models.download_models(model=model_name)
            except Exception:
                pass  # Model may already exist

            annotate_kwargs = {
                "model": model_name,
                "majority_voting": majority_voting,
            }
            if over_clustering and "leiden" in adata.obs.columns:
                annotate_kwargs["over_clustering"] = adata.obs["leiden"]

            predictions = celltypist.annotate(
                adata,
                **annotate_kwargs,
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to load model {model_name}. "
                f"Available models: {celltypist.models.get_available_models()}. "
                f"Error: {e}"
            )

        # Extract predictions
        if hasattr(predictions, "predicted_labels"):
            labels = predictions.predicted_labels
            if hasattr(labels, "iloc"):
                # DataFrame: take first column (or 'majority_voting' if present)
                if "majority_voting" in labels.columns:
                    adata.obs["celltypist_label"] = labels["majority_voting"].values
                else:
                    adata.obs["celltypist_label"] = labels.iloc[:, 0].values
            elif hasattr(labels, "values"):
                adata.obs["celltypist_label"] = labels.values
            else:
                adata.obs["celltypist_label"] = labels
        elif hasattr(predictions, "obs"):
            adata.obs["celltypist_label"] = predictions.obs.get(
                "predicted_labels",
                predictions.obs.get("majority_voting", predictions.obs.iloc[:, 0]),
            )
        else:
            adata.obs["celltypist_label"] = predictions

        # Alias for backward compatibility
        adata.obs["cell_type_celltypist"] = adata.obs["celltypist_label"]

        self.logger.info(
            f"CellTypist annotation complete. "
            f"Unique predictions: {adata.obs['celltypist_label'].nunique()}"
        )

        return adata

    def annotate_majority_voting(
        self,
        adata: ad.AnnData,
        model_name: str = "Immune_All_Low.pkl",
    ) -> ad.AnnData:
        """
        Annotate cells with majority voting for robust predictions.

        Majority voting aggregates predictions from multiple random samplings
        for increased confidence and robustness.

        Args:
            adata: Annotated data matrix.
            model_name: CellTypist model name. Default: 'Immune_All_Low.pkl'.

        Returns:
            Modified adata with cell_type_celltypist annotations.

        Raises:
            ImportError: If celltypist not installed.
        """
        return self.annotate(adata, model_name=model_name, majority_voting=True)

    def map_to_cell_ontology(
        self,
        labels: pd.Series,
    ) -> pd.Series:
        """
        Map CellTypist labels to Cell Ontology IDs.

        Uses a predefined mapping from common cell type names to standardized
        Cell Ontology CL: identifiers. Unmapped labels remain unchanged.

        Args:
            labels: Series of cell type labels from CellTypist.

        Returns:
            Series with Cell Ontology IDs. Unmapped labels kept as original.

        Example:
            >>> labels = pd.Series(["B cell", "T cell", "Plasma cell"])
            >>> cl_ids = annotator.map_to_cell_ontology(labels)
            >>> cl_ids
            0    CL:0000236
            1    CL:0000084
            2    CL:0000786
        """
        self.logger.info(
            f"Mapping {len(labels)} labels to Cell Ontology..."
        )

        mapped = labels.map(
            lambda x: self.CELLTYPIST_TO_CL.get(str(x).strip(), str(x))
        )

        n_mapped = (mapped != labels).sum()
        self.logger.info(
            f"Mapped {n_mapped}/{len(labels)} labels to Cell Ontology IDs"
        )

        return mapped
