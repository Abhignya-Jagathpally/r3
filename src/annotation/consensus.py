"""
Consensus cell type annotation for the R3-MM pipeline.

This module implements consensus annotation across multiple annotation methods
(marker-based, CellTypist, scANVI) using majority voting and uncertainty flagging.
"""

import logging
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConsensusAnnotator:
    """
    Consensus cell type annotation across multiple methods.

    Combines annotations from different methods (marker-based, CellTypist, scANVI)
    using majority voting to produce robust, high-confidence cell type assignments.

    Attributes:
        None (stateless class)

    Example:
        >>> from src.annotation import ConsensusAnnotator
        >>> consensus = ConsensusAnnotator()
        >>> adata = consensus.build_consensus(
        ...     adata,
        ...     methods=['marker', 'celltypist', 'scanvi']
        ... )
    """

    # Canonical method column names
    METHOD_COLUMNS = {
        "marker": "cell_type_marker",
        "celltypist": "cell_type_celltypist",
        "scanvi": "scanvi_pred",
    }

    def __init__(self):
        """Initialize ConsensusAnnotator."""
        self.logger = logger

    def annotate(
        self,
        adata: ad.AnnData,
        annotation_keys: List[str],
        confidence_threshold: float = 0.5,
    ) -> ad.AnnData:
        """
        Annotate cells using consensus across specified annotation columns.

        Args:
            adata: AnnData with annotation columns in obs
            annotation_keys: List of obs column names to build consensus from
            confidence_threshold: Minimum agreement fraction for high confidence

        Returns:
            Modified adata with 'consensus_label', 'cell_type', and agreement scores
        """
        cols_to_use = [k for k in annotation_keys if k in adata.obs.columns]
        if not cols_to_use:
            raise ValueError(f"None of {annotation_keys} found in adata.obs")

        self.logger.info(f"Building consensus from {len(cols_to_use)} methods: {cols_to_use}")

        annotations = adata.obs[cols_to_use].values
        consensus_labels = []
        confidence_scores = []
        n_agreement = []

        for i in range(len(annotations)):
            cell_annotations = annotations[i]
            valid = [a for a in cell_annotations if pd.notna(a) and str(a).strip()]

            if not valid:
                consensus_labels.append("Unknown")
                confidence_scores.append(0.0)
                n_agreement.append(0)
                continue

            unique_labels, counts = np.unique(valid, return_counts=True)
            max_idx = counts.argmax()
            consensus_labels.append(unique_labels[max_idx])
            confidence_scores.append(counts[max_idx] / len(valid))
            n_agreement.append(int(counts[max_idx]))

        adata.obs["consensus_label"] = consensus_labels
        adata.obs["cell_type"] = consensus_labels  # canonical label
        adata.obs["cell_type_consensus"] = consensus_labels  # alias
        adata.obs["annotation_confidence"] = confidence_scores
        adata.obs["annotation_agreement"] = n_agreement

        n_high = sum(1 for c in confidence_scores if c >= confidence_threshold)
        self.logger.info(
            f"Consensus complete. {n_high}/{len(consensus_labels)} high-confidence cells. "
            f"Unique types: {len(set(consensus_labels))}"
        )

        return adata

    def build_consensus(
        self,
        adata: ad.AnnData,
        methods: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        min_methods: int = 2,
    ) -> ad.AnnData:
        """
        Build consensus annotation across methods.

        Uses majority voting across specified annotation methods. Flags cells
        with low confidence (no clear majority) or disagreement.

        Args:
            adata: Annotated data matrix with cell type columns from
                various annotation methods.
            methods: List of methods to use. Valid: ['marker', 'celltypist', 'scanvi'].
                Default: ['marker', 'celltypist', 'scanvi'].
            confidence_threshold: Minimum fraction of votes for high confidence.
                Default: 0.5 (>50% agreement).
            min_methods: Minimum number of methods that must agree. Default: 2.

        Returns:
            Modified adata with:
                - adata.obs['cell_type_consensus']: Consensus annotations
                - adata.obs['annotation_confidence']: Confidence scores
                - adata.obs['n_methods_agree']: Number of methods agreeing

        Raises:
            ValueError: If invalid method names provided.
            ValueError: If required annotation columns not in adata.obs.
        """
        if methods is None:
            methods = ["marker", "celltypist", "scanvi"]

        # Validate methods
        invalid_methods = set(methods) - set(self.METHOD_COLUMNS.keys())
        if invalid_methods:
            raise ValueError(
                f"Invalid methods: {invalid_methods}. "
                f"Valid: {list(self.METHOD_COLUMNS.keys())}"
            )

        # Get column names for requested methods
        cols_to_use = []
        for method in methods:
            col = self.METHOD_COLUMNS[method]
            if col not in adata.obs.columns:
                self.logger.warning(
                    f"Column '{col}' for method '{method}' not found. Skipping."
                )
                continue
            cols_to_use.append(col)

        if len(cols_to_use) < min_methods:
            raise ValueError(
                f"Need at least {min_methods} methods available, "
                f"but only {len(cols_to_use)} found."
            )

        self.logger.info(
            f"Building consensus from {len(cols_to_use)} methods: {cols_to_use}"
        )

        # Extract annotations
        annotations = adata.obs[cols_to_use].values

        # Majority voting
        consensus_labels = []
        confidence_scores = []
        n_agreement = []

        for i in range(len(annotations)):
            cell_annotations = annotations[i]

            # Remove NaN values
            valid_annotations = cell_annotations[~pd.isna(cell_annotations)]

            if len(valid_annotations) == 0:
                consensus_labels.append("Unknown")
                confidence_scores.append(0.0)
                n_agreement.append(0)
                continue

            # Count votes
            unique_labels, counts = np.unique(
                valid_annotations, return_counts=True
            )
            max_count = counts.max()
            max_label = unique_labels[counts.argmax()]
            confidence = max_count / len(valid_annotations)

            consensus_labels.append(max_label)
            confidence_scores.append(confidence)
            n_agreement.append(max_count)

        # Add to adata
        adata.obs["cell_type_consensus"] = consensus_labels
        adata.obs["annotation_confidence"] = confidence_scores
        adata.obs["n_methods_agree"] = n_agreement

        # Summary statistics
        n_high_conf = (np.array(confidence_scores) >= confidence_threshold).sum()
        self.logger.info(
            f"Consensus complete. "
            f"{n_high_conf}/{len(consensus_labels)} cells with "
            f"confidence >= {confidence_threshold}. "
            f"Unique types: {len(np.unique(consensus_labels))}"
        )

        return adata

    def get_uncertain_cells(
        self,
        adata: ad.AnnData,
        confidence_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Get cells with low confidence annotations.

        Args:
            adata: AnnData with consensus annotations.
            confidence_threshold: Confidence below which cells are uncertain.
                Default: 0.5.

        Returns:
            DataFrame with uncertain cells and their conflicting annotations.

        Raises:
            ValueError: If consensus not found in adata.obs.
        """
        if "annotation_confidence" not in adata.obs.columns:
            raise ValueError(
                "annotation_confidence not found. Run build_consensus() first."
            )

        uncertain = adata.obs["annotation_confidence"] < confidence_threshold
        uncertain_idx = adata.obs.index[uncertain]

        result = adata.obs.loc[uncertain_idx, [
            "cell_type_consensus",
            "annotation_confidence",
            "n_methods_agree",
        ]].copy()

        self.logger.info(
            f"Found {len(result)} uncertain cells "
            f"(confidence < {confidence_threshold})"
        )

        return result

    def get_disagreement_cells(
        self,
        adata: ad.AnnData,
    ) -> pd.DataFrame:
        """
        Get cells where annotation methods disagree (n_methods_agree < total).

        Args:
            adata: AnnData with consensus annotations.

        Returns:
            DataFrame with disagreement metrics.

        Raises:
            ValueError: If consensus not found.
        """
        if "n_methods_agree" not in adata.obs.columns:
            raise ValueError(
                "n_methods_agree not found. Run build_consensus() first."
            )

        # Count total methods for each cell
        method_cols = [
            col for col in adata.obs.columns
            if col.startswith("cell_type_") or col == "scanvi_pred"
        ]

        n_total_methods = (
            ~adata.obs[method_cols].isna()
        ).sum(axis=1)

        disagreement = adata.obs["n_methods_agree"] < n_total_methods

        result = adata.obs.loc[disagreement, [
            "cell_type_consensus",
            "annotation_confidence",
            "n_methods_agree",
        ]].copy()
        result["n_methods_total"] = n_total_methods[disagreement]

        self.logger.info(
            f"Found {disagreement.sum()} cells with method disagreement"
        )

        return result
