"""
Marker-based cell type annotation for the R3-MM pipeline.

This module implements marker gene-based cell type annotation specifically
tailored for Multiple Myeloma bone marrow samples, with hardcoded marker
gene sets for common cell types.
"""

import logging
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


class MarkerAnnotator:
    """
    Marker gene-based cell type annotation.

    Uses predefined marker gene sets for MM-relevant cell types to score
    and annotate cells. Hardcoded with comprehensive marker genes for
    bone marrow composition.

    Attributes:
        markers (Dict[str, List[str]]): Cell type to marker genes mapping.
    """

    # Hardcoded marker gene sets for Multiple Myeloma bone marrow
    MARKERS = {
        "Plasma cells": [
            "SDC1",
            "CD138",
            "TNFRSF17",
            "BCMA",
            "XBP1",
            "IRF4",
            "PRDM1",
        ],
        "T cells": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
        "NK cells": ["NCAM1", "NKG7", "GNLY", "KLRD1"],
        "Monocytes": ["CD14", "LYZ", "CST3", "FCGR3A"],
        "B cells": ["CD79A", "MS4A1", "CD20", "CD19"],
        "Erythroid": ["HBA1", "HBB", "GYPA"],
        "HSC/progenitors": ["CD34", "KIT", "THY1"],
        "Osteoclasts": ["ACP5", "CTSK", "MMP9"],
        "Mast cells": ["KIT", "CPA3", "TPSAB1"],
    }

    def __init__(self, markers: Optional[Dict[str, List[str]]] = None):
        """Initialize MarkerAnnotator with marker sets.

        Args:
            markers: Custom marker dict mapping cell_type → gene list.
                If None, uses hardcoded MM markers.
        """
        self.markers = markers if markers is not None else self.MARKERS.copy()
        self.logger = logger

    def get_marker_dict(self) -> Dict[str, List[str]]:
        """
        Get the marker gene dictionary.

        Returns:
            Dictionary mapping cell type names to marker gene lists.
        """
        return self.markers.copy()

    def score_markers(
        self,
        adata: ad.AnnData,
        score_method: str = "scanpy",
        n_jobs: int = 4,
    ) -> ad.AnnData:
        """
        Score marker genes for each cell type in each cell.

        Uses scanpy's score_genes for marker scoring. Adds columns to
        adata.obs with scores for each cell type.

        Args:
            adata: Annotated data matrix. Should have gene names in var_names.
            score_method: Scoring method. Default: 'scanpy' (uses sc.tl.score_genes).
            n_jobs: Number of parallel jobs. Default: 4.

        Returns:
            Modified adata with marker scores in adata.obs as
            'marker_score_{cell_type}'.

        Raises:
            ValueError: If any marker genes not found in adata.
        """
        # Check gene presence
        missing_genes = set()
        for cell_type, genes in self.markers.items():
            for gene in genes:
                if gene not in adata.var_names:
                    missing_genes.add(gene)

        if missing_genes:
            self.logger.warning(
                f"Missing {len(missing_genes)} marker genes: {missing_genes}"
            )

        self.logger.info(
            f"Scoring {len(self.markers)} cell types using {score_method}..."
        )

        for cell_type, genes in self.markers.items():
            # Filter to genes present in data
            genes_present = [g for g in genes if g in adata.var_names]
            if not genes_present:
                self.logger.warning(
                    f"No marker genes found for {cell_type}. Skipping."
                )
                adata.obs[f"marker_score_{cell_type}"] = np.nan
                continue

            self.logger.info(
                f"  Scoring {cell_type} with {len(genes_present)}/{len(genes)} genes..."
            )
            sc.tl.score_genes(adata, genes_present, score_name=f"marker_score_{cell_type}")

        self.logger.info("Marker scoring complete.")
        return adata

    def annotate(
        self,
        adata: ad.AnnData,
        threshold: float = 0.5,
        score_first: bool = True,
    ) -> ad.AnnData:
        """
        Annotate cells based on marker scores.

        Assigns each cell to the cell type with maximum marker score,
        or "Unknown" if all scores below threshold.

        Args:
            adata: Annotated data matrix with marker scores (or will compute them).
            threshold: Minimum score to assign a cell type. Cells with max score
                below this get "Unknown". Default: 0.5.
            score_first: If True, compute scores. If False, use existing scores
                from adata.obs. Default: True.

        Returns:
            Modified adata with:
                - adata.obs['cell_type_marker']: Annotated cell types
                - adata.obs['marker_score_max']: Maximum marker score per cell

        Raises:
            ValueError: If score_first=False but scores not in adata.obs.
        """
        if score_first:
            self.score_markers(adata)

        # Check for scores
        score_cols = [col for col in adata.obs.columns
                      if col.startswith("marker_score_")]
        if not score_cols:
            raise ValueError(
                "No marker scores found in adata.obs and score_first=False. "
                "Call score_markers() first or set score_first=True."
            )

        self.logger.info(
            f"Annotating cells with threshold={threshold}..."
        )

        # Extract scores
        scores = adata.obs[score_cols].values
        cell_types = [col.replace("marker_score_", "") for col in score_cols]

        # Find max score and cell type per cell
        max_scores = np.nanmax(scores, axis=1)
        max_indices = np.nanargmax(scores, axis=1)

        # Assign cell types
        assignments = np.array(
            [cell_types[idx] if max_scores[i] >= threshold else "Unknown"
             for i, idx in enumerate(max_indices)]
        )

        adata.obs["marker_annotation"] = assignments
        adata.obs["cell_type_marker"] = assignments  # alias
        adata.obs["marker_confidence"] = max_scores
        adata.obs["marker_score_max"] = max_scores  # alias

        n_assigned = (assignments != "Unknown").sum()
        self.logger.info(
            f"Annotation complete. {n_assigned}/{len(assignments)} cells assigned. "
            f"Mean max score: {max_scores[~np.isnan(max_scores)].mean():.3f}"
        )

        return adata
