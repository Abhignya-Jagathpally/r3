"""
Cell Ontology mapping and validation for the R3-MM pipeline.

This module provides standardized mapping from common cell type names to
Cell Ontology (CL) identifiers for data standardization and interoperability.
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CellOntologyMapper:
    """
    Standardized cell type mapping to Cell Ontology identifiers.

    Maintains a comprehensive mapping of common cell type names (from various
    annotation sources) to standardized Cell Ontology (CL) IDs for improved
    data interoperability and standardization.

    Attributes:
        label_to_cl (Dict[str, str]): Mapping from cell type names to CL IDs.
        cl_to_label (Dict[str, str]): Reverse mapping from CL IDs to names.
    """

    # Comprehensive mapping for bone marrow and immune cell types
    LABEL_TO_CL = {
        # Plasma cells and B cells
        "Plasma cell": "CL:0000786",
        "Plasma cells": "CL:0000786",
        "Plasmacyte": "CL:0000786",
        "B cell": "CL:0000236",
        "B cells": "CL:0000236",
        "Memory B cell": "CL:0000813",
        "Memory B cells": "CL:0000813",
        "Naive B cell": "CL:0000817",
        "Naive B cells": "CL:0000817",
        "Mature B cell": "CL:0000993",
        # T cells
        "T cell": "CL:0000084",
        "T cells": "CL:0000084",
        "CD4+ T cell": "CL:0000624",
        "CD4+ T cells": "CL:0000624",
        "Helper T cell": "CL:0000624",
        "CD8+ T cell": "CL:0000625",
        "CD8+ T cells": "CL:0000625",
        "Cytotoxic T cell": "CL:0000625",
        "Regulatory T cell": "CL:0000819",
        "Regulatory T cells": "CL:0000819",
        "Naive T cell": "CL:0000898",
        "Memory T cell": "CL:0000813",
        # NK cells
        "NK cell": "CL:0000930",
        "NK cells": "CL:0000930",
        "Natural killer cell": "CL:0000930",
        # Monocytes and macrophages
        "Monocyte": "CL:0000576",
        "Monocytes": "CL:0000576",
        "Macrophage": "CL:0000235",
        "Macrophages": "CL:0000235",
        "Classical monocyte": "CL:0000860",
        "Intermediate monocyte": "CL:0000861",
        "Non-classical monocyte": "CL:0000862",
        # Dendritic cells
        "Dendritic cell": "CL:0000451",
        "Dendritic cells": "CL:0000451",
        "Myeloid dendritic cell": "CL:0000784",
        "Plasmacytoid dendritic cell": "CL:0000785",
        # Granulocytes
        "Neutrophil": "CL:0000775",
        "Neutrophils": "CL:0000775",
        "Eosinophil": "CL:0000771",
        "Eosinophils": "CL:0000771",
        "Basophil": "CL:0000768",
        "Basophils": "CL:0000768",
        # Mast cells
        "Mast cell": "CL:0000097",
        "Mast cells": "CL:0000097",
        # Stem and progenitor cells
        "HSC": "CL:0000037",
        "Hematopoietic stem cell": "CL:0000037",
        "Hematopoietic progenitor": "CL:0000049",
        "Progenitor": "CL:0000049",
        "Common myeloid progenitor": "CL:0000050",
        "Common lymphoid progenitor": "CL:0000051",
        # Erythroid
        "Erythrocyte": "CL:0000232",
        "Erythrocytes": "CL:0000232",
        "Red blood cell": "CL:0000232",
        "RBC": "CL:0000232",
        "Erythroid": "CL:0000762",
        # Megakaryocyte
        "Megakaryocyte": "CL:0000557",
        "Megakaryocytes": "CL:0000557",
        "Platelet": "CL:0000233",
        # Osteoclast and bone cells
        "Osteoclast": "CL:0000182",
        "Osteoclasts": "CL:0000182",
        "Osteoblast": "CL:0000185",
        "Osteoblasts": "CL:0000185",
        # Stromal/fibroblast
        "Fibroblast": "CL:0000057",
        "Fibroblasts": "CL:0000057",
        "Stromal cell": "CL:0000057",
        "Endothelial cell": "CL:0000115",
        "Endothelial cells": "CL:0000115",
        # Generic/unknown
        "Unknown": "CL:0000000",
        "Unassigned": "CL:0000000",
    }

    def __init__(self):
        """Initialize CellOntologyMapper with label-to-CL mapping."""
        self.label_to_cl = self.LABEL_TO_CL.copy()
        self.cl_to_label = {v: k for k, v in self.label_to_cl.items()}
        self.logger = logger

    def map(self, labels, **kwargs) -> list:
        """Map cell type labels to Cell Ontology IDs (convenience wrapper).

        Args:
            labels: List or Series of cell type labels.

        Returns:
            List of mapped labels.
        """
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        result = self.map_labels(labels, **kwargs)
        return result.tolist()

    def map_labels(
        self,
        labels: pd.Series,
        case_sensitive: bool = False,
        keep_unmapped: bool = True,
    ) -> pd.Series:
        """
        Map cell type labels to Cell Ontology IDs.

        Args:
            labels: Series of cell type labels to map.
            case_sensitive: If False, performs case-insensitive matching.
                Default: False.
            keep_unmapped: If True, keeps unmapped labels unchanged.
                If False, maps unmapped to "CL:0000000" (unknown).
                Default: True.

        Returns:
            Series with Cell Ontology IDs.

        Example:
            >>> labels = pd.Series(["B cell", "T cell", "Unknown type"])
            >>> mapper = CellOntologyMapper()
            >>> mapped = mapper.map_labels(labels)
            >>> mapped
            0    CL:0000236
            1    CL:0000084
            2    Unknown type  # or CL:0000000 if keep_unmapped=False
        """
        self.logger.info(f"Mapping {len(labels)} labels to Cell Ontology...")

        if not case_sensitive:
            # Build case-insensitive mapping
            mapping = {}
            for key, value in self.label_to_cl.items():
                mapping[key.lower()] = value

            def map_func(x):
                if pd.isna(x):
                    return "CL:0000000" if not keep_unmapped else x
                x_str = str(x).strip().lower()
                if x_str in mapping:
                    return mapping[x_str]
                return "CL:0000000" if not keep_unmapped else str(x)

            mapped = labels.map(map_func)
        else:
            def map_func(x):
                if pd.isna(x):
                    return "CL:0000000" if not keep_unmapped else x
                x_str = str(x).strip()
                if x_str in self.label_to_cl:
                    return self.label_to_cl[x_str]
                return "CL:0000000" if not keep_unmapped else x_str

            mapped = labels.map(map_func)

        n_mapped = (mapped != labels).sum()
        self.logger.info(f"Mapped {n_mapped}/{len(labels)} labels")

        return mapped

    def get_label_name(self, cl_id: str) -> Optional[str]:
        """
        Get canonical cell type name for a Cell Ontology ID.

        Args:
            cl_id: Cell Ontology ID (e.g., "CL:0000236").

        Returns:
            Canonical cell type name, or None if not found.

        Example:
            >>> mapper = CellOntologyMapper()
            >>> mapper.get_label_name("CL:0000236")
            'B cell'
        """
        # Get first matching name (canonical)
        for label, cid in self.label_to_cl.items():
            if cid == cl_id:
                return label
        return None

    def validate_labels(
        self,
        labels: pd.Series,
        valid_cl_ids: Optional[set] = None,
    ) -> bool:
        """
        Validate that labels are valid Cell Ontology IDs or mapped.

        Args:
            labels: Series of labels to validate.
            valid_cl_ids: Set of allowed CL IDs. If None, uses all known IDs.
                Default: None.

        Returns:
            True if all labels are valid Cell Ontology IDs or in mapping.

        Example:
            >>> labels = pd.Series(["CL:0000236", "CL:0000084"])
            >>> mapper = CellOntologyMapper()
            >>> mapper.validate_labels(labels)
            True
        """
        if valid_cl_ids is None:
            valid_cl_ids = set(self.label_to_cl.values())

        valid_labels = labels.isin(valid_cl_ids)
        n_valid = valid_labels.sum()

        if n_valid < len(labels):
            invalid = labels[~valid_labels].unique()
            self.logger.warning(
                f"Found {len(invalid)} invalid labels: {invalid}"
            )

        self.logger.info(
            f"Validation complete: {n_valid}/{len(labels)} valid"
        )

        return n_valid == len(labels)

    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get complete label-to-CL mapping.

        Returns:
            Dictionary of all label -> CL ID mappings.
        """
        return self.label_to_cl.copy()

    def get_reverse_mappings(self) -> Dict[str, str]:
        """
        Get complete CL-to-label reverse mapping.

        Returns:
            Dictionary of all CL ID -> label mappings.
        """
        return self.cl_to_label.copy()
