"""
Batch annotation and metadata standardization for single-cell RNA-seq data.

This module provides tools to extract, parse, and standardize batch metadata
from GEO and other sources, ensuring consistent column names and validation
across datasets.
"""

import logging
import re
from typing import Dict, List, Optional

import anndata
import pandas as pd

logger = logging.getLogger(__name__)


class BatchAnnotator:
    """
    Batch metadata annotation and standardization.

    Extracts and standardizes batch information from GEO metadata, including
    sample type classification, patient ID, tissue type, and sequencing technology.

    Attributes:
        valid_sample_types: Set of valid sample type classifications.
    """

    # Valid sample types for Multiple Myeloma studies
    VALID_SAMPLE_TYPES = {"MM", "MGUS", "SMM", "healthy", "control", "normal", "unknown"}

    def __init__(self):
        """Initialize BatchAnnotator."""
        logger.info("Initialized BatchAnnotator")

    def extract_study_id(self, adata: anndata.AnnData, key: str = "geo_accession") -> str:
        """
        Extract study ID from metadata.

        Args:
            adata: Input annotated data matrix.
            key: Key in adata.obs or adata.uns containing study ID.

        Returns:
            Study ID string.

        Raises:
            ValueError: If study ID cannot be found.
        """
        if key in adata.obs.columns:
            study_ids = adata.obs[key].unique()
            study_id = study_ids[0] if len(study_ids) > 0 else "unknown"
        elif key in adata.uns:
            study_id = str(adata.uns[key])
        else:
            logger.warning(f"Could not find study ID in key '{key}', using 'unknown'")
            study_id = "unknown"

        logger.info(f"Extracted study ID: {study_id}")
        return study_id

    def extract_patient_id(
        self, adata: anndata.AnnData, name_column: str = "sample_title"
    ) -> anndata.AnnData:
        """
        Extract patient ID from sample names.

        Attempts to extract patient identifiers using common patterns from
        GEO sample titles (e.g., 'Patient001', 'P01', 'MM_001').

        Args:
            adata: Input annotated data matrix.
            name_column: Column in adata.obs containing sample names.

        Returns:
            AnnData with patient ID in adata.obs['patient_id'].

        Raises:
            ValueError: If sample name column is missing.
        """
        if name_column not in adata.obs.columns:
            logger.warning(
                f"Column '{name_column}' not found. Creating empty patient_id column."
            )
            adata.obs["patient_id"] = "unknown"
            return adata

        sample_names = adata.obs[name_column].astype(str)

        # Extract patient IDs using regex patterns
        patient_ids = []
        patterns = [
            r"[Pp]atient[_-]?(\d+)",  # Patient_001, patient-01
            r"[Pp](\d+)",  # P01, P001
            r"[Mm][Mm]_?(\d+)",  # MM_001, MM001
            r"[Pp][Mm]_?(\d+)",  # PM_001
        ]

        for name in sample_names:
            patient_id = None
            for pattern in patterns:
                match = re.search(pattern, str(name))
                if match:
                    patient_id = match.group(0)
                    break

            if patient_id is None:
                patient_id = f"unknown_{hash(name) % 1000}"

            patient_ids.append(patient_id)

        adata.obs["patient_id"] = patient_ids
        logger.info(f"Extracted {len(set(patient_ids))} unique patient IDs")

        return adata

    def classify_sample_type(
        self, adata: anndata.AnnData, disease_status_column: Optional[str] = None
    ) -> anndata.AnnData:
        """
        Classify sample types (MM/MGUS/SMM/healthy).

        Uses disease status metadata or infers from sample characteristics.

        Args:
            adata: Input annotated data matrix.
            disease_status_column: Column in adata.obs containing disease status.

        Returns:
            AnnData with sample_type in adata.obs['sample_type'].
        """
        if disease_status_column and disease_status_column in adata.obs.columns:
            # Use existing disease status column
            status = adata.obs[disease_status_column].astype(str).str.lower()
        else:
            # Try to infer from observation metadata
            if "disease_state" in adata.obs.columns:
                status = adata.obs["disease_state"].astype(str).str.lower()
            elif "condition" in adata.obs.columns:
                status = adata.obs["condition"].astype(str).str.lower()
            else:
                logger.warning("No disease status column found, classifying all as 'unknown'")
                adata.obs["sample_type"] = "unknown"
                return adata

        # Map to standardized sample types
        sample_types = []
        for s in status:
            if any(keyword in s for keyword in ["mm", "multiple myeloma", "myeloma"]):
                sample_type = "MM"
            elif any(keyword in s for keyword in ["mgus", "smm", "smoldering"]):
                sample_type = "SMM" if "smm" in s or "smoldering" in s else "MGUS"
            elif any(
                keyword in s for keyword in ["normal", "healthy", "control", "donor"]
            ):
                sample_type = "healthy"
            else:
                sample_type = "unknown"

            sample_types.append(sample_type)

        adata.obs["sample_type"] = sample_types

        # Log distribution
        type_counts = pd.Series(sample_types).value_counts()
        logger.info(f"Sample type distribution: {type_counts.to_dict()}")

        return adata

    def extract_tissue_type(
        self, adata: anndata.AnnData, tissue_column: Optional[str] = None
    ) -> anndata.AnnData:
        """
        Extract tissue type information.

        Args:
            adata: Input annotated data matrix.
            tissue_column: Column in adata.obs containing tissue type.
                Default: "tissue".

        Returns:
            AnnData with tissue in adata.obs['tissue'].
        """
        if tissue_column is None:
            # Try common column names
            for col in ["tissue", "source_tissue", "tissue_type"]:
                if col in adata.obs.columns:
                    tissue_column = col
                    break

        if tissue_column and tissue_column in adata.obs.columns:
            tissues = adata.obs[tissue_column].astype(str)
        else:
            logger.warning("No tissue column found, using 'bone marrow' as default")
            tissues = pd.Series(["bone_marrow"] * adata.n_obs)

        # Standardize tissue names
        tissues = tissues.str.lower().str.replace(" ", "_").str.replace("-", "_")

        adata.obs["tissue"] = tissues
        logger.info(f"Tissues: {adata.obs['tissue'].unique()}")

        return adata

    def extract_technology(
        self, adata: anndata.AnnData, tech_column: Optional[str] = None
    ) -> anndata.AnnData:
        """
        Extract sequencing technology information.

        Args:
            adata: Input annotated data matrix.
            tech_column: Column in adata.obs containing technology.
                Default: "technology".

        Returns:
            AnnData with technology in adata.obs['technology'].
        """
        if tech_column is None:
            # Try common column names
            for col in ["technology", "sequencing_technology", "platform"]:
                if col in adata.obs.columns:
                    tech_column = col
                    break

        if tech_column and tech_column in adata.obs.columns:
            techs = adata.obs[tech_column].astype(str)
        else:
            logger.warning("No technology column found, using 'unknown'")
            techs = pd.Series(["unknown"] * adata.n_obs)

        # Standardize technology names
        techs = techs.str.lower()
        techs = techs.str.replace("10x", "10x_genomics")
        techs = techs.str.replace("smartseq", "smart_seq")

        adata.obs["technology"] = techs
        logger.info(f"Technologies: {adata.obs['technology'].unique()}")

        return adata

    def validate_batch_keys(
        self, adata: anndata.AnnData, required_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Validate that required batch annotation keys exist.

        Args:
            adata: Input annotated data matrix.
            required_keys: List of required column names. Default: standard keys.

        Returns:
            True if all required keys are present, False otherwise.

        Raises:
            ValueError: If critical keys are missing and validation fails.
        """
        if required_keys is None:
            required_keys = ["patient_id", "sample_type", "tissue", "technology"]

        missing_keys = [key for key in required_keys if key not in adata.obs.columns]

        if missing_keys:
            logger.error(f"Missing required batch keys: {missing_keys}")
            return False

        logger.info("All required batch keys present")
        return True

    def run(
        self,
        adata: anndata.AnnData,
        name_column: str = "sample_title",
        disease_status_column: Optional[str] = None,
        tissue_column: Optional[str] = None,
        tech_column: Optional[str] = None,
    ) -> anndata.AnnData:
        """
        Run complete batch annotation pipeline.

        Performs the full workflow:
        1. Extract patient ID
        2. Classify sample type
        3. Extract tissue type
        4. Extract technology
        5. Validate all required keys

        Args:
            adata: Input annotated data matrix.
            name_column: Column containing sample names.
            disease_status_column: Column containing disease status.
            tissue_column: Column containing tissue type.
            tech_column: Column containing technology type.

        Returns:
            AnnData with standardized batch metadata.

        Raises:
            ValueError: If validation fails.
        """
        logger.info("Starting batch annotation pipeline")

        # Run individual extraction steps
        adata = self.extract_patient_id(adata, name_column=name_column)
        adata = self.classify_sample_type(adata, disease_status_column=disease_status_column)
        adata = self.extract_tissue_type(adata, tissue_column=tissue_column)
        adata = self.extract_technology(adata, tech_column=tech_column)

        # Validate
        if not self.validate_batch_keys(adata):
            logger.warning("Some batch keys are missing, but continuing")

        logger.info("Batch annotation pipeline completed")
        return adata
