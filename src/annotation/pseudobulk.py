"""
Pseudobulk aggregation for the R3-MM pipeline.

This module implements pseudobulk aggregation, which sums raw counts for cells
grouped by patient × cell type (or other groupings). This is critical for
downstream bulk-level analysis and pseudobulk differential expression testing.

The pseudobulk matrix X_{p,c,g}^{pseudo} = sum_{j in cells(p,c)} x_{j,g}
aggregates raw counts from individual cells to patient-cell type level.
"""

import logging
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


class PseudobulkAggregator:
    """
    Pseudobulk aggregation from single-cell counts.

    Aggregates single-cell count matrices to pseudobulk level by summing
    raw counts across cells grouped by patient and cell type (or other groupings).

    This enables bulk-level analysis, pseudobulk differential expression testing,
    and creation of patient-level feature matrices.

    Attributes:
        None (stateless class)

    Example:
        >>> from src.annotation import PseudobulkAggregator
        >>> agg = PseudobulkAggregator()
        >>> pseudobulk = agg.aggregate(
        ...     adata,
        ...     patient_key='patient_id',
        ...     celltype_key='cell_type_consensus',
        ... )
    """

    def __init__(self):
        """Initialize PseudobulkAggregator."""
        self.logger = logger

    def aggregate(
        self,
        adata: ad.AnnData,
        patient_key: str = "patient_id",
        celltype_key: str = "cell_type_consensus",
        layer: str = "counts",
    ) -> ad.AnnData:
        """
        Aggregate cells by patient × cell type.

        Sums raw counts from all cells in each patient-cell type group
        to create a pseudobulk matrix where each row is a (patient, cell_type)
        pair and columns are genes.

        Args:
            adata: Single-cell AnnData with raw counts.
            patient_key: Obs key with patient IDs. Default: 'patient_id'.
            celltype_key: Obs key with cell type annotations. Default: 'cell_type_consensus'.
            layer: Layer containing raw counts. Default: 'counts'.

        Returns:
            New AnnData with:
                - obs: (patient_id, cell_type) tuples as rows
                - X: Aggregated count matrix
                - obs columns: patient_id, cell_type, n_cells (cell count)
                - var: Same as input (gene names)

        Raises:
            ValueError: If layer, patient_key, or celltype_key not found.
            ValueError: If data is sparse but not CSR format for efficiency.
        """
        if layer not in adata.layers:
            raise ValueError(
                f"layer '{layer}' not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        if patient_key not in adata.obs.columns:
            raise ValueError(
                f"patient_key '{patient_key}' not found in adata.obs"
            )
        if celltype_key not in adata.obs.columns:
            raise ValueError(
                f"celltype_key '{celltype_key}' not found in adata.obs"
            )

        self.logger.info(
            f"Aggregating by {patient_key} × {celltype_key} "
            f"using layer '{layer}'..."
        )

        # Get count matrix
        X = adata.layers[layer]
        if sparse.issparse(X):
            if not isinstance(X, sparse.csr_matrix):
                X = X.tocsr()
        else:
            X = np.asarray(X)

        # Get grouping info
        patients = adata.obs[patient_key].values
        celltypes = adata.obs[celltype_key].values

        # Create unique group labels
        groups = list(zip(patients, celltypes))
        unique_groups = sorted(set(groups))

        self.logger.info(
            f"Found {len(unique_groups)} unique "
            f"{patient_key}-{celltype_key} combinations"
        )

        # Aggregate counts using groupby for efficiency
        group_labels = pd.DataFrame({
            patient_key: patients,
            celltype_key: celltypes,
        })
        grouped = group_labels.groupby([patient_key, celltype_key])

        agg_matrix = []
        group_info = []

        for (patient, celltype), idx in grouped.groups.items():
            indices = idx.values
            n_cells = len(indices)

            if sparse.issparse(X):
                group_counts = np.asarray(X[indices].sum(axis=0)).flatten()
            else:
                group_counts = np.asarray(X[indices].sum(axis=0)).flatten()

            agg_matrix.append(group_counts)
            group_info.append({
                patient_key: patient,
                celltype_key: celltype,
                "n_cells": n_cells,
            })

        # Create aggregated AnnData
        agg_X = np.vstack(agg_matrix)
        agg_obs = pd.DataFrame(group_info).set_index(
            [patient_key, celltype_key]
        )

        pseudobulk_adata = ad.AnnData(
            X=agg_X,
            obs=agg_obs,
            var=adata.var.copy(),
        )

        self.logger.info(
            f"Aggregation complete. Pseudobulk shape: {pseudobulk_adata.shape}. "
            f"Total cells aggregated: {pseudobulk_adata.obs['n_cells'].sum()}"
        )

        return pseudobulk_adata

    def aggregate_by_compartment(
        self,
        adata: ad.AnnData,
        patient_key: str = "patient_id",
        celltype_key: str = "cell_type_consensus",
        compartment_key: str = "compartment",
        layer: str = "counts",
    ) -> ad.AnnData:
        """
        Aggregate cells by patient × compartment × cell type.

        Useful when samples have multiple compartments (e.g., bone marrow, blood).

        Args:
            adata: Single-cell AnnData with raw counts.
            patient_key: Obs key with patient IDs. Default: 'patient_id'.
            celltype_key: Obs key with cell types. Default: 'cell_type_consensus'.
            compartment_key: Obs key with compartment labels. Default: 'compartment'.
            layer: Layer with raw counts. Default: 'counts'.

        Returns:
            New AnnData with (patient, compartment, cell_type) rows.

        Raises:
            ValueError: If required keys not found.
        """
        if compartment_key not in adata.obs.columns:
            raise ValueError(
                f"compartment_key '{compartment_key}' not found in adata.obs"
            )

        self.logger.info(
            f"Aggregating by {patient_key} × {compartment_key} × {celltype_key}..."
        )

        # Get matrices
        X = adata.layers[layer]
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        # Get grouping info
        patients = adata.obs[patient_key].values
        compartments = adata.obs[compartment_key].values
        celltypes = adata.obs[celltype_key].values

        # Create unique groups
        groups = list(zip(patients, compartments, celltypes))
        unique_groups = sorted(set(groups))

        self.logger.info(f"Found {len(unique_groups)} unique group combinations")

        # Aggregate using groupby for efficiency
        group_labels = pd.DataFrame({
            patient_key: patients,
            compartment_key: compartments,
            celltype_key: celltypes,
        })
        grouped = group_labels.groupby([patient_key, compartment_key, celltype_key])

        agg_matrix = []
        group_info = []

        for (patient, compartment, celltype), idx in grouped.groups.items():
            indices = idx.values
            n_cells = len(indices)

            if sparse.issparse(X):
                group_counts = np.asarray(X[indices].sum(axis=0)).flatten()
            else:
                group_counts = np.asarray(X[indices].sum(axis=0)).flatten()

            agg_matrix.append(group_counts)
            group_info.append({
                patient_key: patient,
                compartment_key: compartment,
                celltype_key: celltype,
                "n_cells": n_cells,
            })

        # Create AnnData
        agg_X = np.vstack(agg_matrix)
        agg_obs = pd.DataFrame(group_info).set_index(
            [patient_key, compartment_key, celltype_key]
        )

        pseudobulk_adata = ad.AnnData(
            X=agg_X,
            obs=agg_obs,
            var=adata.var.copy(),
        )

        self.logger.info(
            f"Compartment aggregation complete. Shape: {pseudobulk_adata.shape}"
        )

        return pseudobulk_adata

    def to_parquet(
        self,
        pseudobulk_adata: ad.AnnData,
        output_path: str,
        include_cell_counts: bool = True,
    ) -> None:
        """
        Export pseudobulk to parquet format as patient-level feature table.

        Creates a parquet file with genes as columns and (patient, cell_type)
        as row identifiers, suitable for downstream bulk analysis.

        Args:
            pseudobulk_adata: Pseudobulk AnnData from aggregate().
            output_path: Path to write parquet file (e.g., 'pseudobulk.parquet').
            include_cell_counts: If True, includes n_cells as additional column.
                Default: True.

        Raises:
            ImportError: If pyarrow or pandas not available.
            IOError: If unable to write to output_path.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow required for parquet export. "
                "Install with: pip install pyarrow"
            )

        self.logger.info(f"Exporting pseudobulk to {output_path}...")

        import pyarrow as pa
        import pyarrow.parquet as pq

        chunk_size = 1000
        writer = None
        total_rows = 0

        for start in range(0, pseudobulk_adata.n_obs, chunk_size):
            end = min(start + chunk_size, pseudobulk_adata.n_obs)
            chunk = pseudobulk_adata[start:end]
            X = chunk.X.toarray() if hasattr(chunk.X, "toarray") else chunk.X
            df = pd.DataFrame(X, columns=pseudobulk_adata.var_names)

            if include_cell_counts and "n_cells" in pseudobulk_adata.obs.columns:
                df.insert(0, "n_cells", pseudobulk_adata.obs["n_cells"].values[start:end])

            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            total_rows += end - start

        if writer:
            writer.close()

        self.logger.info(
            f"Exported {total_rows} samples × {pseudobulk_adata.n_vars} features"
        )

    def compute_cell_fractions(
        self,
        adata: ad.AnnData,
        patient_key: str = "patient_id",
        celltype_key: str = "cell_type_consensus",
    ) -> pd.DataFrame:
        """
        Compute cell type composition (fractions) per patient.

        Args:
            adata: Single-cell AnnData.
            patient_key: Obs key with patient IDs. Default: 'patient_id'.
            celltype_key: Obs key with cell types. Default: 'cell_type_consensus'.

        Returns:
            DataFrame with shape (n_patients, n_celltypes) containing
            fractional composition.

        Example:
            >>> agg = PseudobulkAggregator()
            >>> fractions = agg.compute_cell_fractions(adata)
            >>> fractions
                           Plasma cell  T cell  B cell
            patient_001         0.45     0.35     0.20
            patient_002         0.50     0.30     0.20
        """
        self.logger.info(
            f"Computing cell type fractions per {patient_key}..."
        )

        # Count cells per group
        counts = adata.obs.groupby(
            [patient_key, celltype_key]
        ).size().unstack(fill_value=0)

        # Compute fractions
        fractions = counts.div(counts.sum(axis=1), axis=0)

        self.logger.info(
            f"Computed fractions for {fractions.shape[0]} patients "
            f"and {fractions.shape[1]} cell types"
        )

        return fractions
