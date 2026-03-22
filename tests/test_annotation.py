"""
Unit tests for the annotation module.

Tests marker-based annotation, pseudobulk aggregation, cell ontology mapping,
and consensus annotation.
"""

import logging
import unittest
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.annotation import (
    CellOntologyMapper,
    CellTypistAnnotator,
    ConsensusAnnotator,
    MarkerAnnotator,
    PseudobulkAggregator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMarkerAnnotator(unittest.TestCase):
    """Test marker-based cell type annotation."""

    def setUp(self) -> None:
        """Create synthetic test data."""
        np.random.seed(42)
        n_obs = 500
        n_vars = 2000

        # Create synthetic count data
        X = np.random.poisson(lam=0.5, size=(n_obs, n_vars)).astype(np.float32)

        # Create gene names with marker genes
        marker_genes = {
            "Plasma cells": ["SDC1", "CD138", "XBP1"],
            "T cells": ["CD3D", "CD3E", "CD8A"],
            "B cells": ["CD79A", "MS4A1", "CD19"],
        }
        all_marker_genes = set()
        for genes in marker_genes.values():
            all_marker_genes.update(genes)

        other_genes = [
            f"GENE_{i}" for i in range(n_vars - len(all_marker_genes))
        ]
        var_names = sorted(list(all_marker_genes)) + other_genes

        # Create AnnData
        self.adata = ad.AnnData(X=X, var=pd.DataFrame(index=var_names))

        # Set higher expression for specific cell types in some cells
        for i in range(100):
            # Plasma cells
            for gene in ["SDC1", "CD138", "XBP1"]:
                if gene in var_names:
                    self.adata.X[i, self.adata.var_names.get_loc(gene)] = 5.0

        for i in range(100, 250):
            # T cells
            for gene in ["CD3D", "CD3E", "CD8A"]:
                if gene in var_names:
                    self.adata.X[i, self.adata.var_names.get_loc(gene)] = 5.0

        for i in range(250, 350):
            # B cells
            for gene in ["CD79A", "MS4A1", "CD19"]:
                if gene in var_names:
                    self.adata.X[i, self.adata.var_names.get_loc(gene)] = 5.0

    def test_marker_dict(self) -> None:
        """Test marker dictionary retrieval."""
        annotator = MarkerAnnotator()
        markers = annotator.get_marker_dict()

        self.assertIn("Plasma cells", markers)
        self.assertIn("SDC1", markers["Plasma cells"])

    def test_score_markers(self) -> None:
        """Test marker gene scoring."""
        annotator = MarkerAnnotator()
        adata = annotator.score_markers(self.adata)

        # Check that score columns were added
        score_cols = [col for col in adata.obs.columns
                      if col.startswith("marker_score_")]
        self.assertGreater(len(score_cols), 0)

        # Plasma cells should have higher scores in first 100 cells
        plasma_scores = adata.obs["marker_score_Plasma cells"].values[:100]
        background_scores = adata.obs["marker_score_Plasma cells"].values[250:]
        self.assertGreater(plasma_scores.mean(), background_scores.mean())

    def test_annotate(self) -> None:
        """Test cell type annotation."""
        annotator = MarkerAnnotator()
        adata = annotator.annotate(self.adata, threshold=0.5)

        # Check annotation columns
        self.assertIn("cell_type_marker", adata.obs.columns)
        self.assertIn("marker_score_max", adata.obs.columns)

        # Check that annotation works
        annotations = adata.obs["cell_type_marker"].unique()
        self.assertGreater(len(annotations), 0)


class TestCellOntologyMapper(unittest.TestCase):
    """Test Cell Ontology mapping."""

    def setUp(self) -> None:
        """Initialize mapper."""
        self.mapper = CellOntologyMapper()

    def test_map_labels(self) -> None:
        """Test label to CL ID mapping."""
        labels = pd.Series(["B cell", "T cell", "Plasma cell"])
        mapped = self.mapper.map_labels(labels)

        self.assertEqual(mapped.iloc[0], "CL:0000236")  # B cell
        self.assertEqual(mapped.iloc[1], "CL:0000084")  # T cell
        self.assertEqual(mapped.iloc[2], "CL:0000786")  # Plasma cell

    def test_case_insensitive_mapping(self) -> None:
        """Test case-insensitive mapping."""
        labels = pd.Series(["b cell", "T CELL", "PLASMA CELL"])
        mapped = self.mapper.map_labels(labels, case_sensitive=False)

        self.assertEqual(mapped.iloc[0], "CL:0000236")
        self.assertEqual(mapped.iloc[1], "CL:0000084")
        self.assertEqual(mapped.iloc[2], "CL:0000786")

    def test_get_label_name(self) -> None:
        """Test reverse mapping from CL ID to label."""
        label = self.mapper.get_label_name("CL:0000236")
        self.assertEqual(label, "B cell")

    def test_validate_labels(self) -> None:
        """Test label validation."""
        valid_labels = pd.Series(["CL:0000236", "CL:0000084"])
        result = self.mapper.validate_labels(valid_labels)
        self.assertTrue(result)

    def test_all_mappings(self) -> None:
        """Test retrieving all mappings."""
        mappings = self.mapper.get_all_mappings()
        self.assertIn("B cell", mappings)
        self.assertEqual(mappings["B cell"], "CL:0000236")

        reverse = self.mapper.get_reverse_mappings()
        self.assertIn("CL:0000236", reverse)


class TestConsensusAnnotator(unittest.TestCase):
    """Test consensus annotation across methods."""

    def setUp(self) -> None:
        """Create test data with multiple annotation methods."""
        np.random.seed(42)
        n_obs = 100

        # Create AnnData with multiple annotation columns
        X = np.random.randn(n_obs, 50)
        obs_data = {
            "cell_type_marker": np.random.choice(
                ["B cell", "T cell", "Plasma cell"],
                n_obs
            ),
            "cell_type_celltypist": np.random.choice(
                ["B cell", "T cell", "Plasma cell"],
                n_obs
            ),
            "scanvi_pred": np.random.choice(
                ["B cell", "T cell", "Plasma cell"],
                n_obs
            ),
        }

        # Add some disagreement
        for i in range(10):
            obs_data["cell_type_marker"][i] = "B cell"
            obs_data["cell_type_celltypist"][i] = "T cell"
            obs_data["scanvi_pred"][i] = "Plasma cell"

        self.adata = ad.AnnData(X=X, obs=pd.DataFrame(obs_data))

    def test_build_consensus(self) -> None:
        """Test consensus building."""
        consensus = ConsensusAnnotator()
        adata = consensus.build_consensus(
            self.adata,
            methods=["marker", "celltypist", "scanvi"],
        )

        # Check new columns
        self.assertIn("cell_type_consensus", adata.obs.columns)
        self.assertIn("annotation_confidence", adata.obs.columns)
        self.assertIn("n_methods_agree", adata.obs.columns)

        # Check confidence ranges
        conf = adata.obs["annotation_confidence"].values
        self.assertTrue((conf >= 0).all())
        self.assertTrue((conf <= 1).all())

    def test_uncertain_cells(self) -> None:
        """Test identification of uncertain cells."""
        consensus = ConsensusAnnotator()
        adata = consensus.build_consensus(self.adata)

        uncertain = consensus.get_uncertain_cells(
            adata,
            confidence_threshold=0.8,
        )

        # Should find some uncertain cells due to disagreement
        self.assertGreater(len(uncertain), 0)

    def test_disagreement_cells(self) -> None:
        """Test identification of disagreement."""
        consensus = ConsensusAnnotator()
        adata = consensus.build_consensus(self.adata)

        disagreement = consensus.get_disagreement_cells(adata)

        # Should find cells with disagreement
        self.assertGreater(len(disagreement), 0)


class TestPseudobulkAggregator(unittest.TestCase):
    """Test pseudobulk aggregation."""

    def setUp(self) -> None:
        """Create test data for pseudobulk."""
        np.random.seed(42)
        n_cells = 1000
        n_genes = 200

        # Create count data
        counts = np.random.poisson(lam=2.0, size=(n_cells, n_genes))

        # Create cell metadata
        obs_data = {
            "patient_id": np.random.choice(
                ["P001", "P002", "P003"],
                n_cells,
            ),
            "cell_type_consensus": np.random.choice(
                ["B cell", "T cell", "Plasma cell"],
                n_cells,
            ),
            "compartment": np.random.choice(
                ["BM", "PB"],
                n_cells,
            ),
        }

        self.adata = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame(
                index=[f"GENE_{i}" for i in range(n_genes)]
            ),
        )
        self.adata.layers["counts"] = counts.copy()

    def test_aggregate_preserves_counts(self) -> None:
        """Test that aggregation preserves total counts."""
        aggregator = PseudobulkAggregator()
        pseudobulk = aggregator.aggregate(
            self.adata,
            patient_key="patient_id",
            celltype_key="cell_type_consensus",
        )

        # Total counts should be preserved
        original_total = self.adata.layers["counts"].sum()
        aggregated_total = pseudobulk.X.sum()

        np.testing.assert_almost_equal(original_total, aggregated_total)

    def test_aggregate_dimensions(self) -> None:
        """Test aggregation dimensions."""
        aggregator = PseudobulkAggregator()
        pseudobulk = aggregator.aggregate(self.adata)

        # Check dimensions
        n_patients = self.adata.obs["patient_id"].nunique()
        n_celltypes = self.adata.obs["cell_type_consensus"].nunique()
        expected_rows = n_patients * n_celltypes

        self.assertEqual(pseudobulk.n_obs, expected_rows)
        self.assertEqual(pseudobulk.n_vars, self.adata.n_vars)

    def test_aggregate_by_compartment(self) -> None:
        """Test aggregation with compartment."""
        aggregator = PseudobulkAggregator()
        pseudobulk = aggregator.aggregate_by_compartment(
            self.adata,
            patient_key="patient_id",
            celltype_key="cell_type_consensus",
            compartment_key="compartment",
        )

        n_patients = self.adata.obs["patient_id"].nunique()
        n_celltypes = self.adata.obs["cell_type_consensus"].nunique()
        n_compartments = self.adata.obs["compartment"].nunique()
        expected_rows = n_patients * n_celltypes * n_compartments

        self.assertEqual(pseudobulk.n_obs, expected_rows)

    def test_cell_fractions(self) -> None:
        """Test cell type fraction computation."""
        aggregator = PseudobulkAggregator()
        fractions = aggregator.compute_cell_fractions(
            self.adata,
            patient_key="patient_id",
            celltype_key="cell_type_consensus",
        )

        # Check shape
        n_patients = self.adata.obs["patient_id"].nunique()
        n_celltypes = self.adata.obs["cell_type_consensus"].nunique()

        self.assertEqual(fractions.shape[0], n_patients)
        self.assertEqual(fractions.shape[1], n_celltypes)

        # Fractions should sum to 1 per patient
        patient_sums = fractions.sum(axis=1)
        np.testing.assert_almost_equal(patient_sums.values, 1.0)

    def test_n_cells_column(self) -> None:
        """Test that n_cells is correctly computed."""
        aggregator = PseudobulkAggregator()
        pseudobulk = aggregator.aggregate(self.adata)

        # Total cells should equal input
        self.assertEqual(
            pseudobulk.obs["n_cells"].sum(),
            self.adata.n_obs,
        )


class TestCellTypistAnnotator(unittest.TestCase):
    """Test CellTypist annotator."""

    def test_cl_mapping(self) -> None:
        """Test Cell Ontology mapping for CellTypist labels."""
        annotator = CellTypistAnnotator()

        labels = pd.Series(["B cell", "T cell", "Plasma cell"])
        mapped = annotator.map_to_cell_ontology(labels)

        self.assertEqual(mapped.iloc[0], "CL:0000236")
        self.assertEqual(mapped.iloc[1], "CL:0000084")
        self.assertEqual(mapped.iloc[2], "CL:0000786")

    def test_import_check(self) -> None:
        """Test celltypist availability check."""
        annotator = CellTypistAnnotator()
        # Should not raise even if celltypist not installed
        available = annotator._check_celltypist()
        self.assertIsInstance(available, bool)


if __name__ == "__main__":
    unittest.main()
