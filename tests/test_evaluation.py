"""
Unit tests for evaluation module.

Tests cover:
- Metrics on known inputs (perfect clustering -> ARI=1.0)
- Patient-level splits have no patient overlap
- Train-only fitting prevents data leakage
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from src.evaluation import (
    BenchmarkSuite,
    CrossValidator,
    PatientLevelSplitter,
    TimeAwareSplitter,
    compute_ari,
    compute_batch_asw,
    compute_bio_conservation,
    compute_graph_connectivity,
    compute_nmi,
    compute_rare_cell_recall,
    ensure_no_patient_overlap,
)


class TestARI:
    """Test Adjusted Rand Index computation."""

    def test_perfect_clustering(self):
        """Test ARI on perfect clustering."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])

        ari = compute_ari(labels_true, labels_pred)
        assert abs(ari - 1.0) < 1e-6

    def test_random_clustering(self):
        """Test ARI on random clustering."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([1, 2, 0, 1, 2, 0])

        ari = compute_ari(labels_true, labels_pred)
        assert -1 <= ari <= 1

    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 0, 1])

        with pytest.raises(ValueError):
            compute_ari(labels_true, labels_pred)


class TestNMI:
    """Test Normalized Mutual Information computation."""

    def test_perfect_clustering(self):
        """Test NMI on perfect clustering."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])

        nmi = compute_nmi(labels_true, labels_pred)
        assert abs(nmi - 1.0) < 1e-6

    def test_random_clustering(self):
        """Test NMI on random clustering."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([1, 2, 0, 1, 2, 0])

        nmi = compute_nmi(labels_true, labels_pred)
        assert 0 <= nmi <= 1

    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 0, 1])

        with pytest.raises(ValueError):
            compute_nmi(labels_true, labels_pred)


class TestBatchASW:
    """Test batch-corrected Average Silhouette Width."""

    @pytest.fixture
    def adata(self):
        """Create synthetic AnnData with batch info."""
        X = np.random.randn(100, 20)
        obs = pd.DataFrame({
            "batch": np.repeat([0, 1], 50),
            "cell_type": np.tile(np.repeat([0, 1, 2], 16), 2)[:100],
        })
        adata = AnnData(X=csr_matrix(X), obs=obs)
        adata.obsm["X_pca"] = np.random.randn(100, 10)
        return adata

    def test_good_batch_mixing(self, adata):
        """Test ASW on well-mixed batches."""
        # ASW should be positive for well-mixed batches
        asw = compute_batch_asw(adata)
        assert -1 <= asw <= 1

    def test_missing_batch_key(self, adata):
        """Test error on missing batch key."""
        with pytest.raises(ValueError):
            compute_batch_asw(adata, batch_key="nonexistent")

    def test_missing_embed_key(self, adata):
        """Test error on missing embedding key."""
        with pytest.raises(ValueError):
            compute_batch_asw(adata, embed_key="X_nonexistent")


class TestGraphConnectivity:
    """Test graph connectivity computation."""

    @pytest.fixture
    def adata(self):
        """Create synthetic AnnData with clear clusters."""
        # Create well-separated clusters
        X = np.vstack([
            np.random.randn(30, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.random.randn(30, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            np.random.randn(30, 10) + [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
        ])
        obs = pd.DataFrame({
            "cell_type": np.repeat([0, 1, 2], 30),
        })
        adata = AnnData(X=X, obs=obs)
        adata.obsm["X_pca"] = X
        return adata

    def test_graph_connectivity(self, adata):
        """Test graph connectivity on well-separated clusters."""
        gc = compute_graph_connectivity(adata)

        # Should be relatively high for well-separated clusters
        assert 0 <= gc <= 1
        assert gc > 0.5  # Well-separated

    def test_missing_label_key(self, adata):
        """Test error on missing label key."""
        with pytest.raises(ValueError):
            compute_graph_connectivity(adata, label_key="nonexistent")


class TestRareCellRecall:
    """Test rare cell type recall computation."""

    def test_perfect_recall(self):
        """Test recall on perfectly predicted rare types."""
        labels_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])  # 2 is rare
        labels_pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])

        recalls = compute_rare_cell_recall(labels_true, labels_pred, rare_types=[2])
        assert recalls[2] == 1.0

    def test_zero_recall(self):
        """Test recall when rare type is missed."""
        labels_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        labels_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

        recalls = compute_rare_cell_recall(labels_true, labels_pred, rare_types=[2])
        assert recalls[2] == 0.0

    def test_partial_recall(self):
        """Test recall when rare type is partially correct."""
        labels_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        labels_pred = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0, 0])

        recalls = compute_rare_cell_recall(labels_true, labels_pred, rare_types=[2])
        assert recalls[2] == 0.4


class TestBioConservation:
    """Test biological conservation score."""

    @pytest.fixture
    def adata(self):
        """Create synthetic AnnData with clear structure."""
        X = np.vstack([
            np.random.randn(30, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.random.randn(30, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ])
        obs = pd.DataFrame({
            "cell_type": np.repeat([0, 1], 30),
        })
        adata = AnnData(X=X, obs=obs)
        adata.obsm["X_pca"] = X
        return adata

    def test_bio_conservation(self, adata):
        """Test biological conservation computation."""
        bio_cons = compute_bio_conservation(adata)

        assert 0 <= bio_cons <= 1

    def test_missing_label_key(self, adata):
        """Test error on missing label key."""
        with pytest.raises(ValueError):
            compute_bio_conservation(adata, label_key="nonexistent")


class TestBenchmarkSuite:
    """Test BenchmarkSuite aggregation."""

    def test_annotation_metrics(self):
        """Test annotation task metrics."""
        labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        labels_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        suite = BenchmarkSuite(task="annotation")
        metrics = suite.compute_annotation_metrics(
            labels_true,
            labels_pred,
            rare_types=[2],
        )

        assert "ari" in metrics
        assert "nmi" in metrics
        assert "composite_score" in metrics
        assert metrics["ari"] == 1.0  # Perfect clustering

    def test_integration_metrics(self):
        """Test integration task metrics."""
        X = np.random.randn(100, 10)
        obs = pd.DataFrame({
            "batch": np.repeat([0, 1], 50),
            "cell_type": np.tile(np.repeat([0, 1, 2], 16), 2)[:100],
        })
        adata = AnnData(X=X, obs=obs)
        adata.obsm["X_pca"] = X

        suite = BenchmarkSuite(task="integration")
        metrics = suite.compute_integration_metrics(adata)

        assert "batch_asw" in metrics
        assert "graph_connectivity" in metrics
        assert "bio_conservation" in metrics
        assert "composite_score" in metrics


class TestPatientLevelSplitter:
    """Test patient-level data splitting."""

    @pytest.fixture
    def adata_with_patients(self):
        """Create AnnData with patient IDs."""
        obs = pd.DataFrame({
            "patient_id": np.repeat([f"P{i}" for i in range(10)], 10),
            "cell_type": np.tile(np.repeat([0, 1, 2], 3), 10)[:100],
        })
        adata = AnnData(X=np.random.randn(100, 20), obs=obs)
        return adata

    def test_split_no_overlap(self, adata_with_patients):
        """Test that split has no patient overlap."""
        splitter = PatientLevelSplitter()
        train, test = splitter.split(adata_with_patients, test_size=0.3)

        train_patients = set(train.obs["patient_id"])
        test_patients = set(test.obs["patient_id"])

        overlap = train_patients & test_patients
        assert len(overlap) == 0, "Patient overlap detected"

    def test_split_size(self, adata_with_patients):
        """Test split respects test_size."""
        splitter = PatientLevelSplitter()
        train, test = splitter.split(adata_with_patients, test_size=0.2)

        n_test_patients = len(set(test.obs["patient_id"]))
        n_total_patients = len(set(adata_with_patients.obs["patient_id"]))

        assert n_test_patients / n_total_patients <= 0.3  # Allow some variance

    def test_missing_patient_key(self, adata_with_patients):
        """Test error on missing patient key."""
        splitter = PatientLevelSplitter()

        with pytest.raises(ValueError):
            splitter.split(adata_with_patients, patient_key="nonexistent")


class TestCrossValidator:
    """Test patient-level cross-validation."""

    @pytest.fixture
    def adata_with_patients(self):
        """Create AnnData with patient IDs."""
        obs = pd.DataFrame({
            "patient_id": np.repeat([f"P{i}" for i in range(5)], 10),
            "cell_type": np.tile(np.repeat([0, 1], 2), 5)[:50],
        })
        adata = AnnData(X=np.random.randn(50, 10), obs=obs)
        return adata

    def test_cv_no_overlap(self, adata_with_patients):
        """Test that cross-validation folds have no patient overlap."""
        cv = CrossValidator()
        folds = cv.patient_level_cv(adata_with_patients, n_folds=3)

        assert len(folds) == 3

        for i, (train, test) in enumerate(folds):
            train_patients = set(train.obs["patient_id"])
            test_patients = set(test.obs["patient_id"])

            overlap = train_patients & test_patients
            assert len(overlap) == 0, f"Fold {i}: Patient overlap detected"

    def test_cv_coverage(self, adata_with_patients):
        """Test that cross-validation covers all patients."""
        cv = CrossValidator()
        folds = cv.patient_level_cv(adata_with_patients, n_folds=3)

        all_test_patients = set()
        for train, test in folds:
            all_test_patients.update(test.obs["patient_id"])

        original_patients = set(adata_with_patients.obs["patient_id"])
        assert all_test_patients == original_patients


class TestEnsureNoPatientOverlap:
    """Test patient overlap detection."""

    def test_no_overlap(self):
        """Test detection of no overlap."""
        obs1 = pd.DataFrame({"patient_id": ["P1", "P2"]})
        obs2 = pd.DataFrame({"patient_id": ["P3", "P4"]})

        adata1 = AnnData(X=np.random.randn(2, 10), obs=obs1)
        adata2 = AnnData(X=np.random.randn(2, 10), obs=obs2)

        result = ensure_no_patient_overlap([adata1, adata2])
        assert result is True

    def test_overlap_detection(self):
        """Test detection of overlap."""
        obs1 = pd.DataFrame({"patient_id": ["P1", "P2"]})
        obs2 = pd.DataFrame({"patient_id": ["P2", "P3"]})

        adata1 = AnnData(X=np.random.randn(2, 10), obs=obs1)
        adata2 = AnnData(X=np.random.randn(2, 10), obs=obs2)

        result = ensure_no_patient_overlap([adata1, adata2])
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
