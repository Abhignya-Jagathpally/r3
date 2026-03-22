"""
Tests for the preprocessing module.

Tests all preprocessing components including QC filtering, doublet detection,
ambient RNA correction, normalization, HVG selection, and batch annotation.
"""

import json
import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    AmbientRNACorrector,
    BatchAnnotator,
    DoubletDetector,
    Normalizer,
    PreprocessingConfig,
    PreprocessingPipeline,
    QCFilter,
)


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData object for testing."""
    np.random.seed(42)

    n_obs = 100
    n_vars = 2000

    # Create count matrix (Poisson-like data)
    X = np.random.poisson(5, size=(n_obs, n_vars)).astype(np.float32)

    # Add some structure
    X[:50, :500] = np.random.poisson(10, size=(50, 500)).astype(np.float32)

    # Create observation metadata
    obs = pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(n_obs)],
            "sample_title": [f"Patient_{i // 10}_cell_{i}" for i in range(n_obs)],
            "disease_state": ["MM"] * 50 + ["healthy"] * 50,
        }
    )

    # Create variable metadata with gene names
    var = pd.DataFrame(
        {
            "gene_name": [f"Gene_{i}" if i % 100 != 0 else f"MT-Gene_{i}" for i in range(n_vars)]
        },
        index=[f"ENSG{i:08d}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata


class TestQCFilter:
    """Test QC filtering functionality."""

    def test_qc_filter_initialization(self):
        """Test QCFilter initialization."""
        qc_filter = QCFilter(min_genes=100, max_genes=6000, min_umi=300)
        assert qc_filter.min_genes == 100
        assert qc_filter.max_genes == 6000
        assert qc_filter.min_umi == 300

    def test_calculate_qc_metrics(self, synthetic_adata):
        """Test QC metrics calculation."""
        qc_filter = QCFilter()
        adata = qc_filter.calculate_qc_metrics(synthetic_adata)

        # Check that metrics were added
        assert "n_genes_by_counts" in adata.obs.columns
        assert "n_counts" in adata.obs.columns
        assert "pct_counts_mito" in adata.obs.columns
        assert "pct_counts_ribo" in adata.obs.columns
        assert "mito" in adata.var.columns
        assert "ribo" in adata.var.columns

    def test_filter_cells(self, synthetic_adata):
        """Test cell filtering."""
        qc_filter = QCFilter(min_genes=5, max_genes=10000, min_umi=1)

        # Calculate metrics first
        adata = qc_filter.calculate_qc_metrics(synthetic_adata)
        n_cells_before = adata.n_obs

        # Filter cells
        adata, stats = qc_filter.filter_cells(adata)

        # Check that filtering happened
        assert adata.n_obs <= n_cells_before
        assert "n_cells_before" in stats.columns
        assert "n_cells_after" in stats.columns

    def test_filter_genes(self, synthetic_adata):
        """Test gene filtering."""
        qc_filter = QCFilter()
        adata = qc_filter.calculate_qc_metrics(synthetic_adata)
        n_genes_before = adata.n_vars

        adata, stats = qc_filter.filter_genes(adata)

        assert adata.n_vars <= n_genes_before
        assert stats["n_genes_after"].values[0] <= stats["n_genes_before"].values[0]

    def test_qc_run_pipeline(self, synthetic_adata):
        """Test complete QC pipeline."""
        qc_filter = QCFilter()
        adata, report = qc_filter.run(synthetic_adata)

        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "qc_metrics" in report
        assert "cell_stats" in report
        assert "gene_stats" in report

    def test_empty_adata_raises_error(self):
        """Test that empty AnnData raises error."""
        empty_adata = anndata.AnnData(X=np.array([]).reshape(0, 0))
        qc_filter = QCFilter()

        with pytest.raises(ValueError):
            qc_filter.run(empty_adata)


class TestDoubletDetector:
    """Test doublet detection functionality."""

    def test_doublet_detector_initialization(self):
        """Test DoubletDetector initialization."""
        detector = DoubletDetector(scrublet_threshold=0.6, doubletfinder_threshold=0.7)
        assert detector.scrublet_threshold == 0.6
        assert detector.doubletfinder_threshold == 0.7

    @pytest.mark.skip(reason="Requires scrublet library")
    def test_detect_scrublet(self, synthetic_adata):
        """Test Scrublet doublet detection."""
        detector = DoubletDetector()
        adata = detector.detect_scrublet(synthetic_adata, expected_doublet_rate=0.05)

        assert "scrublet_score" in adata.obs.columns
        assert "scrublet_doublet" in adata.obs.columns

    def test_consensus_filter_without_predictions_raises_error(self, synthetic_adata):
        """Test that consensus_filter raises error without predictions."""
        detector = DoubletDetector()

        with pytest.raises(ValueError):
            detector.consensus_filter(synthetic_adata)


class TestAmbientRNACorrector:
    """Test ambient RNA correction functionality."""

    def test_corrector_initialization(self):
        """Test AmbientRNACorrector initialization."""
        corrector = AmbientRNACorrector(soupx_fdr=0.1, decontx_delta=20)
        assert corrector.soupx_fdr == 0.1
        assert corrector.decontx_delta == 20

    def test_correct_soupx(self, synthetic_adata):
        """Test SoupX correction."""
        corrector = AmbientRNACorrector()
        adata, stats = corrector.correct_soupx(synthetic_adata)

        assert adata.n_obs == synthetic_adata.n_obs
        assert adata.n_vars == synthetic_adata.n_vars
        assert "soupx_contamination_fraction" in adata.obs.columns
        assert stats["method"] == "soupx"

    def test_correct_decontx(self, synthetic_adata):
        """Test DecontX correction."""
        corrector = AmbientRNACorrector()
        adata, stats = corrector.correct_decontx(synthetic_adata)

        assert adata.n_obs == synthetic_adata.n_obs
        assert adata.n_vars == synthetic_adata.n_vars
        assert "decontx_decontamination_prob" in adata.obs.columns
        assert "decontx_contamination_score" in adata.obs.columns
        assert stats["method"] == "decontx"

    def test_run_with_soupx(self, synthetic_adata):
        """Test run method with SoupX."""
        corrector = AmbientRNACorrector()
        adata, stats = corrector.run(synthetic_adata, method="soupx")

        assert stats["method"] == "soupx"

    def test_run_with_decontx(self, synthetic_adata):
        """Test run method with DecontX."""
        corrector = AmbientRNACorrector()
        adata, stats = corrector.run(synthetic_adata, method="decontx")

        assert stats["method"] == "decontx"

    def test_run_with_invalid_method_raises_error(self, synthetic_adata):
        """Test that invalid method raises error."""
        corrector = AmbientRNACorrector()

        with pytest.raises(ValueError):
            corrector.run(synthetic_adata, method="invalid_method")


class TestNormalizer:
    """Test normalization functionality."""

    def test_normalizer_initialization(self):
        """Test Normalizer initialization."""
        normalizer = Normalizer()
        assert normalizer is not None

    def test_scanpy_normalize(self, synthetic_adata):
        """Test scanpy normalization."""
        normalizer = Normalizer()
        adata, stats = normalizer.scanpy_normalize(synthetic_adata, target_sum=1e4)

        assert "counts" in adata.layers
        assert adata.n_obs == synthetic_adata.n_obs
        assert adata.n_vars == synthetic_adata.n_vars
        assert stats["method"] == "scanpy_log_normalize"

    def test_scran_normalize(self, synthetic_adata):
        """Test SCRAN normalization."""
        normalizer = Normalizer()
        adata, stats = normalizer.scran_normalize(synthetic_adata)

        assert "counts" in adata.layers
        assert adata.n_obs == synthetic_adata.n_obs
        assert adata.n_vars == synthetic_adata.n_vars
        assert stats["method"] == "scran"

    def test_pearson_residuals(self, synthetic_adata):
        """Test Pearson residuals normalization."""
        normalizer = Normalizer()
        adata, stats = normalizer.pearson_residuals(synthetic_adata)

        assert "counts" in adata.layers
        assert adata.n_obs == synthetic_adata.n_obs
        assert adata.n_vars == synthetic_adata.n_vars
        assert stats["method"] == "pearson_residuals"

    def test_select_hvgs(self, synthetic_adata):
        """Test HVG selection."""
        normalizer = Normalizer()

        # First normalize
        adata, _ = normalizer.scanpy_normalize(synthetic_adata)

        # Then select HVGs
        adata, stats = normalizer.select_hvgs(adata, n_top_genes=500, flavor="seurat_v3")

        assert "highly_variable" in adata.var.columns
        assert stats["flavor"] == "seurat_v3"
        assert stats["n_hvgs_selected"] == 500

    def test_hvg_selection_with_n_top_genes_exceeding_total_raises_warning(
        self, synthetic_adata
    ):
        """Test HVG selection with n_top_genes exceeding total genes."""
        normalizer = Normalizer()
        adata, _ = normalizer.scanpy_normalize(synthetic_adata)

        # Should log warning but not fail
        adata, stats = normalizer.select_hvgs(adata, n_top_genes=10000)
        assert stats["n_hvgs_selected"] == adata.n_vars


class TestBatchAnnotator:
    """Test batch annotation functionality."""

    def test_batch_annotator_initialization(self):
        """Test BatchAnnotator initialization."""
        annotator = BatchAnnotator()
        assert annotator is not None

    def test_extract_patient_id(self, synthetic_adata):
        """Test patient ID extraction."""
        annotator = BatchAnnotator()
        adata = annotator.extract_patient_id(synthetic_adata, name_column="sample_title")

        assert "patient_id" in adata.obs.columns
        assert len(adata.obs["patient_id"]) == synthetic_adata.n_obs

    def test_classify_sample_type(self, synthetic_adata):
        """Test sample type classification."""
        annotator = BatchAnnotator()
        adata = annotator.classify_sample_type(synthetic_adata, disease_status_column="disease_state")

        assert "sample_type" in adata.obs.columns
        assert all(st in ["MM", "healthy", "MGUS", "SMM", "unknown"] for st in adata.obs["sample_type"])

    def test_extract_tissue_type(self, synthetic_adata):
        """Test tissue type extraction."""
        annotator = BatchAnnotator()
        synthetic_adata.obs["tissue"] = "Bone Marrow"
        adata = annotator.extract_tissue_type(synthetic_adata, tissue_column="tissue")

        assert "tissue" in adata.obs.columns

    def test_extract_technology(self, synthetic_adata):
        """Test technology extraction."""
        annotator = BatchAnnotator()
        synthetic_adata.obs["technology"] = "10X"
        adata = annotator.extract_technology(synthetic_adata, tech_column="technology")

        assert "technology" in adata.obs.columns

    def test_validate_batch_keys(self, synthetic_adata):
        """Test batch key validation."""
        annotator = BatchAnnotator()

        # Should return False for missing keys
        assert not annotator.validate_batch_keys(synthetic_adata)

        # Add keys
        synthetic_adata.obs["patient_id"] = "P1"
        synthetic_adata.obs["sample_type"] = "MM"
        synthetic_adata.obs["tissue"] = "BM"
        synthetic_adata.obs["technology"] = "10X"

        # Should return True
        assert annotator.validate_batch_keys(synthetic_adata)

    def test_run_pipeline(self, synthetic_adata):
        """Test complete batch annotation pipeline."""
        annotator = BatchAnnotator()
        adata = annotator.run(synthetic_adata)

        assert "patient_id" in adata.obs.columns
        assert "sample_type" in adata.obs.columns
        assert "tissue" in adata.obs.columns
        assert "technology" in adata.obs.columns


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""

    def test_pipeline_initialization(self):
        """Test PreprocessingPipeline initialization."""
        config = PreprocessingConfig(min_genes=100)
        pipeline = PreprocessingPipeline(config)

        assert pipeline.config.min_genes == 100

    def test_pipeline_run(self, synthetic_adata):
        """Test complete pipeline execution."""
        config = PreprocessingConfig(
            min_genes=5,
            max_genes=10000,
            detect_doublets=False,  # Skip doublet detection
            correct_ambient_rna=False,  # Skip ambient RNA correction
            select_hvgs=True,
            n_hvgs=500,
        )
        pipeline = PreprocessingPipeline(config)

        adata, report = pipeline.run(synthetic_adata)

        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert report.n_cells_initial == synthetic_adata.n_obs
        assert report.n_genes_initial == synthetic_adata.n_vars
        assert report.n_cells_final == adata.n_obs
        assert report.n_genes_final == adata.n_vars
        assert "qc_filtering" in report.steps_completed

    def test_freeze_contract(self, synthetic_adata):
        """Test contract freezing."""
        config = PreprocessingConfig(min_genes=100)
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "contract.json"
            pipeline.freeze_contract(str(contract_path))

            # Check that file was created
            assert contract_path.exists()

            # Load and validate contract
            with open(contract_path) as f:
                contract = json.load(f)

            assert contract["frozen"] is True
            assert "config" in contract
            assert contract["config"]["min_genes"] == 100

    def test_pipeline_with_all_steps_enabled(self, synthetic_adata):
        """Test pipeline with all steps enabled."""
        config = PreprocessingConfig(
            detect_doublets=False,  # Skip since it requires external libraries
            correct_ambient_rna=True,
            select_hvgs=True,
            annotate_batch=True,
        )
        pipeline = PreprocessingPipeline(config)

        adata, report = pipeline.run(synthetic_adata)

        # Verify all expected annotations are present
        expected_obs = ["patient_id", "sample_type", "tissue", "technology"]
        for col in expected_obs:
            assert col in adata.obs.columns, f"Missing column: {col}"


class TestPreprocessingConfig:
    """Test preprocessing configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = PreprocessingConfig()

        assert config.min_genes == 200
        assert config.max_genes == 5000
        assert config.min_umi == 500
        assert config.max_mito_pct == 20.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            min_genes=100,
            max_genes=6000,
            n_hvgs=3000,
        )

        assert config.min_genes == 100
        assert config.max_genes == 6000
        assert config.n_hvgs == 3000

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = PreprocessingConfig(min_genes=150)
        config_dict = config.model_dump()

        assert config_dict["min_genes"] == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
