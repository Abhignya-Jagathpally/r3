"""
Tests for agentic tuning modules.

Tests cover:
- Search space sampling and validation
- Contract enforcement
- Experiment runner
- Configuration management
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.agentic.config import AgenticConfig, SearchSpaceConfig, TunerConfig
from src.agentic.contract_enforcer import ContractEnforcer
from src.agentic.experiment_runner import ExperimentRunner, ExperimentTracker
from src.agentic.report_generator import ReportGenerator
from src.agentic.search_space import SearchSpace


class TestSearchSpace:
    """Test hyperparameter search space."""

    def test_random_sampling(self):
        """Test that random sampling produces valid configurations."""
        space = SearchSpace()

        for _ in range(10):
            config = space.sample_config()

            # Check required fields
            assert "model_type" in config
            assert config["model_type"] in ["scvi", "scgpt", "classical", "multimodal"]

            # Check bounds
            assert space.scvi_params["n_latent"][0] <= config["n_latent"]
            assert config["n_latent"] <= space.scvi_params["n_latent"][1]

            assert space.scvi_params["learning_rate"][0] <= config["learning_rate"]
            assert config["learning_rate"] <= space.scvi_params["learning_rate"][1]

    def test_config_validation(self):
        """Test configuration validation."""
        space = SearchSpace()

        # Valid config
        config = space.sample_config()
        is_valid, msg = space.validate_config(config)
        assert is_valid, f"Valid config failed validation: {msg}"

        # Invalid n_latent (out of bounds)
        invalid_config = {**space.sample_config(), "n_latent": 1000}
        is_valid, msg = space.validate_config(invalid_config)
        assert not is_valid
        assert "n_latent" in msg

        # Invalid gene_likelihood
        invalid_config = {**space.sample_config(), "gene_likelihood": "invalid"}
        is_valid, msg = space.validate_config(invalid_config)
        assert not is_valid
        assert "gene_likelihood" in msg

    def test_editable_surface(self):
        """Test editable surface restriction."""
        editable = ["learning_rate", "batch_size"]
        space = SearchSpace(editable_surface=editable)

        assert space.get_editable_params() == editable

    def test_parameter_bounds(self):
        """Test parameter bounds retrieval."""
        space = SearchSpace()
        bounds = space.get_parameter_bounds()

        assert "n_latent" in bounds
        assert "learning_rate" in bounds
        assert "gene_likelihood" in bounds

        # Check bounds are tuples
        assert isinstance(bounds["n_latent"], tuple)
        assert len(bounds["n_latent"]) == 2


class TestContractEnforcer:
    """Test preprocessing contract enforcement."""

    @pytest.fixture
    def contract_file(self):
        """Create temporary contract file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            contract = {
                "version": "1.0",
                "qc_params": {
                    "min_genes": 200,
                    "max_genes": 5000,
                    "max_mito_pct": 20,
                    "min_cells": 3
                },
                "normalization": "scanpy_log",
                "n_hvg": 2000,
            }
            json.dump(contract, f)
            path = f.name
        yield path
        Path(path).unlink()

    def test_load_contract(self, contract_file):
        """Test loading preprocessing contract."""
        enforcer = ContractEnforcer()
        contract = enforcer.load_contract(contract_file)

        assert contract["version"] == "1.0"
        assert contract["qc_params"]["min_genes"] == 200
        assert contract["n_hvg"] == 2000

    def test_load_contract_missing_file(self):
        """Test error handling for missing contract."""
        enforcer = ContractEnforcer()

        with pytest.raises(FileNotFoundError):
            enforcer.load_contract("/nonexistent/path.json")

    def test_verify_frozen_modules(self):
        """Test frozen module verification."""
        enforcer = ContractEnforcer()
        editable = ["learning_rate", "batch_size"]
        frozen = ["preprocessing.*", "data.*"]

        is_valid, msg = enforcer.verify_frozen_modules(editable, frozen)
        assert is_valid

        # Try to edit frozen module
        editable_bad = ["learning_rate", "preprocessing.qc"]
        is_valid, msg = enforcer.verify_frozen_modules(editable_bad, frozen)
        assert not is_valid
        assert "preprocessing.qc" in msg

    def test_compute_data_hash(self):
        """Test data hashing for integrity checks."""
        try:
            import numpy as np
            from anndata import AnnData

            # Create dummy AnnData
            X = np.random.randn(100, 200)
            adata = AnnData(X=X)

            hash1 = ContractEnforcer._compute_data_hash(adata)
            hash2 = ContractEnforcer._compute_data_hash(adata)

            # Same data should have same hash
            assert hash1 == hash2

            # Different data should have different hash
            adata2 = AnnData(X=np.random.randn(100, 200))
            hash3 = ContractEnforcer._compute_data_hash(adata2)
            assert hash1 != hash3

        except ImportError:
            pytest.skip("anndata not installed")


class TestExperimentTracker:
    """Test experiment tracking."""

    def test_log_trial(self):
        """Test logging individual trials."""
        tracker = ExperimentTracker(primary_metric="bio_conservation")

        config = {"learning_rate": 0.001, "batch_size": 64}
        tracker.log_trial(
            trial_id=0,
            config=config,
            metric_value=0.85,
            wallclock_time=120.0,
        )

        assert len(tracker.log) == 1
        assert tracker.log.iloc[0]["bio_conservation"] == 0.85

    def test_best_tracking(self):
        """Test best metric tracking."""
        tracker = ExperimentTracker(primary_metric="bio_conservation")

        for i in range(5):
            config = {"param": i}
            tracker.log_trial(
                trial_id=i,
                config=config,
                metric_value=0.5 + 0.1 * i,
                wallclock_time=100.0,
            )

        best_metric, best_config = tracker.get_best()
        assert best_metric == 0.9
        assert best_config["param"] == 4

    def test_early_stopping(self):
        """Test early stopping criterion."""
        tracker = ExperimentTracker(primary_metric="bio_conservation")

        # Add improving trials
        for i in range(5):
            tracker.log_trial(i, {}, 0.5 + 0.1 * i, 100.0)

        # No early stopping yet
        assert not tracker.early_stop_check(patience=10)

        # Add non-improving trials
        for i in range(5, 15):
            tracker.log_trial(i, {}, 0.85, 100.0)

        # Should trigger early stopping
        assert tracker.early_stop_check(patience=10)


class TestAgenticConfig:
    """Test agentic configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgenticConfig()

        assert config.primary_metric == "bio_conservation"
        assert config.search_budget == 100
        assert config.max_wallclock_hours == 24.0
        assert len(config.editable_surface) > 0
        assert len(config.frozen_modules) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgenticConfig(
            primary_metric="accuracy",
            search_budget=50,
            max_wallclock_hours=12.0,
        )

        assert config.primary_metric == "accuracy"
        assert config.search_budget == 50
        assert config.max_wallclock_hours == 12.0

    def test_config_validation(self):
        """Test config field validation."""
        # Invalid search budget
        with pytest.raises(ValueError):
            AgenticConfig(search_budget=0)

        # Invalid wallclock hours
        with pytest.raises(ValueError):
            AgenticConfig(max_wallclock_hours=-1.0)


class TestReportGenerator:
    """Test report generation."""

    @pytest.fixture
    def sample_log(self):
        """Create sample experiment log."""
        data = {
            "trial_id": range(10),
            "bio_conservation": [0.5 + 0.02 * i for i in range(10)],
            "config_model_type": ["scvi"] * 5 + ["scgpt"] * 5,
            "config_learning_rate": [0.001] * 10,
            "config_n_layers": list(range(1, 11)),
        }
        return pd.DataFrame(data)

    def test_generate_markdown(self, sample_log):
        """Test markdown report generation."""
        generator = ReportGenerator(primary_metric="bio_conservation")
        best_config = {"learning_rate": 0.001}

        report = generator.generate_markdown(sample_log, best_config)

        assert "# AutoResearch Report" in report
        assert "bio_conservation" in report
        assert "best_config.json" in report or "Best Configuration" in report

    def test_leaderboard_generation(self, sample_log):
        """Test leaderboard generation."""
        generator = ReportGenerator(primary_metric="bio_conservation")

        leaderboard = generator._generate_leaderboard(sample_log, top_n=5)

        assert "Rank" in leaderboard
        assert "Trial" in leaderboard
        assert "Metric" in leaderboard

    def test_convergence_analysis(self, sample_log):
        """Test convergence analysis."""
        generator = ReportGenerator(primary_metric="bio_conservation")

        analysis = generator._generate_convergence_analysis(sample_log)

        assert "Mean" in analysis
        assert "Std Dev" in analysis
        assert "Convergence" in analysis

    def test_convergence_data(self, sample_log):
        """Test convergence data generation for plotting."""
        generator = ReportGenerator(primary_metric="bio_conservation")

        data = generator.generate_convergence_data(sample_log)

        assert "trials" in data
        assert "metrics" in data
        assert "best_so_far" in data
        assert len(data["trials"]) == len(sample_log)


class TestSearchSpaceConfig:
    """Test search space configuration."""

    def test_default_search_config(self):
        """Test default search space config."""
        config = SearchSpaceConfig()

        assert config.strategy == "bayesian"
        assert config.n_random_init == 10
        assert config.use_gpu is True
        assert config.n_cpus_per_trial == 4

    def test_tuner_config(self):
        """Test tuner configuration."""
        config = TunerConfig()

        assert config.backend == "ray"
        assert config.n_workers == 4
        assert config.memory_limit_per_worker == 16.0
