"""
Unit tests for configuration management.

Tests for loading, validating, and saving pipeline configurations.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config, save_config


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_success(self, tmp_path):
        """Test successful configuration loading."""
        config_data = {
            "pipeline": {"name": "test-pipeline", "version": "0.1.0"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE123456",
                        "name": "Test Dataset",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 10000,
                        "platform": "10x",
                    }
                ]
            },
            "qc": {"min_genes": 200, "max_genes": 5000},
            "preprocessing": {},
            "integration": {"enabled": True, "methods": []},
            "annotation": {"methods": []},
            "clustering": {},
            "evaluation": {"metrics": []},
            "pseudobulk": {"grouping_variables": []},
            "differential_expression": {"methods": []},
            "agentic": {
                "enabled": True,
                "search_budget": 100,
                "editable_surface": [],
                "frozen_modules": [],
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.pipeline.name == "test-pipeline"
        assert config.pipeline.version == "0.1.0"

    def test_load_config_not_found(self):
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_empty_file(self, tmp_path):
        """Test error with empty config file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError):
            load_config(str(config_file))


class TestConfigSaving:
    """Tests for configuration saving."""

    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config_data = {
            "pipeline": {"name": "save-test", "version": "0.1.0"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE999999",
                        "name": "Save Test",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 5000,
                        "platform": "10x",
                    }
                ]
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        output_file = tmp_path / "saved_config.yaml"

        save_config(config, str(output_file))

        assert output_file.exists()

        # Verify saved config is readable
        loaded_config = load_config(str(output_file))
        assert loaded_config.pipeline.name == "save-test"


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_optimization_direction(self, tmp_path):
        """Test validation of optimization direction."""
        config_data = {
            "pipeline": {"name": "test"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE123",
                        "name": "Test",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 1000,
                        "platform": "10x",
                    }
                ]
            },
            "agentic": {
                "enabled": True,
                "search_budget": 100,
                "editable_surface": [],
                "frozen_modules": [],
                "optimization_direction": "invalid",
            },
        }

        with pytest.raises(ValueError):
            Config(**config_data)

    def test_invalid_outlier_detection(self, tmp_path):
        """Test validation of outlier detection method."""
        config_data = {
            "pipeline": {"name": "test"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE123",
                        "name": "Test",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 1000,
                        "platform": "10x",
                    }
                ]
            },
            "qc": {"outlier_detection": "invalid"},
        }

        with pytest.raises(ValueError):
            Config(**config_data)


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_qc_defaults(self, tmp_path):
        """Test QC parameter defaults."""
        config_data = {
            "pipeline": {"name": "test"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE123",
                        "name": "Test",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 1000,
                        "platform": "10x",
                    }
                ]
            },
        }

        config = Config(**config_data)
        assert config.qc.min_genes == 200
        assert config.qc.max_genes == 5000
        assert config.qc.max_mito_pct == 20

    def test_compute_defaults(self, tmp_path):
        """Test compute parameter defaults."""
        config_data = {
            "pipeline": {"name": "test"},
            "data_sources": {
                "datasets": [
                    {
                        "accession": "GSE123",
                        "name": "Test",
                        "description": "Test",
                        "organism": "Homo sapiens",
                        "tissue": "Bone Marrow",
                        "cell_count_expected": 1000,
                        "platform": "10x",
                    }
                ]
            },
        }

        config = Config(**config_data)
        assert config.compute.gpu_enabled is True
        assert config.compute.batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
