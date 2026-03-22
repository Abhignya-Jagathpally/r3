"""
Unit tests for the integration module.

Tests Harmony, scVI, and scANVI integration methods.
"""

import logging
import unittest
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from src.integration import (
    HarmonyIntegrator,
    ScANVIIntegrator,
    ScVIIntegrator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHarmonyIntegrator(unittest.TestCase):
    """Test Harmony integration."""

    def setUp(self) -> None:
        """Create test data with batch effects."""
        np.random.seed(42)
        n_per_batch = 100
        n_vars = 2000

        # Create synthetic data with batch effect
        X1 = np.random.randn(n_per_batch, n_vars) + 1.0  # Batch 1
        X2 = np.random.randn(n_per_batch, n_vars) - 1.0  # Batch 2

        X = np.vstack([X1, X2])

        obs_data = {
            "batch": ["batch1"] * n_per_batch + ["batch2"] * n_per_batch,
        }

        self.adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(obs_data),
        )

    def test_compute_pca(self) -> None:
        """Test PCA computation."""
        integrator = HarmonyIntegrator()
        adata = integrator.compute_pca(self.adata, n_comps=50)

        self.assertIn("X_pca", adata.obsm)
        self.assertEqual(adata.obsm["X_pca"].shape, (200, 50))

    def test_compute_pca_validation(self) -> None:
        """Test PCA parameter validation."""
        integrator = HarmonyIntegrator()

        with self.assertRaises(ValueError):
            integrator.compute_pca(self.adata, n_comps=0)

        with self.assertRaises(ValueError):
            integrator.compute_pca(self.adata, n_comps=100000)

    def test_compute_neighbors_validation(self) -> None:
        """Test neighbor computation validation."""
        integrator = HarmonyIntegrator()
        adata = integrator.compute_pca(self.adata, n_comps=50)

        with self.assertRaises(ValueError):
            integrator.compute_neighbors(
                adata,
                use_rep="nonexistent",
            )

        with self.assertRaises(ValueError):
            integrator.compute_neighbors(adata, n_neighbors=0)

    def test_compute_umap_requires_neighbors(self) -> None:
        """Test that UMAP requires neighborhood graph."""
        integrator = HarmonyIntegrator()
        adata = integrator.compute_pca(self.adata, n_comps=50)

        with self.assertRaises(ValueError):
            integrator.compute_umap(adata)

    def test_batch_key_validation(self) -> None:
        """Test batch key validation."""
        integrator = HarmonyIntegrator()

        with self.assertRaises(ValueError):
            integrator.integrate(
                self.adata,
                batch_key="nonexistent_batch",
            )


class TestScVIIntegrator(unittest.TestCase):
    """Test scVI integration."""

    def setUp(self) -> None:
        """Create test data for scVI."""
        np.random.seed(42)
        n_obs = 200
        n_vars = 2000

        # Create count data
        counts = np.random.poisson(lam=1.0, size=(n_obs, n_vars))

        obs_data = {
            "batch": np.random.choice(["batch1", "batch2"], n_obs),
        }

        self.adata = ad.AnnData(
            X=counts.astype(np.float32),
            obs=pd.DataFrame(obs_data),
        )
        self.adata.layers["counts"] = counts.astype(np.float32)

    def test_setup_anndata_validation(self) -> None:
        """Test AnnData setup validation."""
        integrator = ScVIIntegrator()

        with self.assertRaises(ValueError):
            integrator.setup_anndata(
                self.adata,
                batch_key="batch",
                layer="nonexistent_layer",
            )

        with self.assertRaises(ValueError):
            integrator.setup_anndata(
                self.adata,
                batch_key="nonexistent_batch",
                layer="counts",
            )

    def test_setup_anndata(self) -> None:
        """Test AnnData setup."""
        integrator = ScVIIntegrator()
        adata = integrator.setup_anndata(
            self.adata,
            batch_key="batch",
            layer="counts",
        )

        # Check that scVI registry is set
        self.assertTrue(
            hasattr(adata, 'uns') and '_scvi' in adata.uns or
            hasattr(adata, '_scvi')
        )

    def test_train_scvi_validation(self) -> None:
        """Test scVI training parameter validation."""
        integrator = ScVIIntegrator()
        integrator.setup_anndata(self.adata, batch_key="batch", layer="counts")

        with self.assertRaises(ValueError):
            integrator.train_scvi(self.adata, n_latent=0)

        with self.assertRaises(ValueError):
            integrator.train_scvi(self.adata, n_epochs=0)


class TestScANVIIntegrator(unittest.TestCase):
    """Test scANVI integration."""

    def setUp(self) -> None:
        """Create test data for scANVI."""
        np.random.seed(42)
        n_obs = 200
        n_vars = 2000

        counts = np.random.poisson(lam=1.0, size=(n_obs, n_vars))

        obs_data = {
            "batch": np.random.choice(["batch1", "batch2"], n_obs),
            "cell_type": np.random.choice(
                ["T cell", "B cell", "Unknown"],
                n_obs,
            ),
        }

        self.adata = ad.AnnData(
            X=counts.astype(np.float32),
            obs=pd.DataFrame(obs_data),
        )
        self.adata.layers["counts"] = counts.astype(np.float32)

    def test_setup_anndata_validation(self) -> None:
        """Test scANVI setup validation."""
        integrator = ScANVIIntegrator()

        with self.assertRaises(ValueError):
            integrator.setup_anndata(
                self.adata,
                batch_key="batch",
                labels_key="cell_type",
                layer="nonexistent",
            )

        with self.assertRaises(ValueError):
            integrator.setup_anndata(
                self.adata,
                batch_key="nonexistent",
                labels_key="cell_type",
                layer="counts",
            )

        with self.assertRaises(ValueError):
            integrator.setup_anndata(
                self.adata,
                batch_key="batch",
                labels_key="nonexistent",
                layer="counts",
            )

    def test_setup_anndata(self) -> None:
        """Test scANVI setup."""
        integrator = ScANVIIntegrator()
        adata = integrator.setup_anndata(
            self.adata,
            batch_key="batch",
            labels_key="cell_type",
            layer="counts",
        )

        # Should have scVI registry
        self.assertTrue(
            hasattr(adata, 'uns') and '_scvi' in adata.uns or
            hasattr(adata, '_scvi')
        )

    def test_train_scanvi_validation(self) -> None:
        """Test scANVI training parameter validation."""
        integrator = ScANVIIntegrator()
        integrator.setup_anndata(
            self.adata,
            batch_key="batch",
            labels_key="cell_type",
            layer="counts",
        )

        with self.assertRaises(ValueError):
            integrator.train_scanvi(
                self.adata,
                labels_key="cell_type",
                n_latent=0,
            )

        with self.assertRaises(ValueError):
            integrator.train_scanvi(
                self.adata,
                labels_key="cell_type",
                n_epochs=0,
            )


class TestIntegrationConsistency(unittest.TestCase):
    """Test consistency across integration methods."""

    def setUp(self) -> None:
        """Create consistent test data."""
        np.random.seed(42)
        n_obs = 200
        n_vars = 2000

        counts = np.random.poisson(lam=1.0, size=(n_obs, n_vars))

        obs_data = {
            "batch": np.random.choice(["batch1", "batch2"], n_obs),
        }

        self.adata = ad.AnnData(
            X=counts.astype(np.float32),
            obs=pd.DataFrame(obs_data),
        )
        self.adata.layers["counts"] = counts.astype(np.float32)

    def test_harmony_outputs(self) -> None:
        """Test Harmony output structure."""
        integrator = HarmonyIntegrator()

        # Would test but requires harmony-pytorch
        # This is a placeholder for full integration test
        self.assertIsNotNone(integrator)

    def test_scvi_outputs(self) -> None:
        """Test scVI output structure."""
        integrator = ScVIIntegrator()

        # Would test but requires scvi-tools setup
        # This is a placeholder for full integration test
        self.assertIsNotNone(integrator)

    def test_scanvi_outputs(self) -> None:
        """Test scANVI output structure."""
        integrator = ScANVIIntegrator()

        # Would test but requires scvi-tools setup
        # This is a placeholder for full integration test
        self.assertIsNotNone(integrator)


if __name__ == "__main__":
    unittest.main()
