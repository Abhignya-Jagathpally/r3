"""
Unit tests for models module.

Tests cover:
- Classical baseline models on synthetic data
- scGPT config validation
- Multimodal fusion dimension handling
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models import (
    ClassicalEnsemble,
    LogisticBaseline,
    MultimodalFuser,
    RandomForestBaseline,
    ScGPTConfig,
    SVMBaseline,
)


class TestLogisticBaseline:
    """Test LogisticBaseline model."""

    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=15, n_classes=3, random_state=42
        )
        return X, y

    def test_initialization(self):
        """Test LogisticBaseline initialization."""
        model = LogisticBaseline(max_iter=500, solver="lbfgs")
        assert model.model is not None
        assert model.scaler is not None

    def test_fit_and_predict(self, data):
        """Test fitting and prediction."""
        X, y = data
        model = LogisticBaseline()

        # Fit
        model.fit(X, y)
        assert model.classes_ is not None
        assert len(model.classes_) == 3

        # Predict
        y_pred = model.predict(X)
        assert len(y_pred) == len(X)
        assert all(pred in model.classes_ for pred in y_pred)

    def test_predict_proba(self, data):
        """Test probability predictions."""
        X, y = data
        model = LogisticBaseline()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), len(model.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_cross_validate(self, data):
        """Test cross-validation."""
        X, y = data
        model = LogisticBaseline()

        results = model.cross_validate(X, y, cv=3)
        assert "test_accuracy" in results
        assert len(results["test_accuracy"]) == 3

    def test_fit_incompatible_shapes(self, data):
        """Test error on incompatible shapes."""
        X, y = data
        model = LogisticBaseline()

        with pytest.raises(ValueError):
            model.fit(X, y[:-1])

    def test_predict_before_fit(self):
        """Test error predicting before fitting."""
        X = np.random.randn(10, 20)
        model = LogisticBaseline()

        with pytest.raises(ValueError):
            model.predict(X)


class TestRandomForestBaseline:
    """Test RandomForestBaseline model."""

    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=15, n_classes=3, random_state=42
        )
        return X, y

    def test_initialization(self):
        """Test RandomForestBaseline initialization."""
        model = RandomForestBaseline(n_estimators=50, max_depth=10)
        assert model.model is not None

    def test_fit_and_predict(self, data):
        """Test fitting and prediction."""
        X, y = data
        model = RandomForestBaseline(n_estimators=10)

        model.fit(X, y)
        assert model.classes_ is not None

        y_pred = model.predict(X)
        assert len(y_pred) == len(X)

    def test_feature_importance(self, data):
        """Test feature importance extraction."""
        X, y = data
        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert importance.sum() > 0

    def test_cross_validate(self, data):
        """Test cross-validation."""
        X, y = data
        model = RandomForestBaseline(n_estimators=10)

        results = model.cross_validate(X, y, cv=3)
        assert "test_accuracy" in results
        assert len(results["test_accuracy"]) == 3


class TestSVMBaseline:
    """Test SVMBaseline model."""

    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=15, n_classes=3, random_state=42
        )
        return X, y

    def test_initialization(self):
        """Test SVMBaseline initialization."""
        model = SVMBaseline(C=1.0, kernel="rbf")
        assert model.model is not None
        assert model.scaler is not None

    def test_fit_and_predict(self, data):
        """Test fitting and prediction."""
        X, y = data
        model = SVMBaseline()

        model.fit(X, y)
        assert model.classes_ is not None

        y_pred = model.predict(X)
        assert len(y_pred) == len(X)

    def test_predict_proba(self, data):
        """Test probability predictions."""
        X, y = data
        model = SVMBaseline()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), len(model.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestClassicalEnsemble:
    """Test ClassicalEnsemble model."""

    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=15, n_classes=3, random_state=42
        )
        return X, y

    def test_initialization(self):
        """Test ClassicalEnsemble initialization."""
        ensemble = ClassicalEnsemble()
        assert ensemble.lr_model is not None
        assert ensemble.rf_model is not None
        assert ensemble.svm_model is not None

    def test_fit_and_predict(self, data):
        """Test ensemble fitting and prediction."""
        X, y = data
        ensemble = ClassicalEnsemble()

        ensemble.fit(X, y)
        assert ensemble.classes_ is not None

        y_pred = ensemble.predict(X)
        assert len(y_pred) == len(X)

    def test_predict_proba(self, data):
        """Test ensemble probability predictions."""
        X, y = data
        ensemble = ClassicalEnsemble()
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), len(ensemble.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_cross_validate(self, data):
        """Test ensemble cross-validation."""
        X, y = data
        ensemble = ClassicalEnsemble()

        results = ensemble.cross_validate(X, y, cv=3)
        assert "logistic" in results
        assert "random_forest" in results
        assert "svm" in results
        assert "ensemble_avg_test_acc" in results


class TestScGPTConfig:
    """Test ScGPTConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScGPTConfig()
        assert config.n_hvg == 3000
        assert config.n_bins == 51
        assert config.hidden_size == 512

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScGPTConfig(n_hvg=5000, n_bins=100, hidden_size=256)
        assert config.n_hvg == 5000
        assert config.n_bins == 100
        assert config.hidden_size == 256

    def test_invalid_n_hvg(self):
        """Test validation of n_hvg."""
        with pytest.raises(ValueError):
            ScGPTConfig(n_hvg=-1)

    def test_invalid_n_bins(self):
        """Test validation of n_bins."""
        with pytest.raises(ValueError):
            ScGPTConfig(n_bins=0)

    def test_device_setting(self):
        """Test device setting."""
        config = ScGPTConfig(device="cpu")
        assert config.device == "cpu"


class TestMultimodalFuser:
    """Test MultimodalFuser."""

    @pytest.fixture
    def embeddings(self):
        """Generate synthetic embeddings."""
        n_samples = 50
        embeddings_dict = {
            "genomics": np.random.randn(n_samples, 64),
            "imaging": np.random.randn(n_samples, 32),
            "clinical": np.random.randn(n_samples, 16),
        }
        return embeddings_dict

    @pytest.fixture
    def labels(self):
        """Generate synthetic labels."""
        return np.random.randint(0, 3, 50)

    def test_initialization(self):
        """Test MultimodalFuser initialization."""
        fuser = MultimodalFuser(fusion_method="concat")
        assert fuser.fusion_method == "concat"

    def test_invalid_method(self):
        """Test error on invalid fusion method."""
        with pytest.raises(ValueError):
            MultimodalFuser(fusion_method="invalid")

    def test_concat_fusion(self, embeddings):
        """Test concatenation fusion."""
        fuser = MultimodalFuser(fusion_method="concat")
        fused = fuser.fuse_embeddings(embeddings)

        expected_dim = sum(e.shape[1] for e in embeddings.values())
        assert fused.shape == (50, expected_dim)

    def test_attention_fusion(self, embeddings):
        """Test attention fusion."""
        fuser = MultimodalFuser(fusion_method="attention")
        fused = fuser.fuse_embeddings(embeddings)

        # All embeddings should be padded to same dimension
        assert fused.shape[0] == 50
        assert fused.shape[1] > 0

    def test_moe_fusion(self, embeddings):
        """Test MoE fusion."""
        fuser = MultimodalFuser(fusion_method="moe")
        fused = fuser.fuse_embeddings(embeddings)

        assert fused.shape[0] == 50
        assert fused.shape[1] > 0

    def test_fusion_empty_dict(self):
        """Test error on empty embeddings dict."""
        fuser = MultimodalFuser()
        with pytest.raises(ValueError):
            fuser.fuse_embeddings({})

    def test_fusion_shape_mismatch(self, embeddings):
        """Test error on mismatched shapes."""
        fuser = MultimodalFuser()
        embeddings["bad"] = np.random.randn(49, 32)  # Wrong n_samples

        with pytest.raises(ValueError):
            fuser.fuse_embeddings(embeddings)

    def test_train_fused_classifier(self, embeddings, labels):
        """Test training classifier on fused embeddings."""
        fuser = MultimodalFuser()
        fused = fuser.fuse_embeddings(embeddings)

        history = fuser.train_fused_classifier(fused, labels, model_type="logistic")
        assert "train_acc" in history

    def test_predict_fused(self, embeddings, labels):
        """Test prediction on fused embeddings."""
        fuser = MultimodalFuser()
        fused = fuser.fuse_embeddings(embeddings)

        fuser.train_fused_classifier(fused, labels)
        preds = fuser.predict_fused(fused)

        assert len(preds) == len(labels)
        assert all(p in np.unique(labels) for p in preds)

    def test_predict_proba_fused(self, embeddings, labels):
        """Test probability predictions on fused embeddings."""
        fuser = MultimodalFuser()
        fused = fuser.fuse_embeddings(embeddings)

        fuser.train_fused_classifier(fused, labels)
        probas = fuser.predict_proba_fused(fused)

        n_classes = len(np.unique(labels))
        assert probas.shape == (len(labels), n_classes)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_predict_before_train(self, embeddings):
        """Test error predicting before training."""
        fuser = MultimodalFuser()
        fused = fuser.fuse_embeddings(embeddings)

        with pytest.raises(ValueError):
            fuser.predict_fused(fused)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
