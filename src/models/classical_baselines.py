"""
Classical baseline models for Multiple Myeloma analysis.

Following the "classical baseline first" principle, this module implements
traditional ML approaches (logistic regression, random forest, SVM) that serve
as strong baselines against which modern foundation models are compared.

These models operate on standard features (PCA, HVG selection, etc.) and are
fast, interpretable, and require minimal hyperparameter tuning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class LogisticBaseline:
    """
    Multinomial Logistic Regression baseline for cell type classification.

    This is a simple, fast, and interpretable baseline that operates on
    PCA-reduced or HVG-selected features. Serves as a strong baseline
    against more complex models.

    Attributes:
        model: Underlying scikit-learn LogisticRegression instance.
        scaler: StandardScaler for feature normalization.
        n_features: Number of input features.
        classes_: Unique class labels.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize Logistic Regression baseline.

        Args:
            max_iter: Maximum iterations for solver. Default: 1000.
            solver: Optimization solver ('lbfgs', 'liblinear', 'newton-cg').
                Default: 'lbfgs'.
            random_state: Random seed. Default: 42.
            verbose: Verbosity level. Default: 0.
        """
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            verbose=verbose,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.n_features = None
        self.classes_ = None

        logger.info(
            f"Initialized LogisticBaseline: solver={solver}, max_iter={max_iter}"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticBaseline":
        """
        Fit logistic regression on training data.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X_train or y_train have incompatible shapes.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples: "
                f"{X_train.shape[0]} vs {y_train.shape[0]}"
            )

        if X_train.shape[0] == 0:
            raise ValueError("X_train must have at least 1 sample")

        self.n_features = X_train.shape[1]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.classes_ = self.model.classes_

        logger.info(
            f"Fitted LogisticBaseline on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features and {len(self.classes_)} classes"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).

        Raises:
            ValueError: If model not fitted or X_test shape incompatible.
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        if X_test.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_test.shape[1]}"
            )

        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        logger.debug(f"Predicted labels for {len(predictions)} samples")
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Class probabilities (n_samples, n_classes).
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).
            cv: Number of folds. Default: 5.

        Returns:
            Dictionary with cross-validation scores (train/test accuracy, etc.).
        """
        from sklearn.metrics import make_scorer

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
        }

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        results = cross_validate(
            self.model,
            self.scaler.fit_transform(X),
            y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
        )

        logger.info(
            f"Cross-validation (cv={cv}): "
            f"mean_test_acc={results['test_accuracy'].mean():.4f} "
            f"(+/- {results['test_accuracy'].std():.4f})"
        )
        return results


class RandomForestBaseline:
    """
    Random Forest classifier baseline for cell type annotation.

    Robust to feature scaling, handles non-linear relationships, and provides
    feature importance estimates. Good for interpretability and moderate-sized
    feature sets.

    Attributes:
        model: Underlying scikit-learn RandomForestClassifier.
        n_features: Number of input features.
        classes_: Unique class labels.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Random Forest baseline.

        Args:
            n_estimators: Number of trees. Default: 100.
            max_depth: Maximum tree depth. Default: None (unlimited).
            min_samples_split: Minimum samples to split. Default: 2.
            min_samples_leaf: Minimum samples per leaf. Default: 1.
            random_state: Random seed. Default: 42.
            n_jobs: Number of parallel jobs (-1 = all cores). Default: -1.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        )
        self.n_features = None
        self.classes_ = None

        logger.info(
            f"Initialized RandomForestBaseline: n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        """
        Fit random forest on training data.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X_train or y_train have incompatible shapes.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples"
            )

        if X_train.shape[0] == 0:
            raise ValueError("X_train must have at least 1 sample")

        self.n_features = X_train.shape[1]

        # Fit model (no scaling needed)
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        logger.info(
            f"Fitted RandomForestBaseline on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features and {len(self.classes_)} classes"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).

        Raises:
            ValueError: If model not fitted or X_test shape incompatible.
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        if X_test.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_test.shape[1]}"
            )

        predictions = self.model.predict(X_test)

        logger.debug(f"Predicted labels for {len(predictions)} samples")
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Class probabilities (n_samples, n_classes).
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X_test)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from fitted model.

        Returns:
            Series with feature importance values.

        Raises:
            ValueError: If model not fitted.
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model must be fitted before getting feature importance")

        return pd.Series(self.model.feature_importances_)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).
            cv: Number of folds. Default: 5.

        Returns:
            Dictionary with cross-validation scores.
        """
        from sklearn.metrics import make_scorer

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
        }

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        results = cross_validate(
            self.model, X, y, cv=cv_splitter, scoring=scoring, return_train_score=True
        )

        logger.info(
            f"Cross-validation (cv={cv}): "
            f"mean_test_acc={results['test_accuracy'].mean():.4f} "
            f"(+/- {results['test_accuracy'].std():.4f})"
        )
        return results


class SVMBaseline:
    """
    Support Vector Machine (SVM) baseline with RBF kernel.

    Good for high-dimensional data and non-linear classification. Requires
    feature scaling for optimal performance. More computationally expensive
    than logistic regression but often provides better accuracy.

    Attributes:
        model: Underlying scikit-learn SVC instance.
        scaler: StandardScaler for feature normalization.
        n_features: Number of input features.
        classes_: Unique class labels.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = "scale",
        kernel: str = "rbf",
        random_state: int = 42,
    ):
        """
        Initialize SVM baseline.

        Args:
            C: Regularization strength. Default: 1.0.
            gamma: Kernel coefficient ('scale', 'auto', or float). Default: 'scale'.
            kernel: Kernel type ('rbf', 'linear', 'poly'). Default: 'rbf'.
            random_state: Random seed. Default: 42.
        """
        self.model = SVC(
            C=C, gamma=gamma, kernel=kernel, probability=True,
            random_state=random_state, class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.n_features = None
        self.classes_ = None

        logger.info(f"Initialized SVMBaseline: kernel={kernel}, C={C}, gamma={gamma}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMBaseline":
        """
        Fit SVM on training data.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X_train or y_train have incompatible shapes.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples"
            )

        if X_train.shape[0] == 0:
            raise ValueError("X_train must have at least 1 sample")

        self.n_features = X_train.shape[1]

        # Scale features (critical for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.classes_ = self.model.classes_

        logger.info(
            f"Fitted SVMBaseline on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features and {len(self.classes_)} classes"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).

        Raises:
            ValueError: If model not fitted or X_test shape incompatible.
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        if X_test.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_test.shape[1]}"
            )

        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        logger.debug(f"Predicted labels for {len(predictions)} samples")
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on test data.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Class probabilities (n_samples, n_classes).
        """
        if self.model is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).
            cv: Number of folds. Default: 5.

        Returns:
            Dictionary with cross-validation scores.
        """
        from sklearn.metrics import make_scorer

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
        }

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Scale all data consistently
        X_scaled = self.scaler.fit_transform(X)

        results = cross_validate(
            self.model, X_scaled, y, cv=cv_splitter, scoring=scoring, return_train_score=True
        )

        logger.info(
            f"Cross-validation (cv={cv}): "
            f"mean_test_acc={results['test_accuracy'].mean():.4f} "
            f"(+/- {results['test_accuracy'].std():.4f})"
        )
        return results


class ClassicalEnsemble:
    """
    Ensemble of classical baselines using majority voting.

    Combines LogisticRegression, RandomForest, and SVM predictions via
    simple majority voting for improved robustness and generalization.

    Attributes:
        lr_model: LogisticBaseline instance.
        rf_model: RandomForestBaseline instance.
        svm_model: SVMBaseline instance.
    """

    def __init__(self):
        """Initialize ensemble with three classical baselines."""
        self.lr_model = LogisticBaseline()
        self.rf_model = RandomForestBaseline()
        self.svm_model = SVMBaseline()
        self.classes_ = None

        logger.info("Initialized ClassicalEnsemble")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalEnsemble":
        """
        Fit all ensemble members on training data.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.lr_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self.svm_model.fit(X_train, y_train)

        self.classes_ = self.lr_model.classes_

        logger.info(f"Fitted ClassicalEnsemble on {X_train.shape[0]} samples")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict via majority voting across ensemble members.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Predicted labels via majority vote (n_samples,).
        """
        if self.classes_ is None:
            raise ValueError("Ensemble must be fitted before prediction")

        lr_pred = self.lr_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        svm_pred = self.svm_model.predict(X_test)

        # Majority voting
        predictions = np.zeros(len(X_test), dtype=object)
        for i in range(len(X_test)):
            votes = [lr_pred[i], rf_pred[i], svm_pred[i]]
            # Find most common prediction
            from collections import Counter

            predictions[i] = Counter(votes).most_common(1)[0][0]

        logger.debug(f"Predicted labels for {len(predictions)} samples via ensemble")
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Average class probabilities across ensemble members.

        Args:
            X_test: Test features (n_samples, n_features).

        Returns:
            Averaged class probabilities (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise ValueError("Ensemble must be fitted before prediction")

        lr_proba = self.lr_model.predict_proba(X_test)
        rf_proba = self.rf_model.predict_proba(X_test)
        svm_proba = self.svm_model.predict_proba(X_test)

        avg_proba = (lr_proba + rf_proba + svm_proba) / 3.0

        return avg_proba

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Cross-validate all ensemble members.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).
            cv: Number of folds. Default: 5.

        Returns:
            Dictionary with CV results for each member and ensemble average.
        """
        lr_results = self.lr_model.cross_validate(X, y, cv=cv)
        rf_results = self.rf_model.cross_validate(X, y, cv=cv)
        svm_results = self.svm_model.cross_validate(X, y, cv=cv)

        ensemble_avg_test = (
            lr_results["test_accuracy"]
            + rf_results["test_accuracy"]
            + svm_results["test_accuracy"]
        ) / 3.0

        logger.info(
            f"Ensemble CV: mean_test_acc={ensemble_avg_test.mean():.4f} "
            f"(+/- {ensemble_avg_test.std():.4f})"
        )

        return {
            "logistic": lr_results,
            "random_forest": rf_results,
            "svm": svm_results,
            "ensemble_avg_test_acc": ensemble_avg_test,
        }

    def save(self, path: str) -> None:
        """Save ensemble models to disk using joblib."""
        import joblib
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.lr_model, save_dir / "logistic.joblib")
        joblib.dump(self.rf_model, save_dir / "random_forest.joblib")
        joblib.dump(self.svm_model, save_dir / "svm.joblib")
        logger.info(f"Ensemble saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "ClassicalEnsemble":
        """Load ensemble models from disk."""
        import joblib
        from pathlib import Path

        load_dir = Path(path)
        ensemble = cls()
        ensemble.lr_model = joblib.load(load_dir / "logistic.joblib")
        ensemble.rf_model = joblib.load(load_dir / "random_forest.joblib")
        ensemble.svm_model = joblib.load(load_dir / "svm.joblib")
        ensemble.classes_ = ensemble.lr_model.classes_
        logger.info(f"Ensemble loaded from {load_dir}")
        return ensemble
