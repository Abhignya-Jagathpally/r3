"""
Core agentic hyperparameter search runner.

This module implements the main experiment loop that conducts constrained
hyperparameter search within fixed budgets and with frozen preprocessing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from anndata import AnnData
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

from .config import AgenticConfig
from .contract_enforcer import ContractEnforcer
from .search_space import SearchSpace

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiment metrics and results.

    Attributes:
        log: DataFrame with experiment history
        best_metric: Best metric value seen so far
        best_config: Configuration for best metric
    """

    def __init__(self, primary_metric: str):
        """Initialize experiment tracker.

        Args:
            primary_metric: Name of primary metric to optimize.
        """
        self.primary_metric = primary_metric
        self.log = pd.DataFrame()
        self.best_metric = None
        self.best_config = None
        self._update_count = 0

    def log_trial(
        self,
        trial_id: int,
        config: Dict,
        metric_value: float,
        wallclock_time: float,
        gpu_memory_mb: float = 0.0,
        **kwargs
    ) -> None:
        """Log a trial result.

        Args:
            trial_id: Trial number.
            config: Configuration dictionary.
            metric_value: Primary metric value.
            wallclock_time: Time taken for trial in seconds.
            gpu_memory_mb: GPU memory used in MB.
            **kwargs: Additional metrics to log.
        """
        record = {
            "trial_id": trial_id,
            "timestamp": pd.Timestamp.now(),
            "wallclock_time_sec": wallclock_time,
            "gpu_memory_mb": gpu_memory_mb,
            self.primary_metric: metric_value,
            **kwargs,
        }

        # Flatten config into separate columns
        for key, val in config.items():
            record[f"config_{key}"] = val

        self.log = pd.concat(
            [self.log, pd.DataFrame([record])],
            ignore_index=True
        )

        # Update best
        if self.best_metric is None or metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_config = config.copy()
            self._update_count += 1

        logger.info(
            f"Trial {trial_id}: {self.primary_metric}={metric_value:.4f} "
            f"(best={self.best_metric:.4f}, time={wallclock_time:.1f}s)"
        )

    def get_log(self) -> pd.DataFrame:
        """Get experiment log DataFrame.

        Returns:
            DataFrame with all trial results.
        """
        return self.log.copy()

    def get_best(self) -> Tuple[float, Dict]:
        """Get best result so far.

        Returns:
            Tuple of (best_metric_value, best_config).
        """
        return self.best_metric, self.best_config

    def early_stop_check(self, patience: int = 10) -> bool:
        """Check if early stopping criterion is met.

        Stops if no improvement in the last `patience` trials.

        Args:
            patience: Number of trials without improvement to trigger stopping.

        Returns:
            True if should stop, False otherwise.
        """
        if len(self.log) < patience:
            return False

        recent = self.log[self.primary_metric].tail(patience)
        best_in_recent = recent.max()
        best_overall = self.log[self.primary_metric].max()

        return best_in_recent < best_overall


class ExperimentRunner:
    """Run agentic hyperparameter search with fixed budgets.

    Implements the core search loop following Karpathy's autoresearch pattern:
    - One primary metric
    - Fixed search budget
    - Frozen preprocessing
    - Full experiment logging

    Attributes:
        config: Agentic configuration
        search_space: Hyperparameter search space
        tracker: Experiment tracker
        contract_enforcer: Contract enforcement for preprocessing
    """

    def __init__(
        self,
        config: AgenticConfig,
        data: Optional["AnnData"] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        """Initialize experiment runner.

        Args:
            config: AgenticConfig instance.
            data: AnnData object (optional, for contract verification).
            tracker: ExperimentTracker instance (created if not provided).
        """
        self.config = config
        self.data = data
        self.tracker = tracker or ExperimentTracker(config.primary_metric)
        self.search_space = SearchSpace(config.editable_surface)
        self.contract_enforcer = ContractEnforcer(config.preprocessing_contract_path)
        self.start_time = None

    def run_search(
        self,
        strategy: str = "bayesian",
        patience: int = 10,
    ) -> pd.DataFrame:
        """Run full hyperparameter search within budget.

        Executes the main search loop:
        1. Verify preprocessing contract
        2. For each trial up to search_budget:
           a. Sample config
           b. Validate config
           c. Train model
           d. Evaluate and log
           e. Check budgets
           f. Early stopping check

        Args:
            strategy: Search strategy ('random' or 'bayesian').
            patience: Patience for early stopping.

        Returns:
            DataFrame with all trial results sorted by metric.

        Raises:
            ImportError: If required dependencies missing.
            ValueError: If config validation fails.
        """
        if not HAS_ANNDATA and self.data is not None:
            logger.warning("anndata not available, skipping contract verification")
        elif self.data is not None:
            is_valid, msg = self.contract_enforcer.verify_data_integrity(self.data)
            if not is_valid:
                raise ValueError(f"Data contract violation: {msg}")

        # Verify frozen modules
        is_valid, msg = self.contract_enforcer.verify_frozen_modules(
            self.config.editable_surface,
            self.config.frozen_modules
        )
        if not is_valid:
            raise ValueError(f"Frozen module violation: {msg}")

        self.start_time = time.time()
        history = []  # For Bayesian optimization

        trial_id = 0
        while trial_id < self.config.search_budget:
            # Check wallclock budget
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.config.max_wallclock_hours:
                logger.info(
                    f"Reached wallclock budget ({elapsed_hours:.2f}h >= "
                    f"{self.config.max_wallclock_hours}h)"
                )
                break

            # Sample configuration
            if strategy == "bayesian" and history:
                config = self.search_space.sample_config_bayesian(
                    history,
                    direction="maximize"
                )
            else:
                config = self.search_space.sample_config()

            # Validate configuration
            is_valid, error_msg = self.search_space.validate_config(config)
            if not is_valid:
                logger.warning(f"Invalid config: {error_msg}, skipping")
                continue

            # Run trial
            trial_start = time.time()
            metric_value = self.run_single_experiment(config)
            wallclock_time = time.time() - trial_start

            # Log trial
            self.tracker.log_trial(
                trial_id=trial_id,
                config=config,
                metric_value=metric_value,
                wallclock_time=wallclock_time,
            )

            # Add to history for Bayesian optimization
            history.append((config, metric_value))

            # Early stopping check
            if self.tracker.early_stop_check(patience=patience):
                logger.info(
                    f"Early stopping after {trial_id + 1} trials "
                    f"(patience={patience})"
                )
                break

            trial_id += 1

        # Return sorted leaderboard
        log = self.tracker.get_log()
        return log.sort_values(
            by=self.config.primary_metric,
            ascending=False
        )

    def run_single_experiment(self, trial_config: Dict) -> float:
        """Run single experiment: re-cluster with trial params and compute metric.

        Args:
            trial_config: Configuration dictionary with param overrides.

        Returns:
            Primary metric value (silhouette_score by default).
        """
        import scanpy as sc
        from sklearn.metrics import silhouette_score

        if self.data is None:
            raise ValueError("No data provided for experiment")

        adata = self.data.copy()

        # Apply trial config parameters
        resolution = trial_config.get("clustering.leiden.resolution", 1.0)
        n_neighbors = trial_config.get("clustering.umap.n_neighbors", 15)

        # Determine embedding
        embed_key = "X_harmony" if "X_harmony" in adata.obsm else "X_pca"
        if embed_key not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
            embed_key = "X_pca"

        # Compute neighbors and cluster
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embed_key)
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden_trial")

        # Compute metric
        X = adata.obsm[embed_key]
        labels = adata.obs["leiden_trial"].values

        n_unique = len(set(labels))
        if n_unique < 2 or n_unique >= adata.n_obs:
            return -1.0  # Invalid clustering

        # Subsample for speed if dataset is large
        if adata.n_obs > 20000:
            import numpy as np
            idx = np.random.choice(adata.n_obs, 20000, replace=False)
            score = silhouette_score(X[idx], labels[idx], sample_size=None)
        else:
            score = silhouette_score(X, labels)

        logger.info(
            f"Trial: resolution={resolution}, n_neighbors={n_neighbors}, "
            f"clusters={n_unique}, silhouette={score:.4f}"
        )
        return float(score)

    def get_best_config(self) -> Dict:
        """Get best configuration found.

        Returns:
            Configuration dictionary for best trial.
        """
        _, config = self.tracker.get_best()
        return config

    def get_best_metric(self) -> float:
        """Get best metric value found.

        Returns:
            Best metric value.
        """
        metric, _ = self.tracker.get_best()
        return metric

    def get_experiment_log(self) -> pd.DataFrame:
        """Get full experiment log.

        Returns:
            DataFrame with all trial results.
        """
        return self.tracker.get_log()

    def get_leaderboard(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N trials by metric.

        Args:
            top_n: Number of top trials to return.

        Returns:
            DataFrame with top N trials.
        """
        log = self.get_experiment_log()
        return log.nlargest(top_n, self.config.primary_metric)

    def save_results(self, output_dir: str) -> None:
        """Save experiment results to directory.

        Args:
            output_dir: Directory to save results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save log
        log = self.get_experiment_log()
        log.to_csv(output_path / "experiment_log.csv", index=False)

        # Save best config
        best_config = self.get_best_config()
        if best_config:
            import json
            with open(output_path / "best_config.json", "w") as f:
                json.dump(best_config, f, indent=2)

        logger.info(f"Saved results to {output_dir}")
