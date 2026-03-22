"""
Ray Tune integration for parallel hyperparameter search.

This module provides distributed hyperparameter tuning using Ray Tune,
supporting multiple trial execution and early stopping strategies.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .search_space import SearchSpace

logger = logging.getLogger(__name__)

try:
    import ray
    from ray import tune
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class RayTuner:
    """Parallel hyperparameter search using Ray Tune.

    Provides distributed trial execution with ASHA early stopping and support
    for both random and Bayesian hyperparameter search strategies.

    Attributes:
        search_space: SearchSpace instance
        config: Ray Tune search configuration
    """

    def __init__(self, search_space: SearchSpace):
        """Initialize Ray Tuner.

        Args:
            search_space: SearchSpace instance defining parameter ranges.

        Raises:
            ImportError: If Ray is not installed.
        """
        if not HAS_RAY:
            raise ImportError("ray required for RayTuner")

        self.search_space = search_space
        self.config = None
        self.result_grid = None

    def setup_tune_config(self) -> Dict[str, Any]:
        """Convert search space to Ray Tune format.

        Converts SearchSpace parameter definitions to Ray Tune's
        search space format with appropriate distributions.

        Returns:
            Dictionary with Ray Tune configuration.
        """
        tune_config = {}

        # scVI parameters
        tune_config["n_latent"] = tune.randint(
            self.search_space.scvi_params["n_latent"][0],
            self.search_space.scvi_params["n_latent"][1]
        )
        tune_config["n_hidden"] = tune.randint(
            self.search_space.scvi_params["n_hidden"][0],
            self.search_space.scvi_params["n_hidden"][1]
        )
        tune_config["n_layers"] = tune.randint(
            self.search_space.scvi_params["n_layers"][0],
            self.search_space.scvi_params["n_layers"][1]
        )
        tune_config["learning_rate"] = tune.loguniform(
            self.search_space.scvi_params["learning_rate"][0],
            self.search_space.scvi_params["learning_rate"][1]
        )
        tune_config["dropout_rate"] = tune.uniform(
            self.search_space.scvi_params["dropout_rate"][0],
            self.search_space.scvi_params["dropout_rate"][1]
        )
        tune_config["gene_likelihood"] = tune.choice(
            self.search_space.scvi_params["gene_likelihood"]
        )

        # Integration parameters
        tune_config["integration_method"] = tune.choice(
            self.search_space.integration_params["method"]
        )
        tune_config["hvg_count"] = tune.randint(
            self.search_space.integration_params["hvg_count"][0],
            self.search_space.integration_params["hvg_count"][1]
        )

        # Fusion parameters
        tune_config["fusion_method"] = tune.choice(
            self.search_space.fusion_params["method"]
        )
        tune_config["fusion_hidden_dim"] = tune.randint(
            self.search_space.fusion_params["fusion_hidden_dim"][0],
            self.search_space.fusion_params["fusion_hidden_dim"][1]
        )

        self.config = tune_config
        logger.info("Set up Ray Tune configuration")
        return tune_config

    def run_parallel_search(
        self,
        trainable: Callable,
        n_trials: int = 10,
        n_gpus: float = 1.0,
        n_cpus: int = 4,
        metric: str = "bio_conservation",
        mode: str = "max",
        asha_grace_period: int = 5,
    ) -> Optional[Any]:
        """Run parallel hyperparameter search with Ray Tune.

        Executes parallel trials with ASHA scheduler for early stopping.

        Args:
            trainable: Training function or Trainable class.
            n_trials: Number of trials to run.
            n_gpus: GPU fraction per trial.
            n_cpus: CPU cores per trial.
            metric: Metric name to optimize.
            mode: Optimization direction ('max' or 'min').
            asha_grace_period: Number of iterations before ASHA pruning.

        Returns:
            Ray ResultGrid with results.

        Raises:
            ImportError: If Ray is not installed.
            ValueError: If trainable is invalid.
        """
        if not HAS_RAY:
            raise ImportError("ray required for RayTuner")

        if not self.config:
            self.setup_tune_config()

        if not self.config:
            raise ValueError("Could not set up Ray Tune configuration")

        try:
            # ASHA scheduler for early stopping
            asha_scheduler = tune.ASHAScheduler(
                time_attr="training_iteration",
                metric=metric,
                mode=mode,
                max_t=100,  # max iterations
                grace_period=asha_grace_period,
                reduction_factor=2,
            )

            # Run tuning
            self.result_grid = tune.run(
                trainable,
                name="agentic_search",
                config=self.config,
                num_samples=n_trials,
                scheduler=asha_scheduler,
                verbose=1,
                progress_reporter=tune.CLIReporter(
                    metric_columns=[metric]
                ),
                resources_per_trial={
                    "gpu": n_gpus,
                    "cpu": n_cpus,
                },
            )

            logger.info(f"Completed {n_trials} trials with Ray Tune")
            return self.result_grid

        except Exception as e:
            logger.error(f"Ray Tune search failed: {e}")
            raise

    def get_best_trial(self) -> Optional[Dict]:
        """Get best trial from completed search.

        Returns:
            Dictionary with best trial configuration and metrics.

        Raises:
            ValueError: If no search has been run.
        """
        if self.result_grid is None:
            raise ValueError("No completed search. Run run_parallel_search() first.")

        best_result = self.result_grid.get_best_result()

        if best_result is None:
            logger.warning("No valid trials found")
            return None

        return {
            "config": best_result.config,
            "metrics": best_result.metrics,
            "logdir": best_result.logdir,
        }

    def get_results_dataframe(self) -> Optional[Any]:
        """Get results as pandas DataFrame.

        Returns:
            DataFrame with all trial results.

        Raises:
            ValueError: If no search has been run.
        """
        if self.result_grid is None:
            raise ValueError("No completed search. Run run_parallel_search() first.")

        try:
            return self.result_grid.results_df
        except AttributeError:
            logger.warning("Could not extract DataFrame from results")
            return None
