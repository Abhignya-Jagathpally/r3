"""
Dask distributed tuning for hyperparameter search on clusters without Ray.

This module provides hyperparameter optimization using Dask for clusters
or systems where Ray is not available.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .search_space import SearchSpace

logger = logging.getLogger(__name__)

try:
    from dask.distributed import Client
    from dask import delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class DaskTuner:
    """Distributed hyperparameter search using Dask.

    Provides distributed trial execution on Dask clusters without requiring
    Ray. Useful for HPC and cloud environments with existing Dask clusters.

    Attributes:
        search_space: SearchSpace instance
        client: Dask distributed client
        results: DataFrame with trial results
    """

    def __init__(self, search_space: SearchSpace):
        """Initialize Dask Tuner.

        Args:
            search_space: SearchSpace instance defining parameter ranges.

        Raises:
            ImportError: If Dask is not installed.
        """
        if not HAS_DASK:
            raise ImportError("dask[distributed] required for DaskTuner")

        self.search_space = search_space
        self.client = None
        self.results = None

    def setup_client(
        self,
        n_workers: int = 4,
        memory_limit: str = "4GB",
        scheduler: str = "threads",
    ) -> "Client":
        """Set up Dask client for distributed computation.

        Args:
            n_workers: Number of worker processes.
            memory_limit: Memory limit per worker.
            scheduler: Scheduler type ('threads', 'processes', 'synchronous').

        Returns:
            Dask Client instance.

        Raises:
            ImportError: If Dask is not installed.
        """
        if not HAS_DASK:
            raise ImportError("dask[distributed] required")

        if scheduler == "threads":
            # LocalCluster with threads
            try:
                from dask.distributed import LocalCluster
                cluster = LocalCluster(
                    n_workers=n_workers,
                    memory_limit=memory_limit,
                    threads_per_worker=2,
                )
                self.client = Client(cluster)
            except Exception as e:
                logger.warning(f"Failed to create LocalCluster: {e}")
                logger.info("Falling back to synchronous scheduler")
                from dask import config
                config.set(scheduler='synchronous')
        else:
            logger.info(f"Using {scheduler} scheduler")

        logger.info(f"Dask client set up with {n_workers} workers")
        return self.client

    def run_distributed_search(
        self,
        trainable: Callable,
        n_trials: int = 10,
        strategy: str = "bayesian",
    ) -> pd.DataFrame:
        """Run distributed hyperparameter search.

        Executes trials in parallel using Dask delayed tasks.

        Args:
            trainable: Training function that takes config dict and returns metric.
            n_trials: Number of trials to run.
            strategy: Search strategy ('random' or 'bayesian').

        Returns:
            DataFrame with all trial results.

        Raises:
            ValueError: If trainable is invalid.
        """
        if not callable(trainable):
            raise ValueError("trainable must be callable")

        logger.info(f"Starting distributed search with {n_trials} trials")

        # Generate configurations
        configs = []
        history = []

        for trial_id in range(n_trials):
            if strategy == "bayesian" and history:
                config = self.search_space.sample_config_bayesian(
                    history,
                    direction="maximize"
                )
            else:
                config = self.search_space.sample_config()

            # Validate config
            is_valid, msg = self.search_space.validate_config(config)
            if not is_valid:
                logger.warning(f"Trial {trial_id}: Invalid config: {msg}")
                continue

            configs.append((trial_id, config))

        # Create delayed tasks
        delayed_tasks = [
            delayed(trainable)(config)
            for trial_id, config in configs
        ]

        # Compute results
        try:
            results = delayed(lambda *args: args)(*delayed_tasks).compute()

            # Parse results
            trial_results = []
            for (trial_id, config), metric in zip(configs, results):
                if metric is not None:
                    trial_results.append({
                        "trial_id": trial_id,
                        "metric": metric,
                        **{f"config_{k}": v for k, v in config.items()},
                    })
                    history.append((config, metric))

            self.results = pd.DataFrame(trial_results)
            logger.info(f"Completed {len(trial_results)} trials")
            return self.results

        except Exception as e:
            logger.error(f"Distributed search failed: {e}")
            raise

    def scatter_data(self, data: Any, key: str = "data") -> str:
        """Scatter data to Dask workers for shared access.

        Distributes large objects to workers to avoid repeated
        serialization and network transfer.

        Args:
            data: Data object to scatter.
            key: Key to store data under in worker cache.

        Returns:
            Key for retrieving scattered data from workers.

        Raises:
            RuntimeError: If no client is connected.
        """
        if self.client is None:
            raise RuntimeError("No Dask client connected. Call setup_client() first.")

        try:
            scattered = self.client.scatter(data, broadcast=True)
            logger.info(f"Scattered data to workers: {key}")
            return key
        except Exception as e:
            logger.error(f"Failed to scatter data: {e}")
            raise

    def get_results(self) -> Optional[pd.DataFrame]:
        """Get results from completed search.

        Returns:
            DataFrame with all trial results.
        """
        return self.results

    def close(self) -> None:
        """Close Dask client and clean up resources."""
        if self.client is not None:
            self.client.close()
            logger.info("Closed Dask client")
