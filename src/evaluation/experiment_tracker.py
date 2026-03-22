"""
Experiment tracking with MLflow and Weights & Biases support.

Provides unified interface for logging metrics, parameters, models, and artifacts
across different tracking backends. Enables systematic experiment management,
model versioning, and easy comparison across runs.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class BaseTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new experiment run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metric values."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        """Log an artifact (file)."""
        pass

    @abstractmethod
    def log_model(self, model: Any, name: str):
        """Log a model."""
        pass

    @abstractmethod
    def end_run(self):
        """End the current run."""
        pass


class MLflowTracker(BaseTracker):
    """MLflow experiment tracker backend."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking URI. Default: None (local).
            experiment_name: Experiment name. Default: None.
        """
        try:
            import mlflow

            self.mlflow = mlflow
            self.run_id = None
            self.current_experiment = experiment_name
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            logger.info(f"Initialized MLflow tracker (experiment: {experiment_name})")
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start MLflow run."""
        self.mlflow.start_run(run_name=run_name)
        if tags:
            for key, value in tags.items():
                self.mlflow.set_tag(key, value)
        self.run_id = self.mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_name} (id: {self.run_id})")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            try:
                self.mlflow.log_metric(key, float(value))
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        for key, value in params.items():
            try:
                self.mlflow.log_param(key, str(value))
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    def log_artifact(self, path: str):
        """Log artifact file to MLflow."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        try:
            if path.is_dir():
                self.mlflow.log_artifacts(str(path))
            else:
                self.mlflow.log_artifact(str(path))
            logger.debug(f"Logged artifact: {path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_model(self, model: Any, name: str):
        """Log model to MLflow (as artifact)."""
        try:
            import pickle

            model_path = Path(f"/tmp/{name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            self.mlflow.log_artifact(str(model_path), artifact_path="models")
            logger.info(f"Logged model: {name}")
        except Exception as e:
            logger.warning(f"Failed to log model {name}: {e}")

    def end_run(self):
        """End MLflow run."""
        self.mlflow.end_run()
        logger.info(f"Ended MLflow run: {self.run_id}")

    def compare_runs(self, metric_name: str) -> pd.DataFrame:
        """
        Compare runs by a specific metric.

        Args:
            metric_name: Name of metric to compare.

        Returns:
            DataFrame with run comparison.
        """
        try:
            experiment = self.mlflow.get_experiment_by_name(self.current_experiment)
            runs = self.mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            # Extract metric for each run
            run_metrics = []
            for _, run in runs.iterrows():
                if metric_name in run["metrics"]:
                    run_metrics.append(
                        {
                            "run_id": run["run_id"],
                            "run_name": run.get("tags.mlflow.runName", "N/A"),
                            metric_name: run["metrics"][metric_name],
                        }
                    )

            if run_metrics:
                df = pd.DataFrame(run_metrics)
                df = df.sort_values(metric_name, ascending=False)
                logger.info(f"Compared {len(df)} runs by {metric_name}")
                return df
            else:
                logger.warning(f"No runs found with metric {metric_name}")
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to compare runs: {e}")
            return pd.DataFrame()


class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker backend."""

    def __init__(self, project: str = "mm-single-cell", entity: Optional[str] = None):
        """
        Initialize W&B tracker.

        Args:
            project: W&B project name. Default: 'mm-single-cell'.
            entity: W&B entity/team name. Default: None (use default).
        """
        try:
            import wandb

            self.wandb = wandb
            self.project = project
            self.entity = entity
            self.run = None
            logger.info(f"Initialized W&B tracker (project: {project})")
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start W&B run."""
        config = tags if tags else {}
        self.run = self.wandb.init(
            project=self.project, entity=self.entity, name=run_name, config=config
        )
        logger.info(f"Started W&B run: {run_name}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to W&B."""
        if self.run is None:
            logger.warning("No active W&B run")
            return

        try:
            self.wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to W&B."""
        if self.run is None:
            logger.warning("No active W&B run")
            return

        try:
            self.run.config.update(params)
        except Exception as e:
            logger.warning(f"Failed to log params to W&B: {e}")

    def log_artifact(self, path: str):
        """Log artifact to W&B."""
        if self.run is None:
            logger.warning("No active W&B run")
            return

        try:
            artifact = self.wandb.Artifact("run_artifact", type="dataset")
            artifact.add_file(path)
            self.run.log_artifact(artifact)
            logger.debug(f"Logged artifact to W&B: {path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact to W&B: {e}")

    def log_model(self, model: Any, name: str):
        """Log model to W&B."""
        if self.run is None:
            logger.warning("No active W&B run")
            return

        try:
            import pickle

            model_path = Path(f"/tmp/{name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            artifact = self.wandb.Artifact(name, type="model")
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)
            logger.info(f"Logged model to W&B: {name}")
        except Exception as e:
            logger.warning(f"Failed to log model to W&B: {e}")

    def end_run(self):
        """End W&B run."""
        if self.run is None:
            logger.warning("No active W&B run")
            return

        self.run.finish()
        logger.info("Ended W&B run")

    def compare_runs(self, metric_name: str) -> pd.DataFrame:
        """
        Compare runs by a specific metric.

        Not directly supported in W&B API; user should use W&B UI.

        Args:
            metric_name: Metric name (for reference).

        Returns:
            Empty DataFrame with note.
        """
        logger.info(
            f"W&B comparison: use W&B UI to compare runs. "
            f"(Metric: {metric_name})"
        )
        return pd.DataFrame()


class ExperimentTracker:
    """
    Unified experiment tracker supporting MLflow and W&B backends.

    Provides a single interface for logging experiments to different
    tracking systems without changing client code.

    Attributes:
        backend: Underlying tracker (MLflow or W&B).
        backend_type: Type of backend ('mlflow' or 'wandb').
    """

    def __init__(
        self,
        backend: str = "mlflow",
        mlflow_tracking_uri: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            backend: Backend to use ('mlflow' or 'wandb'). Default: 'mlflow'.
            mlflow_tracking_uri: MLflow tracking server URI. Default: None (local).
            wandb_project: W&B project name. Default: None (use default).

        Raises:
            ValueError: If backend is unsupported.
        """
        if backend not in ["mlflow", "wandb"]:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend_type = backend

        if backend == "mlflow":
            if mlflow_tracking_uri:
                try:
                    import mlflow

                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                except Exception as e:
                    logger.warning(f"Could not set MLflow tracking URI: {e}")

            self.backend = MLflowTracker()
        else:  # wandb
            self.backend = WandBTracker(project=wandb_project or "mm-single-cell")

        logger.info(f"Initialized ExperimentTracker with backend: {backend}")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a new experiment run.

        Args:
            run_name: Name for the run.
            tags: Optional tags/metadata dictionary.
        """
        self.backend.start_run(run_name, tags)

    def log_metrics(self, metrics_dict: Dict[str, float]):
        """
        Log metrics from a dictionary.

        Args:
            metrics_dict: Dictionary of metric_name -> value.
        """
        self.backend.log_metrics(metrics_dict)

    def log_params(self, params_dict: Dict[str, Any]):
        """
        Log hyperparameters from a dictionary.

        Args:
            params_dict: Dictionary of param_name -> value.
        """
        self.backend.log_params(params_dict)

    def log_artifact(self, path: str):
        """
        Log an artifact file.

        Args:
            path: Path to file or directory.
        """
        self.backend.log_artifact(path)

    def log_model(self, model: Any, name: str):
        """
        Log a trained model.

        Args:
            model: Model object (any picklable object).
            name: Name for the model.
        """
        self.backend.log_model(model, name)

    def end_run(self):
        """End the current experiment run."""
        self.backend.end_run()

    def compare_runs(self, metric_name: str) -> pd.DataFrame:
        """
        Compare multiple runs by a metric.

        Args:
            metric_name: Metric to compare across runs.

        Returns:
            DataFrame with run comparison.
        """
        return self.backend.compare_runs(metric_name)

    def log_benchmark_results(
        self, results_dict: Dict[str, Union[float, Dict]], name: str = "benchmark_results"
    ):
        """
        Log benchmark results in standard format.

        Args:
            results_dict: Benchmark results (nested dicts OK).
            name: Name for logging. Default: 'benchmark_results'.
        """
        # Flatten nested dicts for logging
        flat_results = {}

        def flatten_dict(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                elif isinstance(v, (int, float)):
                    flat_results[new_key] = v

        flatten_dict(results_dict)

        logger.info(f"Logging benchmark results: {flat_results}")
        self.log_metrics(flat_results)

    def log_config_yaml(self, config_path: str):
        """
        Log experiment configuration from YAML file.

        Args:
            config_path: Path to YAML config file.
        """
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            self.log_params(config)
            logger.info(f"Logged config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to log config: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
