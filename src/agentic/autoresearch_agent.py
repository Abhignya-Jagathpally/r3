"""
High-level AutoResearch agent following Karpathy's autoresearch pattern.

This module implements the complete agentic pipeline orchestrator, combining
search space, experiment runner, and reporting to enable end-to-end
hyperparameter optimization with constrained budgets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from anndata import AnnData
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

from .config import AgenticConfig, SearchSpaceConfig, TunerConfig
from .contract_enforcer import ContractEnforcer
from .experiment_runner import ExperimentRunner, ExperimentTracker
from .report_generator import ReportGenerator
from .search_space import SearchSpace

logger = logging.getLogger(__name__)


class AutoResearchResult:
    """Result of AutoResearch pipeline execution.

    Attributes:
        best_config: Best configuration found
        best_score: Best metric value achieved
        best_params: Best parameter values
        total_trials: Total trials run
        successful_trials: Number of successful trials
        experiment_log: Full experiment history
        report: Markdown report of results
        output_dir: Directory where results are saved
    """

    def __init__(
        self,
        best_config: Dict = None,
        best_score: float = 0.0,
        best_params: Dict = None,
        total_trials: int = 0,
        successful_trials: int = 0,
        experiment_log: pd.DataFrame = None,
        report: str = "",
        output_dir: str = "",
        best_metric: float = None,
    ):
        """Initialize AutoResearchResult."""
        self.best_config = best_config or {}
        self.best_score = best_metric if best_metric is not None else best_score
        self.best_params = best_params or best_config or {}
        self.total_trials = total_trials
        self.successful_trials = successful_trials
        self.experiment_log = experiment_log if experiment_log is not None else pd.DataFrame()
        self.report = report
        self.output_dir = output_dir

    @property
    def best_metric(self):
        """Alias for best_score."""
        return self.best_score

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AutoResearchResult(score={self.best_score:.4f}, "
            f"trials={self.total_trials}, output_dir={self.output_dir})"
        )


class AutoResearchAgent:
    """High-level agent for constrained hyperparameter search.

    Implements Karpathy's autoresearch pattern with:
    - One primary metric
    - Fixed search budget
    - Frozen preprocessing
    - Full experiment logging
    - Automated reporting

    Attributes:
        config: AgenticConfig instance
        pipeline_dir: Root directory of pipeline
        search_space: Hyperparameter search space
        contract_enforcer: Contract enforcement
        tracker: Experiment tracker
    """

    def __init__(
        self,
        config: AgenticConfig = None,
        pipeline_dir: str = ".",
    ):
        """Initialize AutoResearch agent.

        Args:
            config: AgenticConfig instance.
            pipeline_dir: Root directory of R3-MM pipeline.
        """
        self.config = config or AgenticConfig()
        self.pipeline_dir = Path(pipeline_dir)
        self.search_space = SearchSpace(self.config.editable_surface)
        self.contract_enforcer = ContractEnforcer(self.config.frozen_modules)
        self.tracker = ExperimentTracker(self.config.optimization_metric)

        # Create output directory
        self.output_dir = self.pipeline_dir / self.config.experiment_log_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized AutoResearchAgent for {pipeline_dir}")

    def run(
        self,
        data: Optional["AnnData"] = None,
        strategy: str = "bayesian",
        tuner_backend: str = "sequential",
        patience: int = 10,
    ) -> AutoResearchResult:
        """Run complete AutoResearch pipeline.

        Orchestrates:
        1. Load and verify preprocessing contract
        2. Define search space
        3. Run hyperparameter search
        4. Analyze results
        5. Generate report

        Args:
            data: AnnData object with preprocessed data.
            strategy: Search strategy ('random' or 'bayesian').
            tuner_backend: Tuning backend ('ray', 'dask', 'sequential').
            patience: Patience for early stopping.

        Returns:
            AutoResearchResult with best config and report.

        Raises:
            ValueError: If preprocessing contract is violated.
        """
        logger.info("Starting AutoResearch pipeline")

        # Verify preprocessing contract
        if HAS_ANNDATA and data is not None:
            is_valid, msg = self.contract_enforcer.verify_data_integrity(data)
            if not is_valid:
                raise ValueError(f"Preprocessing contract violated: {msg}")
            logger.info("Preprocessing contract verified")

        # Create experiment runner
        runner = self._create_runner(data, tuner_backend)

        # Run search
        logger.info(
            f"Starting hyperparameter search (budget={self.config.search_budget}, "
            f"time={self.config.max_wallclock_hours}h)"
        )
        experiment_log = runner.run_search(
            strategy=strategy,
            patience=patience,
        )

        # Get best results
        best_config = runner.get_best_config()
        best_metric = runner.get_best_metric()

        logger.info(
            f"Search complete: {self.config.primary_metric}={best_metric:.4f}"
        )

        # Generate report
        report = self._generate_report(experiment_log, best_config)

        # Save results
        self._save_results(experiment_log, best_config, report)

        result = AutoResearchResult(
            best_config=best_config,
            best_score=best_metric,
            best_params=best_config,
            total_trials=len(experiment_log) if experiment_log is not None else 0,
            successful_trials=len(experiment_log) if experiment_log is not None else 0,
            experiment_log=experiment_log,
            report=report,
            output_dir=str(self.output_dir),
        )

        logger.info(f"AutoResearch complete: {result}")
        return result

    def _create_runner(
        self,
        data: Optional["AnnData"],
        backend: str,
    ) -> ExperimentRunner:
        """Create experiment runner for given backend.

        Args:
            data: AnnData object.
            backend: Tuning backend.

        Returns:
            ExperimentRunner instance.

        Raises:
            ValueError: If backend is invalid.
        """
        if backend not in ["ray", "dask", "sequential"]:
            raise ValueError(f"Unknown backend: {backend}")

        runner = ExperimentRunner(
            config=self.config,
            data=data,
            tracker=self.tracker,
        )

        logger.info(f"Created ExperimentRunner with {backend} backend")
        return runner

    def _generate_report(
        self,
        experiment_log: pd.DataFrame,
        best_config: Dict,
    ) -> str:
        """Generate markdown report of results.

        Args:
            experiment_log: Full experiment history.
            best_config: Best configuration found.

        Returns:
            Markdown report string.
        """
        generator = ReportGenerator(self.config.primary_metric)
        report = generator.generate_markdown(experiment_log, best_config)
        return report

    def _save_results(
        self,
        experiment_log: pd.DataFrame,
        best_config: Dict,
        report: str,
    ) -> None:
        """Save experiment results to disk.

        Args:
            experiment_log: Full experiment history.
            best_config: Best configuration found.
            report: Markdown report.
        """
        # Save experiment log
        log_path = self.output_dir / "experiment_log.csv"
        experiment_log.to_csv(log_path, index=False)
        logger.info(f"Saved experiment log to {log_path}")

        # Save best config
        config_path = self.output_dir / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        logger.info(f"Saved best config to {config_path}")

        # Save report
        report_path = self.output_dir / "AUTORESEARCH_REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")

        # Save configuration
        config_file = self.output_dir / "agentic_config.json"
        with open(config_file, "w") as f:
            json.dump(self.config.dict(), f, indent=2)

    def export_experiment_log(self, output_path: str) -> None:
        """Export experiment log to file.

        Args:
            output_path: Path to save log file.
        """
        log = self.tracker.get_log()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            log.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            log.to_json(output_path, orient="records")
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

        logger.info(f"Exported experiment log to {output_path}")

    def save_best_model(self, output_dir: str) -> None:
        """Save best model configuration and results.

        Args:
            output_dir: Directory to save model.
        """
        from pathlib import Path
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.result and self.result.best_config:
            with open(output_path / "best_config.json", "w") as f:
                json.dump(self.result.best_config, f, indent=2)
            logger.info(f"Saved best config to {output_path / 'best_config.json'}")
        else:
            logger.warning("No best config available to save")

    def get_results(self) -> Dict:
        """Get summary of results.

        Returns:
            Dictionary with results summary.
        """
        best_metric = self.tracker.best_metric
        best_config = self.tracker.best_config
        log = self.tracker.get_log()

        return {
            "best_metric": best_metric,
            "best_config": best_config,
            "n_trials": len(log),
            "top_10_trials": log.nlargest(10, self.config.primary_metric).to_dict("records"),
        }
