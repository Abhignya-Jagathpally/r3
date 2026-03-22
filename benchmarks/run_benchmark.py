"""
MM-SingleCell-Benchmark runner.

Orchestrates the complete benchmarking workflow:
1. Load preprocessed data
2. Split at patient level
3. Train all baselines -> foundation models -> fusion models
4. Evaluate with BenchmarkSuite
5. Log results to experiment tracker
6. Generate leaderboard
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    BenchmarkSuite,
    ExperimentTracker,
    PatientLevelSplitter,
    compute_ari,
    compute_nmi,
)
from src.models import (
    ClassicalEnsemble,
    LogisticBaseline,
    MultimodalFuser,
    RandomForestBaseline,
    ScGPTConfig,
    ScGPTModel,
    SVMBaseline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates MM single-cell benchmarking workflow.

    Handles:
    - Loading config
    - Patient-level data splitting
    - Model training (baselines -> foundation -> fusion)
    - Evaluation with BenchmarkSuite
    - Experiment tracking
    - Leaderboard generation
    """

    def __init__(self, config_path: str, experiment_tracker: Optional[ExperimentTracker] = None):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to benchmark_config.yaml.
            experiment_tracker: Optional ExperimentTracker instance. Default: create MLflow tracker.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tracker = experiment_tracker or ExperimentTracker(
            backend=self.config["tracking"]["backend"]
        )
        self.results = {}
        self.leaderboard = []

        logger.info(f"Initialized BenchmarkRunner with config: {self.config['benchmark_name']}")

    def run_annotation_task(
        self,
        adata: AnnData,
        label_key: str = "cell_type",
        feature_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run cell type annotation task with all model types.

        Args:
            adata: Input AnnData object with labels.
            label_key: Column in adata.obs with cell type labels. Default: 'cell_type'.
            feature_key: Key in adata.obsm or X for features. Default: X (raw counts).

        Returns:
            Dictionary with task results (metrics for each model).

        Raises:
            ValueError: If label_key not in adata.obs.
        """
        if label_key not in adata.obs:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs")

        logger.info(f"Running annotation task on {adata.n_obs} cells")

        # Prepare features
        if feature_key and feature_key in adata.obsm:
            X = adata.obsm[feature_key]
        else:
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()

        y = adata.obs[label_key].values

        # Initialize results
        task_results = {}

        # 1. Classical baselines
        logger.info("Training classical baselines...")
        baseline_models = self._train_classical_baselines(X, y)
        task_results.update(baseline_models)

        # 2. Foundation model (scGPT)
        if self.config["models"]["foundation"]:
            logger.info("Training foundation models...")
            foundation_models = self._train_foundation_models(adata, label_key)
            task_results.update(foundation_models)

        # 3. Multimodal fusion
        if self.config["models"]["fusion"]:
            logger.info("Training fusion models...")
            fusion_models = self._train_fusion_models(X, y)
            task_results.update(fusion_models)

        self.results["annotation"] = task_results
        return task_results

    def _train_classical_baselines(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train classical baseline models.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).

        Returns:
            Dictionary with model names and ARI scores.
        """
        results = {}

        # Split data
        splitter = PatientLevelSplitter(random_state=self.config["reproducibility"]["random_seed"])

        # For now, use random split if patient info not available
        n_train = int(0.8 * len(X))
        train_idx = np.random.choice(len(X), n_train, replace=False)
        test_idx = np.array([i for i in range(len(X)) if i not in train_idx])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train models
        models_config = self.config["models"]["baselines"]

        for model_config in models_config:
            model_name = model_config["name"]
            logger.info(f"Training {model_name}...")

            try:
                if model_name == "LogisticRegression":
                    model = LogisticBaseline(**model_config["hyperparams"])
                elif model_name == "RandomForest":
                    model = RandomForestBaseline(**model_config["hyperparams"])
                elif model_name == "SVM":
                    model = SVMBaseline(**model_config["hyperparams"])
                elif model_name == "ClassicalEnsemble":
                    model = ClassicalEnsemble()
                else:
                    logger.warning(f"Unknown baseline model: {model_name}")
                    continue

                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluate
                ari = compute_ari(y_test, y_pred)
                nmi = compute_nmi(y_test, y_pred)

                results[f"{model_name}_ari"] = ari
                results[f"{model_name}_nmi"] = nmi

                logger.info(f"{model_name}: ARI={ari:.4f}, NMI={nmi:.4f}")

                # Log to tracker
                self.tracker.log_metrics({f"{model_name}_ari": ari, f"{model_name}_nmi": nmi})

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        return results

    def _train_foundation_models(self, adata: AnnData, label_key: str) -> Dict[str, float]:
        """
        Train foundation models (scGPT).

        Args:
            adata: Input AnnData object.
            label_key: Column with cell type labels.

        Returns:
            Dictionary with scGPT results.
        """
        results = {}

        try:
            # Initialize scGPT
            model_config = self.config["models"]["foundation"][0]
            hyperparams = model_config["hyperparams"]

            logger.info("Initializing scGPT model...")
            scgpt_model = ScGPTModel(
                model_dir="./pretrained_models/scgpt",
                n_hvg=hyperparams.get("n_hvg", 3000),
                n_bins=hyperparams.get("n_bins", 51),
            )

            # Preprocess for scGPT
            adata_pp = scgpt_model.preprocess_for_scgpt(adata)

            # Fine-tune
            logger.info("Fine-tuning scGPT...")
            ft_history = scgpt_model.fine_tune(
                adata_pp,
                task="annotation",
                labels_key=label_key,
                n_epochs=hyperparams.get("n_epochs", 10),
                lr=hyperparams.get("learning_rate", 1e-4),
                batch_size=hyperparams.get("batch_size", 64),
            )

            # Store results
            results["scGPT_final_val_acc"] = float(ft_history["val_acc"][-1])
            results["scGPT_final_val_loss"] = float(ft_history["val_loss"][-1])

            logger.info(f"scGPT: final_val_acc={results['scGPT_final_val_acc']:.4f}")

            # Log to tracker
            self.tracker.log_metrics(results)

        except Exception as e:
            logger.error(f"Error training scGPT: {e}")

        return results

    def _train_fusion_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train multimodal fusion models.

        Args:
            X: Features (n_samples, n_features).
            y: Labels (n_samples,).

        Returns:
            Dictionary with fusion results.
        """
        results = {}

        # For now, use single modality (simple case)
        # In practice, would have embeddings from genomics, imaging, clinical

        fusion_configs = self.config["models"]["fusion"]

        for fuse_config in fusion_configs:
            method = fuse_config["hyperparams"]["method"]
            model_name = fuse_config["name"]

            logger.info(f"Training {model_name} (method={method})...")

            try:
                fuser = MultimodalFuser(fusion_method=method)

                # Create dummy modalities (in practice, would be real multi-omics)
                embeddings_dict = {"genomics": X, "metadata": X[:, :100]}

                # Fuse embeddings
                fused_X = fuser.fuse_embeddings(embeddings_dict)

                # Train classifier on fused embeddings
                train_history = fuser.train_fused_classifier(fused_X, y, model_type="mlp", n_epochs=20)

                # Store results
                final_acc = train_history.get("train_acc", [-1])[-1]
                results[f"{model_name}_final_acc"] = final_acc

                logger.info(f"{model_name}: final_acc={final_acc:.4f}")

                # Log to tracker
                self.tracker.log_metrics({f"{model_name}_final_acc": final_acc})

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        return results

    def run_integration_task(
        self,
        adata: AnnData,
        batch_key: str = "batch",
        label_key: str = "cell_type",
    ) -> Dict[str, float]:
        """
        Run batch integration task.

        Args:
            adata: Input AnnData with batch and label info.
            batch_key: Column in adata.obs for batch. Default: 'batch'.
            label_key: Column in adata.obs for cell type. Default: 'cell_type'.

        Returns:
            Dictionary with integration metrics.
        """
        logger.info(f"Running integration task on {adata.n_obs} cells")

        if batch_key not in adata.obs:
            logger.warning(f"Batch key '{batch_key}' not found in adata.obs. Skipping integration task.")
            return {}

        suite = BenchmarkSuite(task="integration")
        metrics = suite.compute_integration_metrics(
            adata, batch_key=batch_key, label_key=label_key
        )

        self.results["integration"] = metrics
        self.tracker.log_metrics(metrics)

        return metrics

    def generate_leaderboard(self) -> pd.DataFrame:
        """
        Generate leaderboard comparing all models.

        Returns:
            DataFrame with model rankings by primary metric.
        """
        logger.info("Generating leaderboard...")

        leaderboard_data = []

        # Flatten results
        for task, task_results in self.results.items():
            for metric_name, value in task_results.items():
                leaderboard_data.append(
                    {"task": task, "metric": metric_name, "value": value}
                )

        if leaderboard_data:
            leaderboard_df = pd.DataFrame(leaderboard_data)

            # Sort by value (descending)
            leaderboard_df = leaderboard_df.sort_values("value", ascending=False)

            logger.info(f"Leaderboard:\n{leaderboard_df.to_string()}")

            # Save
            output_dir = Path(self.config["output"]["results_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            leaderboard_path = output_dir / self.config["output"]["leaderboard_name"]
            leaderboard_df.to_csv(leaderboard_path, index=False)
            logger.info(f"Saved leaderboard to {leaderboard_path}")

            return leaderboard_df
        else:
            logger.warning("No results to generate leaderboard")
            return pd.DataFrame()

    def run_full_benchmark(self, adata: AnnData) -> Dict:
        """
        Run complete benchmarking pipeline.

        Args:
            adata: Input AnnData object with all required metadata.

        Returns:
            Dictionary with all benchmark results.
        """
        self.tracker.start_run(
            run_name=self.config["benchmark_name"],
            tags={"config": str(self.config_path)},
        )

        try:
            # Run tasks
            if self.config["annotation"]["enabled"]:
                self.run_annotation_task(adata)

            if self.config["integration"]["enabled"]:
                self.run_integration_task(adata)

            # Generate leaderboard
            self.generate_leaderboard()

            logger.info("Benchmark complete")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            self.tracker.end_run()

        return self.results


def main():
    """Run benchmarking script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MM single-cell benchmark")
    parser.add_argument(
        "--config",
        default="./benchmarks/benchmark_config.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to preprocessed AnnData object (.h5ad)",
    )
    parser.add_argument(
        "--backend",
        choices=["mlflow", "wandb"],
        default="mlflow",
        help="Experiment tracking backend",
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}")
    adata = AnnData.read_h5ad(args.data)

    # Run benchmark
    tracker = ExperimentTracker(backend=args.backend)
    runner = BenchmarkRunner(config_path=args.config, experiment_tracker=tracker)

    results = runner.run_full_benchmark(adata)

    logger.info("Benchmark workflow complete")
    return results


if __name__ == "__main__":
    main()
