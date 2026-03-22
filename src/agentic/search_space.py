"""
Hyperparameter search space definition for agentic tuning.

This module defines the hyperparameter search space for different model families
(scVI, scGPT, classical ML, integration, fusion) and provides sampling strategies
including random and Bayesian optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


class SearchSpace:
    """Define and sample hyperparameter search space.

    This class manages the searchable hyperparameter space for the agentic tuning
    system. It enforces bounds and provides both random and Bayesian optimization
    sampling strategies.

    Attributes:
        scvi_params: Search space for scVI model
        scgpt_params: Search space for scGPT model
        classical_params: Search space for classical ML models (SVM, RF)
        integration_params: Search space for integration methods
        fusion_params: Search space for multimodal fusion
        editable_surface: List of parameters that can be modified
    """

    def __init__(self, editable_surface: Optional[List[str]] = None):
        """Initialize search space.

        Args:
            editable_surface: List of parameters that agents can modify.
                If None, uses default set.
        """
        self.editable_surface = editable_surface or self._get_default_editable_surface()

        # scVI hyperparameters
        self.scvi_params = {
            "n_latent": (10, 50),           # latent dimension
            "n_hidden": (64, 256),          # hidden layer dimension
            "n_layers": (1, 5),             # number of layers
            "learning_rate": (1e-5, 1e-2), # learning rate (log scale)
            "dropout_rate": (0.0, 0.3),    # dropout probability
            "gene_likelihood": ["zinb", "nb", "poisson"],  # discrete choices
        }

        # scGPT hyperparameters
        self.scgpt_params = {
            "learning_rate": (1e-5, 1e-3),
            "batch_size": (32, 256),
            "n_epochs": (10, 100),
            "fine_tune_layers": (1, 12),
            "dropout_rate": (0.0, 0.3),
        }

        # Classical ML hyperparameters
        self.classical_params = {
            "svm_C": (1e-3, 1e3),              # SVM regularization (log scale)
            "svm_kernel": ["linear", "rbf", "poly"],
            "rf_n_estimators": (50, 500),
            "rf_max_depth": (5, 30),
            "rf_min_samples_split": (2, 10),
        }

        # Integration hyperparameters
        self.integration_params = {
            "method": ["harmony", "scvi", "scanvi"],
            "hvg_count": (1000, 5000),
        }

        # Fusion hyperparameters
        self.fusion_params = {
            "method": ["concat", "attention", "moe"],
            "fusion_hidden_dim": (64, 512),
        }

    @staticmethod
    def _get_default_editable_surface() -> List[str]:
        """Get default list of editable parameters.

        Returns:
            List of parameter names that can be modified.
        """
        return [
            "model_type",
            "n_latent", "n_hidden", "n_layers",
            "learning_rate", "batch_size", "n_epochs",
            "dropout_rate", "gene_likelihood",
            "integration_method", "hvg_count",
            "fusion_method", "fusion_hidden_dim"
        ]

    def get_editable_params(self) -> List[str]:
        """Get list of editable parameters.

        Returns:
            List of parameter names that agents can modify.
        """
        return self.editable_surface

    def sample_config(self) -> Dict:
        """Sample random configuration from search space.

        Generates a random hyperparameter configuration by uniformly sampling
        from the defined parameter ranges.

        Returns:
            Dictionary with parameter names and sampled values.
        """
        config = {}

        # Model type (required)
        config["model_type"] = np.random.choice(
            ["scvi", "scgpt", "classical", "multimodal"]
        )

        # Sample scVI parameters
        config["n_latent"] = int(np.random.uniform(*self.scvi_params["n_latent"]))
        config["n_hidden"] = int(np.random.uniform(*self.scvi_params["n_hidden"]))
        config["n_layers"] = int(np.random.uniform(*self.scvi_params["n_layers"]))
        config["learning_rate"] = float(
            10 ** np.random.uniform(
                np.log10(self.scvi_params["learning_rate"][0]),
                np.log10(self.scvi_params["learning_rate"][1])
            )
        )
        config["dropout_rate"] = float(
            np.random.uniform(*self.scvi_params["dropout_rate"])
        )
        config["gene_likelihood"] = np.random.choice(
            self.scvi_params["gene_likelihood"]
        )

        # Sample classical parameters
        config["svm_C"] = float(
            10 ** np.random.uniform(
                np.log10(self.classical_params["svm_C"][0]),
                np.log10(self.classical_params["svm_C"][1])
            )
        )
        config["rf_n_estimators"] = int(
            np.random.uniform(*self.classical_params["rf_n_estimators"])
        )
        config["rf_max_depth"] = int(
            np.random.uniform(*self.classical_params["rf_max_depth"])
        )

        # Sample integration parameters
        config["integration_method"] = np.random.choice(
            self.integration_params["method"]
        )
        config["hvg_count"] = int(
            np.random.uniform(*self.integration_params["hvg_count"])
        )

        # Sample fusion parameters
        config["fusion_method"] = np.random.choice(self.fusion_params["method"])
        config["fusion_hidden_dim"] = int(
            np.random.uniform(*self.fusion_params["fusion_hidden_dim"])
        )

        # Additional training parameters
        config["batch_size"] = int(
            np.random.choice([32, 64, 128, 256])
        )
        config["n_epochs"] = int(np.random.uniform(10, 100))

        return config

    def sample_config_bayesian(
        self,
        history: List[Dict],
        direction: str = "maximize"
    ) -> Dict:
        """Sample configuration using Bayesian optimization.

        Uses Optuna with TPE sampler to suggest next configuration based on
        optimization history.

        Args:
            history: List of previous (config, metric) tuples.
            direction: Optimization direction ('maximize' or 'minimize').

        Returns:
            Dictionary with parameter names and suggested values.

        Raises:
            ImportError: If optuna is not installed.
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, falling back to random sampling")
            return self.sample_config()

        if not history:
            logger.info("No history, using random initialization")
            return self.sample_config()

        # Create Optuna study
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Populate study with history
        for trial_config, metric_value in history:
            if isinstance(metric_value, (int, float)):
                trial = optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=float(metric_value),
                    distributions={},
                    user_attrs={},
                    system_attrs={},
                    params=trial_config,
                    datetime_start=None,
                )
                study.add_trial(trial)

        # Ask for next suggestion
        trial = study.ask()
        return dict(trial.params) if trial.params else self.sample_config()

    def validate_config(self, config: Dict) -> Tuple[bool, Optional[str]]:
        """Validate configuration is within bounds.

        Checks that configuration parameters are within defined ranges and
        are valid parameter names.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        errors = []

        # Check required fields
        if "model_type" not in config:
            errors.append("Missing required field: model_type")

        # Validate scVI parameters
        if "n_latent" in config:
            val = config["n_latent"]
            if not (self.scvi_params["n_latent"][0] <= val <= self.scvi_params["n_latent"][1]):
                errors.append(
                    f"n_latent {val} out of range "
                    f"{self.scvi_params['n_latent']}"
                )

        if "n_hidden" in config:
            val = config["n_hidden"]
            if not (self.scvi_params["n_hidden"][0] <= val <= self.scvi_params["n_hidden"][1]):
                errors.append(
                    f"n_hidden {val} out of range "
                    f"{self.scvi_params['n_hidden']}"
                )

        if "n_layers" in config:
            val = config["n_layers"]
            if not (self.scvi_params["n_layers"][0] <= val <= self.scvi_params["n_layers"][1]):
                errors.append(
                    f"n_layers {val} out of range "
                    f"{self.scvi_params['n_layers']}"
                )

        if "learning_rate" in config:
            val = config["learning_rate"]
            lr_min, lr_max = self.scvi_params["learning_rate"]
            if not (lr_min <= val <= lr_max):
                errors.append(
                    f"learning_rate {val} out of range "
                    f"{self.scvi_params['learning_rate']}"
                )

        if "dropout_rate" in config:
            val = config["dropout_rate"]
            if not (self.scvi_params["dropout_rate"][0] <= val <= self.scvi_params["dropout_rate"][1]):
                errors.append(
                    f"dropout_rate {val} out of range "
                    f"{self.scvi_params['dropout_rate']}"
                )

        if "gene_likelihood" in config:
            val = config["gene_likelihood"]
            if val not in self.scvi_params["gene_likelihood"]:
                errors.append(
                    f"gene_likelihood '{val}' not in "
                    f"{self.scvi_params['gene_likelihood']}"
                )

        # Validate integration parameters
        if "integration_method" in config:
            val = config["integration_method"]
            if val not in self.integration_params["method"]:
                errors.append(
                    f"integration_method '{val}' not in "
                    f"{self.integration_params['method']}"
                )

        if "hvg_count" in config:
            val = config["hvg_count"]
            if not (self.integration_params["hvg_count"][0] <= val <= self.integration_params["hvg_count"][1]):
                errors.append(
                    f"hvg_count {val} out of range "
                    f"{self.integration_params['hvg_count']}"
                )

        # Validate fusion parameters
        if "fusion_method" in config:
            val = config["fusion_method"]
            if val not in self.fusion_params["method"]:
                errors.append(
                    f"fusion_method '{val}' not in "
                    f"{self.fusion_params['method']}"
                )

        if "fusion_hidden_dim" in config:
            val = config["fusion_hidden_dim"]
            if not (self.fusion_params["fusion_hidden_dim"][0] <= val <= self.fusion_params["fusion_hidden_dim"][1]):
                errors.append(
                    f"fusion_hidden_dim {val} out of range "
                    f"{self.fusion_params['fusion_hidden_dim']}"
                )

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None

        return is_valid, error_msg

    def get_parameter_bounds(self) -> Dict[str, Tuple]:
        """Get all parameter bounds as (min, max) tuples.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples.
        """
        bounds = {}

        # Merge all parameter spaces
        all_params = {
            **self.scvi_params,
            **self.scgpt_params,
            **self.classical_params,
            **self.integration_params,
            **self.fusion_params,
        }

        for param_name, param_def in all_params.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                bounds[param_name] = param_def
            elif isinstance(param_def, list):
                bounds[param_name] = (0, len(param_def) - 1)  # Categorical index

        return bounds
