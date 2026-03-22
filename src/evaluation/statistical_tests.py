"""
Statistical tests for benchmark comparisons.

Provides bootstrap confidence intervals, permutation tests, and
pairwise significance tests for model comparison.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class BootstrapCI:
    """Bootstrap confidence intervals for evaluation metrics."""

    def __init__(
        self, n_bootstrap: int = 1000, ci_level: float = 0.95, random_state: int = 42
    ):
        """
        Initialize bootstrap CI calculator.

        Args:
            n_bootstrap: Number of bootstrap samples. Default: 1000.
            ci_level: Confidence level (0, 1). Default: 0.95.
            random_state: Random seed for reproducibility. Default: 42.
        """
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def compute_ci(self, scores: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute bootstrap CI for a metric using BCa method.

        Returns (mean, lower, upper).
        Uses bias-corrected and accelerated (BCa) bootstrap.

        Args:
            scores: Array of metric scores (n_samples,).

        Returns:
            Tuple of (mean, ci_lower, ci_upper).
        """
        if len(scores) == 0:
            raise ValueError("scores array is empty")

        mean = float(np.mean(scores))

        # BCa bootstrap
        n = len(scores)
        alpha = 1 - self.ci_level

        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self.rng.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_means < mean))

        # Acceleration
        jack_means = []
        for i in range(n):
            jack_sample = np.delete(scores, i)
            jack_means.append(np.mean(jack_sample))
        jack_mean = np.mean(jack_means)
        jack_diffs = np.array(jack_means) - jack_mean
        numerator = np.sum(jack_diffs**3)
        denominator = 6 * (np.sum(jack_diffs**2) ** 1.5)
        acceleration = numerator / denominator if denominator != 0 else 0

        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper)))

        ci_lower = float(np.quantile(bootstrap_means, p_lower))
        ci_upper = float(np.quantile(bootstrap_means, p_upper))

        logger.debug(
            f"Bootstrap CI: mean={mean:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        return (mean, ci_lower, ci_upper)

    def compare_methods(
        self, scores_a: np.ndarray, scores_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare two methods via paired bootstrap.

        Args:
            scores_a: Scores for method A (n_samples,).
            scores_b: Scores for method B (n_samples,).

        Returns:
            Dict with: mean_diff, ci_lower, ci_upper, p_value, significant.

        Raises:
            ValueError: If arrays have different lengths.
        """
        if len(scores_a) != len(scores_b):
            raise ValueError(
                f"scores_a and scores_b must have same length: "
                f"{len(scores_a)} != {len(scores_b)}"
            )

        if len(scores_a) == 0:
            raise ValueError("scores arrays are empty")

        # Compute differences
        diffs = scores_a - scores_b
        mean_diff = float(np.mean(diffs))

        # Bootstrap CI on differences
        n = len(diffs)
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self.rng.choice(diffs, size=n, replace=True)
            bootstrap_diffs.append(np.mean(bootstrap_sample))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Percentile CI
        alpha = 1 - self.ci_level
        ci_lower = float(np.quantile(bootstrap_diffs, alpha / 2))
        ci_upper = float(np.quantile(bootstrap_diffs, 1 - alpha / 2))

        # Two-tailed p-value: fraction of bootstrap samples on opposite side of zero
        p_value = float(2 * min(np.mean(bootstrap_diffs < 0), np.mean(bootstrap_diffs > 0)))
        significant = ci_lower > 0 or ci_upper < 0

        result = {
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "significant": significant,
        }

        logger.debug(
            f"Bootstrap comparison: mean_diff={mean_diff:.4f}, "
            f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}"
        )
        return result


class PairwiseComparison:
    """Pairwise statistical comparison of multiple methods."""

    @staticmethod
    def wilcoxon_test(
        scores_a: np.ndarray, scores_b: np.ndarray
    ) -> Dict[str, Union[float, bool]]:
        """
        Wilcoxon signed-rank test for paired samples.

        Args:
            scores_a: Scores for method A (n_samples,).
            scores_b: Scores for method B (n_samples,).

        Returns:
            Dict with: statistic, p_value, effect_size (rank-biserial correlation).
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("scores_a and scores_b must have same length")

        if len(scores_a) < 2:
            logger.warning("Wilcoxon test requires at least 2 samples")
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
            }

        # Perform Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(scores_a, scores_b)

        # Effect size: rank-biserial correlation
        # r = 1 - (2*T)/(n*(n+1)) where T is test statistic
        n = len(scores_a)
        effect_size = 1 - (2 * stat) / (n * (n + 1))

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
        }

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values.
            alpha: Significance level. Default: 0.05.

        Returns:
            Tuple of (adjusted_p_values, significant).
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        # Bonferroni adjustment
        adjusted = np.minimum(p_values * n_tests, 1.0)
        significant = adjusted < alpha

        return list(adjusted), list(significant)

    def compare_all_pairs(
        self,
        results: Dict[str, np.ndarray],
        correction: str = "bonferroni",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compare all method pairs with multiple testing correction.

        Args:
            results: Dict mapping method names to score arrays (n_samples,).
            correction: Correction method ('bonferroni', 'none'). Default: 'bonferroni'.
            alpha: Significance level. Default: 0.05.

        Returns:
            DataFrame with columns: method_a, method_b, p_value, p_adjusted,
            significant, effect_size.
        """
        method_names = sorted(results.keys())
        n_methods = len(method_names)

        comparisons = []
        p_values = []

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method_a = method_names[i]
                method_b = method_names[j]

                scores_a = results[method_a]
                scores_b = results[method_b]

                # Wilcoxon test
                test_result = self.wilcoxon_test(scores_a, scores_b)

                comparisons.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "p_value": test_result["p_value"],
                        "effect_size": test_result["effect_size"],
                    }
                )
                p_values.append(test_result["p_value"])

        # Multiple testing correction
        if len(p_values) > 0:
            if correction == "bonferroni":
                adjusted_p, significant = self.bonferroni_correction(p_values, alpha)
            else:
                adjusted_p = p_values
                significant = [p < alpha for p in p_values]

            for i, comp in enumerate(comparisons):
                comp["p_adjusted"] = adjusted_p[i]
                comp["significant"] = significant[i]
        else:
            logger.warning("No comparisons to make")

        df = pd.DataFrame(comparisons)
        logger.info(f"Pairwise comparison: {len(comparisons)} pairs tested")
        return df

    @staticmethod
    def friedman_test(results: Dict[str, np.ndarray]) -> Dict:
        """
        Friedman test for comparing multiple methods across folds.

        Tests null hypothesis that all methods have same distribution.

        Args:
            results: Dict mapping method names to score arrays (n_folds,).

        Returns:
            Dict with: statistic, p_value, significant.
        """
        method_names = sorted(results.keys())

        # Stack scores: (n_folds, n_methods)
        scores_matrix = np.column_stack([results[name] for name in method_names])

        if scores_matrix.shape[0] < 2:
            logger.warning("Friedman test requires at least 2 folds")
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
            }

        # Friedman test
        stat, p_value = stats.friedmanchisquare(*[scores_matrix[:, i] for i in range(scores_matrix.shape[1])])

        significant = p_value < 0.05

        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": significant,
        }

        logger.info(f"Friedman test: stat={stat:.4f}, p={p_value:.4f}")
        return result


class CrossValidationStats:
    """Statistical summaries across CV folds."""

    @staticmethod
    def summarize_folds(fold_scores: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Summarize metric scores across folds.

        Args:
            fold_scores: List of dicts, each mapping metric name -> score.

        Returns:
            DataFrame with columns: metric, mean, std, ci_lower, ci_upper, min, max.
        """
        if not fold_scores:
            raise ValueError("fold_scores is empty")

        # Convert to DataFrame
        df_folds = pd.DataFrame(fold_scores)

        # Compute statistics per metric
        summaries = []
        for metric in df_folds.columns:
            scores = df_folds[metric].values

            # Remove NaN values
            scores = scores[~np.isnan(scores)]

            if len(scores) == 0:
                continue

            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1) if len(scores) > 1 else 0)

            # 95% CI using t-distribution (more appropriate for small samples)
            se = std / np.sqrt(len(scores))
            t_val = stats.t.ppf(0.975, df=len(scores) - 1)
            ci_lower = float(mean - t_val * se)
            ci_upper = float(mean + t_val * se)

            summaries.append(
                {
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "n_folds": len(scores),
                }
            )

        df_summary = pd.DataFrame(summaries)
        logger.info(f"Summarized {len(df_summary)} metrics across {len(scores)} folds")
        return df_summary

    @staticmethod
    def is_significantly_better(
        scores_a: List[float], scores_b: List[float], alpha: float = 0.05
    ) -> bool:
        """
        Test if method A is significantly better than B across folds.

        Uses paired t-test (or Wilcoxon for non-normal data).

        Args:
            scores_a: Scores for method A across folds.
            scores_b: Scores for method B across folds.
            alpha: Significance level. Default: 0.05.

        Returns:
            True if A is significantly better than B (p < alpha, mean_a > mean_b).
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        if len(scores_a) != len(scores_b):
            raise ValueError("scores_a and scores_b must have same length")

        if len(scores_a) < 2:
            logger.warning("is_significantly_better requires at least 2 folds")
            return False

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        # One-tailed Wilcoxon test: is A significantly > B?
        stat, p_value = stats.wilcoxon(scores_a, scores_b)

        # Check: A is better AND statistically significant
        significant_better = (mean_a > mean_b) and (p_value / 2 < alpha)

        logger.debug(
            f"Comparison: mean_a={mean_a:.4f}, mean_b={mean_b:.4f}, "
            f"p={p_value:.4f}, significant_better={significant_better}"
        )
        return bool(significant_better)
