"""
MM-specific benchmark metrics for single-cell analysis.

Defines custom metrics tailored to Multiple Myeloma research:
- Annotation quality (ARI, NMI, rare cell recall)
- Integration quality (batch mixing, graph connectivity, biological conservation)
- Transfer learning performance (cross-dataset prediction)
- Composite benchmark scores

Note: No standard public leaderboard exists for MM single-cell analysis,
so we define our own comprehensive benchmark suite.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index for annotation quality.

    ARI measures similarity between true and predicted cluster assignments.
    Range: [-1, 1], where 1 is perfect agreement and 0 is random.

    Args:
        labels_true: Ground truth labels (n_samples,).
        labels_pred: Predicted labels (n_samples,).

    Returns:
        Adjusted Rand Index (float in [-1, 1]).

    Raises:
        ValueError: If label arrays have different lengths.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")

    ari = adjusted_rand_score(labels_true, labels_pred)

    logger.debug(f"Computed ARI: {ari:.4f}")
    return float(ari)


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information for annotation quality.

    NMI measures the mutual dependence between true and predicted labels,
    normalized by the entropy of the true labels.
    Range: [0, 1], where 1 indicates perfect agreement.

    Args:
        labels_true: Ground truth labels (n_samples,).
        labels_pred: Predicted labels (n_samples,).

    Returns:
        Normalized Mutual Information (float in [0, 1]).

    Raises:
        ValueError: If label arrays have different lengths.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")

    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    logger.debug(f"Computed NMI: {nmi:.4f}")
    return float(nmi)


def compute_batch_asw(
    adata: AnnData, batch_key: str = "batch", embed_key: str = "X_pca"
) -> float:
    """
    Compute batch-corrected Average Silhouette Width.

    Measures how well batch-effect-removed embeddings mix across batches.
    Higher ASW indicates better batch integration.
    Range: [-1, 1], where 1 is ideal.

    Args:
        adata: AnnData object with batch info and embeddings.
        batch_key: Column in adata.obs identifying batch. Default: 'batch'.
        embed_key: Key in adata.obsm for embeddings. Default: 'X_pca'.

    Returns:
        Average Silhouette Width (float in [-1, 1]).

    Raises:
        ValueError: If batch_key not in adata.obs or embed_key not in adata.obsm.
    """
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    if embed_key not in adata.obsm:
        raise ValueError(f"embed_key '{embed_key}' not found in adata.obsm")

    # Compute silhouette width
    from sklearn.metrics import silhouette_score

    embeddings = adata.obsm[embed_key]
    batches = adata.obs[batch_key].values

    # ASW: silhouette score with batch labels
    # Positive ASW indicates good mixing; negative indicates batch clustering
    asw = silhouette_score(embeddings, batches)

    logger.debug(f"Computed batch ASW: {asw:.4f}")
    return float(asw)


def compute_graph_connectivity(
    adata: AnnData, label_key: str = "cell_type", n_neighbors: int = 15
) -> float:
    """
    Compute graph connectivity score for cell type clustering.

    Measures the average graph connectivity within cell types:
    for each cell type, what fraction of k-nearest neighbors are of the same type?
    Range: [0, 1], where 1 indicates perfect separation.

    Args:
        adata: AnnData object with cell type labels and PCA embeddings.
        label_key: Column in adata.obs with cell type labels. Default: 'cell_type'.
        n_neighbors: Number of neighbors for graph. Default: 15.

    Returns:
        Graph connectivity score (float in [0, 1]).

    Raises:
        ValueError: If label_key not in adata.obs or no embeddings available.
    """
    if label_key not in adata.obs:
        raise ValueError(f"label_key '{label_key}' not found in adata.obs")

    # Get embeddings (use PCA if available)
    if "X_pca" in adata.obsm:
        embeddings = adata.obsm["X_pca"]
    elif "X_umap" in adata.obsm:
        embeddings = adata.obsm["X_umap"]
    else:
        raise ValueError("No embeddings found in adata.obsm (X_pca or X_umap)")

    labels = adata.obs[label_key].values
    unique_labels = np.unique(labels)

    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Compute connectivity: for each cell, fraction of k-NN with same label (vectorized)
    knn_labels = labels[indices[:, 1:]]  # All k-NN labels, excluding self
    same_label = (knn_labels == labels.reshape(-1, 1)).mean(axis=1)
    graph_connectivity = float(same_label.mean())

    logger.debug(f"Computed graph connectivity: {graph_connectivity:.4f}")
    return float(graph_connectivity)


def compute_rare_cell_recall(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    rare_types: List[str],
    min_freq: float = 0.01,
) -> Dict[str, float]:
    """
    Compute per-cell-type recall, especially for rare cell types.

    Important for MM: rare cell types (osteoclasts, mast cells, HSC progenitors)
    may be clinically relevant despite low frequency.

    Args:
        labels_true: Ground truth labels (n_samples,).
        labels_pred: Predicted labels (n_samples,).
        rare_types: List of rare cell type names to evaluate. Default: [].
        min_freq: Minimum frequency threshold for "rare" classification.
            Default: 0.01 (1%).

    Returns:
        Dictionary mapping cell type -> recall score.

    Raises:
        ValueError: If label arrays have different lengths.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")

    # Convert to strings for matching
    labels_true_str = np.array([str(l) for l in labels_true])
    labels_pred_str = np.array([str(l) for l in labels_pred])

    recall_dict = {}

    for rare_type in rare_types:
        rare_type_str = str(rare_type)

        # Find cells of this type in ground truth
        mask = labels_true_str == rare_type_str

        if mask.sum() == 0:
            logger.warning(f"Rare type '{rare_type}' not found in labels_true")
            recall_dict[rare_type] = 0.0
            continue

        # Compute recall: TP / (TP + FN)
        # TP = correctly predicted as this type
        # FN = incorrectly predicted as other types
        tp = ((labels_true_str == rare_type_str) & (labels_pred_str == rare_type_str)).sum()
        fn = ((labels_true_str == rare_type_str) & (labels_pred_str != rare_type_str)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_dict[rare_type] = float(recall)

    logger.debug(f"Computed rare cell recalls: {recall_dict}")
    return recall_dict


def compute_bio_conservation(
    adata: AnnData, label_key: str = "cell_type", embed_key: str = "X_pca"
) -> float:
    """
    Compute biological conservation score.

    Measures how well biological cell type structure is preserved in embeddings.
    Combines:
    1. Silhouette width within cell types (isolated label silhouette)
    2. NMI between original and k-NN cluster assignments

    Range: [0, 1], where 1 indicates perfect conservation.

    Args:
        adata: AnnData object with cell type labels and embeddings.
        label_key: Column in adata.obs with cell type labels. Default: 'cell_type'.
        embed_key: Key in adata.obsm for embeddings. Default: 'X_pca'.

    Returns:
        Biological conservation score (float in [0, 1]).

    Raises:
        ValueError: If label_key or embed_key not found.
    """
    if label_key not in adata.obs:
        raise ValueError(f"label_key '{label_key}' not found in adata.obs")

    if embed_key not in adata.obsm:
        raise ValueError(f"embed_key '{embed_key}' not found in adata.obsm")

    from sklearn.metrics import silhouette_samples

    embeddings = adata.obsm[embed_key]
    labels = adata.obs[label_key].values

    # Component 1: Isolated Label Silhouette (ILS)
    # Silhouette width of cells within their respective types
    silhouette_vals = silhouette_samples(embeddings, labels)
    # Convert to [0, 1] range: [-1, 1] -> [0, 1]
    ils = (silhouette_vals.mean() + 1) / 2

    # Component 2: NMI-based conservation
    # Cluster k-NN graph and compute NMI with original labels
    nbrs = NearestNeighbors(n_neighbors=15).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Simple clustering: assign each cell to most common label in its k-NN (vectorized)
    from scipy.stats import mode
    knn_labels = labels[indices]
    knn_cluster_labels = mode(knn_labels, axis=1, keepdims=False).mode

    # Compute NMI (normalized to [0, 1])
    nmi = normalized_mutual_info_score(labels, knn_cluster_labels)

    # Combine: average of ILS and NMI
    bio_conservation = (ils + nmi) / 2

    logger.debug(
        f"Computed bio conservation: ILS={ils:.4f}, NMI={nmi:.4f}, "
        f"Combined={bio_conservation:.4f}"
    )
    return float(bio_conservation)


def compute_transfer_score(
    adata_ref: AnnData,
    adata_query: AnnData,
    model,
    label_key: str = "cell_type",
) -> float:
    """
    Compute transfer learning score on external dataset without retraining.

    Measures how well a model trained on one dataset generalizes to another.
    This is key for assessing biological robustness.

    Args:
        adata_ref: Reference AnnData object (training data).
        adata_query: Query AnnData object (test data).
        model: Trained model with .predict() or .predict(adata) method.
        label_key: Column in adata_query.obs with ground truth labels.
            Default: 'cell_type'.

    Returns:
        Transfer accuracy (float in [0, 1]).

    Raises:
        ValueError: If label_key not in adata_query.obs.
        ValueError: If model has no predict method.
    """
    if label_key not in adata_query.obs:
        raise ValueError(f"label_key '{label_key}' not found in adata_query.obs")

    if not hasattr(model, "predict"):
        raise ValueError("model must have a predict() method")

    # Get ground truth labels
    y_true = adata_query.obs[label_key].values

    # Make predictions
    try:
        # Try calling model.predict(adata) for scGPT-like models
        y_pred = model.predict(adata_query)
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
    except (TypeError, AttributeError):
        # Try X-based prediction for classical models
        X_query = adata_query.X
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()
        y_pred = model.predict(X_query)

    # Convert to strings for comparison
    y_true_str = np.array([str(l) for l in y_true])
    y_pred_str = np.array([str(l) for l in y_pred])

    # Compute accuracy
    accuracy = (y_true_str == y_pred_str).mean()

    logger.debug(f"Computed transfer score: {accuracy:.4f}")
    return float(accuracy)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite aggregating all metrics.

    Computes a set of metrics for annotation, integration, and transfer
    tasks, producing both per-metric scores and a composite benchmark score.

    Attributes:
        task: Which task to evaluate ('annotation', 'integration', 'transfer').
        metrics_dict: Dictionary mapping metric names to computed scores.
        composite_score: Weighted average of all metrics.
    """

    def __init__(self, task: str = "annotation"):
        """
        Initialize benchmark suite.

        Args:
            task: Task to evaluate ('annotation', 'integration', 'transfer', 'all').
                Default: 'annotation'.
        """
        self.task = task
        self.metrics_dict = {}
        self.composite_score = None

        logger.info(f"Initialized BenchmarkSuite for task: {task}")

    def compute_annotation_metrics(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        rare_types: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute annotation task metrics (ARI, NMI, rare cell recall).

        Args:
            labels_true: Ground truth labels.
            labels_pred: Predicted labels.
            rare_types: List of rare cell types. Default: None.
            weights: Per-metric weights for composite score.
                Default: equal weights.

        Returns:
            Dictionary with metric scores and composite score.
        """
        if rare_types is None:
            rare_types = []

        self.metrics_dict = {}

        # Core metrics
        self.metrics_dict["ari"] = compute_ari(labels_true, labels_pred)
        self.metrics_dict["nmi"] = compute_nmi(labels_true, labels_pred)

        # Rare cell recall
        rare_recalls = compute_rare_cell_recall(labels_true, labels_pred, rare_types)
        self.metrics_dict.update({f"rare_{k}_recall": v for k, v in rare_recalls.items()})

        # Composite score
        if weights is None:
            weights = {"ari": 0.5, "nmi": 0.5}

        composite = 0.0
        for metric_name, weight in weights.items():
            if metric_name in self.metrics_dict:
                composite += weight * self.metrics_dict[metric_name]

        self.metrics_dict["composite_score"] = composite
        self.composite_score = composite

        logger.info(
            f"Annotation metrics: ARI={self.metrics_dict['ari']:.4f}, "
            f"NMI={self.metrics_dict['nmi']:.4f}, "
            f"composite={composite:.4f}"
        )
        return self.metrics_dict

    def compute_integration_metrics(
        self,
        adata: AnnData,
        batch_key: str = "batch",
        label_key: str = "cell_type",
        embed_key: str = "X_pca",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute integration task metrics (batch ASW, graph connectivity, bio conservation).

        Args:
            adata: AnnData object with batch and label info.
            batch_key: Column in adata.obs for batch. Default: 'batch'.
            label_key: Column in adata.obs for cell type. Default: 'cell_type'.
            embed_key: Key in adata.obsm for embeddings. Default: 'X_pca'.
            weights: Per-metric weights. Default: equal weights.

        Returns:
            Dictionary with metric scores and composite score.
        """
        self.metrics_dict = {}

        # Core metrics
        self.metrics_dict["batch_asw"] = compute_batch_asw(adata, batch_key, embed_key)
        self.metrics_dict["graph_connectivity"] = compute_graph_connectivity(
            adata, label_key
        )
        self.metrics_dict["bio_conservation"] = compute_bio_conservation(
            adata, label_key, embed_key
        )

        # Composite score
        if weights is None:
            weights = {
                "batch_asw": 0.333,
                "graph_connectivity": 0.333,
                "bio_conservation": 0.334,
            }

        composite = 0.0
        for metric_name, weight in weights.items():
            if metric_name in self.metrics_dict:
                # Normalize batch_asw from [-1, 1] to [0, 1]
                value = self.metrics_dict[metric_name]
                if metric_name == "batch_asw":
                    value = (value + 1) / 2
                composite += weight * value

        self.metrics_dict["composite_score"] = composite
        self.composite_score = composite

        logger.info(
            f"Integration metrics: batch_ASW={self.metrics_dict['batch_asw']:.4f}, "
            f"graph_conn={self.metrics_dict['graph_connectivity']:.4f}, "
            f"bio_cons={self.metrics_dict['bio_conservation']:.4f}, "
            f"composite={composite:.4f}"
        )
        return self.metrics_dict

    def compute_transfer_metrics(
        self,
        adata_ref: AnnData,
        adata_query: AnnData,
        model,
        label_key: str = "cell_type",
    ) -> Dict[str, float]:
        """
        Compute transfer learning metrics.

        Args:
            adata_ref: Reference (training) data.
            adata_query: Query (test) data.
            model: Trained model.
            label_key: Column in adata_query.obs for labels. Default: 'cell_type'.

        Returns:
            Dictionary with transfer score.
        """
        self.metrics_dict = {}

        self.metrics_dict["transfer_score"] = compute_transfer_score(
            adata_ref, adata_query, model, label_key
        )

        self.composite_score = self.metrics_dict["transfer_score"]

        logger.info(f"Transfer metrics: transfer_score={self.composite_score:.4f}")
        return self.metrics_dict

    def run_all(
        self,
        adata: AnnData,
        labels_true_key: str = "cell_type",
        labels_pred_key: str = "cell_type",
        batch_key: str = "dataset",
        embed_key: str = "X_pca",
    ) -> Dict[str, float]:
        """
        Run all applicable benchmark metrics.

        Args:
            adata: AnnData object with embeddings and labels
            labels_true_key: obs column for ground truth labels
            labels_pred_key: obs column for predicted labels
            batch_key: obs column for batch info
            embed_key: obsm key for embeddings

        Returns:
            Dictionary of all computed metrics
        """
        results = {}

        # Annotation metrics (if both label keys exist)
        if labels_true_key in adata.obs.columns and labels_pred_key in adata.obs.columns:
            y_true = adata.obs[labels_true_key].values
            y_pred = adata.obs[labels_pred_key].values
            # Filter out empty predictions
            mask = np.array([str(p).strip() != "" for p in y_pred])
            if mask.sum() > 0:
                try:
                    results["ari"] = compute_ari(y_true[mask], y_pred[mask])
                except Exception as e:
                    logger.warning(f"ARI computation failed: {e}")
                    results["ari"] = float("nan")
                try:
                    results["nmi"] = compute_nmi(y_true[mask], y_pred[mask])
                except Exception as e:
                    logger.warning(f"NMI computation failed: {e}")
                    results["nmi"] = float("nan")
                try:
                    from sklearn.metrics import f1_score, balanced_accuracy_score
                    results["f1_weighted"] = float(
                        f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)
                    )
                    results["balanced_accuracy"] = float(
                        balanced_accuracy_score(y_true[mask], y_pred[mask])
                    )
                except Exception as e:
                    logger.warning(f"F1/balanced accuracy computation failed: {e}")
                    results["f1_weighted"] = float("nan")
                    results["balanced_accuracy"] = float("nan")

        # Integration metrics
        if embed_key in adata.obsm:
            label_key = labels_true_key if labels_true_key in adata.obs.columns else "leiden"
            if label_key in adata.obs.columns:
                try:
                    results["bio_conservation"] = compute_bio_conservation(
                        adata, label_key=label_key, embed_key=embed_key
                    )
                except Exception as e:
                    logger.warning(f"bio_conservation failed: {e}")

                try:
                    results["graph_connectivity"] = compute_graph_connectivity(
                        adata, label_key=label_key
                    )
                except Exception as e:
                    logger.warning(f"graph_connectivity failed: {e}")

            if batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1:
                try:
                    results["batch_asw"] = compute_batch_asw(
                        adata, batch_key=batch_key, embed_key=embed_key
                    )
                except Exception as e:
                    logger.warning(f"batch_asw failed: {e}")

        # Clustering quality (silhouette on cell types)
        if embed_key in adata.obsm and labels_true_key in adata.obs.columns:
            try:
                from sklearn.metrics import silhouette_score
                results["silhouette_score"] = float(silhouette_score(
                    adata.obsm[embed_key],
                    adata.obs[labels_true_key].values,
                ))
            except Exception as e:
                logger.warning(f"silhouette_score failed: {e}")

        self.metrics_dict = results
        logger.info(f"BenchmarkSuite.run_all computed {len(results)} metrics")
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to DataFrame for easy reporting.

        Returns:
            DataFrame with metrics.
        """
        if not self.metrics_dict:
            raise ValueError("No metrics computed yet")

        df = pd.DataFrame([self.metrics_dict])
        return df
