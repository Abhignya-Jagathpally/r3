"""
Patient-level and time-aware data splitting for robust evaluation.

CRITICAL: For Multiple Myeloma studies, we must respect patient structure:
- Train-test splits must be at PATIENT level, not cell level
- Multiple cells from same patient cannot appear in both sets
- Preprocessing (normalization, HVG selection, etc.) must be fit ONLY on train data
  and applied to test data to avoid data leakage
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class PatientLevelSplitter:
    """
    Split data at PATIENT level to prevent data leakage.

    Ensures no cells from the same patient appear in both training and test sets.
    This is critical for MM studies where multiple cells from same patient are sampled.

    Attributes:
        random_state: Random seed for reproducibility.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize patient-level splitter.

        Args:
            random_state: Random seed. Default: 42.
        """
        self.random_state = random_state
        np.random.seed(random_state)

        logger.info(f"Initialized PatientLevelSplitter")

    def split(
        self,
        adata: AnnData,
        patient_key: str = "patient_id",
        test_size: float = 0.2,
        stratify_key: Optional[str] = None,
    ) -> Tuple[AnnData, AnnData]:
        """
        Split data at patient level.

        Args:
            adata: Input AnnData object.
            patient_key: Column in adata.obs identifying patients. Default: 'patient_id'.
            test_size: Fraction of patients for test set. Default: 0.2.
            stratify_key: Optional column to stratify by (e.g., disease_stage).
                Default: None.

        Returns:
            Tuple of (train_adata, test_adata).

        Raises:
            ValueError: If patient_key not in adata.obs.
            ValueError: If stratify_key not in adata.obs.
        """
        if patient_key not in adata.obs:
            raise ValueError(f"patient_key '{patient_key}' not found in adata.obs")

        if stratify_key is not None and stratify_key not in adata.obs:
            raise ValueError(f"stratify_key '{stratify_key}' not found in adata.obs")

        # Get unique patients
        unique_patients = adata.obs[patient_key].unique()
        n_patients = len(unique_patients)
        n_test_patients = max(1, int(np.ceil(n_patients * test_size)))
        n_train_patients = n_patients - n_test_patients

        logger.info(
            f"Splitting {n_patients} patients: "
            f"{n_train_patients} train, {n_test_patients} test"
        )

        # Stratified split if requested
        if stratify_key is not None:
            # Get stratification labels for each patient
            patient_strata = {}
            for patient in unique_patients:
                strata_vals = adata.obs[adata.obs[patient_key] == patient][stratify_key].unique()
                # Use most common stratum for each patient
                from collections import Counter

                most_common = Counter(strata_vals).most_common(1)[0][0]
                patient_strata[patient] = most_common

            strata_list = [patient_strata[p] for p in unique_patients]
            splitter = StratifiedKFold(n_splits=1, shuffle=True, random_state=self.random_state)

            # Get first split from StratifiedKFold to achieve test_size
            # (not perfect, but good approximation)
            train_idx, test_idx = next(splitter.split(unique_patients, strata_list))

            test_patients = unique_patients[test_idx[: n_test_patients]]
            train_patients = unique_patients[train_idx[: n_train_patients]]
        else:
            # Random split
            indices = np.arange(n_patients)
            np.random.shuffle(indices)

            train_indices = indices[:n_train_patients]
            test_indices = indices[n_train_patients:]

            train_patients = unique_patients[train_indices]
            test_patients = unique_patients[test_indices]

        # Return cell-level indices
        train_mask = adata.obs[patient_key].isin(train_patients)
        test_mask = adata.obs[patient_key].isin(test_patients)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        logger.info(
            f"Split complete: train {len(train_idx)} cells ({len(train_patients)} patients), "
            f"test {len(test_idx)} cells ({len(test_patients)} patients)"
        )

        return train_idx, test_idx


class TimeAwareSplitter:
    """
    Split longitudinal data by time point.

    For studies with multiple time points (e.g., pre/post treatment),
    trains on earlier samples and tests on later ones.

    Attributes:
        random_state: Random seed.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize time-aware splitter.

        Args:
            random_state: Random seed. Default: 42.
        """
        self.random_state = random_state
        np.random.seed(random_state)

        logger.info("Initialized TimeAwareSplitter")

    def split(
        self,
        adata: AnnData,
        patient_key: str = "patient_id",
        time_key: str = "timepoint",
        cutoff_date: Optional[str] = None,
    ) -> Tuple[AnnData, AnnData]:
        """
        Split by time point (temporal split).

        Args:
            adata: Input AnnData object.
            patient_key: Column identifying patients. Default: 'patient_id'.
            time_key: Column with time points. Default: 'timepoint'.
            cutoff_date: Date cutoff (ISO format, e.g., '2022-06-01').
                If None, uses median time point. Default: None.

        Returns:
            Tuple of (train_adata, test_adata).

        Raises:
            ValueError: If patient_key or time_key not in adata.obs.
        """
        if patient_key not in adata.obs:
            raise ValueError(f"patient_key '{patient_key}' not found in adata.obs")

        if time_key not in adata.obs:
            raise ValueError(f"time_key '{time_key}' not found in adata.obs")

        # Parse time points
        time_vals = adata.obs[time_key].values

        # Try to parse as dates if string
        if isinstance(time_vals[0], str):
            try:
                import pandas as pd

                time_vals = pd.to_datetime(time_vals)
            except Exception as e:
                logger.warning(f"Could not parse time_key as dates: {e}. Using values as-is.")
                time_vals = pd.Series(time_vals).astype(float).values

        # Determine cutoff
        if cutoff_date is None:
            if isinstance(time_vals[0], (pd.Timestamp, np.datetime64)):
                cutoff = np.median([t.timestamp() for t in time_vals])
                cutoff = pd.Timestamp(cutoff * 1e9)
            else:
                cutoff = np.median(time_vals)
        else:
            cutoff = pd.to_datetime(cutoff_date)

        logger.info(f"Time split at cutoff: {cutoff}")

        # Split by time
        if isinstance(time_vals[0], (pd.Timestamp, np.datetime64)):
            train_mask = adata.obs[time_key].astype("datetime64[ns]") <= cutoff
        else:
            train_mask = adata.obs[time_key] <= cutoff

        train_adata = adata[train_mask].copy()
        test_adata = adata[~train_mask].copy()

        logger.info(
            f"Time split complete: train {train_adata.n_obs} cells, "
            f"test {test_adata.n_obs} cells"
        )

        return train_adata, test_adata


class CrossValidator:
    """
    Patient-level k-fold cross-validation.

    Ensures that across all folds, cells from same patient don't appear
    in different train/test splits.

    Attributes:
        random_state: Random seed.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize cross-validator.

        Args:
            random_state: Random seed. Default: 42.
        """
        self.random_state = random_state
        np.random.seed(random_state)

        logger.info("Initialized CrossValidator")

    def patient_level_cv(
        self,
        adata: AnnData,
        patient_key: str = "patient_id",
        n_folds: int = 5,
        stratify_key: Optional[str] = None,
    ) -> List[Tuple[AnnData, AnnData]]:
        """
        Perform patient-level k-fold cross-validation.

        Args:
            adata: Input AnnData object.
            patient_key: Column identifying patients. Default: 'patient_id'.
            n_folds: Number of folds. Default: 5.
            stratify_key: Optional stratification column. Default: None.

        Returns:
            List of (train_fold, test_fold) AnnData tuples.

        Raises:
            ValueError: If patient_key not in adata.obs.
        """
        if patient_key not in adata.obs:
            raise ValueError(f"patient_key '{patient_key}' not found in adata.obs")

        # Get unique patients
        unique_patients = adata.obs[patient_key].unique()
        n_patients = len(unique_patients)

        if n_folds > n_patients:
            logger.warning(
                f"n_folds ({n_folds}) > n_patients ({n_patients}). "
                f"Using n_folds={n_patients}"
            )
            n_folds = n_patients

        logger.info(f"Patient-level {n_folds}-fold CV on {n_patients} patients")

        # Stratified split if requested
        if stratify_key is not None:
            if stratify_key not in adata.obs:
                raise ValueError(f"stratify_key '{stratify_key}' not found in adata.obs")

            # Get stratum for each patient
            patient_strata = {}
            for patient in unique_patients:
                strata_vals = adata.obs[adata.obs[patient_key] == patient][stratify_key].unique()
                from collections import Counter

                most_common = Counter(strata_vals).most_common(1)[0][0]
                patient_strata[patient] = most_common

            strata_list = np.array([patient_strata[p] for p in unique_patients])
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(unique_patients, strata_list))
        else:
            # Simple k-fold without stratification
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
            # StratifiedKFold requires labels, so use dummy labels
            dummy_labels = np.zeros(n_patients)
            splits = list(splitter.split(unique_patients, dummy_labels))

        # Convert patient indices to AnnData splits
        cv_splits = []
        for fold_idx, (train_patient_indices, test_patient_indices) in enumerate(splits):
            train_patients = unique_patients[train_patient_indices]
            test_patients = unique_patients[test_patient_indices]

            train_mask = adata.obs[patient_key].isin(train_patients)
            test_mask = adata.obs[patient_key].isin(test_patients)

            train_fold = adata[train_mask].copy()
            test_fold = adata[test_mask].copy()

            cv_splits.append((train_fold, test_fold))

            logger.debug(
                f"Fold {fold_idx + 1}/{n_folds}: "
                f"train {train_fold.n_obs} cells ({len(train_patients)} patients), "
                f"test {test_fold.n_obs} cells ({len(test_patients)} patients)"
            )

        return cv_splits

    def cross_validate(
        self,
        adata: AnnData,
        patient_key: str = "patient_id",
        n_folds: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Alias for patient_level_cv returning index arrays."""
        return self.patient_level_cv(adata, patient_key=patient_key, n_folds=n_folds)

    def fit_transform_train_only(
        self,
        train_adata: AnnData,
        test_adata: AnnData,
        preprocessing_steps: List,
    ) -> Tuple[AnnData, AnnData]:
        """
        CRITICAL: Fit preprocessing ONLY on training data, apply to test.

        This prevents data leakage from test set influencing normalization,
        HVG selection, or other preprocessing.

        Args:
            train_adata: Training data for fitting preprocessing.
            test_adata: Test data to transform (without fitting).
            preprocessing_steps: List of preprocessing functions/classes with
                .fit() and .transform() methods (sklearn-like API).

        Returns:
            Tuple of (train_adata_preprocessed, test_adata_preprocessed).

        Example:
            >>> from src.preprocessing import NormalizationPipeline
            >>> steps = [NormalizationPipeline(), HVGSelector(n_genes=3000)]
            >>> train_pp, test_pp = cv.fit_transform_train_only(train, test, steps)
        """
        logger.info(
            f"Fitting preprocessing on train data ({train_adata.n_obs} cells) "
            f"and transforming test ({test_adata.n_obs} cells)"
        )

        train_processed = train_adata.copy()
        test_processed = test_adata.copy()

        for i, step in enumerate(preprocessing_steps):
            logger.debug(f"Step {i + 1}/{len(preprocessing_steps)}: {step.__class__.__name__}")

            # Fit on train
            if hasattr(step, "fit"):
                step.fit(train_processed)
            else:
                logger.warning(f"Step {step} has no fit() method. Skipping.")
                continue

            # Transform train
            if hasattr(step, "transform"):
                train_processed = step.transform(train_processed)
            else:
                logger.warning(f"Step {step} has no transform() method. Skipping.")

            # Transform test with fitted preprocessor (don't refit!)
            if hasattr(step, "transform"):
                test_processed = step.transform(test_processed)

        logger.info(
            f"Preprocessing complete. "
            f"Train: {train_processed.n_obs} cells x {train_processed.n_vars} genes. "
            f"Test: {test_processed.n_obs} cells x {test_processed.n_vars} genes."
        )

        return train_processed, test_processed


def ensure_no_patient_overlap(
    train_adata_or_list, test_adata=None, patient_key: str = "patient_id"
) -> bool:
    """
    Verify that no patients appear in multiple AnnData objects.

    Safety check to catch data leakage during splitting.

    Args:
        adata_list: List of AnnData objects to check.
        patient_key: Column identifying patients. Default: 'patient_id'.

    Returns:
        True if no overlap, False if overlap detected.
    """
    # Support both (train, test, key) and ([adata_list], key) signatures
    if test_adata is not None:
        adata_list = [train_adata_or_list, test_adata]
    elif isinstance(train_adata_or_list, list):
        adata_list = train_adata_or_list
    else:
        adata_list = [train_adata_or_list]

    all_patients = []
    for i, adata in enumerate(adata_list):
        if patient_key not in adata.obs:
            logger.warning(f"adata[{i}] has no '{patient_key}' column")
            continue

        patients = set(adata.obs[patient_key].unique())
        all_patients.append((i, patients))

    # Check pairwise overlaps
    has_overlap = False
    for i in range(len(all_patients)):
        for j in range(i + 1, len(all_patients)):
            idx_i, patients_i = all_patients[i]
            idx_j, patients_j = all_patients[j]

            overlap = patients_i & patients_j
            if overlap:
                logger.error(
                    f"Patient overlap detected between adata[{idx_i}] and adata[{idx_j}]: "
                    f"{overlap}"
                )
                has_overlap = True

    if not has_overlap:
        logger.info(f"No patient overlap detected across {len(adata_list)} datasets")

    return not has_overlap
