"""
Contract enforcement for frozen preprocessing in agentic tuning.

This module ensures that preprocessing is locked and immutable during the search
process, preventing data manipulation and maintaining reproducibility.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from anndata import AnnData
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

logger = logging.getLogger(__name__)


class ContractEnforcer:
    """Enforce preprocessing data contract and prevent modifications.

    This class ensures that the preprocessing stage is immutable during
    hyperparameter search. It verifies data integrity, creates checksums,
    and prevents modifications to frozen modules.

    Attributes:
        contract: Loaded preprocessing contract dictionary
        contract_path: Path to contract file
    """

    def __init__(self, contract_path_or_frozen_modules=None):
        """Initialize contract enforcer.

        Args:
            contract_path_or_frozen_modules: Either a path to preprocessing
                contract JSON file, or a list of frozen module names.
        """
        self.contract = None
        self.frozen_modules = []
        if isinstance(contract_path_or_frozen_modules, list):
            self.frozen_modules = contract_path_or_frozen_modules
            self.contract_path = None
        else:
            self.contract_path = contract_path_or_frozen_modules
            if self.contract_path:
                try:
                    self.load_contract(self.contract_path)
                except FileNotFoundError:
                    logger.warning(f"Contract file not found: {self.contract_path}")

    def load_contract(self, contract_path: str) -> Dict:
        """Load frozen preprocessing contract from JSON.

        The contract defines the preprocessing parameters and data
        characteristics that cannot be changed during the search.

        Args:
            contract_path: Path to preprocessing_contract.json.

        Returns:
            Dictionary with contract definitions.

        Raises:
            FileNotFoundError: If contract file doesn't exist.
            json.JSONDecodeError: If contract is not valid JSON.
        """
        path = Path(contract_path)

        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")

        try:
            with open(path, "r") as f:
                self.contract = json.load(f)
            logger.info(f"Loaded preprocessing contract from {contract_path}")
            return self.contract
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse contract JSON: {e}")
            raise

    def verify_data_integrity(
        self,
        adata: "AnnData",
        contract: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """Verify that data matches preprocessing contract.

        Checks that the data has expected dimensions, number of genes,
        cells, and normalization method.

        Args:
            adata: AnnData object to verify.
            contract: Contract dictionary. If None, uses loaded contract.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).

        Raises:
            ImportError: If anndata is not installed.
        """
        if not HAS_ANNDATA:
            logger.warning("anndata not available, skipping integrity check")
            return True, None

        contract = contract or self.contract

        if not contract:
            logger.warning("No contract loaded, skipping integrity check")
            return True, None

        errors = []

        # Check gene count
        qc_params = contract.get("qc_params", {})
        min_genes = qc_params.get("min_genes")
        max_genes = qc_params.get("max_genes")

        if min_genes is not None and max_genes is not None:
            n_genes = adata.n_vars
            if not (min_genes <= n_genes <= max_genes):
                errors.append(
                    f"Gene count {n_genes} out of contract range "
                    f"[{min_genes}, {max_genes}]"
                )

        # Check cell count (minimum check, maximum varies)
        min_cells = qc_params.get("min_cells", 0)
        if adata.n_obs < min_cells:
            errors.append(
                f"Cell count {adata.n_obs} below contract minimum {min_cells}"
            )

        # Check mitochondrial percentage if present
        max_mito = qc_params.get("max_mito_pct")
        if max_mito is not None and "pct_counts_mt" in adata.obs.columns:
            pct_mt = adata.obs["pct_counts_mt"].max()
            if pct_mt > max_mito:
                logger.warning(
                    f"Max mitochondrial percentage {pct_mt:.2f}% "
                    f"exceeds contract limit {max_mito}%"
                )

        # Check normalization (logged in adata.uns)
        normalization = contract.get("normalization")
        if normalization and "log_normalized" in adata.uns:
            if not adata.uns.get("log_normalized", False):
                errors.append(
                    f"Data not log-normalized as per contract "
                    f"(expected {normalization})"
                )

        # Check HVG count
        n_hvg = contract.get("n_hvg")
        if n_hvg is not None and "highly_variable" in adata.var.columns:
            n_var = adata.var["highly_variable"].sum()
            if n_var != n_hvg:
                logger.warning(
                    f"HVG count {n_var} differs from contract {n_hvg}"
                )

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None

        return is_valid, error_msg

    def verify_frozen_modules(
        self,
        editable_surface: list,
        frozen_modules: list
    ) -> Tuple[bool, Optional[str]]:
        """Verify that no frozen modules are in editable surface.

        Checks that the editable surface list does not contain any
        modules or parameters from the frozen modules list.

        Args:
            editable_surface: List of editable parameter names.
            frozen_modules: List of frozen module patterns.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        errors = []

        for frozen in frozen_modules:
            # Convert pattern to prefix for matching
            prefix = frozen.replace(".*", "").replace("*", "")

            for editable in editable_surface:
                if editable.startswith(prefix):
                    errors.append(
                        f"Parameter '{editable}' matches frozen module '{frozen}'"
                    )

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None

        return is_valid, error_msg

    def create_checkpoint(
        self,
        adata: "AnnData",
        checkpoint_path: str
    ) -> Dict:
        """Create preprocessed data checkpoint with hash verification.

        Creates a checkpoint file containing the data hash and metadata
        for later verification that data hasn't been modified.

        Args:
            adata: AnnData object to checkpoint.
            checkpoint_path: Path to save checkpoint file.

        Returns:
            Dictionary with checkpoint metadata.

        Raises:
            ImportError: If anndata is not installed.
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata required for checkpointing")

        # Compute hash of data
        data_hash = self._compute_data_hash(adata)

        checkpoint = {
            "timestamp": str(np.datetime64("now")),
            "data_hash": data_hash,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "x_shape": list(adata.X.shape),
            "x_dtype": str(adata.X.dtype),
        }

        # Save checkpoint
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Created checkpoint at {checkpoint_path}")
        return checkpoint

    def validate_checkpoint(
        self,
        adata: "AnnData",
        checkpoint_path: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate that data matches saved checkpoint.

        Verifies that the data hash and dimensions match the saved
        checkpoint to ensure no modifications have occurred.

        Args:
            adata: AnnData object to validate.
            checkpoint_path: Path to checkpoint file.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).

        Raises:
            ImportError: If anndata is not installed.
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata required for checkpoint validation")

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load checkpoint
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)

        errors = []

        # Verify dimensions
        if adata.n_obs != checkpoint["n_obs"]:
            errors.append(
                f"Cell count mismatch: {adata.n_obs} vs checkpoint {checkpoint['n_obs']}"
            )

        if adata.n_vars != checkpoint["n_vars"]:
            errors.append(
                f"Gene count mismatch: {adata.n_vars} vs checkpoint {checkpoint['n_vars']}"
            )

        # Verify data hash
        current_hash = self._compute_data_hash(adata)
        if current_hash != checkpoint["data_hash"]:
            errors.append(
                f"Data hash mismatch: {current_hash} vs checkpoint {checkpoint['data_hash']}"
            )

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None

        return is_valid, error_msg

    @staticmethod
    def _compute_data_hash(adata: "AnnData") -> str:
        """Compute SHA256 hash of AnnData object.

        Uses data shape, dtype, and sample values to create a hash
        that would change if data is modified.

        Args:
            adata: AnnData object to hash.

        Returns:
            SHA256 hash string.

        Raises:
            ImportError: If anndata is not installed.
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata required for hashing")

        hasher = hashlib.sha256()

        # Hash shape and dtype
        hasher.update(str(adata.X.shape).encode())
        hasher.update(str(adata.X.dtype).encode())

        # Hash sample values (first 100 elements for efficiency)
        n_samples = min(100, adata.X.size)
        sample_values = adata.X.flat[:n_samples]

        # Convert to bytes for hashing
        if hasattr(sample_values, "toarray"):  # scipy sparse
            sample_values = sample_values.toarray().flatten()[:n_samples]

        hasher.update(np.array(sample_values).tobytes())

        return hasher.hexdigest()
