"""
Data storage management for the R3-MM pipeline.

This module provides a StorageManager class that enforces a three-layer data
staging architecture: raw → standardized → analysis_ready.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manage data storage across three staging layers.

    The storage hierarchy is:
        1. raw: Original downloaded data
        2. standardized: QC-filtered and normalized data
        3. analysis_ready: Integrated and annotated data

    Attributes:
        root_dir: Root directory for all storage layers
        raw_dir: Directory for raw data
        standardized_dir: Directory for standardized data
        analysis_ready_dir: Directory for analysis-ready data
    """

    def __init__(
        self,
        root_dir: str = "./data",
        raw_dir: str = "raw",
        standardized_dir: str = "standardized",
        analysis_ready_dir: str = "analysis_ready",
    ):
        """
        Initialize StorageManager.

        Args:
            root_dir: Root storage directory
            raw_dir: Name of raw data subdirectory
            standardized_dir: Name of standardized data subdirectory
            analysis_ready_dir: Name of analysis-ready data subdirectory
        """
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / raw_dir
        self.standardized_dir = self.root_dir / standardized_dir
        self.analysis_ready_dir = self.root_dir / analysis_ready_dir

        # Create directories
        for directory in [self.raw_dir, self.standardized_dir, self.analysis_ready_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized storage layer: {directory}")

    def write_raw(
        self,
        adata: ad.AnnData,
        name: str,
        formats: List[str] = None,
        compression: str = "gzip",
        level: int = 9,
    ) -> Dict[str, Path]:
        """
        Write data to raw storage layer.

        Args:
            adata: AnnData object to store
            name: Dataset name (used as filename)
            formats: List of output formats (h5ad, parquet, zarr)
            compression: Compression method
            level: Compression level (0-9)

        Returns:
            Dictionary mapping format to output path

        Example:
            >>> sm = StorageManager()
            >>> paths = sm.write_raw(adata, "GSE271107")
        """
        if formats is None:
            formats = ["h5ad"]

        return self._write_to_layer(
            adata=adata,
            layer_dir=self.raw_dir,
            name=name,
            formats=formats,
            compression=compression,
            level=level,
            layer_name="raw",
        )

    def write_standardized(
        self,
        adata: ad.AnnData,
        name: str,
        formats: List[str] = None,
        compression: str = "gzip",
        level: int = 9,
    ) -> Dict[str, Path]:
        """
        Write data to standardized storage layer.

        Args:
            adata: AnnData object to store
            name: Dataset name (used as filename)
            formats: List of output formats
            compression: Compression method
            level: Compression level

        Returns:
            Dictionary mapping format to output path
        """
        if formats is None:
            formats = ["h5ad"]

        return self._write_to_layer(
            adata=adata,
            layer_dir=self.standardized_dir,
            name=name,
            formats=formats,
            compression=compression,
            level=level,
            layer_name="standardized",
        )

    def write_analysis_ready(
        self,
        adata: ad.AnnData,
        name: str,
        formats: List[str] = None,
        compression: str = "gzip",
        level: int = 9,
    ) -> Dict[str, Path]:
        """
        Write data to analysis-ready storage layer.

        Args:
            adata: AnnData object to store
            name: Dataset name
            formats: List of output formats
            compression: Compression method
            level: Compression level

        Returns:
            Dictionary mapping format to output path
        """
        if formats is None:
            formats = ["h5ad"]

        return self._write_to_layer(
            adata=adata,
            layer_dir=self.analysis_ready_dir,
            name=name,
            formats=formats,
            compression=compression,
            level=level,
            layer_name="analysis_ready",
        )

    def _write_to_layer(
        self,
        adata: ad.AnnData,
        layer_dir: Path,
        name: str,
        formats: List[str],
        compression: str,
        level: int,
        layer_name: str,
    ) -> Dict[str, Path]:
        """
        Internal method to write data to a storage layer.

        Args:
            adata: AnnData object
            layer_dir: Target layer directory
            name: Dataset name
            formats: List of formats to save
            compression: Compression method
            level: Compression level
            layer_name: Name of layer (for logging)

        Returns:
            Dictionary mapping format to path
        """
        output_paths = {}

        for fmt in formats:
            try:
                if fmt == "h5ad":
                    path = layer_dir / f"{name}.h5ad"
                    adata.write_h5ad(path, compression=compression)
                    logger.info(f"Wrote {layer_name} h5ad: {path}")

                elif fmt == "parquet":
                    path = layer_dir / f"{name}.parquet"
                    self._write_parquet(adata, path, compression=compression)
                    logger.info(f"Wrote {layer_name} parquet: {path}")

                elif fmt == "zarr":
                    path = layer_dir / f"{name}.zarr"
                    self._write_zarr(adata, path, compression=compression)
                    logger.info(f"Wrote {layer_name} zarr: {path}")

                else:
                    logger.warning(f"Unknown format: {fmt}")
                    continue

                output_paths[fmt] = path

            except Exception as e:
                logger.error(f"Failed to write {fmt} for {name}: {str(e)}")
                continue

        return output_paths

    def read_raw(self, name: str, format: str = "h5ad", backed: bool = False) -> ad.AnnData:
        """
        Read data from raw storage layer.

        Args:
            name: Dataset name
            format: File format (h5ad, parquet, zarr)
            backed: If True and format='h5ad', use backed mode for large files.
                Allows lazy loading. Default: False.

        Returns:
            AnnData object
        """
        return self._read_from_layer(
            layer_dir=self.raw_dir,
            name=name,
            format=format,
            layer_name="raw",
            backed=backed,
        )

    def read_standardized(self, name: str, format: str = "h5ad", backed: bool = False) -> ad.AnnData:
        """
        Read data from standardized storage layer.

        Args:
            backed: If True and format='h5ad', use backed mode for large files.
                Default: False.
        """
        return self._read_from_layer(
            layer_dir=self.standardized_dir,
            name=name,
            format=format,
            layer_name="standardized",
            backed=backed,
        )

    def read_analysis_ready(self, name: str, format: str = "h5ad", backed: bool = False) -> ad.AnnData:
        """
        Read data from analysis-ready storage layer.

        Args:
            backed: If True and format='h5ad', use backed mode for large files.
                Default: False.
        """
        return self._read_from_layer(
            layer_dir=self.analysis_ready_dir,
            name=name,
            format=format,
            layer_name="analysis_ready",
            backed=backed,
        )

    def _read_from_layer(
        self,
        layer_dir: Path,
        name: str,
        format: str,
        layer_name: str,
        backed: bool = False,
    ) -> ad.AnnData:
        """
        Internal method to read data from a storage layer.

        Args:
            layer_dir: Source layer directory
            name: Dataset name
            format: File format
            layer_name: Name of layer (for logging)
            backed: If True and format='h5ad', use backed mode for lazy loading.
                Default: False.

        Returns:
            AnnData object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
        """
        if format == "h5ad":
            path = layer_dir / f"{name}.h5ad"
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            # OPTIMIZATION: Use backed='r' for large files to avoid loading full matrix
            if backed:
                adata = ad.read_h5ad(path, backed='r')
                logger.info(f"Read {layer_name} data (backed mode) from {path}: {adata.shape}")
            else:
                adata = ad.read_h5ad(path)
                logger.info(f"Read {layer_name} data from {path}: {adata.shape}")

        elif format == "parquet":
            path = layer_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            adata = self._read_parquet(path)
            logger.info(f"Read {layer_name} data from {path}: {adata.shape}")

        elif format == "zarr":
            path = layer_dir / f"{name}.zarr"
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            adata = self._read_zarr(path)
            logger.info(f"Read {layer_name} data from {path}: {adata.shape}")

        else:
            raise ValueError(f"Unknown format: {format}")

        return adata

    def _write_parquet(
        self,
        adata: ad.AnnData,
        path: Path,
        compression: str = "gzip",
        chunk_size: int = 10000,
    ) -> None:
        """Write AnnData to Parquet format with chunked conversion to avoid OOM."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        try:
            for start in range(0, adata.n_obs, chunk_size):
                end = min(start + chunk_size, adata.n_obs)
                chunk_X = adata.X[start:end]
                if hasattr(chunk_X, "toarray"):
                    chunk_X = chunk_X.toarray()

                df_chunk = pd.DataFrame(
                    chunk_X,
                    index=adata.obs_names[start:end],
                    columns=adata.var_names,
                )
                table = pa.Table.from_pandas(df_chunk)

                if writer is None:
                    writer = pq.ParquetWriter(str(path), table.schema, compression=compression)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def _read_parquet(self, path: Path) -> ad.AnnData:
        """Read AnnData from Parquet format."""
        df = pd.read_parquet(path, index_col=0)
        adata = ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))
        return adata

    def _write_zarr(
        self,
        adata: ad.AnnData,
        path: Path,
        compression: str = "gzip",
    ) -> None:
        """Write AnnData to Zarr format with tuned chunk sizes."""
        # Tune chunk sizes: ~10k cells per chunk, all genes per chunk
        chunks = (min(10000, adata.n_obs), adata.n_vars)
        adata.write_zarr(path, chunks=chunks)

    def _read_zarr(self, path: Path) -> ad.AnnData:
        """Read AnnData from Zarr format."""
        adata = ad.read_zarr(path)
        return adata

    def list_available(self, layer: str = "raw") -> List[str]:
        """
        List available datasets in a storage layer.

        Args:
            layer: Storage layer ("raw", "standardized", "analysis_ready")

        Returns:
            List of available dataset names

        Example:
            >>> sm = StorageManager()
            >>> datasets = sm.list_available("raw")
        """
        if layer == "raw":
            layer_dir = self.raw_dir
        elif layer == "standardized":
            layer_dir = self.standardized_dir
        elif layer == "analysis_ready":
            layer_dir = self.analysis_ready_dir
        else:
            raise ValueError(f"Unknown layer: {layer}")

        files = list(layer_dir.glob("*.h5ad")) + list(layer_dir.glob("*.parquet")) + list(layer_dir.glob("*.zarr"))
        names = set()

        for file in files:
            # Remove extension to get name
            name = file.stem
            # Handle .zarr.html and other edge cases
            if name.endswith(".zarr"):
                name = name[:-5]
            names.add(name)

        return sorted(list(names))

    def get_storage_info(self, layer: str = "raw") -> Dict[str, Union[int, List[str]]]:
        """
        Get storage information for a layer.

        Args:
            layer: Storage layer

        Returns:
            Dictionary with storage information
        """
        if layer == "raw":
            layer_dir = self.raw_dir
        elif layer == "standardized":
            layer_dir = self.standardized_dir
        elif layer == "analysis_ready":
            layer_dir = self.analysis_ready_dir
        else:
            raise ValueError(f"Unknown layer: {layer}")

        files = list(layer_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "layer": layer,
            "directory": str(layer_dir),
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "available_datasets": self.list_available(layer),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    sm = StorageManager()

    # Check available storage info
    for layer in ["raw", "standardized", "analysis_ready"]:
        info = sm.get_storage_info(layer)
        print(f"\n{layer.upper()} Storage Info:")
        print(f"  Directory: {info['directory']}")
        print(f"  Files: {info['total_files']}")
        print(f"  Size: {info['total_size_gb']} GB")
