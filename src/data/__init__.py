"""
Data handling module for the R3-MM pipeline.

This module provides functionality for downloading, loading, and managing
single-cell RNA-seq data from multiple sources.

Submodules:
    - download: Downloading data from GEO
    - storage: Data storage management with staging layers
"""

from src.data.download import download_gse_data
from src.data.storage import StorageManager

__all__ = ["download_gse_data", "StorageManager"]
