"""
Integration module for the R3-MM pipeline.

This module provides batch effect correction and data integration functionality
using methods like Harmony, scVI, and scANVI.

Classes:
    HarmonyIntegrator: Harmony-based batch effect correction.
    ScVIIntegrator: scVI-based integration with VAE.
    ScANVIIntegrator: scANVI semi-supervised integration and annotation.
"""

from .harmony import HarmonyIntegrator
from .scanvi_integration import ScANVIIntegrator
from .scvi_integration import ScVIIntegrator

__all__ = [
    "HarmonyIntegrator",
    "ScVIIntegrator",
    "ScANVIIntegrator",
]
