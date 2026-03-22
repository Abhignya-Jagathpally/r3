"""
Models module for the R3-MM pipeline.

This module provides machine learning models and training functionality
for downstream analyses, including:
- Classical baselines (LogisticRegression, RandomForest, SVM, Ensemble)
- Foundation models (scGPT wrapper)
- Multimodal fusion strategies

Following the principle: classical baselines first, then foundation models,
then multimodal fusion.
"""

from src.models.classical_baselines import (
    ClassicalEnsemble,
    LogisticBaseline,
    RandomForestBaseline,
    SVMBaseline,
)
from src.models.multimodal_fusion import MultimodalFuser
from src.models.scgpt_wrapper import ScGPTConfig, ScGPTModel

__all__ = [
    "LogisticBaseline",
    "RandomForestBaseline",
    "SVMBaseline",
    "ClassicalEnsemble",
    "ScGPTConfig",
    "ScGPTModel",
    "MultimodalFuser",
]
