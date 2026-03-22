"""
Agentic module for the R3-MM pipeline.

This module provides agentic tuning capabilities for hyperparameter optimization
and pipeline parameter search following Karpathy's autoresearch pattern.

Key components:
    - AgenticConfig: Configuration management
    - SearchSpace: Hyperparameter search space definition
    - ContractEnforcer: Frozen preprocessing contract enforcement
    - ExperimentRunner: Core search loop
    - ExperimentTracker: Trial result tracking
    - RayTuner: Parallel distributed tuning with Ray
    - DaskTuner: Distributed tuning with Dask
    - AutoResearchAgent: High-level orchestration
    - ReportGenerator: Automated report generation
"""

from .autoresearch_agent import AutoResearchAgent, AutoResearchResult
from .config import AgenticConfig, SearchSpaceConfig, TunerConfig
from .contract_enforcer import ContractEnforcer
from .dask_tuner import DaskTuner
from .experiment_runner import ExperimentRunner, ExperimentTracker
from .ray_tuner import RayTuner
from .report_generator import ReportGenerator
from .search_space import SearchSpace

__all__ = [
    # Configuration
    "AgenticConfig",
    "SearchSpaceConfig",
    "TunerConfig",
    # Core components
    "SearchSpace",
    "ContractEnforcer",
    "ExperimentRunner",
    "ExperimentTracker",
    # Distributed tuning
    "RayTuner",
    "DaskTuner",
    # High-level API
    "AutoResearchAgent",
    "AutoResearchResult",
    # Reporting
    "ReportGenerator",
]
