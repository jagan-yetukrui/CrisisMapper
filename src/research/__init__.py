"""
Research and experimentation modules for CrisisMapper.

This package provides advanced research capabilities including model comparison,
experimentation frameworks, and scientific analysis tools.
"""

from .experiment_manager import ExperimentManager
from .model_comparison import ModelComparison
from .metrics_calculator import MetricsCalculator
from .data_analyzer import DataAnalyzer

__all__ = ["ExperimentManager", "ModelComparison", "MetricsCalculator", "DataAnalyzer"]
