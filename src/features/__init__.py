"""
Feature engineering tools for machine learning.

Builds merged supply-chain + sentiment datasets with rolling features.
"""

from src.features.feature_engineering import FeatureEngineeringPipeline

__all__ = ["FeatureEngineeringPipeline"]
