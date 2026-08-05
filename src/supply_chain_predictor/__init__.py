"""AI-powered Supply Chain Disruption Predictor.

Combines industrial-engineering supplier metrics (via a scikit-learn model)
with NLP risk analysis of supply-chain news to estimate near-term disruption
risk for a supplier.
"""

from .model import FEATURE_NAMES, DisruptionModel
from .nlp import NewsRiskAnalyzer
from .predictor import RISK_LEVELS, SupplyChainPredictor

__all__ = [
    "DisruptionModel",
    "FEATURE_NAMES",
    "NewsRiskAnalyzer",
    "SupplyChainPredictor",
    "RISK_LEVELS",
]

__version__ = "0.1.0"
