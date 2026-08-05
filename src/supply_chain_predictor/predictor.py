"""High-level API combining the ML model and NLP news analysis.

`SupplyChainPredictor` is the main entry point used by the Streamlit app and by
tests. It trains the disruption model on the synthetic dataset (unless one is
provided) and blends the model probability with the news-risk score into a
single, explainable disruption assessment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .data import generate_supplier_dataset
from .model import FEATURE_NAMES, DisruptionModel
from .nlp import NewsRiskAnalyzer

# Ordered from safest to most severe. Each entry is (label, upper_bound].
RISK_LEVELS: list[tuple[str, float]] = [
    ("Low", 0.25),
    ("Moderate", 0.5),
    ("High", 0.75),
    ("Critical", 1.01),
]


def _risk_level(score: float) -> str:
    for label, upper in RISK_LEVELS:
        if score < upper:
            return label
    return RISK_LEVELS[-1][0]


def _recommendation(level: str) -> str:
    return {
        "Low": "No action needed. Continue routine monitoring.",
        "Moderate": "Increase monitoring cadence and confirm buffer stock levels.",
        "High": "Qualify an alternate supplier and raise safety stock now.",
        "Critical": "Activate contingency plan: dual-source and expedite inbound orders.",
    }[level]


@dataclass
class DisruptionAssessment:
    """Result of a full supplier disruption assessment."""

    overall_risk: float
    risk_level: str
    recommendation: str
    ml_risk: float
    news_risk: float
    ml_weight: float
    news_weight: float
    top_factors: list[tuple[str, float]] = field(default_factory=list)
    matched_risk_terms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SupplyChainPredictor:
    """Blends supplier-metric ML risk with NLP news risk."""

    def __init__(
        self,
        model: DisruptionModel | None = None,
        analyzer: NewsRiskAnalyzer | None = None,
        news_weight: float = 0.35,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= news_weight <= 1.0:
            raise ValueError("news_weight must be in [0, 1]")
        self.news_weight = news_weight
        self.ml_weight = 1.0 - news_weight
        self.analyzer = analyzer or NewsRiskAnalyzer()
        self.model = model or self._train_default_model(seed)

    @staticmethod
    def _train_default_model(seed: int) -> DisruptionModel:
        df = generate_supplier_dataset(seed=seed)
        model = DisruptionModel(seed=seed)
        model.fit(df[FEATURE_NAMES], df["disrupted"])
        return model

    def assess(
        self,
        supplier_features: dict[str, float],
        news_headlines: list[str] | str | None = None,
    ) -> DisruptionAssessment:
        """Assess disruption risk for one supplier.

        `supplier_features` must contain every key in ``FEATURE_NAMES``.
        `news_headlines` is optional free text; when absent, only the ML model
        contributes to the score.
        """
        missing = [f for f in FEATURE_NAMES if f not in supplier_features]
        if missing:
            raise KeyError(f"Missing supplier features: {missing}")

        row = [float(supplier_features[f]) for f in FEATURE_NAMES]
        ml_risk = float(self.model.predict_proba([row])[0])

        news_result = self.analyzer.analyze(news_headlines or [])
        news_risk = news_result.score

        if news_result.headline_count == 0:
            overall = ml_risk
            ml_w, news_w = 1.0, 0.0
        else:
            overall = self.ml_weight * ml_risk + self.news_weight * news_risk
            ml_w, news_w = self.ml_weight, self.news_weight

        contributions = self.model.feature_contributions([row])
        top_factors = sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:3]

        level = _risk_level(overall)
        return DisruptionAssessment(
            overall_risk=round(overall, 4),
            risk_level=level,
            recommendation=_recommendation(level),
            ml_risk=round(ml_risk, 4),
            news_risk=round(news_risk, 4),
            ml_weight=ml_w,
            news_weight=news_w,
            top_factors=top_factors,
            matched_risk_terms=news_result.matched_risk_terms,
        )
