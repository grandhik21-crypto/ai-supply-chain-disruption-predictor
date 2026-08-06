"""
Live prediction service for the dashboard.

Loads the trained XGBoost model and the engineered feature table, then
answers per-supplier questions the Dashboard needs:

  - Risk score
  - Delay (disruption) probability
  - Prediction confidence
  - Historical lead time / sentiment / inventory
  - Top risk factors (from SHAP)
  - Recommended actions
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For settings + result classes
from pathlib import Path  # For file paths

import numpy as np  # Numeric helpers
import pandas as pd  # Tables

from src.data.data_loader import DEFAULT_DATA_PATH  # Raw supplier CSV (for names)
from src.features.feature_engineering import DEFAULT_ML_OUTPUT_PATH
from src.ml.explainability import friendly_name
from src.ml.model import DEFAULT_MODEL_PATH, DisruptionPredictor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SupplierPrediction:
    """One supplier's latest live prediction from the model."""

    supplier_id: str  # e.g. "SUP-1000"
    supplier_name: str  # Company name (if known)
    region: str  # World region
    category: str  # What they supply
    as_of_date: str  # Date of the latest data row used

    risk_score: float  # Business risk score from the data (0–100)
    delay_probability: float  # Model output: chance of disruption (0–1)
    confidence: float  # How sure the model is (0–1)
    prediction_label: str  # "Likely disruption" / "Likely on time"

    lead_time_days: float  # Recent average delivery days
    inventory_coverage_days: float  # Days of stock left
    sentiment_score: float  # Recent news mood (0–1)
    reliability: float  # On-time delivery rate (0–1)
    negative_news_count: int  # Negative articles in last 7 days

    def as_dict(self) -> dict:
        """Return all fields as a plain dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "region": self.region,
            "category": self.category,
            "as_of_date": self.as_of_date,
            "risk_score": self.risk_score,
            "delay_probability": self.delay_probability,
            "confidence": self.confidence,
            "prediction_label": self.prediction_label,
            "lead_time_days": self.lead_time_days,
            "inventory_coverage_days": self.inventory_coverage_days,
            "sentiment_score": self.sentiment_score,
            "reliability": self.reliability,
            "negative_news_count": self.negative_news_count,
        }


@dataclass
class SupplierPredictionService:
    """
    Turns the trained model + feature table into dashboard-ready answers.

    Example:
        service = SupplierPredictionService()
        service.load()
        pred = service.get_supplier_prediction("SUP-1000")
    """

    features_path: Path = field(default_factory=lambda: DEFAULT_ML_OUTPUT_PATH)
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    supplier_csv_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)

    _predictor: DisruptionPredictor | None = field(default=None, init=False, repr=False)
    _features: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _X: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _probabilities: np.ndarray | None = field(default=None, init=False, repr=False)
    _names: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _shap_values: np.ndarray | None = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def model_available(self) -> bool:
        """True when a trained model file exists on disk."""
        return self.model_path.exists()

    def load(self) -> None:
        """Load model, features, and compute predictions for every row once."""
        if self._loaded:
            return

        if not self.model_available:
            raise FileNotFoundError(
                f"No trained model at {self.model_path}. "
                "Run: python3 scripts/train_model.py"
            )

        logger.info("Loading live prediction service...")
        predictor = DisruptionPredictor(
            features_path=self.features_path,
            model_path=self.model_path,
        )
        predictor.load_model()

        features = predictor.load_features()
        X, _ = predictor.prepare_xy(features)
        X = predictor._align_features(X)

        # Predict disruption probability for every supplier-day row at once
        probabilities = predictor.model.predict_proba(X)[:, 1]

        self._predictor = predictor
        self._features = features.reset_index(drop=True)
        self._X = X.reset_index(drop=True)
        self._probabilities = probabilities
        self._names = self._load_supplier_names()
        self._loaded = True

        logger.info(
            "Prediction service ready: %d rows, %d suppliers",
            len(self._features),
            self._features["supplier_id"].nunique(),
        )

    def _load_supplier_names(self) -> dict[str, str]:
        """Map supplier_id → company name using the raw supplier CSV."""
        try:
            raw = pd.read_csv(self.supplier_csv_path)
            raw.columns = raw.columns.str.strip().str.lower()
            if {"supplier_id", "supplier_name"}.issubset(raw.columns):
                return dict(zip(raw["supplier_id"], raw["supplier_name"]))
        except Exception as exc:  # Names are optional
            logger.warning("Could not load supplier names: %s", exc)
        return {}

    def _shap_matrix(self) -> np.ndarray:
        """Compute SHAP values lazily (only when risk factors are requested)."""
        if self._shap_values is not None:
            return self._shap_values

        self.load()
        assert self._X is not None and self._predictor is not None

        import shap  # Imported here so the dashboard loads fast

        explainer = shap.TreeExplainer(self._predictor.model)
        values = explainer.shap_values(self._X)
        if isinstance(values, list):
            values = values[1]
        self._shap_values = np.asarray(values)
        return self._shap_values

    # ------------------------------------------------------------------
    # Supplier list / dropdown
    # ------------------------------------------------------------------

    def get_supplier_ids(self) -> list[str]:
        """All supplier IDs that have feature data, sorted."""
        self.load()
        assert self._features is not None
        return sorted(self._features["supplier_id"].astype(str).unique())

    def get_supplier_label(self, supplier_id: str) -> str:
        """Dropdown label like 'SUP-1000 — Nova Components Ltd.'."""
        name = self._names.get(supplier_id)
        return f"{supplier_id} — {name}" if name else supplier_id

    # ------------------------------------------------------------------
    # Per-supplier data slices
    # ------------------------------------------------------------------

    def get_supplier_history(self, supplier_id: str) -> pd.DataFrame:
        """All feature rows for one supplier, sorted by date, with predictions."""
        self.load()
        assert self._features is not None and self._probabilities is not None

        df = self._features.copy()
        df["delay_probability"] = self._probabilities
        supplier_df = df[df["supplier_id"].astype(str) == str(supplier_id)].copy()
        supplier_df["date"] = pd.to_datetime(supplier_df["date"])
        return supplier_df.sort_values("date").reset_index(drop=True)

    def _latest_row_index(self, supplier_id: str) -> int:
        """Row position (in the full table) of this supplier's newest date."""
        self.load()
        assert self._features is not None

        df = self._features.copy()
        df["_date"] = pd.to_datetime(df["date"])
        mask = df["supplier_id"].astype(str) == str(supplier_id)
        if not mask.any():
            raise ValueError(f"No data for supplier '{supplier_id}'")
        return int(df.loc[mask, "_date"].idxmax())

    def get_supplier_prediction(self, supplier_id: str) -> SupplierPrediction:
        """Live model prediction using this supplier's most recent data row."""
        self.load()
        assert self._features is not None and self._probabilities is not None

        idx = self._latest_row_index(supplier_id)
        row = self._features.loc[idx]
        proba = float(self._probabilities[idx])

        # Confidence = how far the probability is from a coin flip
        confidence = float(max(proba, 1.0 - proba))
        label = "Likely disruption" if proba >= 0.5 else "Likely on time"

        return SupplierPrediction(
            supplier_id=str(row["supplier_id"]),
            supplier_name=self._names.get(str(row["supplier_id"]), "Unknown supplier"),
            region=str(row.get("region", "Unknown")),
            category=str(row.get("category", "Unknown")),
            as_of_date=str(row.get("date", "")),
            risk_score=round(float(pd.to_numeric(row.get("risk_score"), errors="coerce") or 0.0), 1),
            delay_probability=round(proba, 4),
            confidence=round(confidence, 4),
            prediction_label=label,
            lead_time_days=round(
                float(pd.to_numeric(row.get("rolling_lead_time_7d"), errors="coerce") or 0.0), 1
            ),
            inventory_coverage_days=round(
                float(pd.to_numeric(row.get("inventory_coverage"), errors="coerce") or 0.0), 1
            ),
            sentiment_score=round(
                float(
                    pd.to_numeric(row.get("rolling_sentiment_7d"), errors="coerce")
                    if pd.notna(row.get("rolling_sentiment_7d"))
                    else 0.5
                ),
                2,
            ),
            reliability=round(
                float(pd.to_numeric(row.get("supplier_reliability"), errors="coerce") or 0.0), 3
            ),
            negative_news_count=int(
                pd.to_numeric(row.get("negative_news_count_7d"), errors="coerce") or 0
            ),
        )

    def get_lead_time_history(self, supplier_id: str) -> pd.DataFrame:
        """Historical lead time chart data for one supplier."""
        df = self.get_supplier_history(supplier_id)
        out = df[["date", "rolling_lead_time_7d"]].dropna()
        return out.rename(
            columns={"date": "Date", "rolling_lead_time_7d": "Lead Time (days)"}
        )

    def get_sentiment_history(self, supplier_id: str) -> pd.DataFrame:
        """Sentiment trend chart data for one supplier."""
        df = self.get_supplier_history(supplier_id)
        out = df[["date", "rolling_sentiment_7d"]].dropna()
        if out.empty:
            # Fall back to daily sentiment when the rolling column is empty
            out = df[["date", "sentiment_score_daily"]].dropna()
            return out.rename(
                columns={"date": "Date", "sentiment_score_daily": "Sentiment Score"}
            )
        return out.rename(
            columns={"date": "Date", "rolling_sentiment_7d": "Sentiment Score"}
        )

    def get_inventory_history(self, supplier_id: str) -> pd.DataFrame:
        """Inventory coverage chart data for one supplier."""
        df = self.get_supplier_history(supplier_id)
        out = df[["date", "inventory_coverage"]].dropna()
        return out.rename(
            columns={"date": "Date", "inventory_coverage": "Inventory Coverage (days)"}
        )

    def get_delay_probability_history(self, supplier_id: str) -> pd.DataFrame:
        """Model-predicted disruption probability over time."""
        df = self.get_supplier_history(supplier_id)
        out = df[["date", "delay_probability"]].dropna().copy()
        out["delay_probability"] = (out["delay_probability"] * 100).round(1)
        return out.rename(
            columns={"date": "Date", "delay_probability": "Delay Probability (%)"}
        )

    # ------------------------------------------------------------------
    # Explanations + actions
    # ------------------------------------------------------------------

    def get_top_risk_factors(self, supplier_id: str, top_n: int = 6) -> pd.DataFrame:
        """
        Top factors driving this supplier's prediction (from SHAP).

        Positive impact = pushes toward disruption.
        Negative impact = pushes toward on-time.
        """
        self.load()
        assert self._X is not None

        shap_values = self._shap_matrix()
        idx = self._latest_row_index(supplier_id)

        df = pd.DataFrame(
            {
                "Factor": [friendly_name(c) for c in self._X.columns],
                "Impact": shap_values[idx],
                "Value": self._X.iloc[idx].to_numpy(),
            }
        )
        df["abs_impact"] = df["Impact"].abs()
        df["Direction"] = np.where(df["Impact"] > 0, "Increases risk", "Reduces risk")
        top = df.sort_values("abs_impact", ascending=False).head(top_n)
        return top.sort_values("abs_impact", ascending=True).reset_index(drop=True)

    def get_recommended_actions(self, prediction: SupplierPrediction) -> list[dict[str, str]]:
        """
        Turn the prediction into clear next steps a person can act on.

        Each action has a level ("urgent", "watch", "ok") and plain text.
        """
        actions: list[dict[str, str]] = []
        p = prediction

        if p.delay_probability >= 0.7:
            actions.append(
                {
                    "level": "urgent",
                    "text": (
                        f"High disruption risk ({p.delay_probability:.0%}). "
                        "Contact this supplier now and confirm delivery dates."
                    ),
                }
            )
        elif p.delay_probability >= 0.4:
            actions.append(
                {
                    "level": "watch",
                    "text": (
                        f"Moderate disruption risk ({p.delay_probability:.0%}). "
                        "Monitor this supplier weekly."
                    ),
                }
            )
        else:
            actions.append(
                {
                    "level": "ok",
                    "text": (
                        f"Low disruption risk ({p.delay_probability:.0%}). "
                        "No urgent action needed."
                    ),
                }
            )

        if p.inventory_coverage_days < 15:
            actions.append(
                {
                    "level": "urgent",
                    "text": (
                        f"Only {p.inventory_coverage_days:.0f} days of inventory left. "
                        "Place a replenishment order or raise safety stock."
                    ),
                }
            )
        elif p.inventory_coverage_days < 25:
            actions.append(
                {
                    "level": "watch",
                    "text": (
                        f"Inventory coverage is {p.inventory_coverage_days:.0f} days. "
                        "Plan the next order soon."
                    ),
                }
            )

        if p.lead_time_days >= 35:
            actions.append(
                {
                    "level": "watch",
                    "text": (
                        f"Long delivery time ({p.lead_time_days:.0f} days). "
                        "Order earlier or find a closer backup supplier."
                    ),
                }
            )

        if p.reliability < 0.85:
            actions.append(
                {
                    "level": "watch",
                    "text": (
                        f"On-time delivery is only {p.reliability:.0%}. "
                        "Review their performance in the next supplier meeting."
                    ),
                }
            )

        if p.sentiment_score < 0.4 or p.negative_news_count > 0:
            actions.append(
                {
                    "level": "watch",
                    "text": (
                        "Recent news about this supplier looks negative. "
                        "Check for strikes, shortages, or financial trouble."
                    ),
                }
            )

        if p.risk_score >= 70:
            actions.append(
                {
                    "level": "urgent",
                    "text": (
                        f"Risk score is high ({p.risk_score:.0f}/100). "
                        "Consider approving a backup (dual-source) supplier."
                    ),
                }
            )

        return actions

    # ------------------------------------------------------------------
    # Portfolio-level numbers (all suppliers)
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> dict[str, float | int]:
        """Overall numbers across every supplier's latest prediction."""
        self.load()
        predictions = [self.get_supplier_prediction(sid) for sid in self.get_supplier_ids()]

        if not predictions:
            return {
                "supplier_count": 0,
                "avg_delay_probability": 0.0,
                "high_risk_count": 0,
                "avg_lead_time": 0.0,
                "avg_inventory": 0.0,
                "avg_sentiment": 0.0,
            }

        return {
            "supplier_count": len(predictions),
            "avg_delay_probability": round(
                float(np.mean([p.delay_probability for p in predictions])), 4
            ),
            "high_risk_count": int(sum(1 for p in predictions if p.delay_probability >= 0.5)),
            "avg_lead_time": round(float(np.mean([p.lead_time_days for p in predictions])), 1),
            "avg_inventory": round(
                float(np.mean([p.inventory_coverage_days for p in predictions])), 1
            ),
            "avg_sentiment": round(float(np.mean([p.sentiment_score for p in predictions])), 2),
        }

    def get_all_predictions_dataframe(self) -> pd.DataFrame:
        """Table of every supplier's latest prediction (for ranking/tables)."""
        self.load()
        rows = [self.get_supplier_prediction(sid).as_dict() for sid in self.get_supplier_ids()]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("delay_probability", ascending=False).reset_index(drop=True)

    def get_model_metrics(self) -> dict:
        """Saved evaluation metrics from the trained model file."""
        if not self.model_available:
            return {}
        try:
            import joblib

            payload = joblib.load(self.model_path)
            return dict(payload.get("metrics", {}))
        except Exception as exc:
            logger.warning("Could not read model metrics: %s", exc)
            return {}
