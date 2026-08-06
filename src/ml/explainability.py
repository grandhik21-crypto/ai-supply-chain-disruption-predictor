"""
Explainable AI helpers using SHAP.

Explains why the disruption model made a prediction in simple terms:
  - Which factors matter most overall (feature importance)
  - How each factor pushes predictions up or down (SHAP summary)
  - Why one example prediction looks risky or safe (SHAP waterfall)
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For the explainer settings class
from pathlib import Path  # For model / data paths

import numpy as np  # Numeric arrays
import pandas as pd  # Tables
import plotly.express as px  # Easy charts
import plotly.graph_objects as go  # Waterfall and custom charts
import shap  # Explainable AI library

from src.ml.model import DEFAULT_MODEL_PATH, DisruptionPredictor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Friendly names so non-technical users can understand each feature
FRIENDLY_FEATURE_NAMES: dict[str, str] = {
    "risk_score": "Overall risk score",
    "rolling_lead_time_7d": "Average delivery time (7 days)",
    "lead_time_variance_7d": "Delivery time ups and downs",
    "inventory_coverage": "Days of inventory left",
    "supplier_reliability": "On-time delivery rate",
    "rolling_sentiment_7d": "Recent news mood (7 days)",
    "sentiment_velocity": "News mood changing up/down",
    "negative_news_count_7d": "Negative news count (7 days)",
    "sentiment_score_daily": "Today's news mood",
    "article_count": "Number of news articles",
    "region_Asia-Pacific": "Region: Asia-Pacific",
    "region_Europe": "Region: Europe",
    "region_Latin America": "Region: Latin America",
    "region_North America": "Region: North America",
    "category_Chemicals": "Category: Chemicals",
    "category_Electronics": "Category: Electronics",
    "category_Logistics": "Category: Logistics",
    "category_Packaging": "Category: Packaging",
    "category_Raw Materials": "Category: Raw Materials",
}


def friendly_name(feature: str) -> str:
    """Turn a technical column name into plain English."""
    if feature in FRIENDLY_FEATURE_NAMES:
        return FRIENDLY_FEATURE_NAMES[feature]
    # Fallback: replace underscores with spaces
    return feature.replace("_", " ").replace("region ", "Region: ").title()


@dataclass
class ShapExplainer:
    """
    Builds SHAP explanations for the trained disruption model.

    Example:
        explainer = ShapExplainer()
        explainer.load()
        importance_df = explainer.feature_importance()
    """

    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    max_samples: int = 100  # Keep SHAP fast for the website
    random_state: int = 42

    _predictor: DisruptionPredictor = field(init=False, repr=False)
    _explainer: shap.Explainer | None = field(default=None, init=False, repr=False)
    _X: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _shap_values: np.ndarray | None = field(default=None, init=False, repr=False)
    _base_value: float | None = field(default=None, init=False, repr=False)
    _meta: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        """Load the trained model + feature data, then compute SHAP values."""
        if self._loaded:
            return

        logger.info("Loading model and data for SHAP explanations...")
        self._predictor = DisruptionPredictor(model_path=self.model_path)
        self._predictor.load_model()

        raw_df = self._predictor.load_features()
        X, _ = self._predictor.prepare_xy(raw_df)

        # Align columns to what the model expects
        X = self._predictor._align_features(X)

        # Keep supplier/date labels for the dropdown on the website
        meta_cols = [c for c in ("supplier_id", "date") if c in raw_df.columns]
        self._meta = raw_df[meta_cols].copy().reset_index(drop=True)

        # Sample rows so the website stays fast
        if len(X) > self.max_samples:
            X = X.sample(n=self.max_samples, random_state=self.random_state)
            self._meta = self._meta.loc[X.index].reset_index(drop=True)
            X = X.reset_index(drop=True)
        else:
            X = X.reset_index(drop=True)
            self._meta = self._meta.reset_index(drop=True)

        self._X = X

        # TreeExplainer is the right SHAP tool for XGBoost models
        self._explainer = shap.TreeExplainer(self._predictor.model)
        shap_output = self._explainer.shap_values(X)

        # Binary classifiers may return a list [class0, class1] — use class 1 (disruption)
        if isinstance(shap_output, list):
            self._shap_values = np.asarray(shap_output[1])
        else:
            self._shap_values = np.asarray(shap_output)

        expected = self._explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            self._base_value = float(np.asarray(expected).reshape(-1)[-1])
        else:
            self._base_value = float(expected)

        self._loaded = True
        logger.info(
            "SHAP ready: %d samples, %d features, base_value=%.4f",
            len(self._X),
            self._X.shape[1],
            self._base_value,
        )

    @property
    def sample_labels(self) -> list[str]:
        """Dropdown labels like 'SUP-1000 · 2025-07-01'."""
        self.load()
        assert self._meta is not None
        labels: list[str] = []
        for idx, row in self._meta.iterrows():
            supplier = row.get("supplier_id", f"row-{idx}")
            date = row.get("date", "")
            labels.append(f"{supplier} · {date}")
        return labels

    def feature_importance(self) -> pd.DataFrame:
        """
        Overall feature importance = average |SHAP| value per feature.

        Bigger bar = this factor matters more to the model on average.
        """
        self.load()
        assert self._X is not None and self._shap_values is not None

        importance = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame(
            {
                "feature": self._X.columns,
                "Feature": [friendly_name(c) for c in self._X.columns],
                "Importance": importance,
            }
        ).sort_values("Importance", ascending=True)
        return df

    def summary_dataframe(self) -> pd.DataFrame:
        """Long-form table used to draw the SHAP summary (beeswarm-style) chart."""
        self.load()
        assert self._X is not None and self._shap_values is not None

        records: list[dict] = []
        for col_idx, feature in enumerate(self._X.columns):
            values = self._X.iloc[:, col_idx].to_numpy()
            shap_vals = self._shap_values[:, col_idx]
            # Normalize feature values 0–1 for color (low vs high)
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            if vmax > vmin:
                normed = (values - vmin) / (vmax - vmin)
            else:
                normed = np.zeros_like(values, dtype=float)

            for i in range(len(values)):
                records.append(
                    {
                        "Feature": friendly_name(feature),
                        "SHAP value": float(shap_vals[i]),
                        "Feature level": float(normed[i]),
                        "Raw value": float(values[i]),
                    }
                )
        return pd.DataFrame(records)

    def waterfall_dataframe(self, row_index: int = 0) -> pd.DataFrame:
        """
        Top factors for one prediction (waterfall chart).

        Positive SHAP = pushes toward disruption risk.
        Negative SHAP = pushes toward safer / normal.
        """
        self.load()
        assert self._X is not None and self._shap_values is not None

        row_index = int(np.clip(row_index, 0, len(self._X) - 1))
        shap_row = self._shap_values[row_index]
        feature_vals = self._X.iloc[row_index]

        df = pd.DataFrame(
            {
                "feature": self._X.columns,
                "Feature": [friendly_name(c) for c in self._X.columns],
                "SHAP value": shap_row,
                "Feature value": feature_vals.to_numpy(),
            }
        )
        df["abs_shap"] = df["SHAP value"].abs()
        df = df.sort_values("abs_shap", ascending=False).head(10)
        # Plot bottom-to-top with smallest of the top-10 at the bottom
        return df.sort_values("abs_shap", ascending=True).reset_index(drop=True)

    def prediction_summary(self, row_index: int = 0) -> dict[str, float | str | int]:
        """Plain-language summary for one selected example."""
        self.load()
        assert self._X is not None and self._predictor.model is not None

        row_index = int(np.clip(row_index, 0, len(self._X) - 1))
        X_row = self._X.iloc[[row_index]]
        proba = float(self._predictor.model.predict_proba(X_row)[0, 1])
        pred = int(proba >= 0.5)
        label = "Higher disruption risk" if pred == 1 else "Lower disruption risk"

        top = self.waterfall_dataframe(row_index).sort_values(
            "abs_shap", ascending=False
        ).head(3)

        drivers = []
        for _, row in top.iterrows():
            direction = "increased" if row["SHAP value"] > 0 else "decreased"
            drivers.append(f"{row['Feature']} {direction} the risk")

        return {
            "row_index": row_index,
            "probability": round(proba, 3),
            "prediction_label": label,
            "top_drivers": drivers,
            "base_value": round(float(self._base_value or 0.0), 4),
        }

    # ------------------------------------------------------------------
    # Plotly charts for Streamlit
    # ------------------------------------------------------------------

    def plot_feature_importance(self) -> go.Figure:
        """Bar chart: which factors matter most overall."""
        df = self.feature_importance()
        fig = px.bar(
            df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="What matters most to the model?",
            color="Importance",
            color_continuous_scale=["#bfdbfe", "#2563eb"],
        )
        fig.update_layout(
            template="plotly_white",
            height=420,
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Average impact on the prediction",
            yaxis_title="",
        )
        return fig

    def plot_shap_summary(self) -> go.Figure:
        """
        Summary chart (beeswarm-style).

        X = how much the factor pushed the prediction.
        Color = whether that factor's value was low or high.
        """
        df = self.summary_dataframe()
        # Keep top features by mean absolute SHAP for readability
        top_features = (
            df.groupby("Feature")["SHAP value"]
            .apply(lambda s: s.abs().mean())
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        plot_df = df[df["Feature"].isin(top_features)].copy()
        # Order features by importance on the Y axis
        order = list(top_features[::-1])

        fig = px.scatter(
            plot_df,
            x="SHAP value",
            y="Feature",
            color="Feature level",
            color_continuous_scale=["#3b82f6", "#ef4444"],
            title="How each factor pushes predictions (SHAP summary)",
            opacity=0.65,
            category_orders={"Feature": order},
        )
        fig.update_traces(marker=dict(size=7))
        fig.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="← Safer          Riskier →",
            yaxis_title="",
            coloraxis_colorbar=dict(title="Low → High value"),
        )
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#94a3b8")
        return fig

    def plot_shap_waterfall(self, row_index: int = 0) -> go.Figure:
        """Waterfall chart explaining one prediction in plain terms."""
        df = self.waterfall_dataframe(row_index)
        summary = self.prediction_summary(row_index)

        fig = go.Figure(
            go.Waterfall(
                name="Impact",
                orientation="h",
                measure=["relative"] * len(df),
                y=df["Feature"],
                x=df["SHAP value"],
                text=[f"{v:+.3f}" for v in df["SHAP value"]],
                textposition="outside",
                connector={"line": {"color": "#cbd5e1"}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#3b82f6"}},
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=480,
            title=(
                f"Why this prediction? "
                f"{summary['prediction_label']} "
                f"({summary['probability']:.0%} chance)"
            ),
            xaxis_title="Impact on disruption risk  (red = riskier, blue = safer)",
            yaxis_title="",
            margin=dict(l=40, r=40, t=70, b=40),
            showlegend=False,
        )
        return fig
