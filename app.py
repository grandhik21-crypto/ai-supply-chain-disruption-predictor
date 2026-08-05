"""Streamlit UI for the AI Supply Chain Disruption Predictor.

Run in development mode with:

    streamlit run app.py

The app trains the disruption model once (cached) and lets a user enter
supplier metrics plus recent news headlines to get an explainable near-term
disruption risk assessment.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from supply_chain_predictor import SupplyChainPredictor
from supply_chain_predictor.data import sample_suppliers

st.set_page_config(
    page_title="AI Supply Chain Disruption Predictor",
    page_icon="🚚",
    layout="wide",
)

RISK_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}


@st.cache_resource(show_spinner="Training disruption model...")
def get_predictor() -> SupplyChainPredictor:
    return SupplyChainPredictor()


def risk_gauge(score: float, level: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%"},
            title={"text": f"Disruption risk — {level}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS.get(level, "#3498db")},
                "steps": [
                    {"range": [0, 25], "color": "#eafaf1"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 75], "color": "#fdf2e9"},
                    {"range": [75, 100], "color": "#fdedec"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def main() -> None:
    st.title("🚚 AI Supply Chain Disruption Predictor")
    st.caption(
        "Blends supplier operations metrics (ML) with NLP analysis of recent "
        "supply-chain news to estimate near-term disruption risk."
    )

    predictor = get_predictor()
    examples = sample_suppliers()

    with st.sidebar:
        st.header("Supplier metrics")
        preset_name = st.selectbox(
            "Load an example supplier", ["Custom"] + examples["name"].tolist()
        )
        if preset_name == "Custom":
            preset = {
                "lead_time_days": 21,
                "on_time_delivery_rate": 0.9,
                "inventory_days_of_supply": 30,
                "supplier_financial_health": 0.7,
                "geopolitical_risk_index": 0.3,
                "demand_volatility": 0.4,
                "single_source": 0,
            }
        else:
            preset = examples.set_index("name").loc[preset_name].to_dict()

        lead_time = st.slider("Lead time (days)", 1, 90, int(preset["lead_time_days"]))
        otd = st.slider(
            "On-time delivery rate", 0.0, 1.0, float(preset["on_time_delivery_rate"]), 0.01
        )
        inventory = st.slider(
            "Inventory days of supply", 1, 90, int(preset["inventory_days_of_supply"])
        )
        health = st.slider(
            "Supplier financial health", 0.0, 1.0, float(preset["supplier_financial_health"]), 0.01
        )
        geo = st.slider(
            "Geopolitical risk index", 0.0, 1.0, float(preset["geopolitical_risk_index"]), 0.01
        )
        volatility = st.slider(
            "Demand volatility", 0.0, 1.0, float(preset["demand_volatility"]), 0.01
        )
        single_source = st.checkbox("Single-source supplier", bool(preset["single_source"]))

    features = {
        "lead_time_days": lead_time,
        "on_time_delivery_rate": otd,
        "inventory_days_of_supply": inventory,
        "supplier_financial_health": health,
        "geopolitical_risk_index": geo,
        "demand_volatility": volatility,
        "single_source": int(single_source),
    }

    st.subheader("Recent supply-chain news")
    news = st.text_area(
        "One headline per line",
        value="Port workers announce strike amid shipping congestion\n"
        "Supplier warns of semiconductor shortage and shipment delays",
        height=120,
    )
    headlines = [line for line in news.splitlines() if line.strip()]

    if st.button("Assess disruption risk", type="primary"):
        result = predictor.assess(features, headlines)

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(
                risk_gauge(result.overall_risk, result.risk_level),
                use_container_width=True,
            )
        with right:
            st.metric("Overall risk", f"{result.overall_risk * 100:.1f}%", result.risk_level)
            st.metric("Model (metrics) risk", f"{result.ml_risk * 100:.1f}%")
            st.metric("News (NLP) risk", f"{result.news_risk * 100:.1f}%")

        st.success(f"Recommendation: {result.recommendation}")

        st.subheader("Top contributing factors (model)")
        for name, contribution in result.top_factors:
            direction = "increases" if contribution > 0 else "decreases"
            st.write(f"- **{name}** {direction} risk (contribution {contribution:+.3f})")

        if result.matched_risk_terms:
            st.subheader("News risk terms detected")
            st.write(", ".join(f"`{t}` ×{c}" for t, c in result.matched_risk_terms.items()))


if __name__ == "__main__":
    main()
