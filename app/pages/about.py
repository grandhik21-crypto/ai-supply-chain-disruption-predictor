"""
About page for the web application.

Presents project overview, key capabilities, architecture diagram, tech
stack, and development roadmap to end users.
"""

from __future__ import annotations

import streamlit as st

from app.pages.base_page import BasePage
from config.settings import get_settings


class AboutPage(BasePage):
    """Project overview, architecture, and roadmap."""

    @property
    def title(self) -> str:
        return "About"

    @property
    def icon(self) -> str:
        return "ℹ️"

    def render_content(self) -> None:
        settings = get_settings()

        st.markdown(
            f"""
            ### {settings.title}

            **Version:** {settings.version}

            An AI-powered platform for predicting and mitigating supply chain disruptions
            using natural language processing, machine learning, and industrial engineering
            analytics.
            """
        )

        st.markdown("---")
        st.markdown("### Key Capabilities")

        capabilities = st.columns(3)
        items = [
            ("🔮", "Disruption Prediction", "Forecast supply chain risks using ML models."),
            ("📰", "Sentiment Analysis", "Monitor news and market signals in real time."),
            ("📦", "Inventory Intelligence", "Track coverage and buffer adequacy."),
            ("🌍", "Regional Analytics", "Compare lead times and risk across regions."),
            ("🏭", "Supplier Scoring", "Rank suppliers by composite risk metrics."),
            ("📈", "Interactive Dashboards", "Explore insights with Plotly visualizations."),
        ]
        for index, (icon, title, desc) in enumerate(items):
            with capabilities[index % 3]:
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

        st.markdown("---")
        st.markdown("### Architecture")

        st.code(
            """
supply_chain_predictor/
├── app/                    # Streamlit UI layer
│   ├── main.py             # Application entry point
│   ├── components/         # Reusable UI components
│   └── pages/              # Page classes (OOP)
├── config/                 # Settings and constants
├── src/
│   ├── data/               # Data providers
│   ├── models/             # Domain models
│   └── services/           # Business logic
└── requirements.txt
            """.strip(),
            language="text",
        )

        st.markdown("---")
        st.markdown("### Roadmap")
        st.markdown(
            """
            - [x] Dashboard foundation with placeholder data
            - [ ] Integrate live supplier and ERP data feeds
            - [ ] Train NLP sentiment pipeline on news APIs
            - [ ] Deploy production ML inference service
            - [ ] Add alerting and notification workflows
            """
        )

        st.markdown("---")
        st.caption("Built with Python · Streamlit · Plotly · Pandas · NumPy")
