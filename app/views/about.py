"""
The About page.

Tells you what this project does, what tools it uses,
and what features are planned next.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

from app.views.base_page import BasePage  # Shared page template
from config.settings import get_settings  # App name and version


class AboutPage(BasePage):
    """Project overview, architecture, and roadmap."""

    @property
    def title(self) -> str:
        return "About"  # Page name in the header

    @property
    def icon(self) -> str:
        return "ℹ️"  # Info emoji

    def render_content(self) -> None:
        settings = get_settings()  # Load app title and version

        # Main project description block
        st.markdown(
            f"""
            ### {settings.title}

            **Version:** {settings.version}

            An AI-powered platform for predicting and mitigating supply chain disruptions
            using natural language processing, machine learning, and industrial engineering
            analytics.
            """
        )

        st.markdown("---")  # Divider
        st.markdown("### Key Capabilities")  # Section heading

        capabilities = st.columns(3)  # Three columns for feature cards
        # List of (emoji, title, short description) for each feature
        items = [
            ("🔮", "Disruption Prediction", "Forecast supply chain risks using ML models."),
            ("📰", "Sentiment Analysis", "Monitor news and market signals in real time."),
            ("📦", "Inventory Intelligence", "Track coverage and buffer adequacy."),
            ("🌍", "Regional Analytics", "Compare lead times and risk across regions."),
            ("🏭", "Supplier Scoring", "Rank suppliers by composite risk metrics."),
            ("📈", "Interactive Dashboards", "Explore insights with Plotly visualizations."),
        ]
        # Show each feature in one of the three columns (wraps after 3)
        for index, (icon, title, desc) in enumerate(items):
            with capabilities[index % 3]:
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

        st.markdown("---")
        st.markdown("### Architecture")  # Section showing folder structure

        # Display the project folder layout as a code block
        st.code(
            """
supply_chain_predictor/
├── app/                    # Streamlit UI layer
│   ├── main.py             # Application entry point
│   ├── components/         # Reusable UI components
│   └── views/              # Page classes (OOP)
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
        st.markdown("### Roadmap")  # Planned future work
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
        st.caption("Built with Python · Streamlit · Plotly · Pandas · NumPy")  # Footer
