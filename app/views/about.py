"""
The About page.

Tells you what this project does, what tools it uses,
and what features are planned next.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

from app.components.ui import metric_card, section_header
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

    @property
    def subtitle(self) -> str:
        return "What this app does, how it works, and what's next"

    def render_content(self) -> None:
        settings = get_settings()  # Load app title and version

        # Main project description block
        st.markdown(
            f"""
            **{settings.title}** · version {settings.version}

            A platform that predicts and helps prevent supply chain disruptions using
            natural language processing, machine learning, and supply chain analytics.
            """
        )

        section_header("✨", "Key capabilities", "What the app can do today.")

        capabilities = st.columns(3)  # Three columns for feature cards
        # List of (emoji, title, short description) for each feature
        items = [
            ("🔮", "Disruption Prediction", "Forecast supply chain risks using ML models."),
            ("📰", "Sentiment Analysis", "Track news and market mood with FinBERT."),
            ("📦", "Inventory Intelligence", "Watch stock coverage and buffer levels."),
            ("🌍", "Regional Analytics", "Compare lead times and risk across regions."),
            ("🏭", "Supplier Scoring", "Rank suppliers by combined risk measures."),
            ("🧠", "Explainable AI", "See why the model made each prediction (SHAP)."),
        ]
        # Show each feature in one of the three columns (wraps after 3)
        for index, (icon, title, desc) in enumerate(items):
            with capabilities[index % 3]:
                metric_card(label=title, value=icon, caption=desc)

        section_header("🧱", "Architecture", "How the project folders fit together.")

        # Display the project folder layout as a code block
        st.code(
            """
supply_chain_predictor/
├── app/                    # Streamlit UI layer
│   ├── main.py             # Home / Dashboard
│   ├── pages/              # Sidebar page links
│   ├── views/              # Real page UI code
│   └── components/         # Charts, KPI cards, UI helpers
├── config/                 # Settings and constants
├── src/
│   ├── data/               # Data loading and cleaning
│   ├── nlp/                # FinBERT sentiment analysis
│   ├── features/           # Feature engineering
│   ├── ml/                 # XGBoost model + SHAP
│   └── services/           # Predictions, reports, analytics
├── scripts/                # Command-line pipelines
└── requirements.txt
            """.strip(),
            language="text",
        )

        section_header("🚀", "Setup order", "Run these once before using the dashboard.")
        st.code(
            """
python3 -m pip install -r requirements.txt
python3 scripts/run_sentiment.py
python3 scripts/run_feature_engineering.py
python3 scripts/train_model.py
python3 -m streamlit run app/main.py --server.port 8501
            """.strip(),
            language="bash",
        )

        section_header("🗺️", "Roadmap", "What's done and what's coming.")
        st.markdown(
            """
            - [x] Dashboard foundation
            - [x] CSV data ingestion and cleaning
            - [x] FinBERT news sentiment analysis
            - [x] Feature engineering pipeline
            - [x] XGBoost disruption model
            - [x] SHAP explainable AI
            - [x] Live predictions on the dashboard
            - [ ] Upload your own supplier data in the app
            - [ ] Live news and ERP data feeds
            - [ ] Email / Slack alerting
            """
        )

        st.divider()
        st.caption(
            "Built with Python · Streamlit · Plotly · Pandas · scikit-learn · XGBoost · SHAP"
        )
