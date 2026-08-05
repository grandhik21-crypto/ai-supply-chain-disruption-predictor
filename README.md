# AI Supply Chain Disruption Predictor

AI-powered Supply Chain Disruption Predictor using NLP, Machine Learning, and Industrial Engineering analytics.

## Overview

A production-quality foundation for predicting supply chain disruptions. The application features a modular Python architecture and a modern Streamlit dashboard with KPI cards, Plotly charts, and sidebar navigation.

## Project Structure

```
├── app/                        # Streamlit UI layer
│   ├── main.py                 # Application entry point & router
│   ├── components/             # Reusable UI components
│   │   ├── sidebar.py          # Sidebar navigation
│   │   ├── kpi_cards.py        # KPI metric cards
│   │   └── charts.py           # Plotly chart factory
│   └── pages/                  # Page classes (OOP)
│       ├── base_page.py        # Abstract base page
│       ├── dashboard.py        # Executive dashboard
│       ├── supplier_analysis.py
│       ├── model_insights.py
│       └── about.py
├── config/                     # Application settings
│   └── settings.py
├── src/                        # Core business logic
│   ├── data/                   # Data ingestion & providers
│   │   ├── data_loader.py      # CSV ingestion pipeline
│   │   ├── preprocessing.py    # Validation & cleaning
│   │   ├── csv_provider.py     # CSV-backed data provider
│   │   └── placeholder_provider.py
│   ├── models/                 # Domain models
│   │   └── metrics.py
│   ├── services/               # Analytics services
│   │   └── analytics_service.py
│   └── utils/                  # Logging & shared utilities
│       └── logging_config.py
├── data/
│   └── raw/
│       └── supply_chain.csv    # Sample dataset
├── .streamlit/
│   └── config.toml             # Streamlit theme & config
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run the dashboard (use python3 -m — the streamlit command may not be on PATH)
python3 -m streamlit run app/main.py
```

Or use the helper script:

```bash
chmod +x run.sh
./run.sh
```

The app opens at `http://localhost:8501`.

> **Note:** On this environment, `python` and bare `streamlit` may not be found. Always use `python3` and `python3 -m streamlit` instead.

## Pages

| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards, risk trends, inventory coverage, sentiment, and disruption forecasts |
| **Supplier Analysis** | Supplier scatter plots, regional lead times, high-risk registry |
| **Model Insights** | Feature importance, confusion matrix, forecast outputs |
| **About** | Project overview, architecture, and roadmap |

## KPI Metrics

- **Risk Score** — Composite supply chain risk index
- **Lead Time** — Average supplier lead time (days)
- **Inventory Coverage** — Days of inventory buffer
- **Sentiment Score** — NLP-derived market sentiment (0–1)

## Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts and visualizations
- **Pandas / NumPy** — Data processing

## License

MIT
