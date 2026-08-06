# AI Supply Chain Disruption Predictor

AI-powered Supply Chain Disruption Predictor using NLP, Machine Learning, and Industrial Engineering analytics.

## Overview

A production-quality foundation for predicting supply chain disruptions. The application features a modular Python architecture and a modern Streamlit dashboard with KPI cards, Plotly charts, and sidebar navigation.

## Project Structure

Each source file has:
1. A short description at the top explaining what the file does
2. Line-by-line `#` comments explaining what each part of the code does

```
├── app/                        # Streamlit UI layer
│   ├── main.py                 # Home page (Dashboard)
│   ├── page_setup.py           # Shared sidebar/setup helpers
│   ├── pages/                  # Streamlit sidebar page links
│   │   ├── 1_Supplier_Analysis.py
│   │   ├── 2_Model_Insights.py
│   │   └── 3_About.py
│   ├── components/             # Reusable UI components
│   │   ├── sidebar.py
│   │   ├── kpi_cards.py
│   │   └── charts.py
│   └── views/                  # Real page UI code (OOP classes)
│       ├── base_page.py
│       ├── dashboard.py
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
│   │   ├── metrics.py
│   │   └── sentiment.py
│   ├── nlp/                    # NLP modules
│   │   └── sentiment.py        # FinBERT sentiment analysis
│   ├── features/               # Feature engineering for ML
│   │   └── feature_engineering.py
│   ├── ml/                     # Machine learning
│   │   ├── model.py            # XGBoost disruption predictor
│   │   └── explainability.py   # SHAP explanations for Streamlit
│   ├── services/               # Analytics services
│   │   └── analytics_service.py
│   └── utils/                  # Logging & shared utilities
│       └── logging_config.py
├── data/
│   └── raw/
│       ├── supply_chain.csv    # Sample supplier dataset
│       └── news_articles.csv   # Sample news for sentiment analysis
│   └── processed/              # Feature / sentiment output CSVs
├── models/                     # Saved trained models (joblib)
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

## Data Ingestion Pipeline

Load and clean CSV supply chain data with validation, date parsing, and missing-value handling:

```python
from src.data import SupplyChainDataLoader

loader = SupplyChainDataLoader()
df = loader.load()  # Returns a cleaned DataFrame
```

Use CSV-backed analytics in the dashboard service:

```python
from src.services.analytics_service import SupplyChainAnalyticsService

service = SupplyChainAnalyticsService(use_csv=True)
kpis = service.get_kpi_metrics()
```

Required CSV columns: `supplier_id`, `supplier_name`, `region`, `category`, `risk_score`, `lead_time_days`, `inventory_coverage_days`, `on_time_delivery_pct`, `sentiment_score`, `order_date`, `last_disruption`.

## Sentiment Analysis (FinBERT)

Analyze supplier news with Hugging Face FinBERT:

```python
from src.nlp.sentiment import SentimentAnalyzer, load_articles_from_csv

articles = load_articles_from_csv("data/raw/news_articles.csv")
analyzer = SentimentAnalyzer()
daily_sentiment = analyzer.run_pipeline(articles)
```

Or run from the command line:

```bash
python3 scripts/run_sentiment.py
```

**Pipeline steps:**
1. Load news articles (list or CSV)
2. Score each article → `sentiment_label`, `confidence`, `sentiment_score` (0–1)
3. Aggregate mean score per supplier per day
4. Save to `data/processed/daily_supplier_sentiment.csv` and `article_sentiment.csv`

## Feature Engineering Pipeline

Merge supply chain + sentiment and engineer ML features:

```bash
python3 scripts/run_feature_engineering.py
```

```python
from src.features.feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()
ml_df = pipeline.run_pipeline()
```

**Join keys:** `supplier_id` + `date`

**Engineered features:**
- `rolling_lead_time_7d` — 7-day rolling mean lead time
- `lead_time_variance_7d` — 7-day rolling variance of lead time
- `inventory_coverage` — days of inventory buffer
- `supplier_reliability` — on-time delivery rate (0–1)
- `rolling_sentiment_7d` — 7-day rolling mean sentiment
- `sentiment_velocity` — day-over-day sentiment change
- `negative_news_count_7d` — rolling count of negative articles

Output: `data/processed/ml_features.csv`

## Machine Learning Pipeline (XGBoost)

Train a shipment disruption classifier:

```bash
python3 scripts/train_model.py
```

```python
from src.ml.model import DisruptionPredictor

predictor = DisruptionPredictor()
metrics = predictor.run_pipeline()
print(metrics.as_dict())
```

**Pipeline steps:**
1. Load engineered features
2. Create / use disruption labels (`0` = normal, `1` = disruption)
3. Train/test split (80/20, stratified)
4. 5-fold cross-validation
5. Hyperparameter tuning (`GridSearchCV`)
6. Evaluate Accuracy, Precision, Recall, F1, ROC-AUC
7. Save model to `models/disruption_xgb.joblib`

## Explainable AI (SHAP)

The **Model Insights** page shows plain-language SHAP explanations:

- **Feature Importance** — which factors matter most overall
- **SHAP Summary** — how each factor pushes predictions safer or riskier
- **SHAP Waterfall** — why one supplier/day was scored that way

```python
from src.ml.explainability import ShapExplainer

explainer = ShapExplainer()
explainer.load()
explainer.plot_feature_importance()
```

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
