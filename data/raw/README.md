# Sample supply chain dataset for the web application.

`supply_chain.csv` is the default CSV loaded by `SupplyChainDataLoader` when
the dashboard runs in CSV mode (`SupplyChainAnalyticsService(use_csv=True)`).

It contains 10 supplier rows with intentional data quality issues (missing
values, mixed date formats) to demonstrate the ingestion pipeline.

Required columns match the schema in `src/data/preprocessing.py`.
