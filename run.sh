#!/usr/bin/env bash
# Helper script to install dependencies and launch the Streamlit web dashboard.
# Usage: ./run.sh

set -euo pipefail

cd "$(dirname "$0")"

python3 -m pip install -r requirements.txt -q
python3 -m streamlit run app/main.py
