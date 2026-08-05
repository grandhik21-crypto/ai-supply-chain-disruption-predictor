#!/usr/bin/env bash
# Easy way to start the website.
# Just run: ./run.sh
# It installs what you need and opens the app in your browser.

set -euo pipefail  # Stop the script if any command fails

cd "$(dirname "$0")"  # Move into the folder where this script lives

python3 -m pip install -r requirements.txt -q  # Install required Python packages (quiet mode)
python3 -m streamlit run app/main.py  # Start the Streamlit web app
