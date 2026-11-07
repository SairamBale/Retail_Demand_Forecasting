#!/usr/bin/env bash
# One-click setup + run (macOS/Linux).  Usage: bash setup_and_run.sh
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
python build_notebook.py
echo "Done. Open notebooks/retail_demand_analysis.ipynb and reports/evaluation_report.md"
