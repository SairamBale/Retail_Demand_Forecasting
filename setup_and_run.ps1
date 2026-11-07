# One-click setup + run (Windows PowerShell).  Usage:  .\setup_and_run.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
python build_notebook.py
Write-Host "Done. Open notebooks/retail_demand_analysis.ipynb and reports/evaluation_report.md"
