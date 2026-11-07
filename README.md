# Retail Demand Forecasting & Replenishment

End-to-end machine-learning project: clean 2 years of daily store/product sales, engineer leakage-safe
time-series features, compare forecasting models, evaluate error by product category, produce a 7-day
forecast and recommend reorder quantities.

> **Dataset note.** The pipeline expects the Kaggle *Retail Store Inventory Forecasting Dataset*
> (`anirudhchauhan/retail-store-inventory-forecasting-dataset`, ~73,100 rows, 2022-2024, synthetic).
> If `data/raw/retail_store_inventory.csv` is **absent**, the project falls back to a generated synthetic
> sample with the same schema (`data/raw/sample_retail_store_inventory.csv`) and prints/records a warning.
> **Before submitting, download the real CSV, put it at `data/raw/retail_store_inventory.csv` and re-run.**
> Every number in `reports/` and `outputs/` is regenerated from whichever file is used.

## Quick start (VS Code)
```powershell
cd retail-demand-forecasting
code .
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py            # ~2-4 min (add --fast for a quick test)
python build_notebook.py          # re-executes the EDA notebook with embedded outputs
pytest -q                         # leakage / logic tests
```
macOS/Linux: `source .venv/bin/activate`. Or run everything with `.\setup_and_run.ps1` / `bash setup_and_run.sh`.

## Project structure
```
data/raw/            raw CSV (never modified)      data/processed/   clean_data.csv
notebooks/           retail_demand_analysis.ipynb  (EDA + results, executed)
src/                 config, data_loader, preprocessing, feature_engineering, train,
                     evaluate, forecasting, replenishment, report, make_sample_data
tests/               pytest checks for leakage-safe lags, chronological split, metrics
models/              final_model.joblib (fit on all data), evaluated_model_train_only.joblib
outputs/             model_comparison, category/product/store_evaluation, test_predictions,
                     forecast, replenishment_recommendations, feature_importance (CSV)
reports/             evaluation_report.md + figures/
run_pipeline.py      orchestrates everything          build_notebook.py   builds + runs the notebook
```

## Method
**Cleaning** - snake_case columns, parsed dates, normalised labels (`'groceries '` -> `Groceries`), duplicate rows/keys removed,
impossible negatives set to missing, missing values forward-filled *within each store-product series* (past only) with series
median as fallback, outliers **flagged not deleted**, cleaned data saved to `data/processed/`.

**Features** - calendar (year, month, day, weekday, week, quarter, weekend, month start/end); lags 1/7/14/28; rolling mean/std 7/14/28
(computed on the target **shifted by one day**, per store-product); price, discount, promotion, price vs competitor; store, product,
category, region, season.
Excluded on purpose: `Demand Forecast` (provider column, strongly tied to actual sales -> leakage), same-day inventory/orders
(depend on sales and on the replenishment decisions we recommend) and weather (not known in advance).

**Validation** - no random split. Chronological by date: 70 % train / 15 % validation / 15 % test. Model selection and tuning
(`RandomizedSearchCV` + `TimeSeriesSplit`) use train/validation only; the test period is used once for reporting.

**Models** - baseline (previous 7-day mean), Linear Regression, Random Forest, HistGradientBoosting, plus the best ML model tuned.
Metrics: MAE, RMSE, MAPE (zero actuals excluded), WAPE, bias - overall and by category / product / store.

**Forecast** - recursive 7-day forecast per store-product (each prediction feeds the next day's lags). Future promotion = 0; future
price/discount = median of the last 28 days (documented, overridable via `future_inputs`). Predictions are clipped at 0.

**Replenishment** - `ROP = lead-time demand + z * sigma * sqrt(L)`, `order = max(0, ROP - inventory position)`; defaults L = 7 days,
service level 95 % (change in `src/config.py`). sigma = std of out-of-sample forecast errors per series. Outputs stockout probability/risk,
days of cover and overstock flag. Outstanding orders/backorders are absent from the data -> assumed 0.

## Results
See `reports/evaluation_report.md` (auto-generated from the actual run) and the executed notebook.

## Limitations
Sales != demand when stock runs out; synthetic source data; one-step-ahead test scores flatter the 7-day recursive forecast;
safety stock assumes independent normal errors; no outstanding-order data; assumed future prices/promotions; one global model with no
cold-start handling; no cost/MOQ/shelf-life constraints. Details in the report (section 8).

## Future improvements
Multi-horizon (direct) models and prediction intervals (quantile regression), censored-demand correction, hierarchical/global-local models,
cost-based newsvendor ordering, real promotion calendars and weather forecasts, MLflow tracking.
