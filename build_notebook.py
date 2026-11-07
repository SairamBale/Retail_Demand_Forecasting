"""Build AND execute notebooks/retail_demand_analysis.ipynb (outputs embedded).  python build_notebook.py"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""# Retail Demand Forecasting & Replenishment - EDA and Results
Exploratory analysis of the raw data, followed by a review of the trained models, forecasts and reorder recommendations.
All numbers and interpretations below are computed from the data file actually loaded (see the first output cell).""")

code("""
import os, sys, warnings
from pathlib import Path
root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(root); sys.path.insert(0, str(root)); warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.data_loader import load_raw
from src.preprocessing import audit, clean
sns.set_theme(style="whitegrid"); plt.rcParams["figure.figsize"] = (11, 4.2)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

raw, fname, synthetic = load_raw()
print("File:", fname, "| SYNTHETIC SAMPLE" if synthetic else "| real dataset")
""")

md("## 1. Dataset overview")
code("""
print("Shape:", raw.shape); display(raw.head()); raw.info()
""")
code("""
display(raw.describe(include="all").T)
a = audit(raw); print({k: a[k] for k in ["date_min","date_max","n_days","n_stores","n_products"]})
print("Interpretation: data covers", a["n_days"], "days (~%.1f years) for %d stores x %d products." % (a["n_days"]/365, a["n_stores"], a["n_products"]))
""")

md("## 2. Data quality: missing values, duplicates, invalid values, inconsistent labels")
code("""
miss = raw.isna().sum(); miss = miss[miss > 0]
print("Missing values:\\n", miss if len(miss) else "none")
print("Exact duplicate rows:", a["duplicates"], "| duplicate (date, store, product) keys:", a["duplicate_keys"])
print("Negative values:", a["negative_values"] or "none")
print("Category labels as found:", a["category_labels"])
""")
code("""
data = clean(raw, save=False)
print("After cleaning:", data.shape, "| remaining NaN:", int(data.isna().sum().sum()), "| outliers flagged (kept):", int(data.is_outlier.sum()))
print("Category labels after cleaning:", sorted(data.category.unique()))
""")

md("## 3. Sales distribution and trend")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data.units_sold, bins=60, kde=True, ax=ax[0]); ax[0].set(title="Distribution of daily units sold (per store-product)", xlabel="Units sold")
sns.boxplot(x=data.units_sold, ax=ax[1]); ax[1].set(title="Boxplot of daily units sold", xlabel="Units sold")
plt.tight_layout(); plt.show()
print(f"Interpretation: median {data.units_sold.median():.0f}, mean {data.units_sold.mean():.0f}; skew = {data.units_sold.skew():.2f} (right-skewed -> a few high-demand days/series; use WAPE/MAE rather than only MAPE).")
""")
code("""
daily = data.groupby("date").units_sold.sum()
fig, ax = plt.subplots(); daily.plot(ax=ax, lw=.8, label="Daily total"); daily.rolling(28).mean().plot(ax=ax, c="r", lw=2, label="28-day average")
ax.set(title="Total daily units sold - all stores & products", xlabel="Date", ylabel="Units sold"); ax.legend(); plt.show()
h1, h2 = daily[: len(daily)//2].mean(), daily[len(daily)//2 :].mean()
print(f"Interpretation: average daily volume changed {100*(h2/h1-1):+.1f}% between the first and second half of the period.")
""")

md("## 4. Seasonality: monthly, weekly and weekday vs weekend")
code("""
m = data.assign(month=data.date.dt.month).groupby("month").units_sold.mean()
w = data.assign(dow=data.date.dt.dayofweek).groupby("dow").units_sold.mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
m.plot(kind="bar", ax=ax[0], color="#4C78A8"); ax[0].set(title="Average units sold by month", xlabel="Month", ylabel="Avg units / store-product-day")
w.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]; w.plot(kind="bar", ax=ax[1], color="#F58518"); ax[1].set(title="Average units sold by weekday", xlabel="Weekday", ylabel="Avg units")
plt.tight_layout(); plt.show()
wk = data.assign(week=data.date.dt.isocalendar().week.astype(int)).groupby("week").units_sold.mean()
wk.plot(figsize=(11,3.5), title="Average units sold by ISO week of year"); plt.xlabel("Week of year"); plt.ylabel("Avg units"); plt.show()
we = data.groupby(data.date.dt.dayofweek >= 5).units_sold.mean()
print(f"Interpretation: peak month = {m.idxmax()} ({m.max():.0f}), lowest = {m.idxmin()} ({m.min():.0f}); weekends sell {100*(we[True]/we[False]-1):+.1f}% vs weekdays -> calendar features are justified.")
""")

md("## 5. Category, store and product comparison")
code("""
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
data.groupby("category").units_sold.sum().sort_values().plot(kind="barh", ax=ax[0], color="#4C78A8"); ax[0].set(title="Total units by category", xlabel="Units")
data.groupby("store_id").units_sold.mean().plot(kind="bar", ax=ax[1], color="#54A24B"); ax[1].set(title="Avg daily units by store", xlabel="Store", ylabel="Units")
data.groupby("product_id").units_sold.mean().sort_values().plot(kind="barh", ax=ax[2], color="#B279A2"); ax[2].set(title="Avg daily units by product", xlabel="Units")
plt.tight_layout(); plt.show()
cs = data.groupby("category").units_sold.sum().sort_values(ascending=False)
print(f"Interpretation: {cs.index[0]} drives {100*cs.iloc[0]/cs.sum():.0f}% of units; busiest store = {data.groupby('store_id').units_sold.mean().idxmax()}. Category differences justify category-level error reporting.")
""")
code("""
cm = data.assign(month=data.date.dt.month).pivot_table(index="category", columns="month", values="units_sold", aggfunc="mean")
sns.heatmap(cm, cmap="YlOrRd", annot=True, fmt=".0f"); plt.title("Average units sold: category x month"); plt.show()
""")

md("## 6. Promotions, discounts and prices")
code("""
p = data.groupby("holiday_promotion").units_sold.mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=p.index.map({0:"No promotion",1:"Promotion"}), y=p.values, ax=ax[0]); ax[0].set(title="Average units: promotion vs none", ylabel="Avg units", xlabel="")
sns.barplot(data=data, x="discount", y="units_sold", ax=ax[1], errorbar=None); ax[1].set(title="Average units by discount %", xlabel="Discount (%)", ylabel="Avg units")
plt.tight_layout(); plt.show()
if 1 in p.index: print(f"Interpretation: promotion days sell {100*(p[1]/p[0]-1):+.1f}% more on average (raw comparison, not causal - promotions may coincide with other effects).")
""")

md("## 7. Inventory analysis")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data.inventory_level, bins=50, ax=ax[0]); ax[0].set(title="Inventory level distribution", xlabel="Inventory level")
data["cover_days"] = data.inventory_level / data.groupby(["store_id","product_id"]).units_sold.transform("mean")
sns.boxplot(data=data, x="category", y="cover_days", ax=ax[1]); ax[1].set(title="Days of cover (inventory / avg daily sales)", xlabel="Category", ylabel="Days"); ax[1].set_ylim(0, data.cover_days.quantile(.99))
plt.tight_layout(); plt.show()
low = (data.inventory_level <= data.units_sold).mean()
print(f"Interpretation: on {100*low:.2f}% of store-product-days sales consumed all available stock.",
      "Demand is CENSORED on those days (sales understate demand) - see limitations." if low > 0.005 else "Stockout censoring looks minimal in this file, but cannot be ruled out in real data.")
""")

md("## 8. Correlation and outliers")
code("""
num = ["units_sold","inventory_level","units_ordered","price","discount","holiday_promotion","competitor_pricing","demand_forecast"]
num = [c for c in num if c in data]
sns.heatmap(data[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation matrix"); plt.show()
""")
code("""
print(f"Outliers flagged (robust z-score > 6 within series): {int(data.is_outlier.sum())} rows - kept, because spikes can be genuine demand.")
display(data[data.is_outlier == 1].nlargest(8, "units_sold")[["date","store_id","product_id","category","units_sold","holiday_promotion","discount"]])
""")

md("""## 9. Is `demand_forecast` safe to use as a feature?
The provider column is checked against the target. A near-perfect relationship means it was derived from actual sales, so it would not be available (or would be leaky) at prediction time. **It is excluded from all models.**""")
code("""
if "demand_forecast" in data:
    c = data[["units_sold","demand_forecast"]].corr().iloc[0,1]
    mae = (data.demand_forecast - data.units_sold).abs().mean()
    print(f"corr(units_sold, demand_forecast) = {c:.3f}; MAE = {mae:.2f} vs units_sold mean {data.units_sold.mean():.1f}")
    print("Verdict:", "suspiciously close to the target -> treat as leakage, EXCLUDED." if c > 0.95 else "not near-perfect, but its provenance is unknown -> EXCLUDED to stay leakage-safe.")
""")

md("## 10. Model results (from `python run_pipeline.py`)")
code("""
cmp_ = pd.read_csv("outputs/model_comparison.csv")
display(cmp_[["model","val_MAE","val_WAPE_%","test_MAE","test_RMSE","test_MAPE_%","test_WAPE_%"]].round(2))
best = cmp_[~cmp_.model.str.startswith("Baseline")].sort_values("val_WAPE_%").iloc[0]
base = cmp_[cmp_.model.str.startswith("Baseline")].iloc[0]
print(f"Selected on validation: {best.model} | test WAPE {best['test_WAPE_%']:.2f}% vs baseline {base['test_WAPE_%']:.2f}%")
""")
code("""
from IPython.display import Image
for f in ["actual_vs_predicted","residuals","category_error","feature_importance"]:
    display(Image(f"reports/figures/{f}.png"))
display(pd.read_csv("outputs/category_evaluation.csv").round(2))
""")

md("## 11. Forecast and replenishment output")
code("""
fc = pd.read_csv("outputs/forecast.csv"); rp = pd.read_csv("outputs/replenishment_recommendations.csv")
print("Forecast rows:", len(fc), "| dates:", fc.date.min(), "to", fc.date.max(), "| min predicted:", fc.predicted_demand.min())
display(rp.head(10)); print(rp.stockout_risk.value_counts().to_dict(), "| overstock flagged:", int(rp.overstock_flag.sum()),
      "| total recommended units:", int(rp.recommended_reorder_qty.sum()))
""")
md("""## 12. Limitations (summary)
Sales are not true demand when stock runs out; the public dataset is synthetic; test scores are one-step-ahead (the 7-day recursive forecast is less accurate);
safety stock approximates multi-day error from one-day errors; outstanding orders/backorders are not in the data (assumed 0). See `reports/evaluation_report.md` section 8.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "retail_demand_analysis.ipynb"
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, out)
print("Notebook written and executed:", out)
