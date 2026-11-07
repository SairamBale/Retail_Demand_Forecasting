"""End-to-end pipeline:  python run_pipeline.py [--fast]

load -> audit -> clean -> features -> chronological split -> train/compare -> tune best
-> evaluate -> 7-day recursive forecast -> replenishment -> report
"""
import argparse
import json
import logging
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import config as C
from src.data_loader import load_raw
from src.evaluate import grouped, plot_actual_vs_pred, plot_category_error, plot_residuals
from src.feature_engineering import FEATURES, build_features, chronological_split
from src.forecasting import forecast
from src.preprocessing import audit, clean
from src.replenishment import error_sigma, recommend
from src.report import write_report
from src.train import baseline_predict, fit_models, save, tune

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")


def main(fast: bool = False) -> None:
    random.seed(C.RANDOM_SEED); np.random.seed(C.RANDOM_SEED); C.ensure_dirs()

    # 1-2. Load, audit, clean
    raw, fname, synthetic = load_raw()
    aud = audit(raw)
    log.info("Audit: %s", json.dumps(aud, default=str)[:600])
    data = clean(raw)

    # 3. Features + chronological split
    feats = build_features(data)
    train, val, test = chronological_split(feats, C.TRAIN_FRAC, C.VAL_FRAC)
    assert train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min(), "split overlaps!"
    split = {k: f"{d['date'].min().date()} to {d['date'].max().date()} ({len(d):,} rows)"
             for k, d in (("train", train), ("val", val), ("test", test))}
    log.info("Split: %s", split)

    # 4. Train + compare, then tune the best ML model (chosen on VALIDATION)
    fitted, comp = fit_models(train, val, test)
    ml = comp[~comp["model"].str.startswith("Baseline")]
    best_untuned = ml.sort_values("val_WAPE_%").iloc[0]["model"]
    log.info("Best on validation before tuning: %s", best_untuned)
    tuned, best_params = tune(best_untuned, train, n_iter=4 if fast else 10)
    from src.evaluate import metrics
    tname = f"{best_untuned} (tuned)"
    comp = pd.concat([comp, pd.DataFrame([{
        "model": tname,
        **{f"val_{k}": v for k, v in metrics(val[C.TARGET], tuned.predict(val[FEATURES])).items()},
        **{f"test_{k}": v for k, v in metrics(test[C.TARGET], tuned.predict(test[FEATURES])).items()}}])], ignore_index=True)
    fitted[tname] = tuned
    ml = comp[~comp["model"].str.startswith("Baseline")]
    best_name = ml.sort_values("val_WAPE_%").iloc[0]["model"]          # selection on validation only
    best = fitted[best_name]
    comp.to_csv(C.OUTPUTS_DIR / "model_comparison.csv", index=False)

    # 5. Evaluate on test
    test = test.copy(); val = val.copy()
    test["pred"] = np.clip(best.predict(test[FEATURES]), 0, None)
    val["pred"] = np.clip(best.predict(val[FEATURES]), 0, None)
    out = test[["date", *C.KEYS, "category", C.TARGET]].rename(columns={C.TARGET: "actual_units_sold"})
    out["baseline_pred"] = baseline_predict(test)
    for n, m in fitted.items():
        out[f"pred_{n}"] = np.clip(m.predict(test[FEATURES]), 0, None).round(2)
    out["pred_best"], out["best_model"] = test["pred"].round(2).values, best_name
    out.to_csv(C.OUTPUTS_DIR / "test_predictions.csv", index=False)

    cat_eval = grouped(test, "category")
    base_t = test.assign(pred=baseline_predict(test))
    cat_eval = cat_eval.merge(grouped(base_t, "category")[["category", "MAE", "WAPE_%"]]
                              .rename(columns={"MAE": "baseline_MAE", "WAPE_%": "baseline_WAPE_%"}), on="category")
    cat_eval = cat_eval.sort_values("WAPE_%").reset_index(drop=True)
    cat_eval.to_csv(C.OUTPUTS_DIR / "category_evaluation.csv", index=False)
    prod_eval = grouped(test, "product_id"); prod_eval.to_csv(C.OUTPUTS_DIR / "product_evaluation.csv", index=False)
    store_eval = grouped(test, "store_id"); store_eval.to_csv(C.OUTPUTS_DIR / "store_evaluation.csv", index=False)
    plot_actual_vs_pred(test, name=best_name); plot_residuals(test); plot_category_error(cat_eval)

    samp = test.sample(min(4000, len(test)), random_state=C.RANDOM_SEED)
    pi = permutation_importance(best, samp[FEATURES], samp[C.TARGET], n_repeats=3, random_state=C.RANDOM_SEED,
                                scoring="neg_mean_absolute_error", n_jobs=1)
    imp = (pd.DataFrame({"feature": FEATURES, "mae_increase": pi.importances_mean})
           .sort_values("mae_increase", ascending=False).reset_index(drop=True))
    imp.to_csv(C.OUTPUTS_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6)); sns.barplot(data=imp.head(15), y="feature", x="mae_increase", ax=ax, color="#4C78A8")
    ax.set(title="Permutation importance (MAE increase when shuffled)", xlabel="MAE increase"); fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "feature_importance.png", dpi=130); plt.close(fig)

    # 6. Persist: evaluated (train-only) model, and a final model refit on ALL data for forecasting
    save(best, "evaluated_model_train_only.joblib")
    final = clone(best).fit(feats[FEATURES], feats[C.TARGET])
    save({"model": final, "features": FEATURES, "model_name": best_name, "params": best_params}, "final_model.joblib")

    # 7. 7-day recursive forecast + replenishment (lead-time demand needs max(7, lead) days)
    horizon = max(C.FORECAST_HORIZON, C.LEAD_TIME_DAYS)
    fc_all = forecast(final, data, horizon=horizon)
    first = sorted(fc_all["date"].unique())[:C.FORECAST_HORIZON]
    fc_all[fc_all["date"].isin(first)].rename(columns={"store_id": "store_id"}).to_csv(C.OUTPUTS_DIR / "forecast.csv", index=False)
    sigma = error_sigma(pd.concat([val, test])[["store_id", "product_id", "category", "units_sold", "pred"]])
    rep = recommend(fc_all, data, sigma)
    rep.to_csv(C.OUTPUTS_DIR / "replenishment_recommendations.csv", index=False)

    # 8. Report
    g = data.assign(promo=data["holiday_promotion"] > 0, weekend=data["date"].dt.dayofweek >= 5)
    pm = g.groupby("promo")[C.TARGET].mean(); wm = g.groupby("weekend")[C.TARGET].mean()
    cu = data.groupby("category")[C.TARGET].sum().sort_values(ascending=False)
    ins = {"promo_uplift": (pm.get(True, np.nan) / pm.get(False, np.nan) - 1) * 100,
           "weekend_uplift": (wm.get(True, np.nan) / wm.get(False, np.nan) - 1) * 100,
           "cat_units": ", ".join(f"{k} ({v:,.0f})" for k, v in cu.items())}
    write_report(dict(audit=aud, comparison=comp, best_name=best_name, best_params=best_params or "n/a (no tunable params)",
                      category_eval=cat_eval, importance=imp, replen=rep, insights=ins, split=split, synthetic=synthetic,
                      file=fname, n_clean=len(data), n_outliers=int(data["is_outlier"].sum()),
                      lead=C.LEAD_TIME_DAYS, sl=C.SERVICE_LEVEL, cover=C.OVERSTOCK_COVER_DAYS))
    pd.set_option("display.width", 200)
    print("\n=== MODEL COMPARISON ===\n", comp.round(3).to_string(index=False))
    print(f"\nBest model: {best_name}\nSynthetic data used: {synthetic}\nDone. See outputs/ and reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer tuning iterations (quick test)")
    main(ap.parse_args().fast)
