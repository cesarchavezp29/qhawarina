"""
train_poverty_models.py — Prompt 2: Poverty Nowcast Model Training & Validation

Models:
  B0: Department historical mean
  B1: Persistence (P_hat = P_{y-1})
  B1b: Department linear trend
  B2: Panel FE + 3 predictors
  M3: Ridge (RidgeCV)
  M4: Elastic Net (ElasticNetCV)
  M5: Gradient Boosting (GBR)

Validation: rolling-origin temporal (train < test_year, test = test_year)
Test years: 2015-2024  |  Vintage quarters: 3, 6, 9, 12
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_poverty_models")

ROOT    = Path("D:/nexus/nexus")
OUT_DIR = ROOT / "data/processed/poverty"

VINTAGE_QUARTERS = [3, 6, 9, 12]
TEST_YEARS       = list(range(2015, 2025))

# ══════════════════════════════════════════════════════════════════════════════
# Load and prep data
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(OUT_DIR / "vintage_panel.parquet")
    panel = panel.rename(columns={"ubigeo": "dept_code"})

    # ── Drop high-corr _ytd_mean features (r > 0.95 with credit_ytd_mean) ──
    drop_means = {"deposits_ytd_mean", "pension_ytd_mean",
                  "tax_revenue_ytd_mean", "ntl_ytd_mean"}
    # Also drop their growth/vol counterparts to avoid leakage from same source
    drop_series_families = {"deposits", "pension", "ntl"}
    drop_cols = set()
    for col in panel.columns:
        family = col.split("_ytd_")[0] if "_ytd_" in col else None
        if family in drop_series_families:
            drop_cols.add(col)

    log.info(f"Dropping correlated feature families: {sorted(drop_series_families)}")
    log.info(f"Dropped columns ({len(drop_cols)}): {sorted(drop_cols)}")

    # ── Winsorize extreme growth rates ──────────────────────────────────────
    winsorize_cols = ["capex_ytd_growth", "mining_ytd_growth", "inflation_ytd_growth"]
    for col in winsorize_cols:
        if col in panel.columns:
            lo = panel[col].quantile(0.01)
            hi = panel[col].quantile(0.99)
            n_clipped = ((panel[col] < lo) | (panel[col] > hi)).sum()
            panel[col] = panel[col].clip(lower=lo, upper=hi)
            log.info(f"Winsorized {col}: [{lo:.2f}, {hi:.2f}], clipped {n_clipped} obs")

    # ── Add mining missing indicator ─────────────────────────────────────────
    panel["mining_missing"] = panel["mining_ytd_mean"].isna().astype(float)

    # ── Build feature list ────────────────────────────────────────────────────
    all_feat = [c for c in panel.columns if c.endswith(("_ytd_mean", "_ytd_growth", "_ytd_vol"))]
    feature_cols = [c for c in all_feat if c not in drop_cols] + ["mining_missing"]
    feature_cols = sorted(set(feature_cols))

    log.info(f"Final feature set ({len(feature_cols)}): {feature_cols}")
    log.info(f"Panel shape after prep: {panel.shape}")
    log.info(f"Years: {sorted(panel.year.unique())}, Depts: {panel.dept_code.nunique()}")
    return panel, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# Feature matrix helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_Xy(df: pd.DataFrame, feature_cols: list, dept_dummies: bool = True):
    """Return (X, y, dept_codes) with NaN→0 imputation and optional dept dummies."""
    y = df["poverty_rate"].values
    X = df[feature_cols].fillna(0).copy()
    if dept_dummies:
        dummies = pd.get_dummies(df["dept_code"], prefix="dept", drop_first=True)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X.values, y, df["dept_code"].values


def align_dummies(X_test_df: pd.DataFrame, train_dummy_cols: list) -> pd.DataFrame:
    for col in train_dummy_cols:
        if col not in X_test_df.columns:
            X_test_df[col] = 0
    return X_test_df[train_dummy_cols]


# ══════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def bench_dept_mean(train: pd.DataFrame, test: pd.DataFrame, vm: int,
                    panel: pd.DataFrame, **_) -> dict:
    tv = train[train["vintage_month"] == vm]
    means = tv.groupby("dept_code")["poverty_rate"].mean()
    tv_test = test[test["vintage_month"] == vm]
    pred = tv_test["dept_code"].map(means)
    return dict(zip(tv_test.index, pred))


def bench_persistence(train: pd.DataFrame, test: pd.DataFrame, vm: int,
                      panel: pd.DataFrame, test_year: int, **_) -> dict:
    prev = panel[(panel["year"] == test_year - 1) & (panel["vintage_month"] == 12)]
    lookup = dict(zip(prev["dept_code"], prev["poverty_rate"]))
    tv_test = test[test["vintage_month"] == vm]
    pred = tv_test["dept_code"].map(lookup)
    return dict(zip(tv_test.index, pred))


def bench_dept_trend(train: pd.DataFrame, test: pd.DataFrame, vm: int, **_) -> dict:
    tv      = train[train["vintage_month"] == vm]
    tv_test = test[test["vintage_month"] == vm]
    preds   = {}
    for dept in tv["dept_code"].unique():
        dt = tv[tv["dept_code"] == dept]
        if len(dt) < 3:
            p = dt["poverty_rate"].mean()
        else:
            lr = LinearRegression().fit(dt[["year"]], dt["poverty_rate"])
            dt_test = tv_test[tv_test["dept_code"] == dept]
            if len(dt_test) == 0:
                continue
            p = lr.predict(dt_test[["year"]])[0]
        for idx in tv_test[tv_test["dept_code"] == dept].index:
            preds[idx] = p
    return preds


def bench_panel_fe3(train: pd.DataFrame, test: pd.DataFrame, vm: int, **_) -> dict:
    tv      = train[train["vintage_month"] == vm].copy()
    tv_test = test[test["vintage_month"] == vm].copy()
    feats   = [f for f in ["electricity_ytd_mean", "credit_ytd_mean", "capex_ytd_mean"]
               if f in tv.columns]
    if not feats or len(tv) < 5:
        return {}
    td = pd.get_dummies(tv["dept_code"], prefix="dept", drop_first=True)
    td_test = pd.get_dummies(tv_test["dept_code"], prefix="dept", drop_first=True)
    X_tr = pd.concat([tv[feats].fillna(0).reset_index(drop=True),
                      td.reset_index(drop=True)], axis=1)
    for col in td.columns:
        if col not in td_test.columns:
            td_test[col] = 0
    X_te = pd.concat([tv_test[feats].fillna(0).reset_index(drop=True),
                      td_test[td.columns].reset_index(drop=True)], axis=1)
    lr = LinearRegression().fit(X_tr, tv["poverty_rate"])
    preds_arr = lr.predict(X_te)
    return dict(zip(tv_test.index, preds_arr))


# ══════════════════════════════════════════════════════════════════════════════
# ML models
# ══════════════════════════════════════════════════════════════════════════════

def fit_ridge(X_tr, y_tr):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000, 5000])),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def fit_enet(X_tr, y_tr):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=3, max_iter=5000, random_state=42,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def fit_gbr(X_tr, y_tr):
    param_grid = {
        "max_depth":        [3, 4],
        "n_estimators":     [100, 200],
        "learning_rate":    [0.05, 0.1],
        "min_samples_leaf": [5, 10],
    }
    gbr = GradientBoostingRegressor(random_state=42, subsample=0.8)
    gs  = GridSearchCV(gbr, param_grid, cv=3, scoring="neg_mean_absolute_error",
                       n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def model_ml(train: pd.DataFrame, test: pd.DataFrame, vm: int,
             feature_cols: list, fitter) -> dict:
    tv      = train[train["vintage_month"] == vm].dropna(subset=["poverty_rate"])
    tv_test = test[test["vintage_month"] == vm].copy()
    if len(tv) < 10 or len(tv_test) == 0:
        return {}

    # Build train matrices with dept dummies
    td = pd.get_dummies(tv["dept_code"], prefix="dept", drop_first=True)
    all_feat = feature_cols + list(td.columns)
    X_tr = pd.concat([tv[feature_cols].fillna(0).reset_index(drop=True),
                      td.reset_index(drop=True)], axis=1).values
    y_tr = tv["poverty_rate"].values

    # Build test matrix
    td_test = pd.get_dummies(tv_test["dept_code"], prefix="dept", drop_first=True)
    for col in td.columns:
        if col not in td_test.columns:
            td_test[col] = 0
    X_te = pd.concat([tv_test[feature_cols].fillna(0).reset_index(drop=True),
                      td_test[td.columns].reset_index(drop=True)], axis=1).values

    fitted = fitter(X_tr, y_tr)
    preds  = fitted.predict(X_te)
    return dict(zip(tv_test.index, preds))


# ══════════════════════════════════════════════════════════════════════════════
# Rolling-origin validation
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    log.info("Starting rolling-origin validation...")
    log.info(f"Test years: {TEST_YEARS}  |  Vintage quarters: {VINTAGE_QUARTERS}")

    model_fns = {
        "dept_mean":   lambda tr, te, vm, ty: bench_dept_mean(tr, te, vm, panel),
        "persistence": lambda tr, te, vm, ty: bench_persistence(tr, te, vm, panel, ty),
        "dept_trend":  lambda tr, te, vm, ty: bench_dept_trend(tr, te, vm),
        "panel_fe3":   lambda tr, te, vm, ty: bench_panel_fe3(tr, te, vm),
        "ridge":       lambda tr, te, vm, ty: model_ml(tr, te, vm, feature_cols, fit_ridge),
        "elastic_net": lambda tr, te, vm, ty: model_ml(tr, te, vm, feature_cols, fit_enet),
        "gbr":         lambda tr, te, vm, ty: model_ml(tr, te, vm, feature_cols, fit_gbr),
    }

    rows = []
    for test_year in TEST_YEARS:
        train = panel[panel["year"] < test_year]
        test  = panel[panel["year"] == test_year]
        log.info(f"  Test year {test_year}: train={len(train)} rows, test={len(test)} rows")

        for vm in VINTAGE_QUARTERS:
            actual_df = test[test["vintage_month"] == vm][
                ["dept_code", "department", "poverty_rate"]
            ].copy()

            for model_name, fn in model_fns.items():
                try:
                    pred_map = fn(train, test, vm, test_year)
                except Exception as e:
                    log.warning(f"    {model_name} vm={vm} failed: {e}")
                    pred_map = {}

                for idx, row in actual_df.iterrows():
                    pred = pred_map.get(idx, np.nan)
                    rows.append({
                        "test_year":     test_year,
                        "vintage_month": vm,
                        "dept_code":     row["dept_code"],
                        "department":    row["department"],
                        "actual":        row["poverty_rate"],
                        "predicted":     pred,
                        "model":         model_name,
                        "error":         pred - row["poverty_rate"] if not np.isnan(pred) else np.nan,
                    })

    results = pd.DataFrame(rows)
    log.info(f"Validation complete: {len(results)} result rows")
    return results


def add_direction_accuracy(results: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add direction_correct column: did predicted change match actual change direction?"""
    # Lookup prev-year actual at vintage 12
    prev_lookup = (
        panel[panel["vintage_month"] == 12]
        .set_index(["dept_code", "year"])["poverty_rate"]
    )
    directions = []
    for _, row in results.iterrows():
        try:
            prev = prev_lookup.loc[(row["dept_code"], row["test_year"] - 1)]
            actual_chg    = row["actual"] - prev
            predicted_chg = row["predicted"] - prev
            directions.append(
                np.sign(actual_chg) == np.sign(predicted_chg)
                if not np.isnan(row["predicted"]) else np.nan
            )
        except KeyError:
            directions.append(np.nan)
    results["direction_correct"] = directions
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: pd.DataFrame):
    sep  = "=" * 80
    sep2 = "-" * 60
    print(f"\n{sep}")
    print("POVERTY NOWCAST — MODEL VALIDATION RESULTS")
    print(f"{sep}\n")

    model_order = ["dept_mean", "persistence", "dept_trend", "panel_fe3",
                   "ridge", "elastic_net", "gbr"]

    # Overall
    print("--- OVERALL (all vintage quarters pooled, 2015-2024) ---")
    print(f"{'Model':15s}  {'RMSE':>6}  {'MAE':>6}  {'Dir%':>5}  {'Rank_r':>7}  {'N':>5}")
    print(sep2)
    for model in model_order:
        m = results[results["model"] == model].dropna(subset=["predicted"])
        if len(m) == 0:
            continue
        rmse  = np.sqrt((m["error"] ** 2).mean())
        mae   = m["error"].abs().mean()
        direc = m["direction_correct"].mean() if "direction_correct" in m else float("nan")
        rho   = spearmanr(m["actual"], m["predicted"])[0]
        print(f"  {model:15s}  {rmse:6.3f}  {mae:6.3f}  {direc:5.1%}  {rho:7.3f}  {len(m):5d}")

    # By vintage quarter
    print(f"\n--- BY VINTAGE QUARTER (RMSE) ---")
    header = f"{'Model':15s}" + "".join(f"  vm={v:02d}" for v in VINTAGE_QUARTERS)
    print(header)
    print(sep2)
    for model in ["persistence", "dept_trend", "panel_fe3", "ridge", "elastic_net", "gbr"]:
        line = f"  {model:15s}"
        for vm in VINTAGE_QUARTERS:
            m = results[(results["model"] == model) & (results["vintage_month"] == vm)].dropna(subset=["predicted"])
            rmse = np.sqrt((m["error"] ** 2).mean()) if len(m) > 0 else float("nan")
            line += f"  {rmse:6.3f}"
        print(line)

    # MAE by vintage
    print(f"\n--- BY VINTAGE QUARTER (MAE) ---")
    print(header)
    print(sep2)
    for model in ["persistence", "dept_trend", "panel_fe3", "ridge", "elastic_net", "gbr"]:
        line = f"  {model:15s}"
        for vm in VINTAGE_QUARTERS:
            m = results[(results["model"] == model) & (results["vintage_month"] == vm)].dropna(subset=["predicted"])
            mae = m["error"].abs().mean() if len(m) > 0 else float("nan")
            line += f"  {mae:6.3f}"
        print(line)

    # COVID stress test
    print("\n--- COVID 2020 STRESS TEST ---")
    covid = results[results["test_year"] == 2020].dropna(subset=["predicted"])
    print(f"{'Model':15s}  {'MAE':>6}  {'RMSE':>6}  {'Dir%':>5}")
    print(sep2)
    for model in model_order:
        m = covid[covid["model"] == model]
        if len(m) == 0:
            continue
        mae  = m["error"].abs().mean()
        rmse = np.sqrt((m["error"] ** 2).mean())
        direc = m["direction_correct"].mean() if "direction_correct" in m else float("nan")
        print(f"  {model:15s}  {mae:6.3f}  {rmse:6.3f}  {direc:5.1%}")

    # Model selection
    print(f"\n--- MODEL SELECTION ---")
    ridge_mae = results[results["model"] == "ridge"].dropna(subset=["predicted"])["error"].abs().mean()
    gbr_mae   = results[results["model"] == "gbr"].dropna(subset=["predicted"])["error"].abs().mean()
    trend_mae = results[results["model"] == "dept_trend"].dropna(subset=["predicted"])["error"].abs().mean()
    enet_mae  = results[results["model"] == "elastic_net"].dropna(subset=["predicted"])["error"].abs().mean()

    print(f"  dept_trend MAE : {trend_mae:.4f}")
    print(f"  Ridge MAE      : {ridge_mae:.4f}")
    print(f"  Elastic Net MAE: {enet_mae:.4f}")
    print(f"  GBR MAE        : {gbr_mae:.4f}")

    if ridge_mae > trend_mae:
        print("  *** WARNING: Ridge does NOT beat dept_trend. Model adds no value. STOPPING. ***")
        return None

    best_ml_mae = min(ridge_mae, enet_mae, gbr_mae)
    if ridge_mae <= gbr_mae + 0.003 and ridge_mae <= enet_mae + 0.003:
        selected = "ridge"
        print(f"  --> SELECTED: Ridge (interpretability wins, within 0.3pp of GBR)")
    else:
        best = min([("ridge", ridge_mae), ("elastic_net", enet_mae), ("gbr", gbr_mae)],
                   key=lambda x: x[1])
        selected = best[0]
        print(f"  --> SELECTED: {selected} (best MAE = {best[1]:.4f})")

    print(f"{sep}\n")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Train final model + vintage-specific models
# ══════════════════════════════════════════════════════════════════════════════

def train_final_models(panel: pd.DataFrame, feature_cols: list, selected: str,
                       results: pd.DataFrame):
    log.info(f"Training final models (selected: {selected})...")

    fitter_map = {"ridge": fit_ridge, "elastic_net": fit_enet, "gbr": fit_gbr}
    fitter     = fitter_map[selected]

    # Retrain on ALL data at vintage 12 (full-year view)
    full_train = panel[panel["vintage_month"] == 12].dropna(subset=["poverty_rate"])
    td = pd.get_dummies(full_train["dept_code"], prefix="dept", drop_first=True)
    X_full = pd.concat([full_train[feature_cols].fillna(0).reset_index(drop=True),
                        td.reset_index(drop=True)], axis=1)
    y_full = full_train["poverty_rate"].values

    # For Ridge/Enet: use pipeline; for GBR: wrap in dict with feature names
    final_pipe = fitter(X_full.values, y_full)

    joblib.dump(
        {"model": final_pipe, "feature_cols": feature_cols,
         "dept_dummy_cols": list(td.columns), "model_name": selected},
        OUT_DIR / f"poverty_model_{selected}.pkl"
    )
    log.info(f"  Saved poverty_model_{selected}.pkl (trained on {len(full_train)} rows)")

    # Coefficients (Ridge / ElasticNet)
    if selected in ("ridge", "elastic_net"):
        step_name = "ridge" if selected == "ridge" else "enet"
        scaler = final_pipe.named_steps["scaler"]
        coefs  = final_pipe.named_steps[step_name].coef_
        all_features = feature_cols + list(td.columns)
        coef_df = pd.DataFrame({
            "feature": all_features,
            "coefficient": coefs,
            "abs_coef": np.abs(coefs),
        }).sort_values("abs_coef", ascending=False)
        coef_df["normalized_weight"] = coef_df["abs_coef"] / coef_df["abs_coef"].sum()
        print("\n--- COEFFICIENTS (top 15) ---")
        print(coef_df.head(15).to_string(index=False))
        coef_df.to_csv(OUT_DIR / "ridge_coefficients.csv", index=False)
        log.info("  Saved ridge_coefficients.csv")

    # Vintage-specific models (vm = 3, 6, 9, 10, 11, 12)
    log.info("  Training vintage-specific models...")
    for vm in [3, 6, 9, 10, 11, 12]:
        train_v = panel[panel["vintage_month"] == vm].dropna(subset=["poverty_rate"])
        if len(train_v) < 10:
            log.warning(f"    vm={vm}: only {len(train_v)} rows — skipping")
            continue
        td_v = pd.get_dummies(train_v["dept_code"], prefix="dept", drop_first=True)
        X_v  = pd.concat([train_v[feature_cols].fillna(0).reset_index(drop=True),
                          td_v.reset_index(drop=True)], axis=1)
        y_v  = train_v["poverty_rate"].values
        pipe_v = fitter(X_v.values, y_v)
        joblib.dump(
            {"model": pipe_v, "feature_cols": feature_cols,
             "dept_dummy_cols": list(td_v.columns), "vintage_month": vm},
            OUT_DIR / f"poverty_model_v{vm:02d}.pkl"
        )
        log.info(f"    Saved poverty_model_v{vm:02d}.pkl ({len(train_v)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=== POVERTY NOWCAST — Prompt 2: Training & Validation ===")

    panel, feature_cols = load_and_prep()

    log.info("Running rolling-origin validation (this takes a few minutes)...")
    results = run_validation(panel, feature_cols)
    results = add_direction_accuracy(results, panel)

    selected = print_results(results)
    if selected is None:
        log.error("Model does not beat benchmark — stopping. Do not proceed to Prompt 3.")
        return

    # Save validation results
    results.to_parquet(OUT_DIR / "validation_results.parquet", index=False)
    log.info("Saved validation_results.parquet")

    train_final_models(panel, feature_cols, selected, results)

    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
