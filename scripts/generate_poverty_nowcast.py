"""
Poverty Nowcast — Production Forecast Generator
Prompt 3: annual nowcast + monthly pressure tracker + rolling 3-month series

Inputs  (data/processed/poverty/):
  vintage_panel.parquet        — training panel (2004-2024)
  monthly_indicators.parquet   — raw monthly wide panel (incl. 2025)
  poverty_model_v*.pkl         — vintage-specific GBR dicts
  enet_coefficients.csv        — pressure tracker weights
  blending_config.json         — alpha=0.7
  validation_results.parquet   — for CI computation

Outputs (exports/data/):
  poverty_nowcast.json
  vintages/poverty_nowcast_{vm:02d}.json
  D:/qhawarina/public/assets/data/poverty_nowcast.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("poverty_nowcast")

ROOT      = Path("D:/nexus/nexus")
PROCESSED = ROOT / "data/processed/poverty"
EXPORTS   = ROOT / "exports/data"
VINTAGES  = EXPORTS / "vintages"

# Publication lags (months) — must match build_poverty_nowcast.py
PUB_LAGS = {
    "electricity": 2, "credit": 1, "deposits": 1, "capex": 1,
    "current_spending": 1, "mining": 2, "pension": 1,
    "tax_revenue": 1, "inflation": 1, "gdp_monthly": 2, "ntl": 3,
}
SERIES_KEYS = list(PUB_LAGS.keys())

POVERTY_REDUCING  = {"electricity", "credit", "deposits", "capex", "current_spending",
                      "mining", "pension", "tax_revenue", "gdp_monthly", "ntl"}
POVERTY_INCREASING = {"inflation"}


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def ytd_features_from_monthly(dept_monthly: pd.DataFrame, series: str,
                               year: int, vintage_month: int) -> dict:
    """Compute YTD stats for one series respecting publication lag."""
    lag           = PUB_LAGS.get(series, 1)
    last_avail    = vintage_month - lag
    out = {f"{series}_ytd_mean": np.nan,
           f"{series}_ytd_growth": np.nan,
           f"{series}_ytd_vol": np.nan}
    if last_avail < 1 or series not in dept_monthly.columns:
        return out

    cur  = dept_monthly.loc[(dept_monthly["year"] == year)
                             & (dept_monthly["month"] <= last_avail), series].dropna().values
    prev = dept_monthly.loc[(dept_monthly["year"] == year - 1)
                             & (dept_monthly["month"] <= last_avail), series].dropna().values

    if len(cur) == 0:
        return out
    mu = float(np.mean(cur))
    out[f"{series}_ytd_mean"] = mu
    if len(prev) > 0:
        pm = float(np.mean(prev))
        if pm != 0:
            out[f"{series}_ytd_growth"] = mu / pm - 1
    if len(cur) > 1 and mu != 0:
        out[f"{series}_ytd_vol"] = float(np.std(cur) / abs(mu))
    return out


def build_ytd_features(monthly: pd.DataFrame, year: int,
                        vintage_month: int) -> pd.DataFrame:
    """Build YTD feature rows for all departments at a given (year, vintage_month)."""
    dept_codes = sorted(monthly[monthly["dept_code"] != "00"]["dept_code"].unique())
    by_dept    = {d: g.reset_index(drop=True) for d, g in monthly.groupby("dept_code")}
    rows = []
    for dept_code in dept_codes:
        dm  = by_dept.get(dept_code, pd.DataFrame())
        row = {"dept_code": dept_code, "year": year, "vintage_month": vintage_month}
        for s in SERIES_KEYS:
            row.update(ytd_features_from_monthly(dm, s, year, vintage_month))
        row["mining_missing"] = 1 if np.isnan(row.get("mining_ytd_mean", np.nan)) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _pressure_scores_all(monthly: pd.DataFrame, coefs: pd.DataFrame) -> tuple:
    """
    Compute pressure scores for every dept-month in `monthly`.
    Returns (monthly_with_scores, weights_dict).
    """
    indicator_cols = [c for c in monthly.columns
                      if c not in ("dept_code", "year", "month", "date")]

    # Weight mapping: monthly col -> normalized_weight from _ytd_mean ENet coef
    raw_w = {}
    for col in indicator_cols:
        row = coefs[coefs["feature"] == f"{col}_ytd_mean"]
        raw_w[col] = float(row.iloc[0]["normalized_weight"]) if len(row) > 0 else 0.0
    # Re-normalise over positive weights only
    total_w = sum(v for v in raw_w.values() if v > 0)
    weights = {k: (v / total_w if v > 0 else 0.0) for k, v in raw_w.items()}

    # Global standardisation (on departmental rows only — excl. national '00')
    hist = monthly[monthly["dept_code"] != "00"]
    global_stats = {}
    for col in indicator_cols:
        vals = hist[col].dropna()
        global_stats[col] = (vals.mean(), vals.std())

    m = monthly.copy()
    for col in indicator_cols:
        mu, sigma = global_stats[col]
        m[col] = (m[col] - mu) / sigma if sigma > 0 else 0.0

    # Sign alignment — flip poverty-reducing indicators
    for col in indicator_cols:
        if col in POVERTY_REDUCING:
            m[col] = -m[col]

    # Additive deseasonalisation (monthly means over 2015-2024)
    h_std = m[(m["year"].between(2015, 2024)) & (m["dept_code"] != "00")]
    seasonal = h_std.groupby("month")[indicator_cols].mean()
    for col in indicator_cols:
        if col in seasonal.columns:
            m[col] = m[col] - m["month"].map(seasonal[col].to_dict()).fillna(0)

    # Weighted pressure score
    m["pressure_score"] = 0.0
    for col in indicator_cols:
        w = weights.get(col, 0.0)
        if w > 0:
            m["pressure_score"] += w * m[col].fillna(0)

    return m, weights


# ══════════════════════════════════════════════════════════════════════════════
# Part A — Annual Nowcast
# ══════════════════════════════════════════════════════════════════════════════

def generate_annual_nowcast(monthly: pd.DataFrame, panel: pd.DataFrame,
                             config: dict, vintage_override: int | None = None) -> tuple:
    alpha = config["blend_alpha"]

    # Only departments that have poverty targets in the panel
    valid_depts = set(panel["dept_code"].unique())

    # Determine vintage month from 2025 data availability
    m2025 = monthly[(monthly["year"] == 2025)
                    & (monthly["dept_code"] != "00")
                    & (monthly["dept_code"].isin(valid_depts))]
    # Use max month where credit (lag=1) is non-null for majority of depts
    credit_coverage = (m2025.groupby("month")["credit"]
                       .apply(lambda x: x.notna().mean()))
    good_months = credit_coverage[credit_coverage >= 0.5].index.tolist()
    latest_data_month = max(good_months) if good_months else 10
    vintage_month = min(int(latest_data_month) - 1, 12)
    vintage_month = max(vintage_month, 3)
    log.info(f"Latest usable credit month: {latest_data_month} → vintage_month={vintage_month}")
    if vintage_override is not None:
        vintage_month = vintage_override
        log.info(f"Vintage OVERRIDDEN to {vintage_month}")

    # Load vintage-specific model dict
    model_path = PROCESSED / f"poverty_model_v{vintage_month:02d}.pkl"
    if not model_path.exists():
        available = sorted(PROCESSED.glob("poverty_model_v*.pkl"))
        model_path = available[-1]
        vintage_month = int(model_path.stem.split("v")[-1])
        log.warning(f"Falling back to: {model_path}")
    model_dict    = joblib.load(model_path)
    gbr           = model_dict["model"]
    feature_cols  = model_dict["feature_cols"]
    dummy_cols    = model_dict["dept_dummy_cols"]
    log.info(f"Model v{vintage_month:02d} loaded: {len(feature_cols)} features + {len(dummy_cols)} dept dummies")

    # Construct 2025 YTD features (vintage_panel only has 2004-2024)
    # Filter monthly to valid depts only
    monthly_valid = monthly[monthly["dept_code"].isin(valid_depts)]
    feats = build_ytd_features(monthly_valid, 2025, vintage_month)

    # Add dept dummies (same encoding as training)
    dummies = pd.get_dummies(feats["dept_code"], prefix="dept")
    for col in dummy_cols:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[dummy_cols]

    X = pd.concat([feats[feature_cols].fillna(0).reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    gbr_pred = gbr.predict(X)

    # Attach department names
    dept_names = (panel[panel["vintage_month"] == 12][["dept_code", "department"]]
                  .drop_duplicates("dept_code"))
    feats = feats.merge(dept_names, on="dept_code", how="left")

    # P_{t-1} — 2024 official poverty at vintage 12
    prev = (panel[(panel["year"] == 2024) & (panel["vintage_month"] == 12)]
            [["dept_code", "poverty_rate"]]
            .rename(columns={"poverty_rate": "prev_poverty"}))
    feats = feats.merge(prev, on="dept_code", how="left")

    # Blend
    feats["gbr_prediction"]     = gbr_pred
    feats["blended_prediction"] = (alpha * feats["gbr_prediction"]
                                    + (1 - alpha) * feats["prev_poverty"])

    # CIs from pooled validation SE
    validation = pd.read_parquet(PROCESSED / "validation_results.parquet")
    se = validation[validation["model"] == "gbr"]["error"].std()
    feats["lower_ci"] = (feats["blended_prediction"] - 1.96 * se).clip(lower=0.0)
    feats["upper_ci"] = (feats["blended_prediction"] + 1.96 * se).clip(upper=1.0)

    nat    = feats["blended_prediction"].mean()
    nat_lo = feats["lower_ci"].mean()
    nat_hi = feats["upper_ci"].mean()
    log.info(f"National 2025: {nat*100:.2f}% [{nat_lo*100:.2f}%-{nat_hi*100:.2f}%] "
             f"(2024: {feats['prev_poverty'].mean()*100:.2f}%, "
             f"change: {(nat - feats['prev_poverty'].mean())*100:+.2f}pp)")

    return feats, vintage_month, nat, nat_lo, nat_hi


# ══════════════════════════════════════════════════════════════════════════════
# Part B — Monthly Pressure Tracker
# ══════════════════════════════════════════════════════════════════════════════

def _pressure_label(score: float) -> str:
    if score < -0.5:    return "mejora_significativa"
    elif score < -0.15: return "mejora_moderada"
    elif score < 0.15:  return "estable"
    elif score < 0.5:   return "deterioro_moderado"
    else:               return "deterioro_significativo"


def compute_pressure_tracker(monthly: pd.DataFrame, coefs: pd.DataFrame,
                              vintage_month: int) -> tuple:
    """2025 monthly pressure scores, centered within year."""
    m_std, weights = _pressure_scores_all(monthly, coefs)

    indicator_cols = [c for c in monthly.columns
                      if c not in ("dept_code", "year", "month", "date")]

    # Only include months where gdp_monthly (dominant indicator) is available
    # to avoid spurious pressure readings from data gaps
    valid_months_2025 = (
        monthly[(monthly["year"] == 2025) & (monthly["dept_code"] != "00")]
        .groupby("month")["gdp_monthly"]
        .apply(lambda x: x.notna().mean() >= 0.5)
    )
    usable_months = valid_months_2025[valid_months_2025].index.tolist()
    log.info(f"Usable 2025 months (gdp_monthly >=50% non-null): {sorted(usable_months)}")

    current = m_std[(m_std["year"] == 2025)
                    & (m_std["dept_code"] != "00")
                    & (m_std["month"].isin(usable_months))].copy()
    current = current.sort_values(["dept_code", "month"]).reset_index(drop=True)

    if current.empty:
        log.warning("No 2025 departmental data in pressure tracker")
        current["pressure_centered"] = 0.0
        current["pressure_label"]    = "estable"
        return current, pd.DataFrame(), weights

    # Center: subtract expanding YTD mean within each dept
    current["pressure_centered"] = (
        current.groupby("dept_code")["pressure_score"]
        .transform(lambda x: x - x.expanding().mean())
    )
    current["pressure_label"] = current["pressure_centered"].apply(_pressure_label)

    # Driver attribution
    drivers = []
    for _, row in current.iterrows():
        contribs = {
            col: weights.get(col, 0.0) * row[col]
            for col in indicator_cols
            if weights.get(col, 0.0) > 0 and pd.notna(row.get(col))
        }
        sc = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        drivers.append({
            "dept_code":    row["dept_code"],
            "month":        int(row["month"]),
            "top_positive": [(k, round(v, 4)) for k, v in sc if v > 0][:2],
            "top_negative": [(k, round(v, 4)) for k, v in sc if v < 0][:2],
        })

    log.info(f"2025 pressure labels:\n{current['pressure_label'].value_counts().to_string()}")
    return current, pd.DataFrame(drivers), weights


# ══════════════════════════════════════════════════════════════════════════════
# Part C — Calibrate λ
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_lambda(monthly: pd.DataFrame, panel: pd.DataFrame,
                     coefs: pd.DataFrame) -> float:
    """
    ΔP_{d,y} = λ * ΔS_{d,y} + α_d + ε
    OLS on first-differenced, dept-demeaned panel (2005-2024).
    """
    m_std, _ = _pressure_scores_all(monthly, coefs)

    valid_depts = set(panel["dept_code"].unique())
    # Annual pressure score = mean of monthly scores within each dept-year
    annual_pressure = (
        m_std[(m_std["dept_code"] != "00")
              & (m_std["dept_code"].isin(valid_depts))
              & (m_std["year"].between(2005, 2024))]
        .groupby(["dept_code", "year"])["pressure_score"]
        .mean()
        .reset_index()
        .rename(columns={"pressure_score": "S"})
    )

    # Annual poverty at vintage 12 (fraction 0-1)
    poverty = (panel[(panel["vintage_month"] == 12) & (panel["year"].between(2005, 2024))]
               [["dept_code", "year", "poverty_rate"]].copy())

    df = annual_pressure.merge(poverty, on=["dept_code", "year"]).sort_values(
        ["dept_code", "year"])
    df["dS"] = df.groupby("dept_code")["S"].diff()
    df["dP"] = df.groupby("dept_code")["poverty_rate"].diff()
    df = df.dropna(subset=["dS", "dP"])

    # Department FE via demeaning
    dm = df.groupby("dept_code")[["dS", "dP"]].transform("mean")
    df["dS_dm"] = df["dS"] - dm["dS"]
    df["dP_dm"] = df["dP"] - dm["dP"]

    # OLS (no intercept after demeaning)
    n           = len(df)
    var_dS      = (df["dS_dm"] ** 2).sum()
    lambda_hat  = (df["dS_dm"] * df["dP_dm"]).sum() / var_dS if var_dS > 0 else 0.0
    resid       = df["dP_dm"] - lambda_hat * df["dS_dm"]
    se_lambda   = (np.sqrt((resid ** 2).sum() / max(n - 1, 1))
                   / np.sqrt(var_dS)) if var_dS > 0 else np.nan
    ss_tot      = ((df["dP_dm"] - df["dP_dm"].mean()) ** 2).sum()
    r2          = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0

    # Sanity: lambda * std(monthly pressure) → implied within-year swing (in pp)
    monthly_pressure_std = (
        m_std[(m_std["year"] == 2025) & (m_std["dept_code"] != "00")]
        ["pressure_score"].std()
    )
    implied_swing_pp = abs(lambda_hat) * monthly_pressure_std * 100  # convert fraction to pp

    print("\n--- LAMBDA CALIBRATION ---")
    print(f"  Lambda  : {lambda_hat:.6f}")
    print(f"  SE      : {se_lambda:.6f}")
    print(f"  R^2     : {r2:.4f}")
    print(f"  N obs   : {n}")
    print(f"  std(monthly_pressure)   : {monthly_pressure_std:.4f}")
    print(f"  lambda * std -> swing   : {implied_swing_pp:.2f} pp")

    if abs(lambda_hat) > 20:
        log.warning(f"Lambda={lambda_hat:.4f} > 20 — capping")
        lambda_hat = 20.0 * np.sign(lambda_hat)
    if implied_swing_pp > 10:
        log.warning(f"Implied swing {implied_swing_pp:.2f}pp > 10pp — investigate")
    else:
        print(f"  Sanity  : OK ({implied_swing_pp:.2f}pp within-year swing)")

    return lambda_hat


# ══════════════════════════════════════════════════════════════════════════════
# Part D — Monthly Poverty + Rolling 3-month
# ══════════════════════════════════════════════════════════════════════════════

def compute_monthly_poverty(annual_df: pd.DataFrame, pressure_df: pd.DataFrame,
                             lambda_hat: float) -> tuple:
    """P_{d,m} = P^A_{d,2025} + λ * S_tilde_{d,m}"""
    rows = []
    for _, ann in annual_df.iterrows():
        dc   = ann["dept_code"]
        pann = float(ann["blended_prediction"])
        dept_p = pressure_df[pressure_df["dept_code"] == dc].sort_values("month")
        for _, p in dept_p.iterrows():
            pmon = float(np.clip(pann + lambda_hat * p["pressure_centered"], 0, 1))
            rows.append({
                "dept_code":            dc,
                "department":           ann.get("department", ""),
                "year":                 2025,
                "month":                int(p["month"]),
                "poverty_monthly":      pmon,
                "poverty_annual_anchor": pann,
                "pressure_score":       float(p["pressure_centered"]),
                "pressure_label":       p["pressure_label"],
            })

    df = pd.DataFrame(rows).sort_values(["dept_code", "month"])
    df["poverty_rolling_3m"] = df.groupby("dept_code")["poverty_monthly"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    national = (df.groupby("month")
                .agg(poverty_monthly=("poverty_monthly", "mean"),
                     poverty_rolling_3m=("poverty_rolling_3m", "mean"),
                     pressure_score=("pressure_score", "mean"))
                .reset_index())
    national["pressure_label"] = national["pressure_score"].apply(_pressure_label)

    return df, national


# ══════════════════════════════════════════════════════════════════════════════
# Part E — Export JSON
# ══════════════════════════════════════════════════════════════════════════════

def export_poverty_json(annual_df: pd.DataFrame, monthly_dept: pd.DataFrame,
                         monthly_national: pd.DataFrame, vintage_month: int,
                         drivers_df: pd.DataFrame, panel: pd.DataFrame) -> dict:

    historical = (
        panel[panel["vintage_month"] == 12]
        .groupby("year")["poverty_rate"].mean().reset_index()
        .rename(columns={"poverty_rate": "official"})
    )

    validation = pd.read_parquet(PROCESSED / "validation_results.parquet")
    gbr_val = validation[validation["model"] == "gbr"]

    def pct(v):  # fraction → percent, rounded 1dp
        return round(float(v) * 100, 1)

    output = {
        "metadata": {
            "target_year": 2025,
            "vintage_month": f"2025-{vintage_month:02d}",
            "months_available": vintage_month,
            "coverage_note": (
                "Cobertura mixta: algunas series hasta "
                + ("nov 2025" if vintage_month == 12 else f"mes {vintage_month - 1} 2025")
                + ", otras hasta "
                + ("dic 2025" if vintage_month == 12 else f"mes {vintage_month} 2025")
                + (", inflaci\u00f3n hasta ene 2026" if vintage_month == 12 else "")
                + ". Rezagos de publicaci\u00f3n del BCRP aplicados por serie."
            ),
            "model": "gbr_blended",
            "blend_alpha": 0.7,
            "last_updated": datetime.now().isoformat(),
            "methodology_version": "2.0",
        },
        "national": {
            "poverty_nowcast":       pct(annual_df["blended_prediction"].mean()),
            "poverty_2024_official": pct(annual_df["prev_poverty"].mean()),
            "change_pp":             round(pct(annual_df["blended_prediction"].mean())
                                           - pct(annual_df["prev_poverty"].mean()), 1),
            "lower_ci":              pct(annual_df["lower_ci"].mean()),
            "upper_ci":              pct(annual_df["upper_ci"].mean()),
            "pressure_current":      (monthly_national.iloc[-1]["pressure_label"]
                                      if len(monthly_national) > 0 else "estable"),
            "pressure_score":        (round(float(monthly_national.iloc[-1]["pressure_score"]), 3)
                                      if len(monthly_national) > 0 else 0.0),
        },
        "departments": [],
        "revision_path": [],
        "monthly_series": [],
        "backtest": {
            "model": "gbr_blended",
            "validation_years": "2015-2024",
            "rmse": round(float(np.sqrt((gbr_val["error"] ** 2).mean())) * 100, 2),
            "mae":  round(float(gbr_val["error"].abs().mean()) * 100, 2),
            "directional_accuracy": round(float(gbr_val["direction_correct"].mean()), 2),
            "vintage_mae": {
                f"Q{i+1}": round(float(gbr_val[gbr_val["vintage_month"] == vm]
                                        ["error"].abs().mean()) * 100, 2)
                for i, vm in enumerate([3, 6, 9, 12])
            },
        },
        "historical_annual": [
            {"year": int(r["year"]), "official": pct(r["official"]), "nowcast": None}
            for _, r in historical.iterrows()
        ],
    }

    # Departments
    for _, row in annual_df.iterrows():
        dc = row["dept_code"]
        dp = monthly_dept[monthly_dept["dept_code"] == dc]
        pressure_label = (dp.sort_values("month").iloc[-1]["pressure_label"]
                          if len(dp) > 0 else "estable")
        output["departments"].append({
            "department":           row.get("department", ""),
            "dept_code":            dc,
            "poverty_nowcast":      pct(row["blended_prediction"]),
            "poverty_2024_official": pct(row["prev_poverty"]),
            "change_pp":            round(pct(row["blended_prediction"]) - pct(row["prev_poverty"]), 1),
            "lower_ci":             pct(row["lower_ci"]),
            "upper_ci":             pct(row["upper_ci"]),
            "pressure_label":       pressure_label,
        })

    # Monthly national series
    for _, row in monthly_national.iterrows():
        output["monthly_series"].append({
            "month":              f"2025-{int(row['month']):02d}",
            "national_rolling3m": pct(row["poverty_rolling_3m"]),
            "national_monthly":   pct(row["poverty_monthly"]),
            "pressure_score":     round(float(row["pressure_score"]), 3),
            "pressure_label":     row["pressure_label"],
        })

    # Revision path from vintage archive
    for vf in sorted(VINTAGES.glob("poverty_nowcast_*.json")):
        try:
            v = json.load(open(vf))
            output["revision_path"].append({
                "vintage": v["metadata"]["vintage_month"],
                "national_nowcast": v["national"]["poverty_nowcast"],
            })
        except Exception:
            pass

    # Save all destinations
    EXPORTS.mkdir(parents=True, exist_ok=True)
    VINTAGES.mkdir(parents=True, exist_ok=True)

    export_path = EXPORTS / "poverty_nowcast.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Exported: {export_path}")

    vintage_path = VINTAGES / f"poverty_nowcast_{vintage_month:02d}.json"
    with open(vintage_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Vintage archived: {vintage_path}")

    web_path = Path("D:/qhawarina/public/assets/data/poverty_nowcast.json")
    web_path.parent.mkdir(parents=True, exist_ok=True)
    with open(web_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Web copy: {web_path}")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Poverty Nowcast Generator")
    parser.add_argument("--vintage", type=int, default=None,
                        help="Override vintage month (1-12). Default: auto-detect from data.")
    args = parser.parse_args()

    log.info("=== POVERTY NOWCAST — Prompt 3: Production Forecast ===")

    # Load shared data; normalise ubigeo → dept_code
    panel   = pd.read_parquet(PROCESSED / "vintage_panel.parquet").rename(
                  columns={"ubigeo": "dept_code"})
    monthly = pd.read_parquet(PROCESSED / "monthly_indicators.parquet").rename(
                  columns={"ubigeo": "dept_code"})
    config  = json.load(open(PROCESSED / "blending_config.json"))
    coefs   = pd.read_csv(PROCESSED / "enet_coefficients.csv")

    print("\n=== DATA SUMMARY ===")
    print(f"  Vintage panel: {panel.shape}, years {panel['year'].min()}-{panel['year'].max()}")
    print(f"  Monthly:       {monthly.shape}, years {monthly['year'].min()}-{monthly['year'].max()}")
    print(f"  2025 months in monthly: {sorted(monthly[monthly['year']==2025]['month'].unique())}")
    m2025 = monthly[(monthly['year']==2025) & (monthly['dept_code']!='00')]
    print(f"  2025 non-null counts per series:")
    for s in SERIES_KEYS:
        nn = m2025[s].notna().sum()
        print(f"    {s:22s}: {nn}/{len(m2025)}")

    # Part A
    print("\n=== PART A: ANNUAL NOWCAST ===")
    annual_df, vintage_month, nat, nat_lo, nat_hi = generate_annual_nowcast(
        monthly, panel, config, vintage_override=args.vintage)

    # Part B
    print("\n=== PART B: PRESSURE TRACKER ===")
    pressure_df, drivers_df, weights = compute_pressure_tracker(
        monthly, coefs, vintage_month)
    non_zero_w = {k: round(v, 4) for k, v in weights.items() if v > 0}
    print(f"  Pressure weights (sum=1): {non_zero_w}")
    print(f"  2025 pressure range: [{pressure_df['pressure_centered'].min():.4f}, "
          f"{pressure_df['pressure_centered'].max():.4f}]")
    print(f"  2025 pressure std: {pressure_df['pressure_centered'].std():.4f}")

    # Part C
    print("\n=== PART C: LAMBDA CALIBRATION ===")
    lambda_hat = calibrate_lambda(monthly, panel, coefs)

    # Part D
    print("\n=== PART D: MONTHLY POVERTY ESTIMATES ===")
    monthly_dept, monthly_national = compute_monthly_poverty(
        annual_df, pressure_df, lambda_hat)

    # Part E
    print("\n=== PART E: EXPORT ===")
    output = export_poverty_json(
        annual_df, monthly_dept, monthly_national,
        vintage_month, drivers_df, panel)

    # ── Full results printout ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POVERTY NOWCAST 2025 — PRODUCTION OUTPUT")
    print("=" * 70)
    print(f"\nNational: {nat*100:.2f}% [{nat_lo*100:.2f}%-{nat_hi*100:.2f}%]")
    print(f"Change vs 2024: {(nat - annual_df['prev_poverty'].mean())*100:+.2f}pp")
    print(f"Vintage: month {vintage_month}")
    print(f"Lambda:  {lambda_hat:.6f}")

    print("\n--- ALL DEPARTMENT NOWCASTS ---")
    for _, r in annual_df.sort_values("blended_prediction", ascending=False).iterrows():
        dc   = r["dept_code"]
        dp   = monthly_dept[monthly_dept["dept_code"] == dc]
        lbl  = (dp.sort_values("month").iloc[-1]["pressure_label"]
                if len(dp) > 0 else "n/a")
        print(f"  {str(r.get('department','')):<20s}  {r['blended_prediction']*100:5.1f}%  "
              f"({(r['blended_prediction']-r['prev_poverty'])*100:+.1f}pp)  [{lbl}]")

    print("\n--- MONTHLY NATIONAL SERIES ---")
    for _, row in monthly_national.iterrows():
        print(f"  2025-{int(row['month']):02d}:  "
              f"rolling3m={row['poverty_rolling_3m']*100:.2f}%  "
              f"monthly={row['poverty_monthly']*100:.2f}%  "
              f"pressure={row['pressure_score']:+.4f} ({row['pressure_label']})")

    print("\n--- PRESSURE DISTRIBUTION (2025, all depts) ---")
    print(pressure_df["pressure_label"].value_counts().to_string())

    print("\n--- SANITY CHECKS ---")
    nat_pct = nat * 100
    status  = "OK" if 15 <= nat_pct <= 40 else "WARNING"
    print(f"  [{status}] National nowcast: {nat_pct:.1f}% (expected 15-40%)")
    status  = "OK" if abs(lambda_hat) <= 20 else "WARNING"
    print(f"  [{status}] Lambda: {lambda_hat:.4f} (expected <=20)")
    n_lbl   = pressure_df["pressure_label"].nunique()
    status  = "OK" if n_lbl > 1 else "WARNING"
    print(f"  [{status}] Distinct pressure labels: {n_lbl} (expected >1)")
    pstd    = pressure_df["pressure_centered"].std()
    status  = "OK" if pstd > 0.001 else "WARNING"
    print(f"  [{status}] Pressure std: {pstd:.4f} (expected >0.001)")

    print(f"\nExported: {EXPORTS / 'poverty_nowcast.json'}")
    print(f"Archived: {VINTAGES / f'poverty_nowcast_{vintage_month:02d}.json'}")
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
