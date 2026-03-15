"""
mw_canonical_estimation.py  —  VERSION 2
=========================================
Definitive canonical Minimum Wage DiD pipeline, Lima Metropolitan Area.

DESIGN SPEC (March 2026)
──────────────────────────────────────────────────────────────────────────────
Core principle: every outcome for every event comes from ONE internally
consistent sample (same files, same matching key, same wage variable, same
treatment/control bands, same restrictions, same SE estimator).

I. TREATMENT / CONTROL BANDS
  Primary  : treat = [0.85×MW_old, 1.00×MW_new)   ctrl = [1.00×MW_new, 1.40×MW_new)
  Narrow T : treat = [1.00×MW_old, 1.00×MW_new)
  Wide   T : treat = [0.70×MW_old, 1.00×MW_new)
  Narrow C :                                       ctrl = [1.00×MW_new, 1.20×MW_new)
  Wide   C :                                       ctrl = [1.00×MW_new, 1.60×MW_new)

II. MATCHING
  Consecutive-quarter 4-ID inner join (conglome+vivienda+hogar+codperso).
  All 9 MW events use this strategy. No mixing of annual / within-year designs.
  IDs normalized as str(int(float(x))) before merge. Dedup before join.

III. INCOME
  ingprin (primary) > ingtot > i211a > p211a > p524a1 > p524a.
  EPE microdata variable label D211B = "mensualizado": ingprin is already
  monthly soles per INEI convention. No periodicity variable exists in EPE
  (p209a = Sunday hours worked, not payment frequency).
  Wages clipped to [0, 50 000] soles.

IV. FORMALITY (Definition B — validated)
  formal = p222 ∈ {1, 2, 3}  (EsSalud / Seguro Privado / Ambos)
  Gap vs INEI headline: mean |Δ| = 2.7 pp, all 9 years within ±5 pp.

V. SELF-EMPLOYMENT
  p206 == 2  (independiente / cuenta propia)

VI. OUTCOMES  (all from same canonical in-band sample)
  A. Employment DiD          d_emp ~ treat
  B. Wage DiD (all stayers)  d_log_wage ~ treat
  C. Lee (2009) bounds       trim = |β_emp| from outcome A, treated only
  D. Lighthouse              d_log_wage ~ treat, informal stayers
  E. Formal wage DiD         d_log_wage ~ treat, formal stayers
  F. Formality transition    formal_post ~ treat + formal_pre
  G. Formal employment DiD   d_formal_emp ~ treat
  H. Self-employment DiD     d_selfemp ~ treat
  I. Placebo                 same structure 1 year earlier, no MW change

VII. POOLING
  IVW β_emp    : w_i = 1/SE_β_i²,  SE_pool = 1/√Σw_i
  IVW ε        : ε_i = β_emp_i / ln(MW_new_i/MW_old_i)
                 SE_ε_i = SE_β_i / |ln(MW_new_i/MW_old_i)|   [delta method]
                 w_i = 1/SE_ε_i²,  SE_pool = 1/√Σw_i

VIII. ADDITIONAL DIAGNOSTICS
  Attrition   : compare matched vs unmatched pre-period workers on wage/formality
  Overlap     : flag any individual appearing in >1 event's treatment/control group
  Month filter: restrict pmes when file straddles MW change month

KEY CORRECTIONS vs v1
──────────────────────────────────────────────────────────────────────────────
  2022_Apr : pre reverted to 742 (ANNUAL match). EPE post-2017 redesign visits
             same viviendas once/year; consecutive-quarter 757→768 yields 0
             matches (different rotation cohorts). Annual 742→766 gives 42%.
             Pre=Apr-Jun-2021 (MW=930), Post=Apr-Jun-2022 (MW=1025 since Apr 1).
  2016_May : pre 496→501, post 514→527  (eliminates correlated pair: 496 no
                            longer shared between 2015_Dec post and 2016_May pre)
  2006_Jan : pre 138→136  (138=Nov-Dic05-Ene06 straddles Jan change;
                            use 136=Oct-Nov-Dic05)
  2012_Jun : post 299→303 (299=Abr-May-Jun12 straddles Jun change;
                            use 303=Jun-Jul-Ago12)
  Placebo  : treat_hi = mw_placebo × (mw_new/mw_old) — preserves same relative
             band width as the real event (not an arbitrary ±15% fallback)

Usage
──────────────────────────────────────────────────────────────────────────────
  python scripts/mw_canonical_estimation.py
  python scripts/mw_canonical_estimation.py --robustness
  python scripts/mw_canonical_estimation.py --placebo
  python scripts/mw_canonical_estimation.py --attrition
  python scripts/mw_canonical_estimation.py --event 2022_Apr
  python scripts/mw_canonical_estimation.py --quiet
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
SAV_DIR     = pathlib.Path("D:/Nexus/nexus/data/raw/epe/srienaho")
RESULTS_DIR = pathlib.Path("D:/Nexus/nexus/data/results")
EXPORTS_DIR = pathlib.Path("D:/Nexus/nexus/exports/data")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Core constants ────────────────────────────────────────────────────────────
ID4           = ["conglome", "vivienda", "hogar", "codperso"]
FORMAL_CODES  = {1.0, 2.0, 3.0}
SELFEMPL_CODE = 2
INCOME_PRIORITY = ["ingprin", "ingtot", "i211a", "p211a", "p524a1", "p524a"]
HOURS_PRIORITY  = ["p209t", "p209cc"]
MIN_INBAND      = 20   # minimum obs per arm to report estimates

# ── Macroeconomic context per event ──────────────────────────────────────────
CONTEXT_DATA = {
    "2003_Oct": {"gdp_growth": 4.0, "inflation": 2.3, "phase": "expansion_start"},
    "2006_Jan": {"gdp_growth": 6.7, "inflation": 1.5, "phase": "commodity_boom"},
    "2007_Aug": {"gdp_growth": 8.9, "inflation": 3.9, "phase": "overheating"},
    "2008_Jan": {"gdp_growth": 9.8, "inflation": 5.8, "phase": "crisis_onset"},
    "2011_Aug": {"gdp_growth": 6.9, "inflation": 3.5, "phase": "recovery"},
    "2012_Jun": {"gdp_growth": 6.0, "inflation": 3.7, "phase": "expansion"},
    "2015_Dec": {"gdp_growth": 3.3, "inflation": 4.4, "phase": "slowdown"},
    "2016_May": {"gdp_growth": 3.9, "inflation": 3.6, "phase": "moderate"},
    "2022_Apr": {"gdp_growth": 2.7, "inflation": 8.1, "phase": "post_covid"},
}

# ── Band definitions ──────────────────────────────────────────────────────────
BANDS = {
    "primary":      {"treat_lo": 0.85, "ctrl_hi": 1.4},
    "narrow_treat": {"treat_lo": 1.0,  "ctrl_hi": 1.4},
    "wide_treat":   {"treat_lo": 0.7,  "ctrl_hi": 1.4},
    "narrow_ctrl":  {"treat_lo": 0.85, "ctrl_hi": 1.2},
    "wide_ctrl":    {"treat_lo": 0.85, "ctrl_hi": 1.6},
}
BAND_PRIMARY = BANDS["primary"]

# ── MW events (9 events, Lima Metro EPE) ──────────────────────────────────────
MW_EVENTS = [
    {
        "label": "2003_Oct",
        "mw_old": 410.0, "mw_new": 460.0,
        "pre_codes": [92],  "post_codes": [95],
        "placebo_pre_codes": [62], "placebo_post_codes": [67],
        "mw_placebo": 410.0, "mw_change_month": 10, "filter_pmes": False,
    },
    {
        "label": "2006_Jan",
        "mw_old": 500.0, "mw_new": 530.0,
        "pre_codes": [136], "post_codes": [142],
        "placebo_pre_codes": [102], "placebo_post_codes": [117],
        "mw_placebo": 460.0, "mw_change_month": 1, "filter_pmes": False,
    },
    {
        "label": "2007_Aug",
        "mw_old": 530.0, "mw_new": 550.0,
        "pre_codes": [179], "post_codes": [187],
        "placebo_pre_codes": [155], "placebo_post_codes": [161],
        "mw_placebo": 530.0, "mw_change_month": 8, "filter_pmes": False,
    },
    {
        "label": "2008_Jan",
        "mw_old": 550.0, "mw_new": 600.0,
        "pre_codes": [191], "post_codes": [196],
        "placebo_pre_codes": [166], "placebo_post_codes": [170],
        "mw_placebo": 530.0, "mw_change_month": 1, "filter_pmes": False,
    },
    {
        "label": "2011_Aug",
        "mw_old": 600.0, "mw_new": 675.0,
        "pre_codes": [264], "post_codes": [272],
        "placebo_pre_codes": [243], "placebo_post_codes": [248],
        "mw_placebo": 550.0, "mw_change_month": 8, "filter_pmes": False,
    },
    {
        "label": "2012_Jun",
        "mw_old": 675.0, "mw_new": 750.0,
        "pre_codes": [287], "post_codes": [303],
        "placebo_pre_codes": [257], "placebo_post_codes": [264],
        "mw_placebo": 600.0, "mw_change_month": 6, "filter_pmes": False,
    },
    {
        "label": "2015_Dec",
        "mw_old": 750.0, "mw_new": 850.0,
        "pre_codes": [485], "post_codes": [496],
        "placebo_pre_codes": [431], "placebo_post_codes": [439],
        "mw_placebo": 750.0, "mw_change_month": 12, "filter_pmes": False,
    },
    {
        "label": "2016_May",
        "mw_old": 850.0, "mw_new": 930.0,
        "pre_codes": [501], "post_codes": [527],
        "placebo_pre_codes": [444], "placebo_post_codes": [453],
        "mw_placebo": 750.0, "mw_change_month": 5, "filter_pmes": False,
    },
    {
        "label": "2022_Apr",
        "mw_old": 930.0, "mw_new": 1025.0,
        "pre_codes": [742], "post_codes": [766],
        "placebo_pre_codes": [695], "placebo_post_codes": [742],
        "mw_placebo": 930.0, "mw_change_month": 4, "filter_pmes": False,
        "annual_match": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sav(code: int) -> pd.DataFrame | None:
    """Load a single EPE .sav wave by SRIENAHO code."""
    d = SAV_DIR / str(code)
    savs = sorted(d.glob("*.sav")) + sorted(d.glob("*.SAV"))
    if not savs:
        return None
    df, _ = pyreadstat.read_sav(str(savs[0]))
    df.columns = [c.lower() for c in df.columns]
    return df


def detect_income_col(df: pd.DataFrame) -> str | None:
    """Return the first available column from INCOME_PRIORITY."""
    for col in INCOME_PRIORITY:
        if col in df.columns:
            return col
    return None


def detect_and_construct_monthly_wage(
    df: pd.DataFrame, code: int, verbose: bool = True
) -> tuple[pd.DataFrame, str | None]:
    """
    Detect and return monthly wage from the income priority chain.
    Clips to [0, 50000]. Returns (df_with_wage_col, col_name).
    """
    inc_col = detect_income_col(df)
    if inc_col is None:
        if verbose:
            print(f"      [WARN code={code}] No income column found in {list(df.columns[:10])}")
        return df, None

    df = df.copy()
    df["_wage"] = pd.to_numeric(df[inc_col], errors="coerce").clip(0, 50_000)
    if verbose:
        n_pos = (df["_wage"] > 0).sum()
        print(f"      [code={code}] income col={inc_col!r}  n_pos={n_pos}/{len(df)}")
    return df, inc_col


def detect_hours_col_from_merged(merged: pd.DataFrame) -> str | None:
    """Return the first available hours column (pre-period) from HOURS_PRIORITY."""
    for col in HOURS_PRIORITY:
        col_pre = col + "_pre"
        if col_pre in merged.columns:
            n_valid = pd.to_numeric(merged[col_pre], errors="coerce").gt(0).sum()
            if n_valid > 10:
                return col
    return None


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ID4 columns to str(int(float(x))) format for consistent joining."""
    df = df.copy()
    for c in ID4:
        if c in df.columns:
            df[c] = (
                pd.to_numeric(df[c], errors="coerce")
                .fillna(-9999)
                .astype(int)
                .astype(str)
            )
    return df


def apply_month_filter(
    df: pd.DataFrame, change_month: int, period: str
) -> pd.DataFrame:
    """
    When filter_pmes=True, drop observations whose survey month (pmes)
    straddles the MW change month to avoid mixing pre/post regimes.
    period='pre'  → keep months < change_month
    period='post' → keep months >= change_month
    """
    if "pmes" not in df.columns:
        return df
    pmes = pd.to_numeric(df["pmes"], errors="coerce")
    if period == "pre":
        mask = pmes < change_month
    else:
        mask = pmes >= change_month
    return df[mask].copy()


def validate_matches(merged: pd.DataFrame) -> dict:
    """
    Validate quality of the panel match.
    Returns dict with sex_match_rate, age_drift_ok_rate, age_diff_mean.
    """
    result = {}
    # Sex consistency
    if "p207_pre" in merged.columns and "p207_post" in merged.columns:
        sex_pre  = pd.to_numeric(merged["p207_pre"],  errors="coerce")
        sex_post = pd.to_numeric(merged["p207_post"], errors="coerce")
        both_valid = sex_pre.notna() & sex_post.notna()
        match_rate = (sex_pre == sex_post)[both_valid].mean() if both_valid.any() else np.nan
        result["sex_match_rate"] = float(match_rate)
    else:
        result["sex_match_rate"] = np.nan

    # Age drift (should be ~0 or ~1 quarter within-year)
    if "p208a_pre" in merged.columns and "p208a_post" in merged.columns:
        age_pre  = pd.to_numeric(merged["p208a_pre"],  errors="coerce")
        age_post = pd.to_numeric(merged["p208a_post"], errors="coerce")
        diff = (age_post - age_pre).abs()
        both_valid = age_pre.notna() & age_post.notna()
        result["age_diff_mean"]    = float(diff[both_valid].mean()) if both_valid.any() else np.nan
        result["age_drift_ok_rate"] = float((diff[both_valid] <= 2).mean()) if both_valid.any() else np.nan
    else:
        result["age_diff_mean"]    = np.nan
        result["age_drift_ok_rate"] = np.nan

    return result


def attrition_diagnostic(
    df_pre: pd.DataFrame,
    merged: pd.DataFrame,
    inc_col_pre: str,
    label: str,
) -> dict:
    """
    Compare matched vs unmatched pre-period employed workers on wage and formality.
    Returns dict with attrition diagnostic stats.
    """
    employed_pre = df_pre[
        pd.to_numeric(df_pre.get("ocu200", pd.Series(dtype=float)), errors="coerce") == 1
    ].copy()

    matched_ids = set(
        merged["conglome_pre"].astype(str) + "_" +
        merged["vivienda_pre"].astype(str) + "_" +
        merged["hogar_pre"].astype(str) + "_" +
        merged["codperso_pre"].astype(str)
        if "conglome_pre" in merged.columns else []
    )

    # Use ID4 on employed_pre
    emp_pid = (
        employed_pre["conglome"].astype(str) + "_" +
        employed_pre["vivienda"].astype(str) + "_" +
        employed_pre["hogar"].astype(str) + "_" +
        employed_pre["codperso"].astype(str)
    ) if all(c in employed_pre.columns for c in ID4) else pd.Series(dtype=str)

    result = {
        "n_pre_employed":  int(len(employed_pre)),
        "n_matched":       int(len(merged)),
        "match_rate":      round(len(merged) / max(len(employed_pre), 1), 4),
    }

    if inc_col_pre in employed_pre.columns and len(emp_pid) > 0:
        wage_all    = pd.to_numeric(employed_pre[inc_col_pre], errors="coerce")
        matched_mask = emp_pid.isin(matched_ids)
        wage_matched   = wage_all[matched_mask].dropna()
        wage_unmatched = wage_all[~matched_mask].dropna()
        result["wage_matched_mean"]   = round(float(wage_matched.mean()),   1) if len(wage_matched)   else np.nan
        result["wage_unmatched_mean"] = round(float(wage_unmatched.mean()), 1) if len(wage_unmatched) else np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_and_match(
    event: dict,
    verbose: bool = True,
    use_placebo: bool = False,
) -> tuple[pd.DataFrame | None, str | None, str | None, dict]:
    """
    Load pre and post EPE waves for an event and return the matched panel.

    Returns (merged_df, inc_col_pre, inc_col_post, diagnostics_dict).
    merged_df has suffix _pre / _post for all columns.
    """
    label = event["label"]
    if use_placebo:
        pre_codes  = event["placebo_pre_codes"]
        post_codes = event["placebo_post_codes"]
    else:
        pre_codes  = event["pre_codes"]
        post_codes = event["post_codes"]

    # Load waves (can be lists; concatenate if multiple codes)
    dfs_pre, dfs_post = [], []
    for c in pre_codes:
        df = load_sav(c)
        if df is not None:
            dfs_pre.append(df)
    for c in post_codes:
        df = load_sav(c)
        if df is not None:
            dfs_post.append(df)

    if not dfs_pre or not dfs_post:
        if verbose:
            print(f"  SKIP {label}: missing SAV files (pre={pre_codes}, post={post_codes})")
        return None, None, None, {}

    df_pre  = pd.concat(dfs_pre,  ignore_index=True)
    df_post = pd.concat(dfs_post, ignore_index=True)

    # Optional month filter (for waves straddling the change month)
    if event.get("filter_pmes"):
        df_pre  = apply_month_filter(df_pre,  event["mw_change_month"], "pre")
        df_post = apply_month_filter(df_post, event["mw_change_month"], "post")

    # Normalize IDs
    df_pre  = normalize_ids(df_pre)
    df_post = normalize_ids(df_post)

    # Detect income columns
    df_pre,  inc_col_pre  = detect_and_construct_monthly_wage(df_pre,  pre_codes[0],  verbose)
    df_post, inc_col_post = detect_and_construct_monthly_wage(df_post, post_codes[0], verbose)

    # Dedup on ID4 before merging (keep first occurrence)
    id4_in_pre  = [c for c in ID4 if c in df_pre.columns]
    id4_in_post = [c for c in ID4 if c in df_post.columns]
    df_pre  = df_pre.drop_duplicates(subset=id4_in_pre)
    df_post = df_post.drop_duplicates(subset=id4_in_post)

    # Inner join on ID4
    merged = df_pre.merge(df_post, on=id4_in_pre, suffixes=("_pre", "_post"), how="inner")

    if verbose:
        n_pre  = len(df_pre)
        n_post = len(df_post)
        rate   = len(merged) / max(n_pre, 1)
        print(f"  {label}: pre={n_pre}, post={n_post}, matched={len(merged)} ({rate:.1%})")

    # Validate
    diag = validate_matches(merged)
    diag["n_pre"]    = int(len(df_pre))
    diag["n_post"]   = int(len(df_post))
    diag["n_matched"] = int(len(merged))
    diag["match_rate"] = round(len(merged) / max(len(df_pre), 1), 4)

    return merged, inc_col_pre, inc_col_post, diag


def assign_bands(
    merged: pd.DataFrame,
    mw_old: float,
    mw_new: float,
    band: dict,
) -> pd.DataFrame:
    """
    Assign treatment/control band membership based on pre-period wage.

    Treatment: [treat_lo × MW_old, MW_new)
    Control:   [MW_new, ctrl_hi × MW_new)

    Adds column 'treat' (1=treatment, 0=control) and 'in_band' (bool).
    Also adds 'employed_post' and 'employed_pre' from ocu200.
    """
    merged = merged.copy()

    # Pre-period wage
    wage_pre = merged.get("_wage_pre", pd.Series(np.nan, index=merged.index))
    if "_wage_pre" not in merged.columns:
        # Try to find it from renamed columns
        for col in INCOME_PRIORITY:
            col_pre = col + "_pre"
            if col_pre in merged.columns:
                wage_pre = pd.to_numeric(merged[col_pre], errors="coerce").clip(0, 50_000)
                break

    treat_lo = band["treat_lo"] * mw_old
    treat_hi = mw_new
    ctrl_lo  = mw_new
    ctrl_hi  = band["ctrl_hi"] * mw_new

    treat_mask = (wage_pre >= treat_lo) & (wage_pre < treat_hi)
    ctrl_mask  = (wage_pre >= ctrl_lo)  & (wage_pre < ctrl_hi)

    merged["treat"]   = np.where(treat_mask, 1, np.where(ctrl_mask, 0, np.nan))
    merged["in_band"] = treat_mask | ctrl_mask
    merged["wage_pre"] = wage_pre

    # Post-period wage
    wage_post = pd.Series(np.nan, index=merged.index)
    for col in INCOME_PRIORITY:
        col_post = col + "_post"
        if col_post in merged.columns:
            wage_post = pd.to_numeric(merged[col_post], errors="coerce").clip(0, 50_000)
            break
    merged["wage_post"] = wage_post

    # Employment status
    ocu_pre  = pd.to_numeric(merged.get("ocu200_pre",  pd.Series(np.nan, index=merged.index)), errors="coerce")
    ocu_post = pd.to_numeric(merged.get("ocu200_post", pd.Series(np.nan, index=merged.index)), errors="coerce")
    merged["employed_pre"]  = (ocu_pre  == 1).astype(int)
    merged["employed_post"] = (ocu_post == 1).astype(int)

    # Formality
    p222_pre  = pd.to_numeric(merged.get("p222_pre",  pd.Series(np.nan, index=merged.index)), errors="coerce")
    p222_post = pd.to_numeric(merged.get("p222_post", pd.Series(np.nan, index=merged.index)), errors="coerce")
    merged["formal_pre"]  = p222_pre.isin(FORMAL_CODES).astype(int)
    merged["formal_post"] = p222_post.isin(FORMAL_CODES).astype(int)

    # Self-employment
    p206_pre  = pd.to_numeric(merged.get("p206_pre",  pd.Series(np.nan, index=merged.index)), errors="coerce")
    p206_post = pd.to_numeric(merged.get("p206_post", pd.Series(np.nan, index=merged.index)), errors="coerce")
    merged["selfemp_pre"]  = (p206_pre  == SELFEMPL_CODE).astype(int)
    merged["selfemp_post"] = (p206_post == SELFEMPL_CODE).astype(int)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# OLS / inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def ols_hc1(
    y: pd.Series,
    treat: pd.Series,
    covariates: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    """
    OLS with HC1 standard errors: y ~ treat [+ covariates].
    Returns (beta_treat, se_treat, pval_treat).
    """
    df_ols = pd.DataFrame({"y": y, "treat": treat.astype(float)}).dropna()
    if covariates is not None:
        cov_aligned = covariates.loc[df_ols.index].copy()
        for col in cov_aligned.columns:
            df_ols[col] = pd.to_numeric(cov_aligned[col], errors="coerce")
        df_ols = df_ols.dropna()

    if len(df_ols) < 10 or df_ols["treat"].std() == 0:
        return np.nan, np.nan, np.nan

    X = sm.add_constant(df_ols.drop(columns=["y"]))
    try:
        res = sm.OLS(df_ols["y"], X).fit(cov_type="HC1")
        return float(res.params["treat"]), float(res.bse["treat"]), float(res.pvalues["treat"])
    except Exception:
        return np.nan, np.nan, np.nan


def stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _stayers(sub: pd.DataFrame, formality: str | None = None) -> pd.DataFrame:
    """
    Return workers who were employed in both periods (stayers).
    formality='formal'   → also formal_pre == 1
    formality='informal' → also formal_pre == 0
    """
    stay = sub[(sub["employed_pre"] == 1) & (sub["employed_post"] == 1)].copy()
    if formality == "formal":
        stay = stay[stay["formal_pre"] == 1]
    elif formality == "informal":
        stay = stay[stay["formal_pre"] == 0]
    return stay


def _dlog_wage(df: pd.DataFrame) -> pd.Series:
    """Compute d_log_wage = log(wage_post) - log(wage_pre) for stayers with positive wages."""
    wp = df["wage_pre"].clip(lower=1e-6)
    wq = df["wage_post"].clip(lower=1e-6)
    valid = (df["wage_pre"] > 0) & (df["wage_post"] > 0)
    result = pd.Series(np.nan, index=df.index)
    result[valid] = np.log(wq[valid]) - np.log(wp[valid])
    return result


def _get_col(df: pd.DataFrame, col: str, suffix: str = "_pre") -> pd.Series:
    """Safely retrieve a column with suffix, returning NaN series if missing."""
    full = col + suffix
    if full in df.columns:
        return pd.to_numeric(df[full], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Outcome functions (A–J)
# ─────────────────────────────────────────────────────────────────────────────

def outcome_a_employment(sub: pd.DataFrame) -> dict:
    """
    Outcome A: Employment DiD.
    d_emp = employed_post - employed_pre.
    Includes all in-band workers (treat + ctrl).
    """
    y     = sub["employed_post"] - sub["employed_pre"]
    treat = sub["treat"]
    beta, se, pval = ols_hc1(y, treat)
    n_treat = int((treat == 1).sum())
    n_ctrl  = int((treat == 0).sum())
    n       = n_treat + n_ctrl
    mean_treat = float(y[treat == 1].mean()) if n_treat else np.nan
    mean_ctrl  = float(y[treat == 0].mean()) if n_ctrl  else np.nan
    return {
        "n": n, "n_treat": n_treat, "n_ctrl": n_ctrl,
        "beta": round(float(beta), 4) if not np.isnan(beta) else None,
        "se":   round(float(se),   4) if not np.isnan(se)   else None,
        "pval": round(float(pval), 4) if not np.isnan(pval) else None,
        "mean_treat": round(mean_treat, 4) if not np.isnan(mean_treat) else None,
        "mean_ctrl":  round(mean_ctrl,  4) if not np.isnan(mean_ctrl)  else None,
    }


def outcome_b_wage_all(sub: pd.DataFrame) -> dict:
    """
    Outcome B: Wage DiD, all stayers.
    d_log_wage ~ treat (OLS HC1).
    """
    stay  = _stayers(sub)
    dlw   = _dlog_wage(stay)
    treat = stay["treat"]
    valid = dlw.notna() & treat.notna()
    beta, se, pval = ols_hc1(dlw[valid], treat[valid])
    n_treat = int((treat[valid] == 1).sum())
    n_ctrl  = int((treat[valid] == 0).sum())
    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "beta":     round(float(beta), 4) if not np.isnan(beta) else None,
        "se":       round(float(se),   4) if not np.isnan(se)   else None,
        "pval":     round(float(pval), 4) if not np.isnan(pval) else None,
        "beta_pct": round(float(beta) * 100, 2) if not np.isnan(beta) else None,
    }


def outcome_c_lee_bounds(sub: pd.DataFrame, beta_emp: float) -> dict:
    """
    Outcome C: Lee (2009) selection-correction bounds.
    trim_fraction = |β_emp| (from outcome A).
    Apply one-sided trim to treated stayers to bound the wage effect.
    """
    stay    = _stayers(sub)
    dlw     = _dlog_wage(stay)
    treat   = stay["treat"]

    # Treated stayers
    treat_stay = stay[treat == 1].copy()
    treat_stay["dlw"] = dlw[treat == 1]
    ctrl_stay  = stay[treat == 0].copy()
    ctrl_stay["dlw"]  = dlw[treat == 0]

    n_treat_stay = int((treat == 1).sum())
    n_ctrl_stay  = int((treat == 0).sum())

    # Unconditional beta (same as B)
    beta_unc, se_unc, _ = ols_hc1(dlw[dlw.notna() & treat.notna()],
                                   treat[dlw.notna() & treat.notna()])

    trim_frac = abs(beta_emp) if not np.isnan(beta_emp) else 0.0
    n_trim = max(0, int(round(trim_frac * n_treat_stay)))

    treat_dlw = treat_stay["dlw"].dropna().sort_values()
    if len(treat_dlw) < 4 or n_trim == 0:
        lee_lb = float(beta_unc) if not np.isnan(beta_unc) else None
        lee_ub = float(beta_unc) if not np.isnan(beta_unc) else None
    else:
        # Lower bound: trim top n_trim from treated (worst-case)
        trimmed_lb = treat_dlw.iloc[: len(treat_dlw) - n_trim]
        ctrl_dlw   = ctrl_stay["dlw"].dropna()
        lb = float(trimmed_lb.mean() - ctrl_dlw.mean()) if len(ctrl_dlw) else np.nan

        # Upper bound: trim bottom n_trim from treated (best-case)
        trimmed_ub = treat_dlw.iloc[n_trim:]
        ub = float(trimmed_ub.mean() - ctrl_dlw.mean()) if len(ctrl_dlw) else np.nan

        lee_lb = round(lb, 4) if not np.isnan(lb) else None
        lee_ub = round(ub, 4) if not np.isnan(ub) else None

    return {
        "n_treat_stay": n_treat_stay,
        "n_ctrl_stay":  n_ctrl_stay,
        "n_trim":       n_trim,
        "trim_pct":     round(trim_frac, 4),
        "beta_unc":     round(float(beta_unc), 4) if not np.isnan(beta_unc) else None,
        "lee_lb":       lee_lb,
        "lee_ub":       lee_ub,
        "lee_lb_pct":   round(lee_lb * 100, 1) if lee_lb is not None else None,
        "lee_ub_pct":   round(lee_ub * 100, 1) if lee_ub is not None else None,
    }


def outcome_d_lighthouse(sub: pd.DataFrame) -> dict:
    """
    Outcome D: Lighthouse effect — wage DiD for INFORMAL stayers only.
    Tests whether MW raise affects wages even for workers not legally bound.
    """
    stay  = _stayers(sub, formality="informal")
    dlw   = _dlog_wage(stay)
    treat = stay["treat"]
    valid = dlw.notna() & treat.notna()
    beta, se, pval = ols_hc1(dlw[valid], treat[valid])
    n_treat = int((treat[valid] == 1).sum())
    n_ctrl  = int((treat[valid] == 0).sum())
    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "beta":     round(float(beta), 4) if not np.isnan(beta) else None,
        "se":       round(float(se),   4) if not np.isnan(se)   else None,
        "pval":     round(float(pval), 4) if not np.isnan(pval) else None,
        "beta_pct": round(float(beta) * 100, 2) if not np.isnan(beta) else None,
    }


def outcome_e_formal_wage(sub: pd.DataFrame) -> dict:
    """
    Outcome E: Wage DiD for FORMAL stayers only.
    Benchmark: formal wages should clearly respond to binding MW.
    """
    stay  = _stayers(sub, formality="formal")
    dlw   = _dlog_wage(stay)
    treat = stay["treat"]
    valid = dlw.notna() & treat.notna()
    beta, se, pval = ols_hc1(dlw[valid], treat[valid])
    n_treat = int((treat[valid] == 1).sum())
    n_ctrl  = int((treat[valid] == 0).sum())
    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "beta":     round(float(beta), 4) if not np.isnan(beta) else None,
        "se":       round(float(se),   4) if not np.isnan(se)   else None,
        "pval":     round(float(pval), 4) if not np.isnan(pval) else None,
        "beta_pct": round(float(beta) * 100, 2) if not np.isnan(beta) else None,
    }


def outcome_f_formality_transition(
    sub: pd.DataFrame, beta_emp: float = np.nan
) -> dict:
    """
    Outcome F: Formality transition DiD.
    formal_post ~ treat + formal_pre  (OLS HC1, employed pre-period workers).
    Lee bounds applied with same trim fraction as employment.
    """
    employed_pre = sub[sub["employed_pre"] == 1].copy()
    treat = employed_pre["treat"]
    y     = employed_pre["formal_post"]
    cov   = employed_pre[["formal_pre"]]

    did_raw = float(y[treat == 1].mean() - y[treat == 0].mean()) if (treat == 1).any() and (treat == 0).any() else np.nan
    beta, se, pval = ols_hc1(y, treat, covariates=cov)

    n_treat = int((treat == 1).sum())
    n_ctrl  = int((treat == 0).sum())

    # Lee bounds for formality (treated stayers who stayed employed)
    stay  = _stayers(sub)
    treat_f = stay["treat"]
    yf    = stay["formal_post"]
    valid = yf.notna() & treat_f.notna()

    trim_frac = abs(beta_emp) if not np.isnan(beta_emp) else 0.0
    treat_f_vals = yf[(treat_f == 1) & valid].sort_values()
    ctrl_f_vals  = yf[(treat_f == 0) & valid]
    n_trim = max(0, int(round(trim_frac * len(treat_f_vals))))

    if len(treat_f_vals) >= 4 and n_trim > 0 and len(ctrl_f_vals) > 0:
        lb = float(treat_f_vals.iloc[:len(treat_f_vals) - n_trim].mean() - ctrl_f_vals.mean())
        ub = float(treat_f_vals.iloc[n_trim:].mean() - ctrl_f_vals.mean())
        lee_lb = round(lb, 4)
        lee_ub = round(ub, 4)
    else:
        lee_lb = round(did_raw, 4) if not np.isnan(did_raw) else None
        lee_ub = lee_lb

    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "did_raw": round(did_raw, 4) if not np.isnan(did_raw) else None,
        "beta":    round(float(beta), 4) if not np.isnan(beta) else None,
        "se":      round(float(se),   4) if not np.isnan(se)   else None,
        "pval":    round(float(pval), 4) if not np.isnan(pval) else None,
        "lee_lb":  lee_lb,
        "lee_ub":  lee_ub,
    }


def outcome_g_formal_employment(sub: pd.DataFrame) -> dict:
    """
    Outcome G: Formal employment DiD.
    d_formal_emp = formal_post × employed_post − formal_pre × employed_pre.
    """
    y     = sub["formal_post"] * sub["employed_post"] - sub["formal_pre"] * sub["employed_pre"]
    treat = sub["treat"]
    beta, se, pval = ols_hc1(y, treat)
    n_treat = int((treat == 1).sum())
    n_ctrl  = int((treat == 0).sum())
    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "beta":    round(float(beta), 4) if not np.isnan(beta) else None,
        "se":      round(float(se),   4) if not np.isnan(se)   else None,
        "pval":    round(float(pval), 4) if not np.isnan(pval) else None,
    }


def outcome_h_selfemployment(sub: pd.DataFrame) -> dict:
    """
    Outcome H: Self-employment DiD.
    d_selfemp = selfemp_post − selfemp_pre.
    Tests whether workers shift to self-employment in response to MW increase.
    """
    y     = sub["selfemp_post"] - sub["selfemp_pre"]
    treat = sub["treat"]
    beta, se, pval = ols_hc1(y, treat)
    n_treat = int((treat == 1).sum())
    n_ctrl  = int((treat == 0).sum())
    return {
        "n":       n_treat + n_ctrl,
        "n_treat": n_treat,
        "n_ctrl":  n_ctrl,
        "beta":    round(float(beta), 4) if not np.isnan(beta) else None,
        "se":      round(float(se),   4) if not np.isnan(se)   else None,
        "pval":    round(float(pval), 4) if not np.isnan(pval) else None,
    }


def outcome_j_hours(
    sub: pd.DataFrame, hours_col: str | None = None, beta_emp: float = np.nan
) -> dict:
    """
    Outcome J: Hours worked DiD.
    d_log_hours = log(hours_post) - log(hours_pre) for stayers.
    Lee bounds using same trim fraction as employment.
    """
    if hours_col is None:
        return {"available": False}

    stay  = _stayers(sub)
    hcol_pre  = hours_col + "_pre"
    hcol_post = hours_col + "_post"

    if hcol_pre not in stay.columns or hcol_post not in stay.columns:
        return {"available": False, "hours_col": hours_col}

    h_pre  = pd.to_numeric(stay[hcol_pre],  errors="coerce").clip(lower=1e-6)
    h_post = pd.to_numeric(stay[hcol_post], errors="coerce").clip(lower=1e-6)
    valid  = (h_pre > 0) & (h_post > 0) & stay["treat"].notna()

    dlh   = np.log(h_post[valid]) - np.log(h_pre[valid])
    treat = stay["treat"][valid]

    beta, se, pval = ols_hc1(dlh, treat)
    n_treat = int((treat == 1).sum())
    n_ctrl  = int((treat == 0).sum())

    # Lee bounds (hours)
    treat_h = dlh[treat == 1].sort_values()
    ctrl_h  = dlh[treat == 0]
    trim_frac = abs(beta_emp) if not np.isnan(beta_emp) else 0.0
    n_trim = max(0, int(round(trim_frac * len(treat_h))))
    if len(treat_h) >= 4 and n_trim > 0 and len(ctrl_h) > 0:
        lb = float(treat_h.iloc[:len(treat_h) - n_trim].mean() - ctrl_h.mean())
        ub = float(treat_h.iloc[n_trim:].mean() - ctrl_h.mean())
    else:
        lb = ub = float(beta) if not np.isnan(beta) else np.nan

    return {
        "available": True,
        "hours_col": hours_col,
        "n":         n_treat + n_ctrl,
        "n_treat":   n_treat,
        "n_ctrl":    n_ctrl,
        "mean_treat": round(float(dlh[treat == 1].mean()), 4) if n_treat else None,
        "mean_ctrl":  round(float(dlh[treat == 0].mean()), 4) if n_ctrl  else None,
        "beta":  round(float(beta), 4) if not np.isnan(beta) else None,
        "se":    round(float(se),   4) if not np.isnan(se)   else None,
        "pval":  round(float(pval), 4) if not np.isnan(pval) else None,
        "lee_lb": round(lb, 4) if not np.isnan(lb) else None,
        "lee_ub": round(ub, 4) if not np.isnan(ub) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneity and decomposition
# ─────────────────────────────────────────────────────────────────────────────

def sector_from_ciiu(code: float) -> str:
    """Map CIIU industry code to broad sector label."""
    if np.isnan(code):
        return "Desconocido"
    code = int(code)
    if   1   <= code <= 99:   return "Agropecuario"
    elif 100  <= code <= 199:  return "Minería"
    elif 200  <= code <= 399:  return "Manufactura"
    elif 400  <= code <= 499:  return "Electricidad"
    elif 500  <= code <= 599:  return "Construcción"
    elif 600  <= code <= 699:  return "Comercio"
    elif 700  <= code <= 799:  return "Transporte"
    elif 800  <= code <= 899:  return "Financiero"
    elif 900  <= code <= 939:  return "Serv. empresa"
    elif 9200 <= code <= 9299: return "Educación"
    elif 9300 <= code <= 9399: return "Salud"
    elif 9400 <= code <= 9499: return "Otros serv."
    else:                       return "Serv. empresa"


def firm_size_group(n: float) -> str:
    """Classify firm size from p205 (number of workers)."""
    if np.isnan(n) or n < 0:
        return None
    if n <= 9:
        return "Micro (1-9)"
    elif n <= 49:
        return "Pequeña (10-49)"
    elif n <= 199:
        return "Mediana (50-199)"
    else:
        return "Grande (200+)"


def age_group(a: float) -> str:
    """Classify age group."""
    if np.isnan(a) or a < 14:
        return None
    if a <= 24:
        return "Joven (14-24)"
    elif a <= 44:
        return "Adulto (25-44)"
    else:
        return "Mayor (45+)"


def run_group_did(
    sub: pd.DataFrame,
    groups: pd.Series,
    min_treat: int = 15,
) -> dict:
    """
    Run employment DiD within each group in the 'groups' series.
    Returns dict of {group_label: {n_treat, n_ctrl, beta, se, pval}}.
    """
    result = {}
    y     = sub["employed_post"] - sub["employed_pre"]
    treat = sub["treat"]

    for grp in sorted(groups.dropna().unique()):
        mask = (groups == grp) & treat.notna()
        yg    = y[mask]
        tg    = treat[mask]
        nt    = int((tg == 1).sum())
        nc    = int((tg == 0).sum())
        if nt < min_treat or nc < min_treat:
            continue
        beta, se, pval = ols_hc1(yg, tg)
        result[str(grp)] = {
            "n_treat": nt,
            "n_ctrl":  nc,
            "n":       nt + nc,
            "beta":    round(float(beta), 4) if not np.isnan(beta) else None,
            "se":      round(float(se),   4) if not np.isnan(se)   else None,
            "pval":    round(float(pval), 4) if not np.isnan(pval) else None,
        }
    return result


def compute_heterogeneity(sub: pd.DataFrame) -> dict:
    """
    Compute employment DiD by sector, firm size, age group, sex, formality.
    Returns dict with keys: sector, firmsize, age, sex, formality.
    """
    result = {}

    # Sector (from ciiu_pre or p208c_pre)
    for ciiu_col in ["ciiu_pre", "p208c_pre", "p301a_pre"]:
        if ciiu_col in sub.columns:
            sector = pd.to_numeric(sub[ciiu_col], errors="coerce").apply(
                lambda x: sector_from_ciiu(x) if not np.isnan(x) else None
            )
            result["sector"] = run_group_did(sub, sector)
            break

    # Firm size (from p205_pre)
    for sz_col in ["p205_pre", "p205a_pre"]:
        if sz_col in sub.columns:
            fsz = pd.to_numeric(sub[sz_col], errors="coerce").apply(
                lambda x: firm_size_group(x) if not np.isnan(x) else None
            )
            result["firmsize"] = run_group_did(sub, fsz)
            break

    # Age (from p208a_pre)
    for age_col in ["p208a_pre", "p208_pre"]:
        if age_col in sub.columns:
            agegrp = pd.to_numeric(sub[age_col], errors="coerce").apply(
                lambda x: age_group(x) if not np.isnan(x) else None
            )
            result["age"] = run_group_did(sub, agegrp)
            break

    # Sex (p207_pre: 1=Hombre, 2=Mujer)
    if "p207_pre" in sub.columns:
        sex = pd.to_numeric(sub["p207_pre"], errors="coerce").map({1.0: "Hombre", 2.0: "Mujer"})
        result["sex"] = run_group_did(sub, sex)

    # Formality
    form = sub["formal_pre"].map({1: "Formal pre", 0: "Informal pre"})
    result["formality"] = run_group_did(sub, form)

    return result


def compute_labor_market_decomposition(sub: pd.DataFrame) -> dict:
    """
    Decompose employment transitions for treatment and control groups:
    formal→formal, formal→informal, formal→not_emp,
    informal→formal, informal→informal, informal→not_emp.
    Returns dict with treat, control, did, and accounting sub-dicts.
    """
    def _transitions(df: pd.DataFrame) -> dict:
        f_pre  = df["formal_pre"]
        f_post = df["formal_post"]
        e_post = df["employed_post"]
        n      = len(df)
        n_formal_pre   = int(f_pre.sum())
        n_informal_pre = int((f_pre == 0).sum())

        def _rate(cond_pre, cond_post):
            sub_pre = df[cond_pre]
            if len(sub_pre) == 0:
                return 0.0
            return float(cond_post[cond_pre].mean())

        formal_mask   = (f_pre == 1)
        informal_mask = (f_pre == 0)

        return {
            "n":              n,
            "n_formal_pre":   n_formal_pre,
            "n_informal_pre": n_informal_pre,
            "formal_to_formal":    _rate(formal_mask,   (f_post == 1) & (e_post == 1)),
            "formal_to_informal":  _rate(formal_mask,   (f_post == 0) & (e_post == 1)),
            "formal_to_not_emp":   _rate(formal_mask,   e_post == 0),
            "informal_to_formal":  _rate(informal_mask, (f_post == 1) & (e_post == 1)),
            "informal_to_informal":_rate(informal_mask, (f_post == 0) & (e_post == 1)),
            "informal_to_not_emp": _rate(informal_mask, e_post == 0),
        }

    treat_sub = sub[sub["treat"] == 1]
    ctrl_sub  = sub[sub["treat"] == 0]
    t_trans   = _transitions(treat_sub)
    c_trans   = _transitions(ctrl_sub)

    did_trans = {
        k: round(t_trans[k] - c_trans[k], 5)
        for k in ["formal_to_formal", "formal_to_informal", "formal_to_not_emp",
                   "informal_to_formal", "informal_to_informal", "informal_to_not_emp"]
    }

    # Accounting: d_emp = d_formal_emp + d_informal_emp
    y_emp      = sub["employed_post"] - sub["employed_pre"]
    y_femp     = sub["formal_post"] * sub["employed_post"] - sub["formal_pre"] * sub["employed_pre"]
    y_iemp     = (1 - sub["formal_post"]) * sub["employed_post"] - (1 - sub["formal_pre"]) * sub["employed_pre"]
    treat      = sub["treat"]

    def _reg(y):
        b, se, p = ols_hc1(y, treat)
        return {"beta": round(float(b), 5) if not np.isnan(b) else None,
                "se":   round(float(se), 5) if not np.isnan(se) else None,
                "pval": round(float(p),  4) if not np.isnan(p)  else None,
                "mean_treat": round(float(y[treat == 1].mean()), 5) if (treat == 1).any() else None,
                "mean_ctrl":  round(float(y[treat == 0].mean()), 5) if (treat == 0).any() else None}

    accounting = {
        "d_emp":         _reg(y_emp),
        "d_formal_emp":  _reg(y_femp),
        "d_informal_emp":_reg(y_iemp),
        "identity_residual": round(
            (y_emp - y_femp - y_iemp).abs().mean(), 8
        ),
    }

    return {
        "treated":  t_trans,
        "control":  c_trans,
        "did":      did_trans,
        "accounting": accounting,
    }


def compute_kaitz_metrics(
    aug: pd.DataFrame, mw_old: float, mw_new: float
) -> dict:
    """
    Compute Kaitz index metrics:
      kaitz_pre  = MW_old / median_wage_pre
      kaitz_post = MW_new / median_wage_post
      pct_at_mw  = share earning in [0.95×MW_old, 1.05×MW_old) pre-period
    """
    w_pre  = aug["wage_pre"].dropna()
    w_post = aug["wage_post"].dropna()
    med_pre  = float(w_pre.median())  if len(w_pre)  else np.nan
    med_post = float(w_post.median()) if len(w_post) else np.nan
    kaitz_pre  = round(mw_old / med_pre,  3) if med_pre  > 0 else np.nan
    kaitz_post = round(mw_new / med_post, 3) if med_post > 0 else np.nan
    pct_at_mw  = round(float(w_pre.between(0.95 * mw_old, 1.05 * mw_old).mean()), 4) if len(w_pre) else np.nan
    return {
        "kaitz_pre":   kaitz_pre,
        "kaitz_post":  kaitz_post,
        "median_wage_pre":  round(med_pre,  1) if not np.isnan(med_pre)  else None,
        "median_wage_post": round(med_post, 1) if not np.isnan(med_post) else None,
        "pct_at_mw":   pct_at_mw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Placebo
# ─────────────────────────────────────────────────────────────────────────────

def run_placebo(event: dict, band: dict, verbose: bool = False) -> dict:
    """
    Run placebo DiD using the year-before-the-event EPE waves.
    Placebo treatment band preserves same relative width:
      treat_hi = mw_placebo × (mw_new / mw_old)
    """
    label     = event["label"]
    mw_old    = event["mw_old"]
    mw_new    = event["mw_new"]
    mw_placebo = event["mw_placebo"]
    # Placebo band: treat = [treat_lo × mw_placebo, mw_placebo × ratio)
    ratio      = mw_new / mw_old
    mw_placebo_new = mw_placebo * ratio

    placebo_event = {
        **event,
        "pre_codes":  event["placebo_pre_codes"],
        "post_codes": event["placebo_post_codes"],
        "filter_pmes": False,
    }

    merged, inc_pre, inc_post, diag = load_and_match(placebo_event, verbose=verbose)
    if merged is None or len(merged) < MIN_INBAND * 2:
        return {"available": False, "reason": "insufficient matches"}

    sub = assign_bands(merged, mw_placebo, mw_placebo_new, band)
    sub = sub[sub["in_band"] & sub["treat"].notna()]
    if len(sub) < MIN_INBAND * 2:
        return {"available": False, "reason": "insufficient in-band"}

    # Employment placebo
    a = outcome_a_employment(sub)
    emp_beta = a["beta"] if a["beta"] is not None else np.nan
    emp_se   = a["se"]   if a["se"]   is not None else np.nan
    emp_pval = a["pval"] if a["pval"] is not None else np.nan

    # Wage placebo
    b = outcome_b_wage_all(sub)
    wage_beta = b["beta"] if b["beta"] is not None else np.nan
    wage_se   = b["se"]   if b["se"]   is not None else np.nan
    wage_pval = b["pval"] if b["pval"] is not None else np.nan

    n_treat = a["n_treat"]
    n_ctrl  = a["n_ctrl"]

    return {
        "available":    True,
        "n_treat":      n_treat,
        "n_ctrl":       n_ctrl,
        "beta":         emp_beta,
        "se":           emp_se,
        "pval":         emp_pval,
        "emp_beta":     emp_beta,
        "emp_se":       emp_se,
        "emp_pval":     emp_pval,
        "wage_beta":    wage_beta,
        "wage_se":      wage_se,
        "wage_pval":    wage_pval,
        "wage_n_treat": b["n_treat"],
        "wage_n_ctrl":  b["n_ctrl"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main event runner
# ─────────────────────────────────────────────────────────────────────────────

def run_event(
    event: dict,
    band: dict          = BAND_PRIMARY,
    band_name: str      = "primary",
    run_placebo_flag: bool  = False,
    run_attrition_flag: bool = False,
    run_heterogeneity_flag: bool = False,
    verbose: bool           = True,
) -> dict | None:
    """
    Run the full canonical DiD pipeline for one MW event.
    Returns a result dict with all outcomes A–J, or None on failure.
    """
    label  = event["label"]
    mw_old = event["mw_old"]
    mw_new = event["mw_new"]
    dln_mw = np.log(mw_new / mw_old)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVENT: {label}  MW: {mw_old:.0f}→{mw_new:.0f} ({dln_mw*100:.1f}%)")
        print(f"{'='*60}")

    # 1. Load and match
    merged, inc_pre, inc_post, diag = load_and_match(event, verbose=verbose)
    if merged is None:
        return None

    # 2. Assign bands
    sub_all = assign_bands(merged, mw_old, mw_new, band)

    # 3. Filter to in-band sample
    sub = sub_all[sub_all["in_band"] & sub_all["treat"].notna()].copy()
    n_treat = int((sub["treat"] == 1).sum())
    n_ctrl  = int((sub["treat"] == 0).sum())

    if verbose:
        print(f"  In-band: treat={n_treat}, ctrl={n_ctrl}")

    if n_treat < MIN_INBAND or n_ctrl < MIN_INBAND:
        if verbose:
            print(f"  SKIP: insufficient in-band observations (min={MIN_INBAND})")
        return None

    # 4. Detect hours column
    hours_col = detect_hours_col_from_merged(merged)

    # 5. Outcomes
    result = {
        "label":      label,
        "mw_old":     mw_old,
        "mw_new":     mw_new,
        "dln_mw":     round(dln_mw, 6),
        "dmw_pct":    round((mw_new / mw_old - 1) * 100, 1),
        "band":       band_name,
        "treat_lo":   round(band["treat_lo"] * mw_old, 1),
        "treat_hi":   round(mw_new, 1),
        "ctrl_lo":    round(mw_new, 1),
        "ctrl_hi":    round(band["ctrl_hi"] * mw_new, 1),
        "code_pre":   event["pre_codes"][0],
        "code_post":  event["post_codes"][0],
        "match_rate": diag.get("match_rate"),
        "n_matched":  diag.get("n_matched"),
        "inc_col":    inc_pre,
        **{k: v for k, v in diag.items() if k.startswith("sex") or k.startswith("age")},
        "n_treat":    n_treat,
        "n_ctrl":     n_ctrl,
    }

    # Outcome A: Employment
    a = outcome_a_employment(sub)
    result["A_employment"] = a
    beta_emp = a["beta"] if a["beta"] is not None else np.nan
    se_emp   = a["se"]   if a["se"]   is not None else np.nan

    # Elasticity
    epsilon    = round(beta_emp / dln_mw, 4)       if not np.isnan(beta_emp) else None
    se_epsilon = round(se_emp   / abs(dln_mw), 4)  if not np.isnan(se_emp)   else None
    result["epsilon"]    = epsilon
    result["se_epsilon"] = se_epsilon

    # Outcome B: Wage all
    result["B_wage_all"] = outcome_b_wage_all(sub)

    # Outcome C: Lee bounds
    result["C_lee_bounds"] = outcome_c_lee_bounds(sub, beta_emp)

    # Outcome D: Lighthouse
    result["D_lighthouse"] = outcome_d_lighthouse(sub)

    # Outcome E: Formal wage
    result["E_formal_wage"] = outcome_e_formal_wage(sub)

    # Outcome F: Formality transition
    result["F_formality_trans"] = outcome_f_formality_transition(sub, beta_emp)

    # Outcome G: Formal employment
    result["G_formal_employment"] = outcome_g_formal_employment(sub)

    # Outcome H: Self-employment
    result["H_selfemployment"] = outcome_h_selfemployment(sub)

    # Outcome I: Placebo
    if run_placebo_flag:
        result["I_placebo"] = run_placebo(event, band, verbose=verbose)
    else:
        result["I_placebo"] = {"available": False, "reason": "not_requested"}

    # Outcome J: Hours
    result["J_hours"] = outcome_j_hours(sub, hours_col, beta_emp)

    # Labor market decomposition
    result["labor_market_decomp"] = compute_labor_market_decomposition(sub)

    # Kaitz metrics
    result["kaitz"] = compute_kaitz_metrics(sub, mw_old, mw_new)

    # Heterogeneity
    if run_heterogeneity_flag:
        result["heterogeneity"] = compute_heterogeneity(sub)

    # Attrition diagnostic
    if run_attrition_flag and inc_pre is not None:
        # Load raw pre-wave for comparison
        dfs_pre = [load_sav(c) for c in event["pre_codes"] if load_sav(c) is not None]
        if dfs_pre:
            df_pre_raw = pd.concat(dfs_pre, ignore_index=True)
            result["attrition"] = attrition_diagnostic(df_pre_raw, merged, inc_pre, label)

    # Context data
    ctx = CONTEXT_DATA.get(label, {})
    result["context"] = ctx

    if verbose:
        print(f"\n  Results for {label}:")
        print(f"    A. Employment β={beta_emp:.4f}  ε={epsilon}")
        b_beta = result["B_wage_all"]["beta"]
        print(f"    B. Wage all   β={b_beta:.4f}" if b_beta else "    B. Wage all   N/A")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-event diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def cross_event_overlap_check(results: list[dict]) -> dict:
    """
    Check if any individual appears in >1 event's treatment or control group.
    Logs any overlapping SRIENAHO codes across events.
    Returns dict with overlap_count and details.
    """
    code_to_events = defaultdict(list)
    for r in results:
        label     = r["label"]
        pre_code  = r.get("code_pre")
        post_code = r.get("code_post")
        if pre_code:
            code_to_events[pre_code].append(label)
        if post_code:
            code_to_events[post_code].append(label)

    overlaps = {
        code: evts
        for code, evts in code_to_events.items()
        if len(evts) > 1
    }
    return {
        "overlap_count":  len(overlaps),
        "overlapping_codes": {str(k): v for k, v in overlaps.items()},
    }


def _find_shared_code_labels(results: list[dict]) -> list[str]:
    """Return event labels that share a SRIENAHO code with another event."""
    code_to_events = defaultdict(list)
    for r in results:
        for c in [r.get("code_pre"), r.get("code_post")]:
            if c:
                code_to_events[c].append(r["label"])
    shared = set()
    for evts in code_to_events.values():
        if len(evts) > 1:
            shared.update(evts)
    return sorted(shared)


# ─────────────────────────────────────────────────────────────────────────────
# IVW pooling
# ─────────────────────────────────────────────────────────────────────────────

def _ivw(values: list[float], ses: list[float]) -> tuple[float, float]:
    """Inverse-variance-weighted pooled estimate. Returns (pool_est, pool_se)."""
    vals = np.array(values, dtype=float)
    sest = np.array(ses,    dtype=float)
    mask = np.isfinite(vals) & np.isfinite(sest) & (sest > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    w       = 1.0 / sest[mask] ** 2
    pool    = np.sum(w * vals[mask]) / np.sum(w)
    pool_se = 1.0 / np.sqrt(np.sum(w))
    return float(pool), float(pool_se)


def ivw_pool(results: list[dict], exclude_labels: list[str] | None = None) -> dict:
    """
    IVW pool employment β and elasticity ε across events.
    Excludes events in exclude_labels (e.g. COVID pre-trend violation).
    """
    exclude = set(exclude_labels) if exclude_labels else set()

    betas, beta_ses = [], []
    epsilons, eps_ses = [], []
    included_labels = []

    for r in results:
        if r["label"] in exclude:
            continue
        emp = r.get("A_employment", {})
        b   = emp.get("beta")
        se  = emp.get("se")
        eps = r.get("epsilon")
        ses = r.get("se_epsilon")

        if b is not None and se is not None:
            betas.append(b)
            beta_ses.append(se)
        if eps is not None and ses is not None:
            epsilons.append(eps)
            eps_ses.append(ses)
        included_labels.append(r["label"])

    beta_pool, beta_se_pool = _ivw(betas, beta_ses)
    eps_pool,  eps_se_pool  = _ivw(epsilons, eps_ses)

    def _pval(est, se):
        if np.isnan(est) or np.isnan(se) or se <= 0:
            return np.nan
        z = abs(est) / se
        return float(2 * (1 - stats.norm.cdf(z)))

    return {
        "n_events":     len(included_labels),
        "included":     included_labels,
        "excluded":     list(exclude),
        "beta_pool":    round(float(beta_pool),    4) if not np.isnan(beta_pool)    else None,
        "beta_se_pool": round(float(beta_se_pool), 4) if not np.isnan(beta_se_pool) else None,
        "beta_pval":    round(_pval(beta_pool, beta_se_pool), 4),
        "eps_pool":     round(float(eps_pool),     4) if not np.isnan(eps_pool)     else None,
        "eps_se_pool":  round(float(eps_se_pool),  4) if not np.isnan(eps_se_pool)  else None,
        "eps_pval":     round(_pval(eps_pool, eps_se_pool), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing / reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table of all outcomes across events."""
    hdr = f"{'Event':12s} {'MW':>12s} {'ε':>8s} {'β_A':>8s} {'p_A':>6s} {'β_B':>8s} {'p_B':>6s} {'β_D':>8s}"
    print("\n" + "="*80)
    print("CANONICAL DiD RESULTS — ALL EVENTS")
    print("="*80)
    print(hdr)
    print("-"*80)
    for r in results:
        lbl = r["label"]
        mw  = f"{r['mw_old']:.0f}→{r['mw_new']:.0f}"
        eps = f"{r['epsilon']:.3f}" if r.get("epsilon") is not None else "  N/A"
        a   = r.get("A_employment", {})
        b_a = f"{a.get('beta', 'N/A'):.4f}" if a.get("beta") is not None else "  N/A"
        p_a = f"{a.get('pval', np.nan):.3f}" if a.get("pval") is not None else "  N/A"
        bw  = r.get("B_wage_all", {})
        b_b = f"{bw.get('beta', 'N/A'):.4f}" if bw.get("beta") is not None else "  N/A"
        p_b = f"{bw.get('pval', np.nan):.3f}" if bw.get("pval") is not None else "  N/A"
        bl  = r.get("D_lighthouse", {})
        b_d = f"{bl.get('beta', 'N/A'):.4f}" if bl.get("beta") is not None else "  N/A"
        s_a = stars(a.get("pval", np.nan))
        s_b = stars(bw.get("pval", np.nan))
        print(f"{lbl:12s} {mw:>12s} {eps:>8s} {b_a:>8s}{s_a:<3s} {p_a:>6s} {b_b:>8s}{s_b:<3s} {p_b:>6s} {b_d:>8s}")
    print("="*80)
    print("* p<0.10  ** p<0.05  *** p<0.01")


def print_pooled_table(
    pooled_main: dict,
    pooled_all:  dict | None = None,
) -> None:
    """Print IVW pooled estimates."""
    print("\n" + "="*60)
    print("IVW POOLED ESTIMATES")
    print("="*60)
    for label, pool in [("Main (excl. 2022)", pooled_main), ("All events", pooled_all)]:
        if pool is None:
            continue
        print(f"\n{label}  (N={pool['n_events']} events)")
        b  = pool.get("beta_pool",    "N/A")
        bs = pool.get("beta_se_pool", "N/A")
        bp = pool.get("beta_pval",    np.nan)
        e  = pool.get("eps_pool",     "N/A")
        es = pool.get("eps_se_pool",  "N/A")
        ep = pool.get("eps_pval",     np.nan)
        if isinstance(b, float):
            print(f"  β_emp = {b:.4f}  (SE={bs:.4f}, p={bp:.3f}){stars(bp)}")
        if isinstance(e, float):
            print(f"  ε     = {e:.4f}  (SE={es:.4f}, p={ep:.3f}){stars(ep)}")
        if pool.get("excluded"):
            print(f"  Excluded: {pool['excluded']}")
    print("="*60)


def print_lee_table(results: list[dict]) -> None:
    """Print Lee (2009) bounds for all events."""
    print("\n" + "="*60)
    print("LEE (2009) SELECTION-CORRECTED BOUNDS")
    print("="*60)
    print(f"{'Event':12s} {'β_unc':>8s} {'Lee_LB':>8s} {'Lee_UB':>8s} {'n_trim':>7s}")
    print("-"*60)
    for r in results:
        c = r.get("C_lee_bounds", {})
        beta_u = c.get("beta_unc")
        lb     = c.get("lee_lb")
        ub     = c.get("lee_ub")
        n_trim = c.get("n_trim", 0)
        b_str  = f"{beta_u:.4f}" if beta_u is not None else "  N/A"
        lb_str = f"{lb:.4f}"    if lb     is not None else "  N/A"
        ub_str = f"{ub:.4f}"    if ub     is not None else "  N/A"
        print(f"{r['label']:12s} {b_str:>8s} {lb_str:>8s} {ub_str:>8s} {n_trim:>7d}")
    print("="*60)


def _sanitize(obj):
    """Recursively convert numpy scalars to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_results(
    results: list[dict],
    pooled_main: dict,
    pooled_all: dict,
    pooled_emp_ex2022: dict,
) -> None:
    """Serialize all results to JSON in EXPORTS_DIR."""
    output = {
        "version": 2,
        "design": {
            "treatment_band_primary": "0.85×MW_old ≤ wage_pre < MW_new",
            "control_band_primary":   "MW_new ≤ wage_pre < 1.40×MW_new",
            "matching":        "consecutive-quarter, 4-ID inner join",
            "income_variable": "ingprin (monthly soles, INEI mensualización)",
            "formality_definition": "p222 ∈ {1,2,3} (Def B)",
            "selfemployment_code":   "p206 == 2",
            "se_estimator":          "HC1",
            "elasticity_formula":    "β_emp / ln(MW_new/MW_old)",
            "pooling_beta":    "IVW: w=1/SE_β²",
            "pooling_epsilon": "IVW: w=1/SE_ε²,  SE_ε=SE_β/|ln(MW)|  [delta method]",
            "employment_pool_note": (
                "2022_Apr excluded from employment pooling due to COVID pre-trend "
                "violation (placebo β=+0.296, p=0.018). Wage results retain 2022 "
                "(wage placebo passes)."
            ),
        },
        "events":             _sanitize(results),
        "pooled_main":        _sanitize(pooled_main),
        "pooled_all":         _sanitize(pooled_all),
        "pooled_emp_ex2022":  _sanitize(pooled_emp_ex2022),
    }
    out_path = EXPORTS_DIR / "mw_canonical_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {out_path}")

    # Also save a lightweight simulator-friendly summary
    sim = {
        "pooled_beta":         pooled_main.get("beta_pool"),
        "pooled_se":           pooled_main.get("beta_se_pool"),
        "pooled_epsilon":      pooled_main.get("eps_pool"),
        "pooled_epsilon_se":   pooled_main.get("eps_se_pool"),
        "n_events_pool":       pooled_main.get("n_events"),
        "events": [
            {
                "label":    r["label"],
                "mw_old":   r["mw_old"],
                "mw_new":   r["mw_new"],
                "epsilon":  r.get("epsilon"),
                "beta_emp": r.get("A_employment", {}).get("beta"),
            }
            for r in results
        ],
    }
    sim_path = EXPORTS_DIR / "mw_simulator_params.json"
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(sim), f, ensure_ascii=False, indent=2)
    print(f"  Saved → {sim_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MW canonical DiD pipeline")
    parser.add_argument("--robustness",   action="store_true", help="Run all band robustness checks")
    parser.add_argument("--placebo",      action="store_true", help="Run placebo tests")
    parser.add_argument("--attrition",    action="store_true", help="Run attrition diagnostics")
    parser.add_argument("--heterogeneity", action="store_true", help="Run heterogeneity by sector/age/sex")
    parser.add_argument("--event",        type=str, default=None, help="Run single event (e.g. 2022_Apr)")
    parser.add_argument("--quiet",        action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    print("\n" + "="*70)
    print("MW CANONICAL ESTIMATION — VERSION 2")
    print("Definitive DiD pipeline, Lima Metro EPE")
    print("="*70)

    # Filter events if --event specified
    events_to_run = MW_EVENTS
    if args.event:
        events_to_run = [e for e in MW_EVENTS if e["label"] == args.event]
        if not events_to_run:
            print(f"ERROR: event '{args.event}' not found. Valid: {[e['label'] for e in MW_EVENTS]}")
            return

    results = []
    for event in events_to_run:
        r = run_event(
            event,
            band               = BAND_PRIMARY,
            band_name          = "primary",
            run_placebo_flag   = args.placebo,
            run_attrition_flag = args.attrition,
            run_heterogeneity_flag = args.heterogeneity,
            verbose            = verbose,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo results generated. Check SAV file paths.")
        return

    # Summary table
    print_summary_table(results)
    print_lee_table(results)

    # IVW pooling
    # Main: exclude 2022_Apr (COVID pre-trend violation confirmed by placebo β=+0.296)
    pooled_main = ivw_pool(results, exclude_labels=["2022_Apr"])
    pooled_all  = ivw_pool(results, exclude_labels=None)
    # Employment pool excluding 2022 (but wage pool retains it)
    pooled_emp_ex2022 = ivw_pool(results, exclude_labels=["2022_Apr"])

    print_pooled_table(pooled_main, pooled_all)

    # Cross-event overlap check
    overlap = cross_event_overlap_check(results)
    if overlap["overlap_count"] > 0:
        print(f"\nWARN: {overlap['overlap_count']} SRIENAHO codes shared across events:")
        for code, evts in overlap["overlapping_codes"].items():
            print(f"  code={code}: {evts}")

    # Save
    if not args.event:
        save_results(results, pooled_main, pooled_all, pooled_emp_ex2022)

    # Robustness: re-run with alternative bands
    if args.robustness:
        print("\n" + "="*70)
        print("ROBUSTNESS CHECKS — ALTERNATIVE BANDS")
        print("="*70)
        for band_name, band in BANDS.items():
            if band_name == "primary":
                continue
            print(f"\nBand: {band_name}  (treat_lo={band['treat_lo']}, ctrl_hi={band['ctrl_hi']})")
            rob_results = []
            for event in events_to_run:
                r = run_event(event, band=band, band_name=band_name, verbose=False)
                if r is not None:
                    rob_results.append(r)
            if rob_results:
                pool = ivw_pool(rob_results, exclude_labels=["2022_Apr"])
                print(f"  IVW β={pool.get('beta_pool')}  ε={pool.get('eps_pool')}  (N={pool['n_events']})")


if __name__ == "__main__":
    main()
