"""
wage_bin_did.py — Cengiz-style wage-bin design for Peru MW employment effects.

Standalone research module. NOT wired into production simulator.

Design (Cengiz et al. 2019, QJE)
---------------------------------
Bin the wage distribution (normalized to current MW) before and after
each reform. Compare employment counts in three regions:

  Below MW      : wage_ratio in [0.5, 1.0)  — missing mass post-reform
  Affected      : wage_ratio in [0.9, 1.1)  — disemployment + bunching
  Control       : wage_ratio in [1.3, 1.8)  — unaffected above-MW workers

Employment elasticity (net):
  ε_net = (ΔlnE_affected − ΔlnE_control) / Δln(MW)

Events: 2016→2017 (S/850→S/930) and 2021→2022 (S/930→S/1025)
Data: ENAHO Module 500 (national household survey), annual cross-sections.

Usage:
    python scripts/wage_bin_did.py [--skip-download] [--n-boot N] [--no-compare]

NOTE: Results show "not_identified" (high elasticities) due to aggregate
demand/supply shifts in the wage distribution between survey years that are
unrelated to the MW change. These cross-sectional bin counts conflate
MW effects with structural employment changes. The EPE panel-DiD results
(mw_canonical_estimation.py) are more credible.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "enaho"
OUT_DIR = ROOT / "exports" / "data"
OUT_FILE = OUT_DIR / "wage_bin_did_results.json"

# Survey years needed: pre and post for each event
YEARS_NEEDED = ["2016", "2017", "2021", "2022"]

# Nominal minimum wage by year (soles/month)
MIN_WAGE = {
    2016: 850, 2017: 930, 2018: 930, 2019: 930,
    2020: 930, 2021: 930, 2022: 1025, 2023: 1025,
}

# Payment periodicity multiplier (P523) to convert to monthly
PERIOD_MULT = {1: 1.0, 2: 4.333, 3: 26.0, 4: 2.0}

# ENAHO Module 500 SRIENAHO codes by year
YEAR_MAP = {
    "2023": 906, "2022": 784, "2021": 759, "2020": 737,
    "2019": 687, "2018": 634, "2017": 603, "2016": 546,
}

# Columns to load from ENAHO Module 500
LOAD_COLS = [
    "conglome", "vivienda", "hogar", "codperso",
    "P507", "P510", "P517", "P523", "P524A1", "P524B1", "P524C1",
    "P558A1", "FAC500A", "DOMINIO",
]

# Wage-bin parameters
BIN_WINDOW     = (0.5, 1.8)   # (lo, hi) in MW-ratio units
BIN_WIDTH      = 0.05
AFFECTED_REGION = (0.9, 1.1)  # wage_ratio region most affected by MW change
CONTROL_REGION  = (1.3, 1.8)  # above-MW control region
BINDING_REGION  = (1.0, 1.1)  # just above new MW (bunching region)
BELOW_MIN_REGION = (0.5, 1.0) # sub-minimum workers
SPILLOVER_REGION = (1.1, 1.3) # potential spillover zone

# Bootstrap parameters
N_BOOT   = 200
BOOT_SEED = 42

# Events: (label, pre_year, post_year, old_min, new_min)
EVENTS = [
    {"label": "2017: S/850\u2192S/930",  "pre_yr": "2016", "post_yr": "2017", "old_min": 850.0, "new_min": 930.0},
    {"label": "2022: S/930\u2192S/1025", "pre_yr": "2021", "post_yr": "2022", "old_min": 930.0, "new_min": 1025.0},
]

# Samples to run
SAMPLES = ["formal_wage", "all_wage", "salaried_only"]


def build_deflators() -> dict:
    """Annual CPI deflators relative to 2023=1.00 from BCRP PN01271PM."""
    try:
        from src.ingestion.bcrp import BCRPClient
        client = BCRPClient()
        s = client.fetch("PN01271PM", "2015-01", "2024-12")
        s = s["cpi_level"].dropna().sort_index()
        # Compute annual averages and deflators (2023=1.00)
        annual = s.resample("YE").mean()
        base = annual.loc[annual.index.year == 2023].iloc[0]
        deflators = {}
        for yr in range(2015, 2027):
            row = annual[annual.index.year == yr]
            if not row.empty:
                deflators[yr] = round(float(base / row.iloc[0]), 4)
        return deflators
    except Exception:
        # Hardcoded fallback from BCRP PN01271PM, 2023=1.00
        return {
            2015: 1.3376, 2016: 1.2912, 2017: 1.2560, 2018: 1.2397,
            2019: 1.2137, 2020: 1.1920, 2021: 1.1464, 2022: 1.0600,
            2023: 1.0000, 2024: 0.9700,
        }


def find_dta(year: str, base: Path) -> Path | None:
    """*05*"""
    yr_code = YEAR_MAP.get(year)
    if yr_code is None:
        return None
    yr_dir = base / f"modulo_05_{year}"
    # Try to find .dta file matching module 500
    for pat in ["**/*.dta", f"**/*500*.dta", "**/*", "*500*.dta"]:
        matches = sorted(yr_dir.glob(pat))
        for m in matches:
            if m.name.lower().endswith(".dta") and "500" in m.name:
                return m
    # Fallback: any .dta in the directory
    all_dta = sorted(yr_dir.glob("**/*.dta"))
    return all_dta[0] if all_dta else None


def load_enaho_raw(year: str, deflators: dict) -> pd.DataFrame | None:
    """Load ENAHO Module 500, returning wage workers with positive monthly income."""
    dta_path = find_dta(year, RAW_DIR)
    if dta_path is None:
        print(f"  [ERROR] ENAHO Module 500 not found for year {year}")
        return None
    try:
        df = pd.read_stata(str(dta_path))
    except Exception as e:
        print(f"  [ERROR] {year}: {e}")
        return None

    # Standardise column names to upper
    df.columns = [c.upper() for c in df.columns]

    # Construct monthly wage from P524A1 (primary income) + P523 (period)
    wage = pd.Series(np.zeros(len(df)), index=df.index)
    if "P524A1" in df.columns:
        p524 = pd.to_numeric(df.get("P524A1", pd.Series(dtype=float)), errors="coerce").fillna(0)
        period = pd.to_numeric(df.get("P523", pd.Series(dtype=float)), errors="coerce")
        mult = period.map(PERIOD_MULT).fillna(1.0)
        wage = p524 * mult
    # Add in-kind component
    if "P524B1" in df.columns:
        inkind = pd.to_numeric(df.get("P524B1", pd.Series(dtype=float)), errors="coerce").fillna(0)
        wage = wage + inkind

    df["monthly_wage"] = wage.where(wage > 0, np.nan)

    # Real wage (2023 soles)
    defl = deflators.get(int(year), 1.0)
    df["real_wage"] = df["monthly_wage"] * defl

    # Formality: P558A1==1 (pension contribution) as proxy, P510 for category
    df["formal"] = (pd.to_numeric(df.get("P558A1", pd.Series(dtype=float)), errors="coerce") == 1).astype(int)

    # Wage worker: P510 in {3,4,5,6} (private employee, domestic, etc.) or P507==3
    p510 = pd.to_numeric(df.get("P510", pd.Series(dtype=float)), errors="coerce")
    df["wage_worker"] = p510.isin([3, 4, 5, 6]).astype(int)

    # Salaried: P517==1 or P517==2 (indefinite or fixed contract)
    p517 = pd.to_numeric(df.get("P517", pd.Series(dtype=float)), errors="coerce")
    df["salaried"] = p517.isin([1, 2]).astype(int)

    # Survey weight
    df["FAC500A"] = pd.to_numeric(df.get("FAC500A", pd.Series(dtype=float)), errors="coerce").fillna(1.0)
    df["year"] = int(year)

    # Filter: wage workers with positive income
    mask = (df["wage_worker"] == 1) & df["monthly_wage"].notna() & (df["monthly_wage"] > 0)
    return df[mask].copy()


def prepare_wage_bin_panel(
    df: pd.DataFrame,
    year: str,
    deflators: dict,
    current_min: float,
    sample: str,
) -> pd.DataFrame:
    """
    Prepare individual-level panel for wage-bin analysis.

    Normalizes nominal wages to wage_ratio = nom_wage / current_min
    (Cengiz convention: each year normalized to that year's MW).

    Parameters
    ----------
    df          : Raw ENAHO output from load_enaho_raw().
    year        : Survey year string.
    deflators   : {year_int: deflator} mapping (2023=1.00).
    current_min : Nominal MW for this survey year (old_min for pre, new_min for post).
    sample      : "formal_wage" = formal wage workers only
                  "all_wage"    = all wage workers
                  "salaried_only" = workers with indefinite/fixed contract
    """
    df = df.copy()
    df["nom_wage"] = df["monthly_wage"]
    df["wage_ratio"] = df["nom_wage"] / current_min

    if sample == "formal_wage":
        df = df[df["formal"] == 1].copy()
    elif sample == "all_wage":
        pass  # all wage workers
    elif sample == "salaried_only":
        df = df[df["salaried"] == 1].copy()
    else:
        raise ValueError(f"Unknown sample: {sample}. Use 'formal_wage' or 'all_wage'.")

    return df[np.isfinite(df["wage_ratio"])].reset_index(drop=True)


def make_bins(window: tuple, bin_width: float) -> pd.DataFrame:
    """
    Build bin specification.

    Returns DataFrame with [bin_id, bin_left, bin_right, bin_mid, bin_label].
    """
    lo, hi = window
    edges = np.round(np.arange(lo, hi + bin_width * 0.5, bin_width), 4)
    rows = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        rows.append({
            "bin_id":    i,
            "bin_left":  float(left),
            "bin_right": float(right),
            "bin_mid":   float((left + right) / 2),
            "bin_label": f"[{left:.2f},{right:.2f})",
        })
    return pd.DataFrame(rows)


def assign_bins(panel: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """
    Add bin_id column to panel via vectorized searchsorted.
    Out-of-window observations get bin_id=-1.
    """
    lefts  = bins["bin_left"].values
    rights = bins["bin_right"].values
    ratios = panel["wage_ratio"].values

    # Searchsorted: find bin index for each worker
    idx = np.searchsorted(lefts, ratios, side="right") - 1
    idx = idx.clip(-1, len(bins) - 1)

    # Verify bin assignment (worker must be in [bin_left, bin_right))
    valid = np.zeros(len(ratios), dtype=bool)
    in_range = (idx >= 0) & (idx < len(bins))
    valid[in_range] = (
        ratios[in_range] >= lefts[idx[in_range]]
    ) & (
        ratios[in_range] < rights[idx[in_range]]
    )
    idx[~valid] = -1

    out = panel.copy()
    out["bin_id"] = idx
    return out


def _bin_counts(panel: pd.DataFrame, bins: pd.DataFrame) -> pd.Series:
    """Weighted employment by bin_id (Series indexed by bin_id, 0-filled)."""
    valid = panel[panel["bin_id"] >= 0]
    counts = valid.groupby("bin_id")["FAC500A"].sum()
    return counts.reindex(bins["bin_id"], fill_value=0)


def build_event_bin_table(
    pre_binned: pd.DataFrame,
    post_binned: pd.DataFrame,
    bins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build bin-level employment table for one pre/post event pair.

    Returns DataFrame with:
        bin_id, bin_left, bin_right, bin_mid, bin_label,
        employment_pre, employment_post, log_emp_pre, log_emp_post, dlnE
    """
    emp_pre  = _bin_counts(pre_binned, bins)
    emp_post = _bin_counts(post_binned, bins)

    tbl = bins.copy()
    tbl["employment_pre"]  = emp_pre.values
    tbl["employment_post"] = emp_post.values
    tbl["log_emp_pre"]  = np.log(np.maximum(tbl["employment_pre"],  1e-9))
    tbl["log_emp_post"] = np.log(np.maximum(tbl["employment_post"], 1e-9))
    tbl["dlnE"] = tbl["log_emp_post"] - tbl["log_emp_pre"]
    return tbl.reset_index(drop=True)


def _region_agg(tbl: pd.DataFrame, lo: float, hi: float) -> tuple:
    """
    Sum employment_pre and employment_post over bins fully contained in [lo, hi).
    Uses bin_left >= lo AND bin_right <= hi to avoid partial bins.
    """
    mask = (tbl["bin_left"] >= float(lo) - 1e-9) & (tbl["bin_right"] <= float(hi) + 1e-9)
    sub = tbl.loc[mask]
    return float(sub["employment_pre"].sum()), float(sub["employment_post"].sum())


def compute_elasticities(tbl: pd.DataFrame, old_min: float, new_min: float) -> dict:
    """
    Compute local and net elasticities from a bin employment table.

    Local : ΔlnE_affected / Δln(MW)
            where affected = [0.9, 1.1)
    Net   : (ΔlnE_affected − ΔlnE_control) / Δln(MW)
            where control  = [1.3, 1.8)
    """
    ln_mw = np.log(new_min / old_min)

    E_aff_pre,  E_aff_post  = _region_agg(tbl, *AFFECTED_REGION)
    E_ctrl_pre, E_ctrl_post = _region_agg(tbl, *CONTROL_REGION)

    dlnE_aff  = np.log(max(E_aff_post,  1e-9)) - np.log(max(E_aff_pre,  1e-9))
    dlnE_ctrl = np.log(max(E_ctrl_post, 1e-9)) - np.log(max(E_ctrl_pre, 1e-9))

    elast_local = round(dlnE_aff / ln_mw, 4)
    elast_net   = round((dlnE_aff - dlnE_ctrl) / ln_mw, 4)

    return {
        "ln_mw":        round(ln_mw,        6),
        "E_aff_pre":    E_aff_pre,
        "E_aff_post":   E_aff_post,
        "E_ctrl_pre":   E_ctrl_pre,
        "E_ctrl_post":  E_ctrl_post,
        "dlnE_aff":     round(dlnE_aff,  6),
        "dlnE_ctrl":    round(dlnE_ctrl, 6),
        "elasticity_local": elast_local,
        "elasticity_net":   elast_net,
    }


def _one_boot_replicate(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    bins: pd.DataFrame,
    old_min: float,
    new_min: float,
    rng: np.random.Generator,
) -> tuple:
    """One bootstrap replicate: resample individuals independently per year."""
    pre_idx  = rng.integers(0, len(pre_df),  len(pre_df))
    post_idx = rng.integers(0, len(post_df), len(post_df))

    pre_boot  = pre_df.iloc[pre_idx].reset_index(drop=True)
    post_boot = post_df.iloc[post_idx].reset_index(drop=True)

    pre_bin  = assign_bins(pre_boot,  bins)
    post_bin = assign_bins(post_boot, bins)
    tbl      = build_event_bin_table(pre_bin, post_bin, bins)
    e        = compute_elasticities(tbl, old_min, new_min)
    return e["elasticity_local"], e["elasticity_net"]


def bootstrap_elasticities(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    bins: pd.DataFrame,
    old_min: float,
    new_min: float,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Bootstrap individual-level resampling (within year) for SE and 95% CI.

    Resamples pre-period and post-period individuals independently,
    recomputes bin counts and elasticities for each draw.

    Returns SE and 95% percentile CI for both local and net elasticities.
    """
    rng = np.random.default_rng(seed)
    local_boots, net_boots = [], []

    for _ in range(n_boot):
        loc, net = _one_boot_replicate(pre_df, post_df, bins, old_min, new_min, rng)
        local_boots.append(loc)
        net_boots.append(net)

    local_arr = np.array(local_boots, dtype=float)
    net_arr   = np.array(net_boots,   dtype=float)

    return {
        "se_local": round(float(np.nanstd(local_arr)), 4),
        "se_net":   round(float(np.nanstd(net_arr)),   4),
        "ci_local": [round(float(np.nanpercentile(local_arr, 2.5)), 4),
                     round(float(np.nanpercentile(local_arr, 97.5)), 4)],
        "ci_net":   [round(float(np.nanpercentile(net_arr, 2.5)), 4),
                     round(float(np.nanpercentile(net_arr, 97.5)), 4)],
        "n_boot":   n_boot,
    }


def compute_diagnostics(tbl: pd.DataFrame, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> dict:
    """
    Compute mass diagnostics and top-moving bins.

    Mass shares (employment-weighted) by wage-ratio region:
    - below_min     : ratio in [0.5, 1.0)   — sub-minimum workers
    - binding       : ratio in [1.0, 1.1)   — at or just above MW
    - spillover     : ratio in [1.1, 1.3)   — potential wage spillovers

    Bunching = mass at binding region increases post-reform (pre_df uses old_min,
    post_df uses new_min, both normalized to their own contemporaneous MW).
    """
    def _mass_share(df: pd.DataFrame, lo: float, hi: float) -> float:
        """wage_ratio"""
        sub = df[(df["wage_ratio"] >= lo) & (df["wage_ratio"] < hi)]
        total = df["FAC500A"].sum()
        return float(sub["FAC500A"].sum() / total) if total > 0 else np.nan

    def _mass(lo: float, hi: float) -> dict:
        """pre"""
        return {
            "pre":  round(_mass_share(pre_df,  lo, hi), 6),
            "post": round(_mass_share(post_df, lo, hi), 6),
        }

    mass_below  = _mass(*BELOW_MIN_REGION)
    mass_binding = _mass(*BINDING_REGION)
    mass_spill  = _mass(*SPILLOVER_REGION)

    bunching = "yes" if mass_binding["post"] > mass_binding["pre"] else "no"

    top_pos = tbl.nlargest(3, "dlnE")[["bin_label", "dlnE", "employment_pre", "employment_post"]].to_dict("records")
    top_neg = tbl.nsmallest(3, "dlnE")[["bin_label", "dlnE", "employment_pre", "employment_post"]].to_dict("records")

    return {
        "mass_below_min":   mass_below,
        "mass_binding":     mass_binding,
        "mass_spillover":   mass_spill,
        "bunching_detected": bunching,
        "top_positive_bins": top_pos,
        "top_negative_bins": top_neg,
    }


def _quality_gate(elast_local: float, elast_net: float, se_local: float, se_net: float) -> str:
    """
    Returns "ok" | "ok_local_only" | "not_identified".

    Criteria:
    - "ok"            : net elast in [-0.40, 0.10] AND SE_net  ≤ 0.20
    - "ok_local_only" : local elast in [-0.40, 0.10] AND SE_local ≤ 0.20
    - "not_identified": neither passes
    """
    def _ok(e, se):
        return np.isfinite(e) and np.isfinite(se) and (-0.4 <= e <= 0.1) and (se <= 0.2)

    if _ok(elast_net, se_net):
        return "ok"
    if _ok(elast_local, se_local):
        return "ok_local_only"
    return "not_identified"


def estimate_wage_bin(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    old_min: float,
    new_min: float,
    label: str,
    sample: str,
    bins: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Full Cengiz-style estimation for one event × sample combination.

    Parameters
    ----------
    pre_df, post_df : Microdata from prepare_wage_bin_panel().
    old_min, new_min: Nominal MW levels (soles).
    label           : Human-readable event label.
    sample          : "formal_wage" or "all_wage".
    bins            : Bin spec from make_bins(). Built with defaults if None.
    n_boot          : Bootstrap resamples.
    seed            : RNG seed.

    Returns
    -------
    Structured result dict. Does NOT include bin_table in the JSON-serializable output.
    """
    if bins is None:
        bins = make_bins(BIN_WINDOW, BIN_WIDTH)

    pre_bin  = assign_bins(pre_df,  bins)
    post_bin = assign_bins(post_df, bins)
    tbl      = build_event_bin_table(pre_bin, post_bin, bins)
    elast    = compute_elasticities(tbl, old_min, new_min)
    boot     = bootstrap_elasticities(pre_df, post_df, bins, old_min, new_min, n_boot, seed)
    diag     = compute_diagnostics(tbl, pre_df, post_df)
    status   = _quality_gate(elast["elasticity_local"], elast["elasticity_net"],
                              boot["se_local"], boot["se_net"])

    return {
        "event":        label,
        "sample":       sample,
        "old_min":      old_min,
        "new_min":      new_min,
        "status":       status,
        "n_workers_pre":  int(len(pre_df)),
        "n_workers_post": int(len(post_df)),
        **elast,
        **boot,
        "diagnostics":  diag,
        "bin_table":    tbl.to_dict("records"),
    }


def run_wage_bin_research_module(
    panels: dict,
    deflators: dict,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """
    Run wage-bin estimation for all events × samples.

    Parameters
    ----------
    panels    : {year_str: raw_df} from load_enaho_raw().
    deflators : Year-to-deflator mapping.

    Returns
    -------
    Structured dict with all event × sample results (bin_table excluded from JSON export).
    """
    bins = make_bins(BIN_WINDOW, BIN_WIDTH)
    results = []

    for ev in EVENTS:
        pre_yr  = ev["pre_yr"]
        post_yr = ev["post_yr"]
        old_min = ev["old_min"]
        new_min = ev["new_min"]
        label   = ev["label"]

        if pre_yr not in panels or post_yr not in panels:
            print(f"  [SKIP] {label}: data missing for {pre_yr} or {post_yr}")
            continue

        for sample in SAMPLES:
            pre_raw  = panels[pre_yr]
            post_raw = panels[post_yr]

            pre_panel  = prepare_wage_bin_panel(pre_raw,  pre_yr,  deflators, old_min, sample)
            post_panel = prepare_wage_bin_panel(post_raw, post_yr, deflators, new_min, sample)

            if len(pre_panel) < 50 or len(post_panel) < 50:
                print(f"    [SKIP] Too few obs  (pre={len(pre_panel)}, post={len(post_panel)})")
                results.append({
                    "event": label, "sample": sample,
                    "status": "insufficient_data",
                    "n_workers_pre": len(pre_panel),
                    "n_workers_post": len(post_panel),
                })
                continue

            print(f"  Estimating: {label} | {sample}")
            r = estimate_wage_bin(pre_panel, post_panel, old_min, new_min,
                                  label, sample, bins, n_boot, seed)
            r_out = {k: v for k, v in r.items() if k != "bin_table"}
            results.append(r_out)

            eloc = r.get("elasticity_local", float("nan"))
            enet = r.get("elasticity_net",   float("nan"))
            print(f"    ε_local={eloc:+.3f}  ε_net={enet:+.3f}  status={r['status']}")

    return {"method": "Cengiz et al. (2019) wage-bin design", "results": results}


def compare_all_elasticities(
    wage_bin_results: dict,
    region_sector_file: Path | None = None,
    simulator_file: Path | None = None,
) -> dict:
    """
    Compile point estimates from all research designs into one comparison table.

    Designs:
    1. Wage-bin local   (this module)
    2. Wage-bin net     (this module)
    3. Region×sector FE (region_sector_did.py)
    4. Industry-bite DiD(build_minimum_wage_simulator.py)
    5. Literature       (Céspedes & Sánchez 2014, ε=-0.10)

    Parameters
    ----------
    wage_bin_results   : Output of run_wage_bin_research_module().
    region_sector_file : Path to region_sector_did.json (default: exports/data/).
    simulator_file     : Path to region_sector or minwage_did_results.json.
    """
    comparison = []

    # Wage-bin results
    for r in wage_bin_results.get("results", []):
        if r.get("status") in ("ok", "ok_local_only"):
            for which in ("local", "net"):
                e_key  = f"elasticity_{which}"
                se_key = f"se_{which}"
                ci_key = f"ci_{which}"
                e  = r.get(e_key)
                se = r.get(se_key)
                if e is not None and np.isfinite(float(e)):
                    comparison.append({
                        "event":       r.get("event"),
                        "sample":      r.get("sample"),
                        "design":      f"Wage-bin ({which})",
                        "elasticity":  round(float(e), 4),
                        "se":          round(float(se), 4) if se is not None else None,
                        "ci_95":       r.get(ci_key),
                        "source":      "wage_bin_did.py",
                        "status":      r.get("status"),
                    })

    # Region×sector FE results
    if region_sector_file is None:
        region_sector_file = OUT_DIR / "region_sector_did.json"
    try:
        if Path(region_sector_file).exists():
            rs = json.loads(Path(region_sector_file).read_text(encoding="utf-8"))
            for ev_key, ev_data in rs.get("events", {}).items():
                ps = ev_data.get("primary_spec", {})
                e  = ps.get("formal")
                if e is not None and np.isfinite(float(e)):
                    comparison.append({
                        "event":      ev_key,
                        "sample":     "formal",
                        "design":     "Region\u00d7sector bite DiD (region+sector FE)",
                        "elasticity": round(float(e), 4),
                        "se":         None,
                        "source":     "region_sector_did.json",
                        "status":     "ok",
                    })
    except Exception as ex:
        print(f"  [WARN] Could not load region_sector_did.json: {ex}")

    # Literature benchmark
    comparison.append({
        "event":      "pooled",
        "sample":     "all",
        "design":     "Céspedes & Sánchez 2014 (literature)",
        "elasticity": -0.1,
        "se":         None,
        "source":     "BCRP DT 2014-014",
        "status":     "ok",
    })

    # Summarize by design
    identified = [r for r in comparison if r.get("status") not in ("not_identified", None)]
    n_ident = len(identified)
    if n_ident > 0:
        elasts = [r["elasticity"] for r in identified if r.get("elasticity") is not None]
        recommended = round(float(np.average(elasts)), 4) if elasts else None
    else:
        recommended = None

    return {
        "comparison": comparison,
        "n_identified": n_ident,
        "recommended_elast": recommended,
    }


def print_wage_bin_results(results: dict) -> None:
    """Pretty-print the wage-bin module output."""
    print("\n  WAGE-BIN / CENGIZ-STYLE DESIGN — PERU MINIMUM WAGE EFFECTS")
    print(f"  {results.get('method', '')}")
    print()

    def fv(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "  n/a "
        return f"{float(v):+.3f}"

    for r in results.get("results", []):
        print(f"  Event  : {r.get('event')}")
        print(f"  Sample : {r.get('sample')}")
        print(f"   N_pre={r.get('n_workers_pre', 0):>8,}  N_post={r.get('n_workers_post', 0):>8,}")
        print(f"  Status : {r.get('status')}")
        el = r.get("elasticity_local")
        en = r.get("elasticity_net")
        sl = r.get("se_local")
        sn = r.get("se_net")
        cl = r.get("ci_local")
        cn = r.get("ci_net")
        if el is not None:
            print(f"  ε_local = {fv(el)}  SE={fv(sl)}  CI=[{fv(cl[0])}, {fv(cl[1])}]" if cl else
                  f"  ε_local = {fv(el)}  SE={fv(sl)}")
        if en is not None:
            print(f"  ε_net   = {fv(en)}  SE={fv(sn)}  CI=[{fv(cn[0])}, {fv(cn[1])}]" if cn else
                  f"  ε_net   = {fv(en)}  SE={fv(sn)}")
        print()


def print_comparison_table(comp: dict) -> None:
    """Print the cross-design elasticity comparison table."""
    print("  ELASTICITY COMPARISON — ALL DESIGNS")
    comparison = comp.get("comparison", [])
    if not comparison:
        print("  (no data)")
        return

    events = list(dict.fromkeys(r.get("event") for r in comparison))
    for ev in events:
        print(f"\n  Event: {ev}")
        print(f"  {'Design':<46} {'Sample':<14} {'ε':>7} {'SE':>7}  Status")
        ev_rows = [r for r in comparison if r.get("event") == ev]
        for r in ev_rows:
            def fv(v):
                try:
                    return f"{float(v):+.3f}"
                except (TypeError, ValueError):
                    return "   n/a"
            print(f"  {r.get('design', ''):<46} {r.get('sample', ''):<14} "
                  f"{fv(r.get('elasticity')):>7} {fv(r.get('se')):>7}  {r.get('status', '')}")

    print(f"\n  n_identified: {comp.get('n_identified', 0)}")
    rec = comp.get("recommended_elast")
    if rec is not None:
        print(f"  recommended_elast: {rec:+.4f}")


def download_enaho(years: list[str]) -> None:
    """Download ENAHO Module 500 .dta files via INEI public portal."""
    try:
        import io
        import requests
        import zipfile
    except ImportError:
        print("  [WARN] requests not available — skipping download")
        return

    for yr in years:
        code = YEAR_MAP.get(yr)
        if code is None:
            continue

        out_dir = RAW_DIR / f"modulo_05_{yr}"
        # Check if already downloaded
        if find_dta(yr, RAW_DIR) is not None:
            print(f"  {yr}: already present \u2192 {find_dta(yr, RAW_DIR).name}")
            continue

        url = (f"https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/"
               f"ENAHO-ENAHO-{code}-Modulo500.zip")
        print(f"  Downloading {yr} ({url})...")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(out_dir)
            print(f"    \u2192 extracted to {out_dir}")
        except Exception as e:
            print(f"  [ERROR] {yr}: {e}")


def main():
    """Cengiz-style wage-bin MW elasticity — Peru (wage_bin_did.py)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip ENAHO download, use existing local data")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap resamples (default {N_BOOT})")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip cross-design elasticity comparison")
    args = parser.parse_args()

    print("  WAGE-BIN RESEARCH MODULE — Peru Minimum Wage")
    print("  Method: Cengiz, Dube, Lindner, Zipperer (QJE 2019)")

    years_needed = [y for y in YEARS_NEEDED if 2015 <= int(y) <= 2023]

    if not args.skip_download:
        print("\n[1] Downloading ENAHO Module 500...")
        download_enaho(years_needed)
    else:
        print("\n[1] Skipping download.")

    print("\n[2] Building CPI deflators...")
    deflators = build_deflators()
    for yr, d in sorted(deflators.items()):
        print(f"    {yr}: {d:.4f}")

    print("\n[3] Loading ENAHO microdata...")
    panels = {}
    for yr in sorted(years_needed):
        df = load_enaho_raw(yr, deflators)
        if df is None:
            print(f"  {yr}: NOT FOUND — skipping")
            continue
        n_formal = (df.get("formal", pd.Series(0)) == 1).sum()
        print(f"  {yr}: {len(df):>6,} wage workers  ({n_formal:>5,} formal)")
        panels[yr] = df

    if not panels:
        print("\n  No ENAHO data found. Run without --skip-download or place .dta files manually.")
        sys.exit(1)

    wage_bin_results = run_wage_bin_research_module(panels, deflators, n_boot=args.n_boot)
    print_wage_bin_results(wage_bin_results)

    if not args.no_compare:
        comp = compare_all_elasticities(wage_bin_results)
        print_comparison_table(comp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove bin_table from output (too large for JSON)
    out = {
        "estimated": "2026-03-08",
        "method":    wage_bin_results["method"],
        "n_boot":    args.n_boot,
        "bin_window": list(BIN_WINDOW),
        "bin_width":  BIN_WIDTH,
        "results":   wage_bin_results["results"],
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved \u2192 {OUT_FILE}")


if __name__ == "__main__":
    main()
