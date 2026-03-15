"""build_poverty_nowcast.py — Prompt 1: Build vintage panel for poverty nowcast.

Creates:
  data/processed/poverty/vintage_panel.parquet      — pseudo-real-time training panel
  data/processed/poverty/monthly_indicators.parquet — raw monthly wide panel by dept
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_poverty_nowcast")

ROOT      = Path("D:/nexus/nexus")
RAW_BCRP  = ROOT / "data/raw/bcrp"
PROC_DEPT = ROOT / "data/processed/departmental"
PROC_NTL  = ROOT / "data/processed"
TARGETS   = ROOT / "data/targets"
OUT_DIR   = ROOT / "data/processed/poverty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication lags (months after period ends before data is public) ─────────
PUB_LAGS = {
    "electricity":      2,
    "credit":           1,
    "deposits":         1,
    "capex":            1,
    "current_spending": 1,
    "mining":           2,
    "pension":          1,
    "tax_revenue":      1,
    "inflation":        1,
    "gdp_monthly":      2,
    "ntl":              3,
}
SERIES_KEYS = list(PUB_LAGS.keys())

# ── Category → predictor mapping ─────────────────────────────────────────────
CAT_MAP = {
    "credit_by_department":           "credit",
    "deposits_by_department":         "deposits",
    "electricity_by_department":      "electricity",
    "government_capex_local":         "_capex_local",
    "government_capex_regional":      "_capex_regional",
    "government_spending_local":      "_spending_local",
    "government_spending_regional":   "_spending_regional",
    "pension_affiliates_by_department": "pension",
    "tax_revenue_by_department":      "tax_revenue",
    "mining_copper_by_department":    "_mining_copper",
    "mining_gold_by_department":      "_mining_gold",
    "mining_lead_by_department":      "_mining_lead",
    "mining_silver_by_department":    "_mining_silver",
    "mining_zinc_by_department":      "_mining_zinc",
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Build wide monthly panel by department
# ══════════════════════════════════════════════════════════════════════════════

def load_dept_panel() -> pd.DataFrame:
    """Load processed departmental panel; return long frame (ubigeo, date, predictor, value)."""
    log.info("Loading departmental panel...")
    panel = pd.read_parquet(PROC_DEPT / "panel_departmental_monthly.parquet")

    panel = panel[panel["category"].isin(CAT_MAP)].copy()
    panel["predictor"] = panel["category"].map(CAT_MAP)
    panel["date"]      = pd.to_datetime(panel["date"])
    panel["ubigeo"]    = panel["ubigeo"].astype(str).str.zfill(2)

    # Drop aggregate rows (empty ubigeo)
    panel = panel[panel["ubigeo"].str.match(r"^\d{2}$")].copy()

    # Aggregate: sum within (ubigeo, date, predictor) — handles multi-series per dept
    agg = (
        panel.groupby(["ubigeo", "date", "predictor"])["value_raw"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"value_raw": "value"})
    )
    log.info(f"  Dept panel: {len(agg)} rows, {agg.ubigeo.nunique()} ubigeos, "
             f"{agg.date.min().date()} → {agg.date.max().date()}")
    return agg


def combine_components(agg: pd.DataFrame) -> pd.DataFrame:
    """Sum mining metals into 'mining'; sum capex parts into 'capex'; spending into 'current_spending'."""
    groups = {
        "mining":           [p for p in agg["predictor"].unique() if p.startswith("_mining")],
        "capex":            ["_capex_local", "_capex_regional"],
        "current_spending": ["_spending_local", "_spending_regional"],
    }
    for target, parts in groups.items():
        sub = agg[agg["predictor"].isin(parts)]
        if sub.empty:
            log.warning(f"  No data for {parts} → {target} will be NaN")
            continue
        combined = (
            sub.groupby(["ubigeo", "date"])["value"]
            .sum(min_count=1)
            .reset_index()
            .assign(predictor=target)
        )
        agg = agg[~agg["predictor"].isin(parts)].copy()
        agg = pd.concat([agg, combined], ignore_index=True)
        log.info(f"  Combined {parts} → '{target}': {len(combined)} rows")
    return agg


def load_national_series(dept_ubigeos: list) -> pd.DataFrame:
    """Load national GDP index and CPI; replicate to all departments."""
    log.info("Loading national series...")
    frames = []

    # GDP monthly index (2007=100)
    gdp = pd.read_parquet(RAW_BCRP / "bcrp_national_gdp_indicators.parquet")
    gdp_idx = gdp[gdp["series_code"] == "PN01770AM"][["date", "value"]].copy()
    gdp_idx["date"] = pd.to_datetime(gdp_idx["date"])
    gdp_idx["predictor"] = "gdp_monthly"
    log.info(f"  GDP index: {len(gdp_idx)} rows, {gdp_idx.date.min().date()} → {gdp_idx.date.max().date()}")
    frames.append(gdp_idx)

    # CPI monthly % change (Lima Metropolitan)
    inf = pd.read_parquet(RAW_BCRP / "bcrp_national_inflation.parquet")
    cpi = inf[inf["series_code"] == "PN01271PM"][["date", "value"]].copy()
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi["predictor"] = "inflation"
    log.info(f"  Inflation: {len(cpi)} rows, {cpi.date.min().date()} → {cpi.date.max().date()}")
    frames.append(cpi)

    national = pd.concat(frames, ignore_index=True)

    # Replicate to all departments
    replicated = pd.concat(
        [national.assign(ubigeo=u) for u in dept_ubigeos],
        ignore_index=True
    )
    log.info(f"  National series replicated to {len(dept_ubigeos)} depts: {len(replicated)} rows")
    return replicated


def load_ntl() -> pd.DataFrame:
    """Load NTL department monthly CSV; convert DEPT_CODE int → ubigeo string."""
    log.info("Loading NTL data...")
    ntl = pd.read_csv(PROC_NTL / "ntl_monthly_department.csv")
    ntl["date"]      = pd.to_datetime(ntl["date"])
    ntl["ubigeo"]    = ntl["DEPT_CODE"].astype(int).astype(str).str.zfill(2)
    ntl["predictor"] = "ntl"
    ntl = ntl[["ubigeo", "date", "predictor", "ntl_mean"]].rename(columns={"ntl_mean": "value"})
    log.info(f"  NTL: {len(ntl)} rows, {ntl.date.min().date()} → {ntl.date.max().date()}")
    return ntl


def build_monthly_wide(dept_ubigeos: list) -> pd.DataFrame:
    """Build wide monthly panel: index=(ubigeo, date), columns=SERIES_KEYS."""
    agg = load_dept_panel()
    agg = combine_components(agg)

    # Keep only final predictor names (drop underscore-prefixed intermediates)
    dept_preds = {"credit", "deposits", "electricity", "capex", "current_spending",
                  "mining", "pension", "tax_revenue"}
    agg = agg[agg["predictor"].isin(dept_preds)].copy()

    national  = load_national_series(dept_ubigeos)
    ntl       = load_ntl()

    all_long = pd.concat([agg, national, ntl], ignore_index=True)

    log.info("Pivoting to wide format...")
    wide = (
        all_long.pivot_table(index=["ubigeo", "date"], columns="predictor",
                             values="value", aggfunc="sum")
        .reset_index()
    )
    wide.columns.name = None

    # Ensure all SERIES_KEYS exist
    for k in SERIES_KEYS:
        if k not in wide.columns:
            wide[k] = np.nan
            log.warning(f"  Column '{k}' missing after pivot — filled NaN")

    wide["date"]  = pd.to_datetime(wide["date"])
    wide["year"]  = wide["date"].dt.year
    wide["month"] = wide["date"].dt.month

    log.info(f"Wide panel: {wide.shape}, {wide.ubigeo.nunique()} depts, "
             f"{wide.date.min().date()} → {wide.date.max().date()}")
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Load poverty targets
# ══════════════════════════════════════════════════════════════════════════════

def load_targets() -> pd.DataFrame:
    df = pd.read_parquet(TARGETS / "poverty_departmental.parquet")
    df["ubigeo"] = df["department_code"].astype(str).str.zfill(2)
    df = df[["ubigeo", "department_name", "year", "poverty_rate"]].copy()
    log.info(f"Targets: {len(df)} rows, years {df.year.min()}–{df.year.max()}, "
             f"{df.ubigeo.nunique()} depts")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build pseudo-real-time vintage panel
# ══════════════════════════════════════════════════════════════════════════════

def ytd_features(dept_wide: pd.DataFrame, series: str, year: int,
                 vintage_month: int) -> dict:
    """Compute YTD stats for one series at a given vintage, respecting pub lag."""
    lag            = PUB_LAGS.get(series, 1)
    last_available = vintage_month - lag  # last month of data available at this vintage

    out = {f"{series}_ytd_mean": np.nan,
           f"{series}_ytd_growth": np.nan,
           f"{series}_ytd_vol": np.nan}

    if last_available < 1 or series not in dept_wide.columns:
        return out

    cur_mask  = (dept_wide["year"] == year)     & (dept_wide["month"] <= last_available)
    prev_mask = (dept_wide["year"] == year - 1) & (dept_wide["month"] <= last_available)

    cur_vals  = dept_wide.loc[cur_mask,  series].dropna().values
    prev_vals = dept_wide.loc[prev_mask, series].dropna().values

    if len(cur_vals) == 0:
        return out

    ytd_mean = float(np.mean(cur_vals))
    out[f"{series}_ytd_mean"] = ytd_mean

    if len(prev_vals) > 0:
        prev_mean = float(np.mean(prev_vals))
        if prev_mean != 0:
            out[f"{series}_ytd_growth"] = ytd_mean / prev_mean - 1

    if len(cur_vals) > 1 and ytd_mean != 0:
        out[f"{series}_ytd_vol"] = float(np.std(cur_vals) / abs(ytd_mean))

    return out


def build_vintage_panel(wide: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    log.info("Building vintage panel...")

    years        = sorted(targets["year"].unique())
    dept_ubigeos = sorted(targets["ubigeo"].unique())
    total        = len(dept_ubigeos) * len(years) * 12
    done         = 0
    rows         = []

    # Pre-group wide panel by ubigeo for speed
    wide_by_dept = {u: grp.reset_index(drop=True) for u, grp in wide.groupby("ubigeo")}

    for ubigeo in dept_ubigeos:
        dept_wide = wide_by_dept.get(ubigeo, pd.DataFrame())
        dept_name = targets.loc[targets["ubigeo"] == ubigeo, "department_name"].iloc[0]

        for year in years:
            tgt = targets.loc[(targets["ubigeo"] == ubigeo) & (targets["year"] == year),
                              "poverty_rate"]
            if tgt.empty:
                done += 12
                continue
            poverty_rate = float(tgt.iloc[0])

            for vintage_month in range(1, 13):
                row = {
                    "ubigeo":             ubigeo,
                    "department":         dept_name,
                    "year":               year,
                    "vintage_month":      vintage_month,
                    "months_available":   max(0, vintage_month - 1),
                    "share_year_observed": (vintage_month - 1) / 12,
                    "poverty_rate":       poverty_rate,
                }
                for series in SERIES_KEYS:
                    row.update(ytd_features(dept_wide, series, year, vintage_month))

                rows.append(row)
                done += 1
                if done % 1000 == 0:
                    log.info(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    panel = pd.DataFrame(rows)
    log.info(f"Vintage panel built: {panel.shape}")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Quality checks
# ══════════════════════════════════════════════════════════════════════════════

def quality_checks(panel: pd.DataFrame):
    sep = "=" * 60
    log.info(f"\n{sep}")
    log.info("QUALITY CHECKS")
    log.info(sep)
    log.info(f"Shape: {panel.shape}")
    log.info(f"Depts: {panel.ubigeo.nunique()}, "
             f"Years: {panel.year.nunique()} ({panel.year.min()}–{panel.year.max()}), "
             f"Vintages: {panel.vintage_month.nunique()}")

    log.info("\nMissing fraction per _ytd_mean feature:")
    for col in sorted(panel.columns):
        if col.endswith("_ytd_mean"):
            pct = panel[col].isna().mean() * 100
            log.info(f"  {col:35s}: {pct:5.1f}% missing")

    pr = panel["poverty_rate"] * 100
    log.info(f"\nPoverty rate: mean={pr.mean():.1f}%, std={pr.std():.1f}%, "
             f"min={pr.min():.1f}%, max={pr.max():.1f}%")

    log.info("\nRows per vintage month:")
    for vm, cnt in panel.groupby("vintage_month").size().items():
        log.info(f"  month {vm:2d}: {cnt}")

    jan = panel[panel["vintage_month"] == 1]
    dec = panel[panel["vintage_month"] == 12]
    log.info(f"\nJan vintage non-null rate: {jan.notna().mean().mean():.2f}")
    log.info(f"Dec vintage non-null rate: {dec.notna().mean().mean():.2f}")
    log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=== BUILD POVERTY NOWCAST — Prompt 1: Vintage Panel ===")

    targets      = load_targets()
    dept_ubigeos = sorted(targets["ubigeo"].unique())
    log.info(f"Departments ({len(dept_ubigeos)}): {dept_ubigeos}")

    # Build wide monthly panel
    wide = build_monthly_wide(dept_ubigeos)

    # Save monthly_indicators.parquet
    monthly_cols = ["ubigeo", "year", "month", "date"] + SERIES_KEYS
    monthly_out  = wide[[c for c in monthly_cols if c in wide.columns]].copy()
    out_monthly  = OUT_DIR / "monthly_indicators.parquet"
    monthly_out.to_parquet(out_monthly, index=False)
    log.info(f"\nSaved monthly_indicators.parquet: {monthly_out.shape} → {out_monthly}")

    # Log coverage per series
    log.info("\nMonthly indicator coverage (non-null rows):")
    for k in SERIES_KEYS:
        if k in monthly_out.columns:
            nn  = monthly_out[k].notna().sum()
            tot = len(monthly_out)
            d_min = monthly_out.loc[monthly_out[k].notna(), "date"].min()
            d_max = monthly_out.loc[monthly_out[k].notna(), "date"].max()
            log.info(f"  {k:20s}: {nn}/{tot} non-null  ({d_min.date()} → {d_max.date()})")

    # Build vintage panel
    panel     = build_vintage_panel(wide, targets)
    out_panel = OUT_DIR / "vintage_panel.parquet"
    panel.to_parquet(out_panel, index=False)
    log.info(f"\nSaved vintage_panel.parquet: {panel.shape} → {out_panel}")

    quality_checks(panel)
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
