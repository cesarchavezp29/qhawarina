"""Generate departmental GDP nowcasts using NTL spatial weights + indicator differentials.

Method (v2 — post-audit restructure):
1. NTL ROLE — spatial weights only.
   Audit confirmed cross-sectional R²=0.81 (NTL level ~ VAB level) but median r=0.09
   for NTL growth ~ VAB growth. NTL captures economic STOCK, not FLOW.
   NTL shares are computed from dry-season months (May–Oct) to avoid rainy-season
   cloud contamination (VIIRS VNP46A3 affected by cloud cover in Nov–Apr).

2. GROWTH SIGNAL — panel indicators only.
   Composite = equal-weight mean of credit YoY, electricity YoY, tax revenue YoY
   (last N_MONTHS months). Indicators with |YoY| > 50% excluded (electricity splice fix).

3. FORMULA:
   gdp_dept = gdp_national + max_adj × tanh(z_score / scale)
   where z_score = (composite - cross-sectional median) / robust_scale (IQR-based).
   tanh replaces hard ±2σ clip — smooth compression, strict monotonicity,
   no two departments can produce identical outputs.
   NTL does NOT enter the growth adjustment. NTL only determines weights
   for the re-centering constraint (NTL-weighted avg = national GDP).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))

DATA_DIR    = PROJ_ROOT / "data"
EXPORTS_DIR = PROJ_ROOT / "exports" / "data"

# Number of recent months to average indicators
N_MONTHS = 3
# Dry season months — used for NTL shares (cloud-free)
DRY_SEASON_MONTHS = {5, 6, 7, 8, 9, 10}   # May–October
# tanh adjustment parameters
TANH_MAX_ADJ = 4.0   # asymptotic cap on adj in pp (reached at z → ∞)
TANH_SCALE   = 2.0   # z-score at which adj ≈ 3.05 pp (tanh(1) × 4)
# Categories used for composite activity index
ACTIVITY_CATS = [
    "credit_by_department",
    "electricity_by_department",
    "tax_revenue_by_department",
]

# Ubigeo → canonical department name (matches GeoJSON NOMBDEP in title case)
DEPT_NAMES = {
    "01": "Amazonas",    "02": "Ancash",      "03": "Apurímac",   "04": "Arequipa",
    "05": "Ayacucho",    "06": "Cajamarca",   "07": "Callao",     "08": "Cusco",
    "09": "Huancavelica","10": "Huánuco",     "11": "Ica",        "12": "Junín",
    "13": "La Libertad", "14": "Lambayeque",  "15": "Lima",       "16": "Loreto",
    "17": "Madre de Dios","18": "Moquegua",   "19": "Pasco",      "20": "Piura",
    "21": "Puno",        "22": "San Martín",  "23": "Tacna",      "24": "Tumbes",
    "25": "Ucayali",
}


def calculate_ntl_shares(ntl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NTL-based spatial weights from dry-season months only.

    Uses May–October months from the most recent available dry season to avoid
    VIIRS cloud contamination (rainy season Nov–Apr depresses interior departments
    and inflates coastal Lima readings). Cross-sectional R²=0.81 justifies NTL
    as a spatial weight; it is NOT used as a growth signal.
    """
    ntl_col = "ntl_sum_eqarea" if "ntl_sum_eqarea" in ntl_df.columns else "ntl_sum"

    df = ntl_df.copy()
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    # Find the most recent calendar year that has ≥3 dry season months available
    dry = df[df["month"].isin(DRY_SEASON_MONTHS)].copy()
    available = dry.groupby("year")["month"].nunique()
    valid_years = available[available >= 3].index
    if len(valid_years) == 0:
        raise ValueError("No year with ≥3 dry season months found in NTL data")
    dry_year = int(valid_years.max())
    dry_months_used = sorted(dry[dry["year"] == dry_year]["month"].unique())
    print(f"   [NTL] Using dry season months: {dry_year} months {dry_months_used}")

    recent_dry = dry[dry["year"] == dry_year].copy()

    dept_ntl = (
        recent_dry.groupby(["DEPT_CODE", "department"])[ntl_col]
        .mean()          # mean across dry months (stable per-dept level)
        .reset_index()
        .rename(columns={ntl_col: "ntl_mean"})
    )
    total_ntl = dept_ntl["ntl_mean"].sum()
    dept_ntl["ntl_share"] = dept_ntl["ntl_mean"] / total_ntl
    dept_ntl = dept_ntl.rename(columns={"ntl_mean": "ntl_sum"})

    print(f"   [NTL] Top 5 by share: " +
          ", ".join(f"{r['department']}={r['ntl_share']:.3f}"
                    for _, r in dept_ntl.nlargest(5, "ntl_share").iterrows()))
    return dept_ntl.sort_values("DEPT_CODE")


def build_activity_composite(dept_panel: pd.DataFrame) -> pd.Series:
    """Return composite activity YoY per department (latest N_MONTHS).

    Electricity splice fix: BCRP electricity series has base-period anomalies
    (mid-2024 depressed due to El Niño drought / hydro shortfall), causing
    YoY of +120% to +280% in 2025. Any indicator |YoY| > 50% is excluded.
    Remaining 1–2 indicators carry the composite. If all extreme → NaN
    → department receives adj=0 (national rate).

    Returns a Series indexed by ubigeo with mean composite YoY.
    """
    INDICATOR_CAP_PCT = 50.0

    sub = dept_panel[dept_panel["category"].isin(ACTIVITY_CATS)].copy()
    sub = sub[sub["ubigeo"].isin(DEPT_NAMES.keys())]
    latest  = sub["date"].max()
    cutoff  = latest - pd.DateOffset(months=N_MONTHS - 1)
    sub     = sub[sub["date"] >= cutoff]

    # Flag and exclude extreme YoY values (splice artefacts)
    extreme_mask = sub["value_yoy"].abs() > INDICATOR_CAP_PCT
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        extreme_rows = sub[extreme_mask][["ubigeo", "category", "date", "value_yoy"]]
        print(f"   [Composite] Flagged {n_extreme} extreme obs (|YoY| > {INDICATOR_CAP_PCT}%):")
        for _, r in extreme_rows.sort_values("value_yoy").iterrows():
            dept = DEPT_NAMES.get(r["ubigeo"], r["ubigeo"])
            print(f"     {dept:18s}  {r['category']:35s}  {r['date'].date()}  "
                  f"{r['value_yoy']:+.1f}%  → excluded")
        sub.loc[extreme_mask, "value_yoy"] = np.nan

    # Pivot: dept × category → mean yoy (NaN excluded automatically)
    pivot = (
        sub.groupby(["ubigeo", "category"])["value_yoy"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    composite = pivot.mean(axis=1, skipna=True)

    # Depts where all indicators extreme → NaN → will receive adj=0
    all_nan = pivot.isna().all(axis=1)
    composite[all_nan] = np.nan
    if all_nan.any():
        print(f"   [Composite] All indicators extreme: "
              f"{[DEPT_NAMES.get(c,c) for c in all_nan[all_nan].index]} → national rate")

    return composite


def compute_dept_adjustments(
    composite: pd.Series,
    ntl_shares: pd.DataFrame,
    national_gdp: float,
) -> pd.DataFrame:
    """Compute dept-specific GDP nowcasts with tanh adjustment.

    Formula:
        z        = (composite_yoy - cross_median) / robust_scale
        adj_pp   = TANH_MAX_ADJ × tanh(z / TANH_SCALE)
        gdp_dept = gdp_national + adj_pp  (before re-centering)

    tanh properties vs hard clip:
    - Strictly monotonic → no two departments with different composites
      can produce identical adj_pp (solves the identical-values problem)
    - Smooth compression → extreme outliers dampened, not truncated
    - Asymptotic ±TANH_MAX_ADJ pp → bounded spread
    - z=0 → adj=0; z=TANH_SCALE → adj ≈ 0.76 × TANH_MAX_ADJ pp

    NTL shares enter ONLY the re-centering step (weighted avg = national GDP).
    They do NOT affect individual department z-scores or adjustments.
    """
    result = ntl_shares.copy()
    result = result[result["DEPT_CODE"].isin(DEPT_NAMES.keys())].copy()
    result["ubigeo"]        = result["DEPT_CODE"]
    result["composite_yoy"] = result["ubigeo"].map(composite).values

    # Robust cross-sectional z-score (IQR-based, less sensitive to outliers)
    valid       = result[result["composite_yoy"].notna()]
    med         = valid["composite_yoy"].median()
    iqr         = valid["composite_yoy"].quantile(0.75) - valid["composite_yoy"].quantile(0.25)
    robust_scale = iqr / 1.349 if iqr > 0 else 1.0   # IQR/1.349 ≈ σ for normal distribution

    result["z_score"] = (result["composite_yoy"] - med) / robust_scale

    # tanh adjustment — strictly monotonic, no hard clip, no identical outputs
    result["adj"] = TANH_MAX_ADJ * np.tanh(result["z_score"] / TANH_SCALE)

    # Depts with no indicator data → adj = 0 (national rate)
    no_data = result["composite_yoy"].isna()
    result.loc[no_data, "adj"]     = 0.0
    result.loc[no_data, "z_score"] = 0.0

    # Dept GDP nowcast before re-centering
    result["gdp_yoy"] = national_gdp + result["adj"]

    # Re-center: NTL-weighted average must equal national GDP exactly
    w_avg_adj         = (result["adj"] * result["ntl_share"]).sum()
    result["adj"]     -= w_avg_adj
    result["gdp_yoy"]  = national_gdp + result["adj"]

    result["gdp_contribution"] = result["gdp_yoy"] * result["ntl_share"]
    result["department"]       = result["ubigeo"].map(DEPT_NAMES)

    return result[
        ["DEPT_CODE", "department", "gdp_yoy", "adj", "ntl_share",
         "gdp_contribution", "composite_yoy", "z_score"]
    ].sort_values("DEPT_CODE")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== GDP Regional Nowcast Generator (v2 — dry-season NTL + tanh adj) ===\n")

    # 1. National nowcast
    print("1. Loading national GDP nowcast...")
    with open(EXPORTS_DIR / "gdp_nowcast.json", "r", encoding="utf-8") as f:
        gdp_nowcast = json.load(f)
    target_period = gdp_nowcast["nowcast"]["target_period"]
    national_gdp  = gdp_nowcast["nowcast"]["value"]
    print(f"   National: {national_gdp:.2f}% ({target_period})")

    # 2. NTL shares (dry season — spatial weights only)
    print("\n2. Loading NTL data and computing dry-season spatial weights...")
    ntl_df = pd.read_parquet(DATA_DIR / "processed" / "ntl_monthly_department.parquet")
    ntl_df["date"] = pd.to_datetime(ntl_df["date"])
    print(f"   {len(ntl_df)} obs, {ntl_df['department'].nunique()} depts, "
          f"latest: {ntl_df['date'].max().date()}")
    ntl_shares = calculate_ntl_shares(ntl_df)

    # 3. Activity composite (growth signal — indicators only, no NTL)
    print("\n3. Building activity composite from panel indicators...")
    dept_panel = pd.read_parquet(
        DATA_DIR / "processed" / "departmental" / "panel_departmental_monthly.parquet"
    )
    dept_panel["date"]   = pd.to_datetime(dept_panel["date"])
    dept_panel["ubigeo"] = dept_panel["ubigeo"].astype(str).str.zfill(2)

    composite = build_activity_composite(dept_panel)
    print(f"   Composite available: {composite.notna().sum()} / {len(DEPT_NAMES)} depts")
    print(f"   Range: [{composite.min():.2f}%, {composite.max():.2f}%]  "
          f"median={composite.median():.2f}%")

    # 4. Dept nowcasts
    print("\n4. Computing tanh-adjusted departmental nowcasts...")
    dept_nowcasts = compute_dept_adjustments(composite, ntl_shares, national_gdp)

    weighted_avg = (dept_nowcasts["gdp_yoy"] * dept_nowcasts["ntl_share"]).sum()
    print(f"   NTL-weighted avg: {weighted_avg:.4f}% (target: {national_gdp:.2f}%)")
    print(f"   Range: [{dept_nowcasts['gdp_yoy'].min():.2f}%, "
          f"{dept_nowcasts['gdp_yoy'].max():.2f}%]")

    # Check for identical values — should be zero after tanh
    from collections import Counter
    yoy_rounded = [round(float(v), 2) for v in dept_nowcasts["gdp_yoy"]]
    dupes = {v: c for v, c in Counter(yoy_rounded).items() if c > 1}
    if dupes:
        print(f"   WARNING: identical gdp_yoy values (rounded to 2dp): {dupes}")
    else:
        print(f"   OK: all 25 department nowcasts are unique")

    print(f"\n   Top 5:")
    for _, row in dept_nowcasts.nlargest(5, "gdp_yoy").iterrows():
        print(f"     {row['department']:20s}: {row['gdp_yoy']:.2f}%  "
              f"(adj {row['adj']:+.2f}pp, z={row['z_score']:.2f})")
    print(f"\n   Bottom 5:")
    for _, row in dept_nowcasts.nsmallest(5, "gdp_yoy").iterrows():
        print(f"     {row['department']:20s}: {row['gdp_yoy']:.2f}%  "
              f"(adj {row['adj']:+.2f}pp, z={row['z_score']:.2f})")

    # 5. Export JSON
    print("\n5. Exporting JSON...")
    output = {
        "metadata": {
            "method": "NTL dry-season spatial weights + tanh-adjusted indicator differentials",
            "target_period": target_period,
            "national_gdp_yoy": round(float(national_gdp), 2),
            "n_departments": len(dept_nowcasts),
            "ntl_basis": "dry-season months (May-Oct), most recent year",
            "indicator_months": N_MONTHS,
            "indicators": ACTIVITY_CATS,
            "adjustment_formula": (
                f"adj = {TANH_MAX_ADJ} × tanh(z / {TANH_SCALE}); "
                f"z = (composite - cross_median) / IQR_scale"
            ),
            "note": (
                f"NTL (VIIRS VNP46A3) serves as spatial weights reflecting each department's "
                f"share of economic activity (cross-sectional R²=0.81). Growth signals come "
                f"from credit, electricity, and tax revenue — indicators with direct economic "
                f"content. NTL growth rates showed insufficient correlation with departmental "
                f"GDP growth (median r=0.09) and are not used as growth predictors. "
                f"Indicators with |YoY|>50% excluded (electricity splice artefact fix). "
                f"tanh adjustment ensures no two departments produce identical growth rates."
            ),
        },
        "departmental_nowcasts": [
            {
                "dept_code":      row["DEPT_CODE"],
                "department":     row["department"],
                "gdp_yoy":        round(float(row["gdp_yoy"]), 2),
                "adj_pp":         round(float(row["adj"]), 2),
                "ntl_share":      round(float(row["ntl_share"]), 4),
                "gdp_contribution": round(float(row["gdp_contribution"]), 4),
                "composite_yoy":  round(float(row["composite_yoy"]), 2)
                                  if pd.notna(row["composite_yoy"]) else None,
                "z_score":        round(float(row["z_score"]), 3),
            }
            for _, row in dept_nowcasts.iterrows()
        ],
    }

    out_path = EXPORTS_DIR / "gdp_regional_nowcast.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   Exported to {out_path}")


if __name__ == "__main__":
    main()
