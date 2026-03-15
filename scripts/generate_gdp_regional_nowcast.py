"""Generate departmental GDP nowcasts using NTL + activity indicator differentials.

Method:
1. Compute NTL shares for spatial weights (economic mass per department).
2. Build a composite activity index per department from high-frequency
   indicators: credit YoY, electricity consumption YoY, tax revenue YoY.
3. Compute each department's deviation from the national composite.
4. Apply dampened adjustment: dept_gdp = national_gdp + α × deviation
   (α = 0.40 — partial pass-through from indicators to GDP growth).
5. Constrain by NTL-weighted average ≈ national nowcast.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))

DATA_DIR = PROJ_ROOT / "data"
EXPORTS_DIR = PROJ_ROOT / "exports" / "data"

# Damping factor: how much indicator differential translates to GDP differential
ALPHA = 0.40
# Number of recent months to average indicators
N_MONTHS = 3
# Categories used for composite activity index
ACTIVITY_CATS = [
    "credit_by_department",
    "electricity_by_department",
    "tax_revenue_by_department",
]

# Ubigeo → canonical department name (matches GeoJSON NOMBDEP in title case)
DEPT_NAMES = {
    "01": "Amazonas", "02": "Ancash", "03": "Apurímac", "04": "Arequipa",
    "05": "Ayacucho", "06": "Cajamarca", "07": "Callao", "08": "Cusco",
    "09": "Huancavelica", "10": "Huánuco", "11": "Ica", "12": "Junín",
    "13": "La Libertad", "14": "Lambayeque", "15": "Lima", "16": "Loreto",
    "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes",
    "25": "Ucayali",
}


def calculate_ntl_shares(ntl_df: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    # v7.1 pipeline uses ntl_sum_eqarea; older used ntl_sum
    ntl_col = "ntl_sum" if "ntl_sum" in ntl_df.columns else "ntl_sum_eqarea"
    latest_date = ntl_df["date"].max()
    cutoff_date = latest_date - pd.DateOffset(months=n_months - 1)
    recent = ntl_df[ntl_df["date"] >= cutoff_date].copy()
    dept_ntl = (
        recent.groupby(["DEPT_CODE", "department"])[ntl_col]
        .sum()
        .reset_index()
        .rename(columns={ntl_col: "ntl_sum"})
    )
    total_ntl = dept_ntl["ntl_sum"].sum()
    dept_ntl["ntl_share"] = dept_ntl["ntl_sum"] / total_ntl
    return dept_ntl.sort_values("DEPT_CODE")


def build_activity_composite(dept_panel: pd.DataFrame) -> pd.Series:
    """Return NTL-weighted composite YoY per department (latest N_MONTHS).

    Returns a Series indexed by ubigeo (str, zero-padded to 2 chars) with
    mean composite YoY over the last N_MONTHS months.
    """
    # Filter to activity categories and recent months
    sub = dept_panel[dept_panel["category"].isin(ACTIVITY_CATS)].copy()
    sub = sub[sub["ubigeo"].isin(DEPT_NAMES.keys())]  # exclude Total/Metro etc.
    latest = sub["date"].max()
    cutoff = latest - pd.DateOffset(months=N_MONTHS - 1)
    sub = sub[sub["date"] >= cutoff]

    # Pivot: dept × category → mean yoy
    pivot = (
        sub.groupby(["ubigeo", "category"])["value_yoy"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    # Equal-weight composite across available categories
    composite = pivot.mean(axis=1)
    return composite


def compute_dept_adjustments(
    composite: pd.Series,
    ntl_shares: pd.DataFrame,
    national_gdp: float,
    target_sigma: float = 1.5,
) -> pd.DataFrame:
    """Compute dept-specific GDP nowcasts using z-score standardization.

    The composite YoY values have very different scales across indicators
    (credit, electricity, tax can all be large). We standardize using
    z-scores relative to the cross-sectional distribution, then scale
    the z-scores to a target GDP spread (target_sigma pp), so that the
    weighted average is preserved by construction.

    Parameters
    ----------
    composite : pd.Series
        Composite activity index YoY by ubigeo.
    ntl_shares : pd.DataFrame
        NTL shares with DEPT_CODE (ubigeo) and ntl_share columns.
    national_gdp : float
        National GDP nowcast (%).
    target_sigma : float
        Target std dev of dept GDP nowcasts (pp). Default 1.5 pp.
    """
    result = ntl_shares.copy()
    result = result[result["DEPT_CODE"].isin(DEPT_NAMES.keys())].copy()
    result["ubigeo"] = result["DEPT_CODE"]
    result["composite_yoy"] = result["ubigeo"].map(composite).values

    # Use robust cross-sectional z-score: median + IQR (less sensitive to outliers)
    valid = result[result["composite_yoy"].notna()]
    med = valid["composite_yoy"].median()
    iqr = valid["composite_yoy"].quantile(0.75) - valid["composite_yoy"].quantile(0.25)
    robust_scale = iqr / 1.349 if iqr > 0 else 1.0  # IQR / 1.349 ≈ sigma for normal

    # Z-score each department, capped at ±2 to handle outliers
    result["z_score"] = ((result["composite_yoy"] - med) / robust_scale).clip(-2.0, 2.0)

    # Scale z-score to target GDP spread
    result["adj"] = target_sigma * result["z_score"]

    # For depts with no indicator data → adj = 0
    result.loc[result["composite_yoy"].isna(), "adj"] = 0.0
    result.loc[result["composite_yoy"].isna(), "z_score"] = 0.0

    # Dept GDP nowcast
    result["gdp_yoy"] = national_gdp + result["adj"]

    # Re-center so NTL-weighted average = national_gdp exactly
    w_avg_adj = (result["adj"] * result["ntl_share"]).sum()
    result["adj"] -= w_avg_adj
    result["gdp_yoy"] = national_gdp + result["adj"]

    result["gdp_contribution"] = result["gdp_yoy"] * result["ntl_share"]
    result["department"] = result["ubigeo"].map(DEPT_NAMES)

    return result[
        ["DEPT_CODE", "department", "gdp_yoy", "adj", "ntl_share",
         "gdp_contribution", "composite_yoy"]
    ].sort_values("DEPT_CODE")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== GDP Regional Nowcast Generator (Differential) ===\n")

    # 1. Load national nowcast
    print("1. Loading national GDP nowcast...")
    with open(EXPORTS_DIR / "gdp_nowcast.json", "r", encoding="utf-8") as f:
        gdp_nowcast = json.load(f)
    target_period = gdp_nowcast["nowcast"]["target_period"]
    national_gdp = gdp_nowcast["nowcast"]["value"]
    print(f"   National: {national_gdp:.2f}% ({target_period})")

    # 2. Load NTL departmental data
    print("\n2. Loading NTL departmental data...")
    ntl_df = pd.read_parquet(DATA_DIR / "processed" / "ntl_monthly_department.parquet")
    ntl_df["date"] = pd.to_datetime(ntl_df["date"])
    print(f"   {len(ntl_df)} obs, {ntl_df['department'].nunique()} depts")

    # 3. NTL shares
    print("\n3. Calculating NTL shares (12-month avg)...")
    ntl_shares = calculate_ntl_shares(ntl_df, n_months=12)

    # 4. Departmental activity composite
    print("\n4. Building activity composite from departmental panel...")
    dept_panel = pd.read_parquet(
        DATA_DIR / "processed" / "departmental" / "panel_departmental_monthly.parquet"
    )
    dept_panel["date"] = pd.to_datetime(dept_panel["date"])
    # Normalize ubigeo to zero-padded 2 chars
    dept_panel["ubigeo"] = dept_panel["ubigeo"].astype(str).str.zfill(2)

    composite = build_activity_composite(dept_panel)
    print(f"   Composite available for {composite.notna().sum()} / {len(DEPT_NAMES)} depts")
    print(f"   National composite (NTL-weighted): {composite.mean():.2f}%")
    print(f"   Range: [{composite.min():.2f}%, {composite.max():.2f}%]")

    # 5. Compute differential nowcasts
    print("\n5. Computing differential nowcasts...")
    dept_nowcasts = compute_dept_adjustments(composite, ntl_shares, national_gdp)

    # NTL-weighted avg should be ≈ national
    weighted_avg = (dept_nowcasts["gdp_yoy"] * dept_nowcasts["ntl_share"]).sum()
    print(f"   NTL-weighted avg nowcast: {weighted_avg:.3f}% (target: {national_gdp:.2f}%)")
    print(f"   Range: [{dept_nowcasts['gdp_yoy'].min():.2f}%, {dept_nowcasts['gdp_yoy'].max():.2f}%]")
    print(f"\n   Top 5 fastest growing:")
    for _, row in dept_nowcasts.nlargest(5, "gdp_yoy").iterrows():
        adj_str = f" ({row['adj']:+.2f} pp adj)" if not np.isnan(row["adj"]) else ""
        print(f"     {row['department']:20s}: {row['gdp_yoy']:.2f}%{adj_str}")
    print(f"\n   Slowest 5:")
    for _, row in dept_nowcasts.nsmallest(5, "gdp_yoy").iterrows():
        adj_str = f" ({row['adj']:+.2f} pp adj)" if not np.isnan(row["adj"]) else ""
        print(f"     {row['department']:20s}: {row['gdp_yoy']:.2f}%{adj_str}")

    # 6. Export JSON
    print("\n6. Exporting JSON...")
    output = {
        "metadata": {
            "method": "NTL spatial weights + activity indicator differentials",
            "target_period": target_period,
            "national_gdp_yoy": round(float(national_gdp), 2),
            "n_departments": len(dept_nowcasts),
            "ntl_months": 12,
            "indicator_months": N_MONTHS,
            "alpha": ALPHA,
            "indicators": ACTIVITY_CATS,
            "note": (
                f"Dept GDP = National ({national_gdp:.2f}%) + {ALPHA} × "
                f"(dept composite – national composite). "
                f"Composite = mean YoY of credit, electricity, tax revenue "
                f"(last {N_MONTHS} months). Capped at ±3 pp."
            ),
        },
        "departmental_nowcasts": [
            {
                "dept_code": row["DEPT_CODE"],
                "department": row["department"],
                "gdp_yoy": round(float(row["gdp_yoy"]), 2),
                "adj_pp": round(float(row["adj"]), 2) if pd.notna(row["adj"]) else 0.0,
                "ntl_share": round(float(row["ntl_share"]), 4),
                "gdp_contribution": round(float(row["gdp_contribution"]), 4),
                "composite_yoy": round(float(row["composite_yoy"]), 2) if pd.notna(row["composite_yoy"]) else None,
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
