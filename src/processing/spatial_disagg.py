"""Spatial disaggregation of department-level nowcasts to district level using NTL.

Uses nighttime lights as spatial weights to distribute department-level
predictions (poverty, GDP, etc.) to 1,891 districts. Analogous to how
disaggregate.py uses Chow-Lin for temporal disaggregation.

Two methods:
  - ntl_share: proportional allocation (for GDP/economic activity)
  - ntl_dasymetric: inverse-NTL weighting (for poverty rates)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.spatial_disagg")


def disaggregate_to_districts(
    dept_values: pd.DataFrame,
    ntl_district: pd.DataFrame,
    target_col: str = "poverty_rate",
    method: str = "ntl_dasymetric",
    year: int | None = None,
) -> pd.DataFrame:
    """Distribute department-level nowcasts to districts using NTL weights.

    Parameters
    ----------
    dept_values : pd.DataFrame
        Department-level values with columns: department_code, year, and target_col.
    ntl_district : pd.DataFrame
        District NTL data (ntl_monthly_district.parquet) with UBIGEO, DEPT_CODE,
        year, ntl_sum.
    target_col : str
        Which column to disaggregate.
    method : str
        'ntl_share' (proportional) or 'ntl_dasymetric' (inverse weighting).
    year : int or None
        Year for NTL weights. If None, uses latest available year.

    Returns
    -------
    pd.DataFrame
        District-level estimates with columns: district_ubigeo, department_code,
        year, {target_col}_nowcast, ntl_weight, ntl_sum.
    """
    ntl = ntl_district.copy()
    ntl["date"] = pd.to_datetime(ntl["date"])
    ntl["year"] = ntl["date"].dt.year

    # Normalise column name: pipeline v7 uses ntl_sum_eqarea, older used ntl_sum
    if "ntl_sum" not in ntl.columns and "ntl_sum_eqarea" in ntl.columns:
        ntl = ntl.rename(columns={"ntl_sum_eqarea": "ntl_sum"})

    # Select year for NTL weights
    if year is None:
        year = ntl["year"].max()
    ntl_year = ntl[ntl["year"] == year].copy()

    # Aggregate NTL to annual by district
    district_annual = (
        ntl_year.groupby(["UBIGEO", "DEPT_CODE"])["ntl_sum"]
        .mean()
        .reset_index()
    )

    # Compute weights within each department
    if method == "ntl_share":
        # Proportional: more light = higher share
        dept_totals = district_annual.groupby("DEPT_CODE")["ntl_sum"].sum()
        district_annual = district_annual.merge(
            dept_totals.rename("dept_total"),
            left_on="DEPT_CODE",
            right_index=True,
        )
        district_annual["weight"] = (
            district_annual["ntl_sum"] / district_annual["dept_total"]
        )
    elif method == "ntl_dasymetric":
        # Inverse NTL: more light = less poverty (within department)
        # Use log(1+ntl) for smoother distribution
        district_annual["inv_ntl"] = 1.0 / np.log1p(
            district_annual["ntl_sum"].clip(lower=0.01)
        )
        dept_inv_totals = district_annual.groupby("DEPT_CODE")["inv_ntl"].sum()
        district_annual = district_annual.merge(
            dept_inv_totals.rename("dept_inv_total"),
            left_on="DEPT_CODE",
            right_index=True,
        )
        district_annual["weight"] = (
            district_annual["inv_ntl"] / district_annual["dept_inv_total"]
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ntl_share' or 'ntl_dasymetric'.")

    # Map department codes to poverty department codes
    # Callao (07) → Lima (15) in poverty data
    dept_vals = dept_values.copy()

    # Merge district weights with department values
    results = []
    for _, district_row in district_annual.iterrows():
        district_ubigeo = district_row["UBIGEO"]
        dept_code = district_row["DEPT_CODE"]
        weight = district_row["weight"]
        ntl_sum_val = district_row["ntl_sum"]

        # Look up department value (handle Callao→Lima merge)
        lookup_dept = "15" if dept_code == "07" else dept_code
        dept_match = dept_vals[dept_vals["department_code"] == lookup_dept]

        if dept_match.empty:
            continue

        dept_val = dept_match[target_col].iloc[0]

        if method == "ntl_share":
            # For level variables: district = dept_total × share
            district_val = dept_val * weight
        else:
            # For rates (poverty): distribute around dept average
            # n_districts in this dept
            n_dists = len(
                district_annual[district_annual["DEPT_CODE"] == dept_code]
            )
            # district_rate = dept_rate × (weight × n_districts)
            # This ensures mean(district_rates) ≈ dept_rate
            district_val = dept_val * weight * n_dists
            # Constrain to [0, 1] for rates (stored as fractions)
            if target_col in ("poverty_rate", "extreme_poverty_rate"):
                district_val = np.clip(district_val, 0.0, 1.0)

        results.append({
            "district_ubigeo": district_ubigeo,
            "department_code": dept_code,
            "year": year,
            f"{target_col}_nowcast": district_val,
            "ntl_weight": weight,
            "ntl_sum": ntl_sum_val,
        })

    result_df = pd.DataFrame(results)

    logger.info(
        "District disaggregation: %d districts, method=%s, year=%d",
        len(result_df), method, year,
    )
    return result_df


def nowcast_districts(
    dept_nowcasts: dict,
    ntl_district_path: Path,
    year: int,
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convenience: take poverty nowcaster output and disaggregate to districts.

    Parameters
    ----------
    dept_nowcasts : dict
        Mapping {dept_code: {target: value}} from a poverty nowcaster.
    ntl_district_path : Path
        Path to ntl_monthly_district.parquet.
    year : int
        Year for NTL weights and output.
    target_cols : list[str] or None
        Which targets to disaggregate. Defaults to poverty targets.

    Returns
    -------
    pd.DataFrame
        District-level nowcasts for all specified targets.
    """
    if target_cols is None:
        target_cols = ["poverty_rate", "extreme_poverty_rate", "mean_consumption"]

    ntl_district = pd.read_parquet(ntl_district_path)

    # Convert dept_nowcasts to DataFrame
    rows = []
    for dept, targets in dept_nowcasts.items():
        row = {"department_code": dept}
        row.update(targets)
        rows.append(row)
    dept_df = pd.DataFrame(rows)

    if dept_df.empty:
        return pd.DataFrame()

    # Disaggregate each target
    all_results = []
    for tc in target_cols:
        if tc not in dept_df.columns:
            continue

        method = "ntl_dasymetric" if tc in ("poverty_rate", "extreme_poverty_rate") else "ntl_share"

        result = disaggregate_to_districts(
            dept_df,
            ntl_district,
            target_col=tc,
            method=method,
            year=year,
        )
        all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    # Merge all targets into single DataFrame
    merged = all_results[0][["district_ubigeo", "department_code", "year", "ntl_weight", "ntl_sum"]]
    for result in all_results:
        nowcast_cols = [c for c in result.columns if c.endswith("_nowcast")]
        for col in nowcast_cols:
            merged[col] = result[col]

    return merged
