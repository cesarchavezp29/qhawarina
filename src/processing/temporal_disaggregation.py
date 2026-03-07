"""Temporal disaggregation for converting annual poverty to quarterly estimates.

Uses Chow-Lin or Denton methods with high-frequency indicators:
- GDP growth (quarterly, proxy for income)
- CPI (quarterly average, for poverty line)
- Unemployment (monthly → quarterly, labor market)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger("nexus.temporal_disagg")


def chow_lin_quarterly(
    annual_values: pd.Series,
    quarterly_indicator: pd.Series,
    rho: float = 0.5
) -> pd.Series:
    """Disaggregate annual series to quarterly using proportional method.

    Simplified approach: distribute annual values proportionally to indicator,
    then adjust to ensure quarterly values average to annual total.

    Parameters
    ----------
    annual_values : Series
        Annual values with DatetimeIndex (year starts).
    quarterly_indicator : Series
        Quarterly indicator with DatetimeIndex (quarter starts).
    rho : float
        Unused (kept for API compatibility).

    Returns
    -------
    Series
        Quarterly estimates that average to annual values.
    """
    # Align years
    years = sorted(annual_values.index.year.unique())
    all_quarters = quarterly_indicator.index

    # Filter quarters to years in annual data
    start_year = min(years)
    end_year = max(years)
    quarters = all_quarters[(all_quarters.year >= start_year) & (all_quarters.year <= end_year)]

    # Get indicator values for these quarters
    indicator_q = quarterly_indicator.reindex(quarters).ffill().bfill()

    # Proportional distribution with annual constraint
    result_values = []
    result_dates = []

    for year in years:
        year_quarters = quarters[quarters.year == year]
        year_indicator = indicator_q.loc[year_quarters]

        # Annual value for this year
        annual_date = pd.Timestamp(f"{year}-01-01")
        if annual_date in annual_values.index:
            annual_val = annual_values[annual_date]
        else:
            annual_val = annual_values.iloc[(annual_values.index.year == year).argmax()]

        # Proportional distribution
        if year_indicator.sum() > 0:
            quarterly_vals = (year_indicator / year_indicator.sum()) * annual_val * len(year_quarters)
        else:
            quarterly_vals = pd.Series([annual_val] * len(year_quarters), index=year_quarters)

        result_values.extend(quarterly_vals.values)
        result_dates.extend(year_quarters)

    result = pd.Series(
        result_values,
        index=pd.DatetimeIndex(result_dates),
        name=annual_values.name or "quarterly_estimate"
    )

    return result


def denton_quarterly(
    annual_values: pd.Series,
    quarterly_indicator: pd.Series,
    variant: str = "additive"
) -> pd.Series:
    """Disaggregate annual series to quarterly using Denton method.

    Denton minimizes period-to-period differences in adjustments,
    producing smooth quarterly series that aggregates to annual totals.

    Parameters
    ----------
    annual_values : Series
        Annual values with DatetimeIndex.
    quarterly_indicator : Series
        Quarterly indicator (e.g., GDP growth).
    variant : str
        'additive' (differences) or 'proportional' (ratios).

    Returns
    -------
    Series
        Smoothed quarterly estimates.

    References
    ----------
    Denton, F. T. (1971). "Adjustment of Monthly or Quarterly Series to
    Annual Totals: An Approach Based on Quadratic Minimization."
    Journal of the American Statistical Association, 66(333), 99-102.
    """
    years = annual_values.index.year.unique()
    quarters = quarterly_indicator.index

    n_quarters = len(quarterly_indicator)
    n_years = len(annual_values)

    # Aggregation matrix
    C = np.zeros((n_years, n_quarters))
    for i, year in enumerate(years):
        year_quarters = quarters[quarters.year == year]
        for q in year_quarters:
            q_idx = quarters.get_loc(q)
            C[i, q_idx] = 1.0

    # Preliminary estimates (proportional to indicator)
    prelim = quarterly_indicator.copy()

    # Scale preliminary to match annual totals
    scaled = np.zeros(n_quarters)
    for i, year in enumerate(years):
        year_mask = quarters.year == year
        year_prelim = prelim[year_mask]
        if year_prelim.sum() > 0:
            scale = annual_values.iloc[i] / year_prelim.sum()
            scaled[year_mask] = year_prelim * scale

    # Denton smoothing: minimize sum of squared first differences
    # subject to C * y = annual_values
    D = np.diff(np.eye(n_quarters), axis=0)  # first difference matrix

    if variant == "proportional":
        # Minimize (D @ (y / prelim))^2
        weights = 1.0 / (prelim.values + 1e-8)
        W = np.diag(weights)
        H = D.T @ D @ W
    else:
        # Minimize (D @ y)^2
        H = D.T @ D

    # Quadratic programming: min 0.5 * y' * H * y  s.t. C * y = annual_values
    # KKT: H*y + C'*lambda = 0, C*y = b
    # Solution: y = H^{-1} * C' * (C * H^{-1} * C')^{-1} * b

    H_inv = np.linalg.pinv(H + 1e-6 * np.eye(n_quarters))  # regularization
    CH = C @ H_inv
    CHC_inv = np.linalg.pinv(CH @ C.T)

    y_smooth = H_inv @ C.T @ CHC_inv @ annual_values.values

    result = pd.Series(
        y_smooth,
        index=quarterly_indicator.index,
        name=annual_values.name or "quarterly_estimate"
    )

    return result


def poverty_quarterly_disagg(
    annual_poverty: pd.DataFrame,
    gdp_quarterly: pd.DataFrame,
    cpi_monthly: pd.DataFrame,
    method: str = "chow-lin"
) -> pd.DataFrame:
    """Disaggregate annual poverty to quarterly using economic indicators.

    Parameters
    ----------
    annual_poverty : DataFrame
        Annual poverty with columns: year, poverty_rate (and optionally department_code).
    gdp_quarterly : DataFrame
        Quarterly GDP with columns: date, gdp_yoy.
    cpi_monthly : DataFrame
        Monthly CPI with columns: date, ipc_monthly_var.
    method : str
        'chow-lin' or 'denton'.

    Returns
    -------
    DataFrame
        Quarterly poverty estimates with columns: date, poverty_rate_quarterly, (department_code).
    """
    # Aggregate CPI to quarterly
    cpi_monthly = cpi_monthly.copy()
    cpi_monthly["date"] = pd.to_datetime(cpi_monthly["date"])
    cpi_monthly = cpi_monthly.set_index("date")

    cpi_quarterly = cpi_monthly.resample("QS")["ipc_monthly_var"].mean()

    # Merge GDP and CPI to create composite indicator
    gdp_quarterly = gdp_quarterly.copy()
    gdp_quarterly["date"] = pd.to_datetime(gdp_quarterly["date"])
    gdp_quarterly = gdp_quarterly.set_index("date")

    # Composite: poverty inversely related to GDP growth, directly to inflation
    # Simple model: poverty ~ -0.5 * GDP_growth + 0.3 * CPI_change
    indicator = pd.DataFrame(index=gdp_quarterly.index)
    indicator["composite"] = (
        -0.5 * gdp_quarterly["gdp_yoy"].fillna(0)
        + 0.3 * cpi_quarterly.reindex(indicator.index).fillna(0)
    )

    # Normalize indicator to be positive
    indicator["composite"] = indicator["composite"] - indicator["composite"].min() + 1.0

    results = []

    # Check if departmental or national
    if "department_code" in annual_poverty.columns:
        # Departmental disaggregation
        for dept in annual_poverty["department_code"].unique():
            dept_annual = annual_poverty[annual_poverty["department_code"] == dept].copy()
            dept_annual = dept_annual.set_index(pd.to_datetime(dept_annual["year"].astype(str) + "-01-01"))

            annual_series = dept_annual["poverty_rate"]

            if method == "chow-lin":
                quarterly = chow_lin_quarterly(annual_series, indicator["composite"])
            else:
                quarterly = denton_quarterly(annual_series, indicator["composite"])

            dept_results = pd.DataFrame({
                "date": quarterly.index,
                "department_code": dept,
                "poverty_rate_quarterly": quarterly.values
            })
            results.append(dept_results)

    else:
        # National disaggregation
        annual_poverty = annual_poverty.copy()
        annual_poverty = annual_poverty.set_index(pd.to_datetime(annual_poverty["year"].astype(str) + "-01-01"))

        annual_series = annual_poverty["poverty_rate"]

        if method == "chow-lin":
            quarterly = chow_lin_quarterly(annual_series, indicator["composite"])
        else:
            quarterly = denton_quarterly(annual_series, indicator["composite"])

        results.append(pd.DataFrame({
            "date": quarterly.index,
            "poverty_rate_quarterly": quarterly.values
        }))

    return pd.concat(results, ignore_index=True)


def poverty_monthly_disagg(
    annual_poverty: pd.DataFrame,
    ntl_monthly: pd.DataFrame,
    cpi_monthly: pd.DataFrame,
    method: str = "chow-lin",
    smoothing: int = 3
) -> pd.DataFrame:
    """Disaggregate annual poverty to monthly using NTL + CPI indicators.

    Parameters
    ----------
    annual_poverty : DataFrame
        Annual poverty with columns: year, poverty_rate.
    ntl_monthly : DataFrame
        Monthly NTL with columns: date, value (log NTL sum).
    cpi_monthly : DataFrame
        Monthly CPI with columns: date, ipc_monthly_var.
    method : str
        'chow-lin' or 'denton'.
    smoothing : int
        Window for moving average smoothing (default 3 months).

    Returns
    -------
    DataFrame
        Monthly poverty estimates with columns: date, poverty_rate_monthly, poverty_rate_smooth.
    """
    # Prepare NTL monthly indicator
    ntl_monthly = ntl_monthly.copy()
    ntl_monthly["date"] = pd.to_datetime(ntl_monthly["date"])
    ntl_monthly = ntl_monthly.set_index("date")

    # Prepare CPI monthly
    cpi_monthly = cpi_monthly.copy()
    cpi_monthly["date"] = pd.to_datetime(cpi_monthly["date"])
    cpi_monthly = cpi_monthly.set_index("date")

    # Create composite monthly indicator
    # Poverty inversely related to NTL (economic activity), directly to inflation
    indicator = pd.DataFrame(index=ntl_monthly.index)
    indicator["composite"] = (
        -0.6 * ntl_monthly["value"].fillna(0)
        + 0.4 * cpi_monthly["ipc_monthly_var"].reindex(indicator.index).fillna(0)
    )

    # Normalize to be positive
    indicator["composite"] = indicator["composite"] - indicator["composite"].min() + 1.0

    # Prepare annual poverty as series
    annual_poverty = annual_poverty.copy()
    annual_poverty = annual_poverty.set_index(
        pd.to_datetime(annual_poverty["year"].astype(str) + "-01-01")
    )
    annual_series = annual_poverty["poverty_rate"]

    # Disaggregate to monthly using proportional method
    years = sorted(annual_series.index.year.unique())
    all_months = indicator.index

    # Filter months to years in annual data
    start_year = min(years)
    end_year = max(years)
    months = all_months[(all_months.year >= start_year) & (all_months.year <= end_year)]

    # Get indicator values
    indicator_m = indicator["composite"].reindex(months).ffill().bfill()

    # Proportional distribution with annual constraint
    result_values = []
    result_dates = []

    for year in years:
        year_months = months[months.year == year]
        year_indicator = indicator_m.loc[year_months]

        # Annual value
        annual_date = pd.Timestamp(f"{year}-01-01")
        if annual_date in annual_series.index:
            annual_val = annual_series[annual_date]
        else:
            annual_val = annual_series.iloc[(annual_series.index.year == year).argmax()]

        # Proportional distribution
        if year_indicator.sum() > 0:
            monthly_vals = (year_indicator / year_indicator.sum()) * annual_val * len(year_months)
        else:
            monthly_vals = pd.Series([annual_val] * len(year_months), index=year_months)

        result_values.extend(monthly_vals.values)
        result_dates.extend(year_months)

    # Create result DataFrame
    result = pd.DataFrame({
        "date": result_dates,
        "poverty_rate_monthly": result_values
    })

    # Apply 3-month moving average smoothing
    result["poverty_rate_smooth"] = (
        result["poverty_rate_monthly"]
        .rolling(window=smoothing, center=True, min_periods=1)
        .mean()
    )

    return result
