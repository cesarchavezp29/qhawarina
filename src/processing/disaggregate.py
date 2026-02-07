"""Temporal disaggregation: quarterly to monthly using Chow-Lin method.

Methodology Documentation
=========================
The Chow-Lin method (Chow & Lin 1971) converts low-frequency (quarterly) data
to high-frequency (monthly) estimates using related high-frequency indicators.

**Mathematical framework:**

Given quarterly GDP values y_q (T×1) and monthly indicator matrix X_m (3T×1),
the model assumes a monthly GDP series y_m = X_m β + u_m with AR(1) errors.

The GLS estimator:
    β_hat = (X' Ω⁻¹ X)⁻¹ X' Ω⁻¹ C' (C Ω⁻¹ C')⁻¹ y_q

where C is the 3T×T aggregation matrix (summing each 3-month block) and
Ω is the AR(1) covariance matrix parameterized by ρ.

The parameter ρ is estimated by maximizing the log-likelihood of the
aggregated residuals.

**Application in NEXUS:**

We disaggregate the quarterly GDP index (PN02516AQ, 2007=100) from ~87
quarterly observations to ~260 monthly estimates using three related monthly
indicators:
    1. Electricity production (PD37966AM) — proxy for industrial activity
    2. Cement consumption growth (PD37967GM) — proxy for construction
    3. PBI global index (PN01770AM) — composite activity index

Reference: Chow, G.C. & Lin, A.L. (1971). "Best Linear Unbiased Interpolation,
Distribution, and Extrapolation of Time Series by Related Series."
Review of Economics and Statistics, 53(4), 372-375.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

logger = logging.getLogger("nexus.disaggregate")


def _build_aggregation_matrix(n_months: int) -> np.ndarray:
    """Build the C aggregation matrix that sums monthly to quarterly.

    C is (n_quarters × n_months) with ones in blocks of 3.
    """
    n_quarters = n_months // 3
    C = np.zeros((n_quarters, n_months))
    for q in range(n_quarters):
        C[q, q * 3: q * 3 + 3] = 1.0
    return C


def _ar1_covariance(n: int, rho: float) -> np.ndarray:
    """Build AR(1) covariance matrix Ω for n observations."""
    idx = np.arange(n)
    Omega = rho ** np.abs(idx[:, None] - idx[None, :])
    return Omega / (1 - rho ** 2)


def chow_lin(
    y_quarterly: pd.Series,
    X_monthly: pd.DataFrame,
    method: str = "max_log",
) -> pd.Series:
    """Chow-Lin temporal disaggregation from quarterly to monthly.

    Parameters
    ----------
    y_quarterly : pd.Series
        Quarterly target variable with quarterly DatetimeIndex.
    X_monthly : pd.DataFrame
        Monthly indicator(s) with monthly DatetimeIndex.
    method : str
        Estimation method for ρ. Currently only 'max_log' (ML) supported.

    Returns
    -------
    pd.Series
        Monthly disaggregated series with DatetimeIndex.
    """
    # Align time frames: trim monthly to complete quarters matching y
    y = y_quarterly.dropna().sort_index()
    X = X_monthly.sort_index()

    # Determine overlapping period
    q_start = y.index[0]
    q_end = y.index[-1]

    # Monthly range: start of first quarter to end of last quarter
    m_start = q_start
    # End = last month of last quarter
    m_end = q_end + pd.offsets.MonthBegin(2)

    X = X.loc[m_start:m_end]

    # Ensure complete quarters
    n_months = len(X)
    n_complete = (n_months // 3) * 3
    X = X.iloc[:n_complete]
    n_months = len(X)
    n_quarters = n_months // 3

    # Trim y to match
    y = y.iloc[:n_quarters]
    if len(y) != n_quarters:
        raise ValueError(
            f"Mismatch: {n_quarters} quarters from monthly data, "
            f"but {len(y)} quarterly observations"
        )

    # Add intercept
    X_mat = np.column_stack([np.ones(n_months), X.values])
    y_vec = y.values

    C = _build_aggregation_matrix(n_months)

    # Optimize ρ via profile log-likelihood
    def neg_log_likelihood(rho):
        if abs(rho) >= 0.999:
            return 1e10
        Omega = _ar1_covariance(n_months, rho)
        CXC = C @ Omega @ C.T
        try:
            CXC_inv = np.linalg.inv(CXC)
        except np.linalg.LinAlgError:
            return 1e10
        # Aggregated residuals
        XC = C @ X_mat
        beta_agg = np.linalg.lstsq(XC, y_vec, rcond=None)[0]
        resid = y_vec - XC @ beta_agg
        # Log-likelihood of aggregated model
        sign, logdet = np.linalg.slogdet(CXC)
        if sign <= 0:
            return 1e10
        return 0.5 * (logdet + resid @ CXC_inv @ resid)

    result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 0.99), method="bounded")
    rho_hat = result.x
    logger.info("Chow-Lin: estimated ρ = %.4f", rho_hat)

    # GLS estimation with optimal ρ
    Omega = _ar1_covariance(n_months, rho_hat)
    CXC = C @ Omega @ C.T
    CXC_inv = np.linalg.inv(CXC)

    # β_hat from aggregated GLS
    XC = C @ X_mat
    XCXC_inv = np.linalg.inv(XC.T @ CXC_inv @ XC)
    beta_hat = XCXC_inv @ XC.T @ CXC_inv @ y_vec

    # Monthly distribution
    y_monthly = X_mat @ beta_hat

    # Add Chow-Lin adjustment: distribute quarterly residuals to months
    u_agg = y_vec - C @ y_monthly
    # Distribution matrix: Ω C' (C Ω C')⁻¹
    D = Omega @ C.T @ CXC_inv
    y_monthly += D @ u_agg

    # Verify aggregation constraint
    y_reagg = C @ y_monthly
    max_diff = np.max(np.abs(y_reagg - y_vec))
    if max_diff > 1e-6:
        logger.warning("Chow-Lin aggregation constraint violated: max diff = %.2e", max_diff)
    else:
        logger.info("Chow-Lin aggregation constraint satisfied (max diff = %.2e)", max_diff)

    return pd.Series(y_monthly, index=X.index, name="gdp_monthly")


def disaggregate_gdp(
    raw_df: pd.DataFrame,
    gdp_quarterly_path: str | None = None,
) -> pd.Series:
    """Disaggregate quarterly GDP index to monthly frequency.

    Uses three monthly BCRP indicators as interpolators:
        1. PD37966AM — Electricity production (GWh)
        2. PD37967GM — Cement consumption (var% 12m)
        3. PN01770AM — PBI global index (2007=100)

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw BCRP monthly data with columns [date, series_code, value].
    gdp_quarterly_path : str or None
        Path to quarterly GDP target parquet. If None, uses default location.

    Returns
    -------
    pd.Series
        Monthly GDP index estimate with DatetimeIndex.
    """
    from config.settings import TARGETS_DIR

    # Load quarterly GDP
    if gdp_quarterly_path is None:
        gdp_quarterly_path = TARGETS_DIR / "gdp_quarterly.parquet"
    gdp_q = pd.read_parquet(gdp_quarterly_path)
    y_q = gdp_q.set_index("date")["gdp_index"].sort_index()

    # Build monthly indicator matrix
    indicator_codes = ["PD37966AM", "PD37967GM", "PN01770AM"]
    indicators = {}
    for code in indicator_codes:
        s = (
            raw_df[raw_df["series_code"] == code]
            .dropna(subset=["value"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["value"]
        )
        indicators[code] = s

    X_monthly = pd.DataFrame(indicators)
    X_monthly = X_monthly.dropna()

    if X_monthly.empty:
        raise ValueError("No monthly indicator data available for disaggregation")

    logger.info(
        "Disaggregating GDP: %d quarterly obs, %d monthly indicators (%d obs each)",
        len(y_q), len(indicator_codes), len(X_monthly),
    )

    return chow_lin(y_q, X_monthly)
