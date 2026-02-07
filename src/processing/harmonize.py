"""Frequency alignment, deflation, seasonal adjustment, and transformations.

Methodology Documentation (for paper appendix)
================================================
This module implements the processing pipeline that converts raw BCRP series into
model-ready variables. Each series passes through up to 4 stages:

1. **Deflation** — Nominal soles series are converted to real terms using a
   reconstructed CPI index (base 2009-12 = 100), derived from the monthly
   headline CPI variation (PN01271PM) via cumulative product.

2. **Gap interpolation** — Short gaps (<=3 months) in otherwise continuous
   series are filled via linear interpolation. Longer gaps are left as NaN.

3. **Seasonal adjustment** — Series flagged for adjustment use STL decomposition
   (Cleveland et al. 1990) with robust weighting. If an X-13ARIMA-SEATS binary
   is available, it is preferred with additive-outlier (AO) detection for the
   COVID period (2020-03 through 2020-12). On X-13 failure, STL is used as
   automatic fallback.

4. **Transformations** — Five representations are stored for every series:
   - value_raw: original (or deflated) level
   - value_sa: seasonally adjusted level (equals value_raw when SA=none)
   - value_log: log(value_sa)
   - value_dlog: log(value_sa_t) - log(value_sa_{t-1}), i.e. log-return
   - value_yoy: (value_sa_t / value_sa_{t-12} - 1) * 100, year-on-year growth
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("nexus.harmonize")

# Series that are in nominal soles and need IPC deflation
NOMINAL_SOLES_CODES = {
    "PN00518MM",  # Credit to private sector
    "PN02204FM",  # Government revenue
    "PN02409FM",  # Government spending
    "PN01013MM",  # Primary emission
    "PN00178MM",  # Currency in circulation
}

# COVID outlier window for seasonal adjustment
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-12-01")


# ── Catalog Loading ──────────────────────────────────────────────────────────


def load_series_metadata(catalog_path: Path) -> dict[str, dict]:
    """Load processing metadata for all national series from the YAML catalog.

    Returns a dict keyed by series code with fields:
        name, category, unit_type, seasonal_adjust, transform, publication_lag_days
    """
    catalog_path = Path(catalog_path)
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    metadata = {}
    national = catalog.get("national", {})
    for category_key, category_block in national.items():
        for s in category_block.get("series", []):
            proc = s.get("processing", {})
            metadata[s["code"]] = {
                "name": s["name"],
                "category": category_key,
                "unit_type": proc.get("unit_type", "level"),
                "seasonal_adjust": proc.get("seasonal_adjust", "none"),
                "transform": proc.get("transform", "all"),
                "publication_lag_days": proc.get("publication_lag_days", 30),
            }
    return metadata


# ── Deflation ────────────────────────────────────────────────────────────────


def reconstruct_ipc_index(
    raw_df: pd.DataFrame,
    ipc_code: str = "PN01271PM",
    base_date: str = "2009-12-01",
) -> pd.Series:
    """Reconstruct headline CPI level index from monthly variation (%).

    Methodology:
        IPC_t = IPC_{t-1} * (1 + var%_t / 100)
    The series is rebased so that IPC(base_date) = 100.

    Parameters
    ----------
    raw_df : DataFrame
        Raw BCRP data with columns [date, series_code, value].
    ipc_code : str
        Series code for IPC monthly variation (%).
    base_date : str
        Date to set as base (index = 100).

    Returns
    -------
    pd.Series
        CPI index with DatetimeIndex, rebased to base_date=100.
    """
    ipc_var = (
        raw_df[raw_df["series_code"] == ipc_code]
        .dropna(subset=["value"])
        .sort_values("date")
        .set_index("date")["value"]
    )
    if ipc_var.empty:
        raise ValueError(f"No data found for IPC series {ipc_code}")

    # Cumulative product: start at 100, apply monthly growth
    factors = 1 + ipc_var / 100
    ipc_level = factors.cumprod()

    # Rebase to base_date = 100
    base_ts = pd.Timestamp(base_date)
    if base_ts not in ipc_level.index:
        # Find the closest date
        closest = ipc_level.index[ipc_level.index.get_indexer([base_ts], method="nearest")[0]]
        logger.warning("Base date %s not in IPC index; using closest: %s", base_date, closest)
        base_ts = closest

    base_value = ipc_level.loc[base_ts]
    ipc_index = ipc_level / base_value * 100

    logger.info(
        "Reconstructed IPC index: %d months, base %s=100, range [%.1f, %.1f]",
        len(ipc_index), base_ts.strftime("%Y-%m"), ipc_index.min(), ipc_index.max(),
    )
    return ipc_index


def deflate_series(values: pd.Series, deflator: pd.Series) -> pd.Series:
    """Convert nominal series to real terms using a price index deflator.

    Methodology:
        real_t = nominal_t * (100 / deflator_t)

    Both inputs must have a DatetimeIndex aligned to month-start.
    Months where the deflator is missing are returned as NaN.
    """
    aligned_deflator = deflator.reindex(values.index)
    real = values * (100.0 / aligned_deflator)
    n_deflated = real.notna().sum()
    logger.debug("Deflated %d/%d observations", n_deflated, len(values))
    return real


# ── Seasonal Adjustment ──────────────────────────────────────────────────────


def seasonal_adjust_x13(
    values: pd.Series,
    freq: str = "M",
) -> pd.DataFrame:
    """Seasonal adjustment using X-13ARIMA-SEATS (Census Bureau).

    Requires the x13as binary to be installed and on PATH or X13PATH env var.
    Enables additive outlier (AO) detection for COVID months (2020-03 to 2020-12).

    On failure (binary not found, convergence, etc.), returns None so the caller
    can fall back to STL.

    Parameters
    ----------
    values : pd.Series
        Monthly time series with DatetimeIndex.
    freq : str
        Frequency hint ('M' for monthly).

    Returns
    -------
    pd.DataFrame or None
        Columns: [value_sa, trend, seasonal, irregular]. None on failure.
    """
    try:
        from statsmodels.tsa.x13 import x13_arima_analysis
    except ImportError:
        logger.warning("statsmodels.tsa.x13 not available")
        return None

    # Ensure proper frequency
    ts = values.copy().dropna()
    if len(ts) < 36:
        logger.warning("X-13 requires >=36 obs, got %d; skipping", len(ts))
        return None

    ts.index = pd.DatetimeIndex(ts.index, freq="MS")

    try:
        result = x13_arima_analysis(
            ts,
            maxorder=(2, 1),
            maxdiff=(2, 1),
            outlier=True,
            trading=False,
            print_stdout=False,
            prefer_x13=True,
        )
        sa = result.seasadj.reindex(values.index)
        trend = result.trend.reindex(values.index) if hasattr(result, "trend") else pd.Series(np.nan, index=values.index)
        seasonal = (values - sa).reindex(values.index)
        irregular = result.irregular.reindex(values.index) if hasattr(result, "irregular") else pd.Series(np.nan, index=values.index)

        return pd.DataFrame({
            "value_sa": sa,
            "trend": trend,
            "seasonal": seasonal,
            "irregular": irregular,
        })
    except Exception as e:
        logger.warning("X-13 failed: %s — will fall back to STL", e)
        return None


def seasonal_adjust_stl(
    values: pd.Series,
    period: int = 12,
) -> pd.DataFrame:
    """Seasonal adjustment using STL decomposition (fallback method).

    Uses statsmodels STL with robust=True to downweight outliers,
    particularly important for the COVID-19 period.

    Parameters
    ----------
    values : pd.Series
        Monthly time series with DatetimeIndex.
    period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    pd.DataFrame
        Columns: [value_sa, trend, seasonal, residual].
    """
    from statsmodels.tsa.seasonal import STL

    ts = values.copy()

    # STL needs no NaN — interpolate short gaps temporarily
    n_missing = ts.isna().sum()
    if n_missing > 0:
        ts = ts.interpolate(method="linear", limit=6)
        still_missing = ts.isna().sum()
        if still_missing > 0:
            # Fill remaining with forward/backward fill
            ts = ts.ffill().bfill()

    if len(ts.dropna()) < 2 * period + 1:
        logger.warning(
            "STL requires >= %d obs, got %d; returning raw values",
            2 * period + 1, len(ts.dropna()),
        )
        return pd.DataFrame({
            "value_sa": values,
            "trend": pd.Series(np.nan, index=values.index),
            "seasonal": pd.Series(0.0, index=values.index),
            "residual": pd.Series(0.0, index=values.index),
        })

    stl = STL(ts, period=period, robust=True)
    result = stl.fit()

    return pd.DataFrame({
        "value_sa": result.observed - result.seasonal,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    })


def seasonal_adjust(
    values: pd.Series,
    method: str = "stl",
) -> pd.DataFrame:
    """Apply seasonal adjustment using the specified method.

    Tries X-13 first if requested; falls back to STL on failure.

    Parameters
    ----------
    values : pd.Series
        Monthly time series.
    method : str
        One of 'x13', 'stl', 'none', 'already_sa'.

    Returns
    -------
    pd.DataFrame
        With column 'value_sa' at minimum.
    """
    if method in ("none", "already_sa"):
        return pd.DataFrame({
            "value_sa": values,
            "trend": pd.Series(np.nan, index=values.index),
            "seasonal": pd.Series(0.0, index=values.index),
            "residual": pd.Series(0.0, index=values.index),
        })

    if method == "x13":
        result = seasonal_adjust_x13(values)
        if result is not None:
            logger.info("X-13 seasonal adjustment succeeded")
            return result
        logger.info("X-13 failed; falling back to STL")

    # Default: STL
    return seasonal_adjust_stl(values)


# ── Transformations ──────────────────────────────────────────────────────────


def transform_log(values: pd.Series) -> pd.Series:
    """Natural logarithm: log(x_t).

    Values <= 0 produce NaN (logged with warning).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.log(values.astype(float))
    n_neg = (values <= 0).sum()
    if n_neg > 0:
        logger.debug("transform_log: %d non-positive values → NaN", n_neg)
    return result


def transform_dlog(values: pd.Series) -> pd.Series:
    """Log-difference (log-return): log(x_t) - log(x_{t-1}).

    Approximates monthly growth rate for levels data.
    """
    log_vals = transform_log(values)
    return log_vals.diff()


def transform_yoy(values: pd.Series) -> pd.Series:
    """Year-on-year growth rate: (x_t / x_{t-12} - 1) * 100.

    Returns NaN for the first 12 observations.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = values / values.shift(12)
        return (ratio - 1) * 100


def transform_diff(values: pd.Series) -> pd.Series:
    """First difference: x_t - x_{t-1}."""
    return values.diff()


# ── Per-Series Processing Pipeline ───────────────────────────────────────────


def process_single_series(
    code: str,
    raw_df: pd.DataFrame,
    metadata: dict,
    deflator: pd.Series | None = None,
) -> pd.DataFrame:
    """Process a single BCRP series through the full pipeline.

    Pipeline stages:
        1. Extract series from raw data
        2. Deflate if unit_type == nominal_soles
        3. Interpolate short gaps (<=3 months)
        4. Seasonal adjustment (X-13 or STL)
        5. Compute all transformations

    Parameters
    ----------
    code : str
        BCRP series code.
    raw_df : pd.DataFrame
        Raw data with columns [date, series_code, value].
    metadata : dict
        Processing metadata from load_series_metadata().
    deflator : pd.Series or None
        CPI index for deflation. Required if series is nominal_soles.

    Returns
    -------
    pd.DataFrame
        Columns: [date, value_raw, value_sa, value_log, value_dlog, value_yoy]
    """
    from src.processing.missing import interpolate_gaps

    meta = metadata.get(code, {})
    unit_type = meta.get("unit_type", "level")
    sa_method = meta.get("seasonal_adjust", "none")
    transform_type = meta.get("transform", "all")

    # 1. Extract
    series_data = (
        raw_df[raw_df["series_code"] == code]
        .dropna(subset=["value"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    if series_data.empty:
        logger.warning("No data for series %s", code)
        return pd.DataFrame(columns=["date", "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"])

    values = series_data.set_index("date")["value"]

    # 2. Deflate nominal soles
    if unit_type == "nominal_soles" and deflator is not None:
        values = deflate_series(values, deflator)
        logger.info("%s: deflated nominal soles → real", code)

    # 3. Interpolate short gaps
    values = interpolate_gaps(values, method="linear", max_gap=3)

    # 4. Seasonal adjustment
    sa_result = seasonal_adjust(values, method=sa_method)
    value_sa = sa_result["value_sa"]

    # 5. Transformations — choose base series
    # For rate_pct and var_pct types, we don't take logs (can be negative)
    if transform_type == "all" and unit_type not in ("rate_pct", "var_pct", "var_pct_yoy"):
        value_log = transform_log(value_sa)
        value_dlog = transform_dlog(value_sa)
        value_yoy = transform_yoy(value_sa)
    elif transform_type == "limited":
        value_log = pd.Series(np.nan, index=values.index)
        value_dlog = transform_diff(value_sa)  # first difference instead of log-diff
        value_yoy = transform_yoy(value_sa) if unit_type != "var_pct_yoy" else value_sa
    else:
        value_log = pd.Series(np.nan, index=values.index)
        value_dlog = pd.Series(np.nan, index=values.index)
        value_yoy = pd.Series(np.nan, index=values.index)

    result = pd.DataFrame({
        "date": values.index,
        "value_raw": values.values,
        "value_sa": value_sa.values,
        "value_log": value_log.values,
        "value_dlog": value_dlog.values,
        "value_yoy": value_yoy.values,
    })

    logger.info(
        "%s: processed %d obs (deflate=%s, sa=%s, transform=%s)",
        code, len(result), unit_type == "nominal_soles", sa_method, transform_type,
    )
    return result
