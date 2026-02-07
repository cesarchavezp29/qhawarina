"""Missing data imputation and ragged-edge diagnostics.

Methodology Documentation
=========================
Economic time series often have ragged edges: some series are published faster
than others, creating a staggered pattern of missing values at the end of the
panel. Additionally, historical gaps may occur due to data revisions or
temporary discontinuations.

This module provides:

1. **interpolate_gaps** — Linear interpolation for short internal gaps (<=3
   months by default). Longer gaps are preserved as NaN to avoid introducing
   spurious trends. Edge NaNs (leading/trailing) are never interpolated.

2. **diagnose_ragged_edge** — Reports the last available date for each series,
   identifying which series are lagging and by how much. This is critical for
   nowcasting: the ragged edge determines which observations are actually
   available at each forecast origin.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.missing")


def interpolate_gaps(
    values: pd.Series,
    method: str = "linear",
    max_gap: int = 3,
) -> pd.Series:
    """Interpolate short internal gaps in a time series.

    Only fills gaps of length <= max_gap. Leading and trailing NaN values
    are never filled. This prevents extrapolation beyond the data range.

    Parameters
    ----------
    values : pd.Series
        Time series with DatetimeIndex. May contain NaN gaps.
    method : str
        Interpolation method ('linear' or 'index'). Default 'linear'.
    max_gap : int
        Maximum consecutive NaN values to interpolate. Gaps longer than
        this are left as NaN. Default 3 (months).

    Returns
    -------
    pd.Series
        Series with short gaps filled, longer gaps and edges preserved.
    """
    if values.isna().sum() == 0:
        return values

    # Identify gap lengths: only interpolate gaps with <= max_gap consecutive NaNs
    is_na = values.isna()
    # Label consecutive NaN groups
    gap_groups = (~is_na).cumsum()
    gap_lengths = is_na.groupby(gap_groups).transform("sum")
    # Mask: only fill where gap length is <= max_gap and it's an internal gap
    fillable = is_na & (gap_lengths <= max_gap)

    result = values.copy()
    interpolated = result.interpolate(method=method, limit_area="inside")
    # Only apply interpolation where gap is short enough
    result[fillable] = interpolated[fillable]

    n_filled = values.isna().sum() - result.isna().sum()
    if n_filled > 0:
        logger.debug("Interpolated %d missing values (max_gap=%d)", n_filled, max_gap)

    return result


def diagnose_ragged_edge(panel_wide: pd.DataFrame) -> pd.DataFrame:
    """Diagnose the ragged edge of a wide-format panel.

    For each column (series), reports:
    - last_date: the last non-NaN observation date
    - n_obs: total non-NaN observations
    - n_missing: total NaN observations
    - lag_months: how many months behind the most recent series

    Parameters
    ----------
    panel_wide : pd.DataFrame
        Wide-format panel with DatetimeIndex rows and series-name columns.

    Returns
    -------
    pd.DataFrame
        Diagnostic table sorted by last_date (most lagged first).
    """
    diagnostics = []
    for col in panel_wide.columns:
        series = panel_wide[col].dropna()
        if series.empty:
            diagnostics.append({
                "series": col,
                "last_date": pd.NaT,
                "first_date": pd.NaT,
                "n_obs": 0,
                "n_missing": len(panel_wide),
                "pct_missing": 100.0,
            })
        else:
            n_total = len(panel_wide.loc[series.index[0]:series.index[-1]])
            n_obs = len(series)
            diagnostics.append({
                "series": col,
                "last_date": series.index[-1],
                "first_date": series.index[0],
                "n_obs": n_obs,
                "n_missing": n_total - n_obs,
                "pct_missing": round((n_total - n_obs) / max(n_total, 1) * 100, 1),
            })

    diag_df = pd.DataFrame(diagnostics)
    if diag_df.empty:
        return diag_df

    # Compute lag relative to the most recent date across all series
    most_recent = diag_df["last_date"].max()
    if pd.notna(most_recent):
        diag_df["lag_months"] = diag_df["last_date"].apply(
            lambda d: ((most_recent.year - d.year) * 12 + most_recent.month - d.month)
            if pd.notna(d) else np.nan
        )
    else:
        diag_df["lag_months"] = np.nan

    diag_df = diag_df.sort_values("last_date", ascending=True, na_position="first")
    return diag_df
