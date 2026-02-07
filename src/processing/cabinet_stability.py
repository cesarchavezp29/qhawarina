"""Cabinet stability component for the Political Instability Index.

Computes days_since_last_cabinet_change as an inverse stability proxy.
A premier who has served 400 days indicates stability; one who has served
15 days indicates a recent crisis.

The z-score is negated so that higher values = higher INSTABILITY.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.cabinet_stability")


def compute_days_since_change(
    cabinet_timeline: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.Series:
    """For each date, compute days since the most recent cabinet change.

    Parameters
    ----------
    cabinet_timeline : DataFrame
        Must have 'start_date' column (each row = a new premier).
    date_range : DatetimeIndex
        Dates to compute the metric for.

    Returns
    -------
    Series indexed by date_range with days_since_last_change values.
    """
    change_dates = cabinet_timeline["start_date"].sort_values().values
    days_since = pd.Series(index=date_range, dtype=float, name="days_since_change")

    for date in date_range:
        past = change_dates[change_dates <= np.datetime64(date)]
        if len(past) == 0:
            days_since[date] = np.nan
        else:
            last = pd.Timestamp(past[-1])
            days_since[date] = (date - last).days

    return days_since


def zscore_rolling(
    series: pd.Series, window: int = 60, min_periods: int = 12,
) -> pd.Series:
    """Compute rolling z-score with a given window (in periods)."""
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_cabinet_instability(
    cabinet_timeline: pd.DataFrame,
    freq: str = "M",
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Compute cabinet instability index at monthly or weekly frequency.

    Parameters
    ----------
    cabinet_timeline : DataFrame with start_date, end_date, premier_name, president
    freq : "M" for monthly, "W-FRI" for weekly
    zscore_window : rolling window for z-score (in periods of freq)

    Returns
    -------
    DataFrame with columns: date, days_since_change, cabinet_zscore
    """
    # Build date range
    min_date = cabinet_timeline["start_date"].min()
    max_date = cabinet_timeline["end_date"].max()

    if freq == "M":
        date_range = pd.date_range(
            start=min_date.to_period("M").to_timestamp(),
            end=max_date,
            freq="MS",
        )
    elif freq.startswith("W"):
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    # Compute days since last change
    days_since = compute_days_since_change(cabinet_timeline, date_range)

    # Invert and z-score: fewer days since change = higher instability
    # Negate so that z_score > 0 means MORE instability than average
    zscore = -zscore_rolling(days_since, window=zscore_window)

    result = pd.DataFrame({
        "date": date_range,
        "days_since_change": days_since.values,
        "cabinet_zscore": zscore.values,
    })

    logger.info(
        "Cabinet instability (%s): %d periods, zscore range [%.2f, %.2f]",
        freq, len(result),
        result["cabinet_zscore"].min() if not result["cabinet_zscore"].isna().all() else 0,
        result["cabinet_zscore"].max() if not result["cabinet_zscore"].isna().all() else 0,
    )
    return result


def compute_monthly_change_count(
    cabinet_timeline: pd.DataFrame,
) -> pd.Series:
    """Count number of cabinet changes per month.

    Useful as an alternative/additional instability signal.
    """
    changes = cabinet_timeline["start_date"].dt.to_period("M").value_counts()
    changes = changes.sort_index()
    # Reindex to full range
    full_range = pd.period_range(changes.index.min(), changes.index.max(), freq="M")
    changes = changes.reindex(full_range, fill_value=0)
    return changes
