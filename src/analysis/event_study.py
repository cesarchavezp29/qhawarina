"""Financial event studies for high-impact political events.

Computes EMBI spread and FX volatility changes in windows around
the 14 high-impact events from the ground truth dataset.
Tests whether unanticipated events produce larger financial reactions.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("nexus.event_study")


def compute_event_window(
    financial_series: pd.Series,
    event_date: pd.Timestamp,
    window_days: int = 7,
) -> dict:
    """Compute financial metric change around an event.

    Parameters
    ----------
    financial_series : Series indexed by date (daily frequency)
    event_date : date of the event
    window_days : days before/after for the window

    Returns
    -------
    dict with pre_mean, post_mean, change, abs_change, max_post
    """
    start = event_date - pd.Timedelta(days=window_days)
    end = event_date + pd.Timedelta(days=window_days)

    pre = financial_series[
        (financial_series.index >= start) & (financial_series.index < event_date)
    ]
    post = financial_series[
        (financial_series.index >= event_date) & (financial_series.index <= end)
    ]

    if pre.empty or post.empty:
        return {
            "pre_mean": np.nan,
            "post_mean": np.nan,
            "change": np.nan,
            "abs_change": np.nan,
            "max_post": np.nan,
        }

    pre_mean = pre.mean()
    post_mean = post.mean()

    return {
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "change": post_mean - pre_mean,
        "abs_change": abs(post_mean - pre_mean),
        "max_post": post.max(),
    }


def run_event_studies(
    high_impact_events: pd.DataFrame,
    embi_daily: pd.Series | None = None,
    fx_daily: pd.Series | None = None,
    window_days: int = 7,
) -> pd.DataFrame:
    """Run event studies for all high-impact events.

    Parameters
    ----------
    high_impact_events : DataFrame from GT with event_id, date, anticipated
    embi_daily : Series of daily EMBI spread, indexed by date
    fx_daily : Series of daily FX rate (PEN/USD), indexed by date

    Returns
    -------
    DataFrame with event-level results: event_id, date, anticipated,
    embi_change, fx_change, etc.
    """
    results = []

    for _, event in high_impact_events.iterrows():
        event_date = pd.Timestamp(event["date"])
        row = {
            "event_id": event["event_id"],
            "date": event_date,
            "event_description": event.get("event_description", ""),
            "anticipated": event.get("anticipated", np.nan),
        }

        if embi_daily is not None and not embi_daily.empty:
            embi_result = compute_event_window(embi_daily, event_date, window_days)
            row["embi_change"] = embi_result["change"]
            row["embi_abs_change"] = embi_result["abs_change"]
            row["embi_max_post"] = embi_result["max_post"]
        else:
            row["embi_change"] = np.nan

        if fx_daily is not None and not fx_daily.empty:
            fx_result = compute_event_window(fx_daily, event_date, window_days)
            row["fx_change"] = fx_result["change"]
            row["fx_abs_change"] = fx_result["abs_change"]
        else:
            row["fx_change"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    logger.info("Event studies: %d events analyzed", len(df))
    return df


def test_anticipated_effect(event_studies: pd.DataFrame) -> dict:
    """Test whether unanticipated events produce larger financial reactions.

    Uses t-test comparing abs_change for anticipated=0 vs anticipated=1.
    """
    df = event_studies.dropna(subset=["anticipated", "embi_change"])
    if df.empty:
        return {"test_possible": False}

    surprise = df[df["anticipated"] == 0]["embi_abs_change"].dropna()
    expected = df[df["anticipated"] == 1]["embi_abs_change"].dropna()

    if len(surprise) < 2 or len(expected) < 2:
        return {
            "test_possible": False,
            "n_surprise": len(surprise),
            "n_expected": len(expected),
        }

    t_stat, p_value = stats.ttest_ind(surprise, expected, equal_var=False)

    return {
        "test_possible": True,
        "n_surprise": len(surprise),
        "n_expected": len(expected),
        "mean_surprise": round(surprise.mean(), 4),
        "mean_expected": round(expected.mean(), 4),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "surprise_larger": surprise.mean() > expected.mean(),
    }
