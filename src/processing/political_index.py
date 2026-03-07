"""Political Instability Index: composite aggregation of 4 components.

Components and weights (monthly):
  - Institutional events (NLP severity):  35%
  - Financial (EMBI + FX volatility):     25%
  - Cabinet stability (days since change): 20%
  - Business confidence (PD12912AM):       20%

Weekly index uses 3 components (no business confidence):
  - Events: 43.75%, Financial: 31.25%, Cabinet: 25%

All components are z-scored on a rolling 60-month window before aggregation.

The index includes two composite measures:
  - composite_index: weighted z-score sum (captures relative anomalies)
  - composite_level: percentile rank of raw severity (captures absolute level)
  - composite_v2: 50/50 blend of z-score and level (robust to dampening)

The z-score approach penalizes sustained crises (baseline rises), while the
level approach preserves absolute severity. composite_v2 balances both.
"""

import logging

import numpy as np
import pandas as pd

from src.processing.cabinet_stability import zscore_rolling

logger = logging.getLogger("nexus.political_index")

# Default weights
MONTHLY_WEIGHTS = {
    "events_zscore": 0.35,
    "financial_zscore": 0.25,
    "cabinet_zscore": 0.20,
    "confidence_zscore": 0.20,
}

WEEKLY_WEIGHTS = {
    "events_zscore": 0.4375,
    "financial_zscore": 0.3125,
    "cabinet_zscore": 0.25,
}


def build_monthly_index(
    events_monthly: pd.DataFrame,
    financial_monthly: pd.DataFrame,
    cabinet_monthly: pd.DataFrame,
    confidence_monthly: pd.DataFrame,
    zscore_window: int = 60,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Build the monthly composite political instability index.

    Parameters
    ----------
    events_monthly : DataFrame with date, events_score (max severity per month)
    financial_monthly : DataFrame with date, financial_score (EMBI z-score + FX vol)
    cabinet_monthly : DataFrame with date, cabinet_zscore
    confidence_monthly : DataFrame with date, confidence_score (business expectations)
    zscore_window : rolling window for z-scoring components
    weights : optional dict overriding default weights

    Returns
    -------
    DataFrame with: date, events_zscore, financial_zscore, cabinet_zscore,
                    confidence_zscore, composite_index
    """
    if weights is None:
        weights = MONTHLY_WEIGHTS

    # Merge all components on date
    idx = pd.DataFrame({"date": events_monthly["date"]})

    # Events component
    idx = idx.merge(
        events_monthly[["date", "events_score"]].rename(
            columns={"events_score": "events_raw"}
        ),
        on="date", how="left",
    )

    # Financial component
    if financial_monthly is not None and not financial_monthly.empty:
        idx = idx.merge(
            financial_monthly[["date", "financial_score"]].rename(
                columns={"financial_score": "financial_raw"}
            ),
            on="date", how="left",
        )
    else:
        idx["financial_raw"] = 0.0

    # Cabinet component (already z-scored)
    idx = idx.merge(
        cabinet_monthly[["date", "cabinet_zscore"]],
        on="date", how="left",
    )

    # Business confidence
    if confidence_monthly is not None and not confidence_monthly.empty:
        idx = idx.merge(
            confidence_monthly[["date", "confidence_score"]].rename(
                columns={"confidence_score": "confidence_raw"}
            ),
            on="date", how="left",
        )
    else:
        idx["confidence_raw"] = np.nan

    idx = idx.sort_values("date").reset_index(drop=True)

    # Z-score each raw component (except cabinet, already z-scored)
    # Events use min_periods=3 so z-scores start early (no warmup zeros needed)
    idx["events_zscore"] = zscore_rolling(
        idx["events_raw"].fillna(0), window=zscore_window, min_periods=3,
    )
    idx["financial_zscore"] = zscore_rolling(
        idx["financial_raw"].fillna(0), window=zscore_window,
    )
    # Confidence: INVERT so lower confidence = higher instability
    if idx["confidence_raw"].notna().any():
        idx["confidence_zscore"] = -zscore_rolling(
            idx["confidence_raw"], window=zscore_window
        )
    else:
        idx["confidence_zscore"] = 0.0

    # Composite z-score: weighted sum (sensitive to rolling window dampening)
    idx["composite_index"] = (
        weights.get("events_zscore", 0.35) * idx["events_zscore"].fillna(0)
        + weights.get("financial_zscore", 0.25) * idx["financial_zscore"].fillna(0)
        + weights.get("cabinet_zscore", 0.20) * idx["cabinet_zscore"].fillna(0)
        + weights.get("confidence_zscore", 0.20) * idx["confidence_zscore"].fillna(0)
    )

    # Level: percentile rank of raw events_score (immune to dampening)
    idx["composite_level"] = idx["events_raw"].rank(pct=True)

    # V2: 50/50 blend of z-score (normalized to [0,1]) and level
    ci = idx["composite_index"]
    ci_min, ci_max = ci.min(), ci.max()
    if ci_max > ci_min:
        ci_norm = (ci - ci_min) / (ci_max - ci_min)
    else:
        ci_norm = 0.5
    idx["composite_v2"] = 0.5 * idx["composite_level"] + 0.5 * ci_norm

    output_cols = [
        "date", "events_zscore", "financial_zscore",
        "cabinet_zscore", "confidence_zscore",
        "composite_index", "composite_level", "composite_v2",
    ]
    result = idx[output_cols].copy()

    logger.info(
        "Monthly index: %d months, composite range [%.2f, %.2f]",
        len(result),
        result["composite_index"].min(),
        result["composite_index"].max(),
    )
    return result


def build_weekly_index(
    events_weekly: pd.DataFrame,
    financial_weekly: pd.DataFrame,
    cabinet_weekly: pd.DataFrame,
    zscore_window: int = 260,  # ~5 years in weeks
    weights: dict | None = None,
) -> pd.DataFrame:
    """Build the weekly composite political instability index.

    Uses 3 components (no business confidence, which is monthly).
    Weights renormalized: events 43.75%, financial 31.25%, cabinet 25%.
    """
    if weights is None:
        weights = WEEKLY_WEIGHTS

    idx = pd.DataFrame({"date": events_weekly["date"]})

    idx = idx.merge(
        events_weekly[["date", "events_score"]].rename(
            columns={"events_score": "events_raw"}
        ),
        on="date", how="left",
    )

    if financial_weekly is not None and not financial_weekly.empty:
        idx = idx.merge(
            financial_weekly[["date", "financial_score"]].rename(
                columns={"financial_score": "financial_raw"}
            ),
            on="date", how="left",
        )
    else:
        idx["financial_raw"] = 0.0

    idx = idx.merge(
        cabinet_weekly[["date", "cabinet_zscore"]],
        on="date", how="left",
    )

    idx = idx.sort_values("date").reset_index(drop=True)

    idx["events_zscore"] = zscore_rolling(
        idx["events_raw"].fillna(0), window=zscore_window, min_periods=12,
    )
    idx["financial_zscore"] = zscore_rolling(
        idx["financial_raw"].fillna(0), window=zscore_window,
    )

    idx["weekly_index"] = (
        weights.get("events_zscore", 0.4375) * idx["events_zscore"].fillna(0)
        + weights.get("financial_zscore", 0.3125) * idx["financial_zscore"].fillna(0)
        + weights.get("cabinet_zscore", 0.25) * idx["cabinet_zscore"].fillna(0)
    )

    # Level and V2 for weekly
    idx["weekly_level"] = idx["events_raw"].rank(pct=True)
    wi = idx["weekly_index"]
    wi_min, wi_max = wi.min(), wi.max()
    if wi_max > wi_min:
        wi_norm = (wi - wi_min) / (wi_max - wi_min)
    else:
        wi_norm = 0.5
    idx["weekly_v2"] = 0.5 * idx["weekly_level"] + 0.5 * wi_norm

    output_cols = [
        "date", "events_zscore", "financial_zscore",
        "cabinet_zscore", "weekly_index", "weekly_level", "weekly_v2",
    ]
    result = idx[output_cols].copy()

    logger.info(
        "Weekly index: %d weeks, range [%.2f, %.2f]",
        len(result),
        result["weekly_index"].min(),
        result["weekly_index"].max(),
    )
    return result


def prepare_events_monthly(
    events_df: pd.DataFrame,
    score_col: str = "severity_claude",
    date_range: pd.DatetimeIndex | None = None,
    warmup_months: int = 0,
) -> pd.DataFrame:
    """Aggregate events to monthly weighted severity sum.

    Uses sum(severity/5) per month rather than max, so months with many
    high-severity events (e.g. Dec 2022 self-coup + protests) score higher
    than months with a single high event. The /5 normalization keeps individual
    event contributions in [0.2, 1.0].

    Fills ALL months from (first event - warmup_months) to last event,
    ensuring continuous series for rolling z-score computation.
    """
    df = events_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    # Weighted sum: each event contributes severity/5
    df["_weight"] = df[score_col] / 5.0
    monthly = df.groupby("month")["_weight"].sum().reset_index()
    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.rename(columns={"_weight": "events_score"})

    # Build continuous monthly range with warmup pad
    if date_range is None:
        start = monthly["date"].min() - pd.DateOffset(months=warmup_months)
        date_range = pd.date_range(start, monthly["date"].max(), freq="MS")

    full = pd.DataFrame({"date": date_range})
    monthly = full.merge(monthly[["date", "events_score"]], on="date", how="left")
    monthly["events_score"] = monthly["events_score"].fillna(0)

    return monthly[["date", "events_score"]]


def prepare_events_weekly(
    events_df: pd.DataFrame,
    score_col: str = "severity_claude",
) -> pd.DataFrame:
    """Aggregate events to weekly (Friday) weighted severity sum."""
    df = events_df.copy()
    df["_weight"] = df[score_col] / 5.0
    df = df.set_index("date")
    weekly = df["_weight"].resample("W-FRI").sum().fillna(0).reset_index()
    weekly.columns = ["date", "events_score"]
    return weekly
