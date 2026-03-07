"""Financial stress component for the Political Instability Index.

Computes a composite financial stress score from:
  1. FX volatility: rolling std of monthly PEN/USD returns (PN01246PM)
  2. Credit spread proxy: TAMN - TIPMN lending-deposit spread (PN07807NM - PN07816NM)
  3. Reserves drawdown: month-over-month change in international reserves (PN00027MM)

Each sub-component is z-scored on a rolling window, then averaged.
Higher values = greater financial stress = higher instability.
"""

import logging

import numpy as np
import pandas as pd

from src.processing.cabinet_stability import zscore_rolling

logger = logging.getLogger("nexus.financial_component")

# Series used for financial stress
FX_SERIES = "PN01246PM"          # PEN/USD exchange rate (level)
TAMN_SERIES = "PN07807NM"       # Active lending rate MN (%)
TIPMN_SERIES = "PN07816NM"      # Passive deposit rate MN (%)
RESERVES_SERIES = "PN00027MM"   # Net international reserves (millions USD)


def build_financial_score(
    panel_long: pd.DataFrame,
    freq: str = "M",
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Build the financial stress score from panel data.

    Parameters
    ----------
    panel_long : DataFrame
        Long-format national panel (date, series_id, value_raw, ...).
    freq : str
        'M' for monthly, 'W' for weekly (weekly uses last available monthly value).
    zscore_window : int
        Rolling window for z-scoring (in periods).

    Returns
    -------
    DataFrame with columns: date, financial_score
    """
    # Pivot relevant series to wide format
    relevant = [FX_SERIES, TAMN_SERIES, TIPMN_SERIES, RESERVES_SERIES]
    subset = panel_long[panel_long["series_id"].isin(relevant)].copy()

    if subset.empty:
        logger.warning("No financial series found in panel")
        return pd.DataFrame(columns=["date", "financial_score"])

    wide = subset.pivot_table(
        index="date", columns="series_id", values="value_raw", aggfunc="first"
    ).sort_index()

    components = []

    # 1. FX volatility: absolute monthly return of PEN/USD
    if FX_SERIES in wide.columns:
        fx = wide[FX_SERIES].dropna()
        fx_return = fx.pct_change().abs()  # absolute return = volatility proxy
        fx_z = zscore_rolling(fx_return, window=zscore_window, min_periods=12)
        components.append(fx_z.rename("fx_vol_z"))
        logger.info("FX volatility: %d obs", fx_z.notna().sum())

    # 2. Credit spread: TAMN - TIPMN (wider spread = stress)
    if TAMN_SERIES in wide.columns and TIPMN_SERIES in wide.columns:
        spread = wide[TAMN_SERIES] - wide[TIPMN_SERIES]
        spread_z = zscore_rolling(spread, window=zscore_window, min_periods=12)
        components.append(spread_z.rename("spread_z"))
        logger.info("Credit spread: %d obs", spread_z.notna().sum())

    # 3. Reserves drawdown: negative MoM change = stress (negate so drawdown > 0)
    if RESERVES_SERIES in wide.columns:
        reserves = wide[RESERVES_SERIES].dropna()
        reserves_change = reserves.pct_change()
        # Negate: reserve decline = positive stress signal
        reserves_stress = -reserves_change
        reserves_z = zscore_rolling(reserves_stress, window=zscore_window, min_periods=12)
        components.append(reserves_z.rename("reserves_z"))
        logger.info("Reserves stress: %d obs", reserves_z.notna().sum())

    if not components:
        logger.warning("No financial components could be computed")
        return pd.DataFrame(columns=["date", "financial_score"])

    # Combine: equal-weighted average of available z-scores
    combined = pd.concat(components, axis=1)
    financial_score = combined.mean(axis=1)  # nanmean across columns

    result = pd.DataFrame({
        "date": financial_score.index,
        "financial_score": financial_score.values,
    }).reset_index(drop=True)

    # Fill NaN at start with 0 (no signal = neutral)
    result["financial_score"] = result["financial_score"].fillna(0.0)

    logger.info(
        "Financial score: %d months, range [%.2f, %.2f]",
        len(result), result["financial_score"].min(), result["financial_score"].max(),
    )
    return result


def build_confidence_score(
    panel_long: pd.DataFrame,
    series_id: str = "PD37981AM",
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Build the business confidence component from panel data.

    Uses PD37981AM (business expectations index, 12-month ahead).
    Higher expectations = lower instability, so the score is INVERTED
    in the composite index builder (political_index.py).

    Parameters
    ----------
    panel_long : DataFrame
        Long-format national panel.
    series_id : str
        Panel series ID for business confidence.
    zscore_window : int
        Rolling z-score window.

    Returns
    -------
    DataFrame with columns: date, confidence_score
    """
    subset = panel_long[panel_long["series_id"] == series_id].copy()

    if subset.empty:
        logger.warning("Business confidence series %s not found in panel", series_id)
        return pd.DataFrame(columns=["date", "confidence_score"])

    subset = subset.sort_values("date").drop_duplicates("date", keep="last")
    confidence = subset.set_index("date")["value_raw"]

    result = pd.DataFrame({
        "date": confidence.index,
        "confidence_score": confidence.values,
    }).reset_index(drop=True)

    logger.info(
        "Confidence score (%s): %d months, range [%.1f, %.1f]",
        series_id, len(result),
        result["confidence_score"].min(), result["confidence_score"].max(),
    )
    return result
