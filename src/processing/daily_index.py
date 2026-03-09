"""Daily instability index — EPU-style severity-weighted proportion.

Methodology follows Baker/Bloom/Davis (2016, QJE) and Caldara/Iacoviello
(2022, AER) adapted for LLM-classified articles with severity scores.

INDEX CONSTRUCTION (4 steps):

  Step 1 — Per-source daily proportion
    For each source i on day t:
      SWP_it = sum(severity_j / 3) / total_articles_it
    This is the severity-weighted share of the news agenda devoted to
    political (or economic) instability. Range: 0 to 1.
    Mathematically equivalent to: proportion × mean_intensity.

  Step 2 — Source standardization (fixed baseline)
    Y_it = SWP_it / sigma_i
    where sigma_i = std(SWP_i) over the full sample.
    Ensures no single source dominates due to higher variance.

  Step 3 — Volume-weighted aggregation across sources
    Z_t = sum(w_it * Y_it)
    where w_it = total_articles_it / sum_i(total_articles_it).
    Sources publishing more articles on day t get proportionally more weight.

  Step 4 — Normalize to mean=100
    Index_t = Z_t × (100 / M)
    where M = mean(Z_t) over the baseline period.
    Value of 200 means twice the average instability coverage.

References:
  - Baker, Bloom & Davis (2016) "Measuring Economic Policy Uncertainty", QJE
  - Caldara & Iacoviello (2022) "Measuring Geopolitical Risk", AER
  - Shapiro, Sudhof & Wilson (2022) "Daily News Sentiment Index", J.Econometrics
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.processing.daily_index")


# ---------------------------------------------------------------------------
# Legacy v1 (mean-severity) — kept for backward compatibility with tests
# ---------------------------------------------------------------------------

def build_daily_index(
    articles_df: pd.DataFrame,
    start_date: str = "2025-01-01",
    zscore_window: int = 90,
    min_periods: int = 7,
) -> pd.DataFrame:
    """Build daily index using legacy mean-severity formula.

    Kept for backward compatibility. New code should use build_daily_index_v2.
    """
    if articles_df.empty:
        logger.warning("No articles provided — returning empty daily index")
        return _empty_index()

    df = articles_df.copy()
    df = df[df["article_category"] != "irrelevant"].copy()

    if df.empty:
        logger.warning("All articles classified as irrelevant — returning empty index")
        return _empty_index()

    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.date

    pol_mask = df["article_category"].isin(["political", "both"])
    econ_mask = df["article_category"].isin(["economic", "both"])

    pol = df[pol_mask].copy()
    econ = df[econ_mask].copy()

    pol_daily = (
        pol.groupby("date")
        .agg(political_score=("article_severity", lambda x: (x / 3).mean()),
             n_articles_political=("article_severity", "count"))
        .reset_index()
    )
    econ_daily = (
        econ.groupby("date")
        .agg(economic_score=("article_severity", lambda x: (x / 3).mean()),
             n_articles_economic=("article_severity", "count"))
        .reset_index()
    )
    total_daily = df.groupby("date").size().reset_index(name="n_articles_total")

    all_dates = set()
    if not pol_daily.empty:
        all_dates.update(pol_daily["date"])
    if not econ_daily.empty:
        all_dates.update(econ_daily["date"])
    if not all_dates:
        return _empty_index()

    start = pd.Timestamp(start_date).date()
    end = max(all_dates)
    date_range = pd.date_range(start, end, freq="D")
    idx = pd.DataFrame({"date": date_range.date})
    idx = idx.merge(pol_daily, on="date", how="left")
    idx = idx.merge(econ_daily, on="date", how="left")
    idx = idx.merge(total_daily, on="date", how="left")
    for col in ["political_score", "economic_score"]:
        idx[col] = idx[col].fillna(0.0)
    for col in ["n_articles_political", "n_articles_economic", "n_articles_total"]:
        idx[col] = idx[col].fillna(0).astype(int)

    idx["political_zscore"] = _zscore_rolling(idx["political_score"], zscore_window, min_periods)
    idx["economic_zscore"] = _zscore_rolling(idx["economic_score"], zscore_window, min_periods)
    idx["political_level"] = idx["political_score"].rank(pct=True)
    idx["economic_level"] = idx["economic_score"].rank(pct=True)
    for prefix in ["political", "economic"]:
        zs = idx[f"{prefix}_zscore"]
        zs_min, zs_max = zs.min(), zs.max()
        if pd.notna(zs_min) and pd.notna(zs_max) and zs_max > zs_min:
            zs_norm = (zs - zs_min) / (zs_max - zs_min)
        else:
            zs_norm = pd.Series(0.5, index=idx.index)
        idx[f"{prefix}_v2"] = 0.5 * idx[f"{prefix}_level"] + 0.5 * zs_norm

    idx["date"] = pd.to_datetime(idx["date"])
    return idx[_OUTPUT_COLS].copy()


# ---------------------------------------------------------------------------
# V2: EPU-style severity-weighted proportion index
# ---------------------------------------------------------------------------

def build_daily_index_v2(
    articles_df: pd.DataFrame,
    start_date: str = "2025-07-01",
    smoothing_window: int = 7,
) -> pd.DataFrame:
    """Build daily instability index using EPU-style proportion methodology.

    Parameters
    ----------
    articles_df : DataFrame
        ALL articles (including irrelevant). Must have columns:
        published, source, and EITHER:
          - new dual-score schema: political_score, economic_score (0-10 floats)
          - legacy schema: article_category, article_severity (str/int)
    start_date : str
        First date for the index (should be when coverage is stable).
    smoothing_window : int
        Window for moving average smoothing (days). Use 1 for no smoothing.

    Returns
    -------
    DataFrame with columns:
        date,
        political_swp, political_index, political_smooth,
        economic_swp, economic_index, economic_smooth,
        n_articles_political, n_articles_economic, n_articles_total,
        n_sources
    """
    if articles_df.empty:
        logger.warning("No articles provided — returning empty index")
        return _empty_index_v2()

    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.normalize().dt.tz_localize(None)

    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    if df.empty:
        logger.warning("No articles after start_date %s", start_date)
        return _empty_index_v2()

    # ── Detect schema: dual-score (new) vs category+severity (legacy) ──
    use_dual_scores = (
        "political_score" in df.columns and "economic_score" in df.columns
    )
    if use_dual_scores:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)
        logger.info("Using dual-score schema (political_score / economic_score, 0-10)")
    else:
        logger.info("Using legacy schema (article_category / article_severity)")

    # ── Step 1: Per-source daily severity-weighted proportion ──────────
    # For each source on each day, compute:
    #   SWP = sum(score/10 for relevant articles) / total_articles
    # This captures both the proportion of political coverage AND its intensity.

    sources = sorted(df["source"].unique())
    logger.info("Sources: %s", sources)

    source_daily = {}
    for src in sources:
        src_df = df[df["source"] == src]
        src_by_day = src_df.groupby("date")

        # Total articles per day for this source (denominator)
        total = src_by_day.size().rename("total")

        if use_dual_scores:
            # Political: sum(political_score/100) for articles with political_score > 0
            # Dividing by 100 maps the 0-100 scale to 0-1, same as old severity/3
            pol_arts = src_df[src_df["political_score"] > 0]
            pol_sev = pol_arts.groupby("date")["political_score"].apply(
                lambda x: (x / 100).sum()
            ).rename("pol_sev_sum")

            # Economic: sum(economic_score/100) for articles with economic_score > 0
            econ_arts = src_df[src_df["economic_score"] > 0]
            econ_sev = econ_arts.groupby("date")["economic_score"].apply(
                lambda x: (x / 100).sum()
            ).rename("econ_sev_sum")
        else:
            # Political: sum(severity/3) for political+both articles
            pol_arts = src_df[src_df["article_category"].isin(["political", "both"])]
            pol_sev = pol_arts.groupby("date")["article_severity"].apply(
                lambda x: (x / 3).sum()
            ).rename("pol_sev_sum")

            # Economic: sum(severity/3) for economic+both articles
            econ_arts = src_df[src_df["article_category"].isin(["economic", "both"])]
            econ_sev = econ_arts.groupby("date")["article_severity"].apply(
                lambda x: (x / 3).sum()
            ).rename("econ_sev_sum")

        combined = pd.DataFrame({"total": total}).join(pol_sev).join(econ_sev).fillna(0)
        combined["pol_swp"] = combined["pol_sev_sum"] / combined["total"]
        combined["econ_swp"] = combined["econ_sev_sum"] / combined["total"]
        combined["source"] = src

        source_daily[src] = combined

    # ── Step 2: Source standardization ─────────────────────────────────
    # Divide each source's SWP by its own standard deviation.
    # This ensures a high-variance source doesn't dominate the aggregate.

    for src in sources:
        sd = source_daily[src]
        for dim in ["pol_swp", "econ_swp"]:
            sigma = sd[dim].std()
            if sigma > 0:
                sd[f"{dim}_std"] = sd[dim] / sigma
            else:
                sd[f"{dim}_std"] = 0.0
            logger.info("  %s %s: mean=%.4f sigma=%.4f", src, dim, sd[dim].mean(), sigma)

    # ── Step 3: Volume-weighted aggregation across sources ─────────────
    # On each day, weight each source by its article volume.
    # Z_t = sum(w_it * Y_it) where w_it = total_articles_it / sum(total)

    # Build continuous date range
    all_dates_in_data = set()
    for sd in source_daily.values():
        all_dates_in_data.update(sd.index)
    if not all_dates_in_data:
        return _empty_index_v2()

    end = max(all_dates_in_data)
    date_range = pd.date_range(start, end, freq="D")

    rows = []
    for day in date_range:
        total_vol = 0
        weighted_pol = 0.0
        weighted_econ = 0.0
        raw_pol_swp = 0.0
        raw_econ_swp = 0.0
        n_pol = 0
        n_econ = 0
        n_total = 0
        n_sources_today = 0

        for src in sources:
            sd = source_daily[src]
            if day not in sd.index:
                continue
            row_src = sd.loc[day]
            vol = int(row_src["total"])
            if vol == 0:
                continue

            n_sources_today += 1
            total_vol += vol
            weighted_pol += vol * row_src["pol_swp_std"]
            weighted_econ += vol * row_src["econ_swp_std"]
            raw_pol_swp += vol * row_src["pol_swp"]
            raw_econ_swp += vol * row_src["econ_swp"]

        if total_vol > 0:
            z_pol = weighted_pol / total_vol
            z_econ = weighted_econ / total_vol
            swp_pol = raw_pol_swp / total_vol
            swp_econ = raw_econ_swp / total_vol
        else:
            z_pol = 0.0
            z_econ = 0.0
            swp_pol = 0.0
            swp_econ = 0.0

        # Count relevant articles across all sources
        day_all = df[df["date"] == day]
        if not day_all.empty:
            if use_dual_scores:
                n_pol = int((day_all["political_score"] > 0).sum())
                n_econ = int((day_all["economic_score"] > 0).sum())
            else:
                n_pol = len(day_all[day_all["article_category"].isin(["political", "both"])])
                n_econ = len(day_all[day_all["article_category"].isin(["economic", "both"])])
            n_total = len(day_all)

        rows.append({
            "date": day,
            "political_z": z_pol,
            "economic_z": z_econ,
            "political_swp": swp_pol,
            "economic_swp": swp_econ,
            "n_articles_political": n_pol,
            "n_articles_economic": n_econ,
            "n_articles_total": n_total,
            "n_sources": n_sources_today,
        })

    result = pd.DataFrame(rows)

    # ── Step 4: Normalize to mean=100 ──────────────────────────────────
    # The baseline is the full sample. Index_t = Z_t × (100 / M).
    # A value of 200 means twice the average instability coverage.

    for dim in ["political", "economic"]:
        z_col = f"{dim}_z"
        idx_col = f"{dim}_index"
        smooth_col = f"{dim}_smooth"

        m = result[z_col].mean()
        if m > 0:
            result[idx_col] = result[z_col] * (100.0 / m)
        else:
            result[idx_col] = 0.0

        # Smoothed version (7-day moving average)
        result[smooth_col] = result[idx_col].rolling(
            window=smoothing_window, min_periods=1, center=False,
        ).mean()

    # Output columns
    output_cols = [
        "date",
        "political_swp", "political_index", "political_smooth",
        "economic_swp", "economic_index", "economic_smooth",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources",
    ]
    result = result[output_cols].copy()

    # Log summary
    logger.info(
        "Daily index v2: %d days, %d sources, %d political, %d economic articles",
        len(result), len(sources),
        result["n_articles_political"].sum(),
        result["n_articles_economic"].sum(),
    )
    logger.info(
        "  Political index: mean=%.1f median=%.1f min=%.1f max=%.1f",
        result["political_index"].mean(),
        result["political_index"].median(),
        result["political_index"].min(),
        result["political_index"].max(),
    )
    logger.info(
        "  Economic index:  mean=%.1f median=%.1f min=%.1f max=%.1f",
        result["economic_index"].mean(),
        result["economic_index"].median(),
        result["economic_index"].min(),
        result["economic_index"].max(),
    )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore_rolling(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


_OUTPUT_COLS = [
    "date",
    "political_score", "political_zscore", "political_level", "political_v2",
    "economic_score", "economic_zscore", "economic_level", "economic_v2",
    "n_articles_political", "n_articles_economic", "n_articles_total",
]


def _empty_index() -> pd.DataFrame:
    """Return an empty DataFrame with the v1 schema."""
    return pd.DataFrame(columns=_OUTPUT_COLS)


def _empty_index_v2() -> pd.DataFrame:
    """Return an empty DataFrame with the v2 schema."""
    return pd.DataFrame(columns=[
        "date",
        "political_swp", "political_index", "political_smooth",
        "economic_swp", "economic_index", "economic_smooth",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources",
    ])
