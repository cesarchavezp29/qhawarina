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
# Source credibility weights (Improvement 3)
# ---------------------------------------------------------------------------

SOURCE_ECO_WEIGHTS = {
    # Tier 1 — serious economic sources
    "gestion": 1.0, "elcomercio": 1.0, "andina": 1.0,
    # Tier 2
    "rpp": 0.7, "larepublica": 0.7, "peru21": 0.7, "correo": 0.7,
    "canaln": 0.7, "caretas": 0.7,
    # Tier 3
    "trome": 0.4, "diariouno": 0.4, "elbuho": 0.4, "inforegion": 0.4,
    "larazon": 0.4, "atv": 0.4, "panamericana": 0.4,
}

SOURCE_POL_WEIGHTS = {
    # Tier 1
    "larepublica": 1.0, "rpp": 1.0, "elcomercio": 1.0, "caretas": 1.0, "canaln": 1.0,
    # Tier 2
    "andina": 0.7, "gestion": 0.7, "peru21": 0.7, "correo": 0.7,
    # Tier 3
    "trome": 0.4, "diariouno": 0.4, "elbuho": 0.4, "inforegion": 0.4,
    "larazon": 0.4, "atv": 0.4, "panamericana": 0.4,
}

_DEFAULT_SOURCE_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

USE_CLUSTERING = False      # TF-IDF deduplication: breaks GPR volume signal
USE_TOPK = False            # Top-K per-source filtering: breaks GPR count logic
USE_SOURCE_WEIGHTS = False  # Source credibility tiers: all sources equal in GPR

# Legacy SWP formula (Caldara & Iacoviello SWP adaptation).
# Set True to revert to old per-source Σ(score^α)/N_articles formula.
LEGACY_SWP = False

# ---------------------------------------------------------------------------
# Intensity + Breadth index parameters (new formula)
# ---------------------------------------------------------------------------

BASELINE_WINDOW = 90      # days for rolling baseline (expanding mean during cold start)
BREADTH_THRESHOLD = 20    # minimum score for article to count toward breadth numerator
BETA_POL = 0.5            # breadth exponent for political index
BETA_ECO = 0.3            # breadth exponent for economic index (lower: eco coverage concentrated
                           # in Gestión/ElComercio econ sections → naturally lower breadth)
BETA = 0.5                # legacy alias (kept for call-site compat)
EMA_ALPHA = 0.3           # EMA smoothing: smooth_t = EMA_ALPHA*smooth_{t-1} + (1-EMA_ALPHA)*raw_t
                           # EMA_ALPHA=0.3 → 70% weight on today, 30% on history


# ---------------------------------------------------------------------------
# Event clustering helper (disabled by default — keep for reference)
# ---------------------------------------------------------------------------

def _cluster_articles_by_day(df: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
    """Cluster similar articles within each day using TF-IDF cosine similarity.

    For each day, articles with cosine similarity >= threshold are merged into
    a cluster. The representative article (highest combined score) is kept;
    other articles are dropped. A cluster_size column tracks how many were merged.

    Parameters
    ----------
    df : DataFrame with at least 'date', 'title', 'political_score', 'economic_score'
    threshold : cosine similarity threshold for same-cluster assignment (default 0.55)

    Returns
    -------
    DataFrame with one row per cluster, cluster_size column added.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    if df.empty:
        df = df.copy()
        df["cluster_size"] = 0
        return df

    n_before = len(df)
    result_rows = []

    for day, day_df in df.groupby("date"):
        day_df = day_df.copy()
        n_day = len(day_df)

        if n_day == 1:
            day_df["cluster_size"] = 1
            result_rows.append(day_df)
            continue

        titles = day_df["title"].fillna("").tolist()

        try:
            vec = TfidfVectorizer(min_df=1, max_df=1.0, sublinear_tf=True)
            tfidf = vec.fit_transform(titles)
            sim_matrix = cos_sim(tfidf)
        except Exception:
            # Fallback: treat all as separate clusters
            day_df["cluster_size"] = 1
            result_rows.append(day_df)
            continue

        # Union-Find clustering
        idx_list = list(range(n_day))
        parent = list(idx_list)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n_day):
            for j in range(i + 1, n_day):
                if sim_matrix[i, j] >= threshold:
                    union(i, j)

        # Group by cluster root
        from collections import defaultdict
        clusters = defaultdict(list)
        for i in idx_list:
            clusters[find(i)].append(i)

        # For each cluster, keep representative (highest combined score)
        day_indices = day_df.index.tolist()
        pol_scores = day_df["political_score"].fillna(0).values
        eco_scores = day_df["economic_score"].fillna(0).values

        cluster_rows = []
        for root, members in clusters.items():
            cluster_size = len(members)
            # Pick member with highest combined score
            combined = [pol_scores[m] + eco_scores[m] for m in members]
            best_local_idx = members[int(np.argmax(combined))]
            best_df_idx = day_indices[best_local_idx]

            row = day_df.loc[best_df_idx].copy()
            # Within cluster: take MAX for each score dimension
            row["political_score"] = max(pol_scores[m] for m in members)
            row["economic_score"] = max(eco_scores[m] for m in members)
            row["cluster_size"] = cluster_size
            cluster_rows.append(row)

        if cluster_rows:
            result_rows.append(pd.DataFrame(cluster_rows))

    if not result_rows:
        df = df.copy()
        df["cluster_size"] = 1
        return df

    clustered = pd.concat(result_rows, ignore_index=True)
    n_after = len(clustered)
    logger.info("Event clustering: %d articles → %d clusters", n_before, n_after)
    return clustered


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
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.date

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
# V2: Intensity × Breadth^β index (new formula, LEGACY_SWP=False)
# ---------------------------------------------------------------------------

def build_daily_index_v2(
    articles_df: pd.DataFrame,
    start_date: str = "2025-07-01",
    # Legacy SWP params (used only when LEGACY_SWP=True)
    smoothing_window: int = 7,
    alpha: float = 1.0,
    # Intensity+breadth params
    baseline_window: int = BASELINE_WINDOW,
    breadth_threshold: int = BREADTH_THRESHOLD,
    beta: float = BETA,            # legacy single-beta (overridden by beta_pol/beta_eco)
    beta_pol: float = BETA_POL,    # breadth exponent for political index
    beta_eco: float = BETA_ECO,    # breadth exponent for economic index
    ema_alpha: float = EMA_ALPHA,
) -> pd.DataFrame:
    """Build daily instability index: IRP = INTENSITY_POL × BREADTH_POL^β_pol.

    Formula (Qhawarina final architecture):
        RAW_INTENSITY_t = Σ(score_i)           — sum of all article scores on day t
        BASELINE_t      = rolling_mean(RAW, N) — N-day window, expanding for cold start
        INTENSITY_t     = (RAW_t / BASELINE_t) × 100
        BREADTH_t       = N_articles_above_threshold / N_total
        IRP_t           = INTENSITY_POL_t × BREADTH_POL_t ^ beta_pol
        IRE_t           = INTENSITY_ECO_t × BREADTH_ECO_t ^ beta_eco

    beta_pol and beta_eco are separate because economic coverage in Peruvian media
    is structurally concentrated (Gestión, El Comercio econ sections), making
    BREADTH_ECO naturally lower than BREADTH_POL for comparable crisis events.
    Lower beta_eco (default 0.3) compensates for this structural concentration.

    Set LEGACY_SWP=True to revert to the old per-source SWP formula.

    Parameters
    ----------
    articles_df : DataFrame with published, source, political_score, economic_score
    start_date  : first date for the index
    baseline_window  : rolling window for normalization baseline (default 90 days)
    breadth_threshold : minimum score for breadth numerator (default 20)
    beta_pol    : breadth exponent for IRP (default 0.5)
    beta_eco    : breadth exponent for IRE (default 0.3)
    ema_alpha   : EMA persistence weight (default 0.3 → 70% today, 30% history)
    """
    if articles_df.empty:
        logger.warning("No articles provided — returning empty index")
        return _empty_index_v2()

    if LEGACY_SWP:
        return _build_swp_index(articles_df, start_date, smoothing_window, alpha)

    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)

    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    if df.empty:
        logger.warning("No articles after start_date %s", start_date)
        return _empty_index_v2()

    # Schema detection
    use_dual = "political_score" in df.columns and "economic_score" in df.columns
    if use_dual:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)
        logger.info("Schema: dual-score (political_score / economic_score, 0-100)")
    else:
        logger.info("Schema: legacy — intensity formula not applicable")

    # Build daily aggregates over continuous date range
    end = df["date"].max()
    date_range = pd.date_range(start, end, freq="D")

    rows = []
    for day in date_range:
        day_df = df[df["date"] == day]
        n_total = len(day_df)

        if n_total == 0:
            rows.append({
                "date": day,
                "raw_intensity_pol": 0.0,
                "raw_intensity_eco": 0.0,
                "n_articles_pol_above_threshold": 0,
                "n_articles_eco_above_threshold": 0,
                "n_articles_total": 0,
                "n_articles_political": 0,
                "n_articles_economic": 0,
                "n_sources": 0,
            })
            continue

        if use_dual:
            pol = day_df["political_score"].fillna(0.0)
            eco = day_df["economic_score"].fillna(0.0)
            raw_pol = float(pol.sum())
            raw_eco = float(eco.sum())
            n_pol_above = int((pol >= breadth_threshold).sum())
            n_eco_above = int((eco >= breadth_threshold).sum())
            n_pol = int((pol > 0).sum())
            n_eco = int((eco > 0).sum())
        else:
            raw_pol = raw_eco = 0.0
            n_pol_above = n_eco_above = n_pol = n_eco = 0

        rows.append({
            "date": day,
            "raw_intensity_pol": raw_pol,
            "raw_intensity_eco": raw_eco,
            "n_articles_pol_above_threshold": n_pol_above,
            "n_articles_eco_above_threshold": n_eco_above,
            "n_articles_total": n_total,
            "n_articles_political": n_pol,
            "n_articles_economic": n_eco,
            "n_sources": int(day_df["source"].nunique()),
        })

    result = pd.DataFrame(rows)

    # Rolling baseline (expanding mean during cold start < baseline_window)
    result["baseline_pol"] = (
        result["raw_intensity_pol"]
        .rolling(window=baseline_window, min_periods=1)
        .mean()
    )
    result["baseline_eco"] = (
        result["raw_intensity_eco"]
        .rolling(window=baseline_window, min_periods=1)
        .mean()
    )

    # Normalized intensity: (raw / baseline) × 100
    # 100 = normal day, 200 = twice average, 50 = unusually calm
    result["intensity_pol"] = np.where(
        result["baseline_pol"] > 0,
        result["raw_intensity_pol"] / result["baseline_pol"] * 100.0,
        0.0,
    )
    result["intensity_eco"] = np.where(
        result["baseline_eco"] > 0,
        result["raw_intensity_eco"] / result["baseline_eco"] * 100.0,
        0.0,
    )

    # Breadth: share of articles above threshold
    n_tot = result["n_articles_total"].astype(float)
    result["breadth_pol"] = np.where(
        n_tot > 0,
        result["n_articles_pol_above_threshold"] / n_tot,
        0.0,
    )
    result["breadth_eco"] = np.where(
        n_tot > 0,
        result["n_articles_eco_above_threshold"] / n_tot,
        0.0,
    )

    # Combined index: intensity × breadth^β (separate exponents per dimension)
    result["irp"] = result["intensity_pol"] * (result["breadth_pol"] ** beta_pol)
    result["ire"] = result["intensity_eco"] * (result["breadth_eco"] ** beta_eco)

    # EMA smoothing: smooth_t = ema_alpha × smooth_{t-1} + (1-ema_alpha) × raw_t
    # pandas ewm(alpha=x) puts weight x on the most recent obs.
    # We want today's weight = (1 - ema_alpha), so pass alpha=(1-ema_alpha).
    ewm_alpha = 1.0 - ema_alpha  # weight on today = 0.7 when ema_alpha=0.3
    result["irp_smooth"] = result["irp"].ewm(alpha=ewm_alpha, adjust=False, min_periods=1).mean()
    result["ire_smooth"] = result["ire"].ewm(alpha=ewm_alpha, adjust=False, min_periods=1).mean()

    # Backward-compatible column names for export pipeline
    result["political_index"] = result["irp"]
    result["economic_index"] = result["ire"]
    result["political_smooth"] = result["irp_smooth"]
    result["economic_smooth"] = result["ire_smooth"]
    result["political_swp"] = result["raw_intensity_pol"]   # legacy compat label
    result["economic_swp"] = result["raw_intensity_eco"]    # legacy compat label
    result["low_coverage"] = result["n_articles_total"] < 25

    output_cols = [
        "date",
        # New schema — all intermediate values for debugging / re-analysis
        "raw_intensity_pol", "raw_intensity_eco",
        "baseline_pol", "baseline_eco",
        "intensity_pol", "intensity_eco",
        "breadth_pol", "breadth_eco",
        "irp_smooth", "ire_smooth",
        "n_articles_pol_above_threshold", "n_articles_eco_above_threshold",
        # Backward-compat aliases
        "political_index", "economic_index",
        "political_smooth", "economic_smooth",
        "political_swp", "economic_swp",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources", "low_coverage",
    ]
    result = result[output_cols].copy()

    valid = result[result["n_articles_total"] > 0]
    logger.info(
        "Intensity+Breadth index (beta_pol=%.1f beta_eco=%.1f, threshold=%d, window=%d): %d days, %d articles",
        beta_pol, beta_eco, breadth_threshold, baseline_window,
        len(result), int(result["n_articles_total"].sum()),
    )
    if len(valid) > 0:
        logger.info(
            "  IRP: mean=%.1f median=%.1f min=%.1f max=%.1f",
            valid["political_index"].mean(), valid["political_index"].median(),
            valid["political_index"].min(), valid["political_index"].max(),
        )
        logger.info(
            "  IRE: mean=%.1f median=%.1f min=%.1f max=%.1f",
            valid["economic_index"].mean(), valid["economic_index"].median(),
            valid["economic_index"].min(), valid["economic_index"].max(),
        )

    return result


# ---------------------------------------------------------------------------
# Legacy SWP implementation (gated behind LEGACY_SWP=True)
# ---------------------------------------------------------------------------

def _build_swp_index(
    articles_df: pd.DataFrame,
    start_date: str = "2025-07-01",
    smoothing_window: int = 7,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Legacy SWP formula: index_t = (1/N_sources) × Σ_s[Σ(score^α)/N_s].

    Kept for reference. Active when LEGACY_SWP=True.
    """
    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)

    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    if df.empty:
        return _empty_index_v2()

    use_dual_scores = "political_score" in df.columns and "economic_score" in df.columns
    if use_dual_scores:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)

    sources = sorted(df["source"].unique())
    source_daily: dict = {}
    for src in sources:
        src_df = df[df["source"] == src]
        total = src_df.groupby("date").size().rename("total")
        if use_dual_scores:
            pol_sev = src_df.groupby("date")["political_score"].apply(
                lambda x: (x.fillna(0.0) ** alpha).sum()
            ).rename("pol_sev_sum")
            eco_sev = src_df.groupby("date")["economic_score"].apply(
                lambda x: (x.fillna(0.0) ** alpha).sum()
            ).rename("econ_sev_sum")
        else:
            pol_arts = src_df[src_df["article_category"].isin(["political", "both"])]
            pol_sev = pol_arts.groupby("date")["article_severity"].apply(
                lambda x: (x ** alpha).sum()
            ).rename("pol_sev_sum")
            econ_arts = src_df[src_df["article_category"].isin(["economic", "both"])]
            eco_sev = econ_arts.groupby("date")["article_severity"].apply(
                lambda x: (x ** alpha).sum()
            ).rename("econ_sev_sum")
        combined = pd.DataFrame({"total": total}).join(pol_sev).join(eco_sev).fillna(0)
        combined["pol_swp"] = combined["pol_sev_sum"] / combined["total"]
        combined["econ_swp"] = combined["econ_sev_sum"] / combined["total"]
        source_daily[src] = combined

    all_dates_in_data: set = set()
    for sd in source_daily.values():
        all_dates_in_data.update(sd.index)
    if not all_dates_in_data:
        return _empty_index_v2()

    end = max(all_dates_in_data)
    date_range = pd.date_range(start, end, freq="D")
    rows = []
    for day in date_range:
        total_vol = 0
        sum_pol_swp = sum_eco_swp = 0.0
        raw_pol_swp = raw_eco_swp = 0.0
        n_pol = n_econ = n_total = n_sources_today = 0
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
            sum_pol_swp += row_src["pol_swp"]
            sum_eco_swp += row_src["econ_swp"]
            raw_pol_swp += vol * row_src["pol_swp"]
            raw_eco_swp += vol * row_src["econ_swp"]
        if n_sources_today > 0:
            z_pol = sum_pol_swp / n_sources_today
            z_econ = sum_eco_swp / n_sources_today
            swp_pol = raw_pol_swp / total_vol if total_vol > 0 else 0.0
            swp_econ = raw_eco_swp / total_vol if total_vol > 0 else 0.0
        else:
            z_pol = z_econ = swp_pol = swp_econ = 0.0
        day_all = df[df["date"] == day]
        if not day_all.empty:
            n_total = len(day_all)
            if use_dual_scores:
                n_pol = int((day_all["political_score"] > 0).sum())
                n_econ = int((day_all["economic_score"] > 0).sum())
        rows.append({
            "date": day, "political_z": z_pol, "economic_z": z_econ,
            "political_swp": swp_pol, "economic_swp": swp_econ,
            "n_articles_political": n_pol, "n_articles_economic": n_econ,
            "n_articles_total": n_total, "n_sources": n_sources_today,
        })
    result = pd.DataFrame(rows)
    MIN_ARTICLES_PER_DAY = 25
    for dim in ["political", "economic"]:
        z_col = f"{dim}_z"
        idx_col = f"{dim}_index"
        smooth_col = f"{dim}_smooth"
        m = result[z_col].mean()
        result[idx_col] = result[z_col] * (100.0 / m) if m > 0 else 0.0
        result[smooth_col] = result[idx_col].ewm(span=smoothing_window, adjust=False, min_periods=1).mean()
    result["low_coverage"] = result["n_articles_total"] < MIN_ARTICLES_PER_DAY
    # Add new-schema columns as 0 for parquet compat
    for col in ["raw_intensity_pol", "raw_intensity_eco", "baseline_pol", "baseline_eco",
                "intensity_pol", "intensity_eco", "breadth_pol", "breadth_eco",
                "irp_smooth", "ire_smooth", "n_articles_pol_above_threshold",
                "n_articles_eco_above_threshold"]:
        result[col] = 0.0
    output_cols = [
        "date",
        "raw_intensity_pol", "raw_intensity_eco",
        "baseline_pol", "baseline_eco",
        "intensity_pol", "intensity_eco",
        "breadth_pol", "breadth_eco",
        "irp_smooth", "ire_smooth",
        "n_articles_pol_above_threshold", "n_articles_eco_above_threshold",
        "political_index", "economic_index",
        "political_smooth", "economic_smooth",
        "political_swp", "economic_swp",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources", "low_coverage",
    ]
    return result[output_cols].copy()


# ---------------------------------------------------------------------------
# Diagnostic output — new Intensity × Breadth^β format
# ---------------------------------------------------------------------------

def print_diagnostic(
    articles_df: pd.DataFrame,
    index_df_10: pd.DataFrame,
    index_df_13: pd.DataFrame = None,   # kept for call-site compat, ignored
    date_strs: list = None,
    start_date: str = "2025-07-01",
    breadth_threshold: int = BREADTH_THRESHOLD,
    beta: float = BETA,
    beta_pol: float = BETA_POL,
    beta_eco: float = BETA_ECO,
) -> None:
    """Print diagnostic output for specified dates.

    Shows all index components (intensity, breadth, combined) and compares
    beta=0.3 / 0.5 / 0.7. Uses intermediate columns from index_df_10.

    Parameters
    ----------
    articles_df  : classified articles (political_score, economic_score, source, title)
    index_df_10  : index DataFrame (must contain new-schema columns)
    date_strs    : list of date strings like ["2026-03-14", ..., "2026-03-18"]
    breadth_threshold : threshold used to build the index (for display)
    beta         : beta used to build the index (for display)
    """
    if date_strs is None:
        date_strs = []

    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)
    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    use_dual = "political_score" in df.columns and "economic_score" in df.columns
    if use_dual:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)

    idx = index_df_10.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")

    # Check if new-schema columns are available
    has_new_schema = "intensity_pol" in idx.columns and "breadth_pol" in idx.columns

    for date_str in date_strs:
        day = pd.Timestamp(date_str)
        day_df = df[df["date"] == day]
        n_total = len(day_df)

        if use_dual and n_total > 0:
            pol = day_df["political_score"].fillna(0.0)
            eco = day_df["economic_score"].fillna(0.0)
            n_pol_above = int((pol >= breadth_threshold).sum())
            n_eco_above = int((eco >= breadth_threshold).sum())
            sum_pol = float(pol.sum())
            sum_eco = float(eco.sum())
            n_pol = int((pol > 0).sum())
            n_eco = int((eco > 0).sum())
        else:
            n_pol_above = n_eco_above = sum_pol = sum_eco = n_pol = n_eco = 0

        n_sources = int(day_df["source"].nunique()) if n_total > 0 else 0

        # Retrieve pre-computed components from index
        def _get(col, default=float("nan")):
            if day in idx.index and col in idx.columns:
                return float(idx.loc[day, col])
            return default

        irp = _get("political_index")
        ire = _get("economic_index")
        irp_sm = _get("irp_smooth", _get("political_smooth"))
        ire_sm = _get("ire_smooth", _get("economic_smooth"))
        raw_pol = _get("raw_intensity_pol")
        raw_eco = _get("raw_intensity_eco")
        baseline_pol = _get("baseline_pol")
        baseline_eco = _get("baseline_eco")
        intensity_pol = _get("intensity_pol")
        intensity_eco = _get("intensity_eco")
        breadth_pol = _get("breadth_pol")
        breadth_eco = _get("breadth_eco")

        print(f"\n{'='*66}")
        print(f"=== {day.strftime('%B %d, %Y')} ===")
        print(f"{'='*66}")
        print(f"Total articles: {n_total}")
        print(f"Articles with pol >= threshold({breadth_threshold}): {n_pol_above}"
              f"  |  Articles with eco >= threshold({breadth_threshold}): {n_eco_above}")

        if has_new_schema:
            print(f"\nCOMPONENT 1 -- INTENSITY:")
            print(f"  RAW_INTENSITY_POL: {raw_pol:.0f}  (sum of all pol scores)")
            print(f"  RAW_INTENSITY_ECO: {raw_eco:.0f}  (sum of all eco scores)")
            print(f"  BASELINE_POL (rolling {BASELINE_WINDOW}d): {baseline_pol:.1f}")
            print(f"  BASELINE_ECO (rolling {BASELINE_WINDOW}d): {baseline_eco:.1f}")
            print(f"  INTENSITY_POL (normalized): {intensity_pol:.1f}")
            print(f"  INTENSITY_ECO (normalized): {intensity_eco:.1f}")
            print(f"\nCOMPONENT 2 -- BREADTH:")
            print(f"  BREADTH_POL: {breadth_pol:.3f}  ({n_pol_above}/{n_total})")
            print(f"  BREADTH_ECO: {breadth_eco:.3f}  ({n_eco_above}/{n_total})")
            print(f"\nCOMBINED INDEX (beta_pol={beta_pol}, beta_eco={beta_eco}):")
            print(f"  IRP = {intensity_pol:.1f} x {breadth_pol:.3f}^{beta_pol} = {irp:.1f}")
            print(f"  IRE = {intensity_eco:.1f} x {breadth_eco:.3f}^{beta_eco} = {ire:.1f}")
        else:
            print(f"\n  IRP: {irp:.1f}   IRE: {ire:.1f}  (old schema — rebuild index for full breakdown)")

        print(f"\nSMOOTHED (EMA alpha={EMA_ALPHA}):")
        print(f"  IRP_smooth: {irp_sm:.1f}")
        print(f"  IRE_smooth: {ire_sm:.1f}")

        # Beta sensitivity comparison (computed from stored intermediate values)
        if has_new_schema and not np.isnan(intensity_pol):
            print(f"\nBeta sensitivity (IRP uses beta_pol, IRE uses beta_eco):")
            print(f"  {'bp':>5}  {'be':>5}  {'IRP':>8}  {'IRE':>8}")
            print(f"  {'-'*32}")
            combos = [(0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.7, 0.5), (0.7, 0.7)]
            for bp, be in combos:
                irp_b = intensity_pol * (breadth_pol ** bp) if not np.isnan(breadth_pol) else float("nan")
                ire_b = intensity_eco * (breadth_eco ** be) if not np.isnan(breadth_eco) else float("nan")
                marker = " <-- current" if (abs(bp - beta_pol) < 0.01 and abs(be - beta_eco) < 0.01) else ""
                print(f"  {bp:>5.1f}  {be:>5.1f}  {irp_b:>8.1f}  {ire_b:>8.1f}{marker}")

        if use_dual and n_total > 0:
            # Top 10 political contributors
            pol_nonzero = day_df[day_df["political_score"] > 0].copy()
            pol_nonzero = pol_nonzero.sort_values("political_score", ascending=False).head(10)
            print(f"\nTop 10 political contributors (of {n_pol} with pol>0):")
            for rank, (_, row) in enumerate(pol_nonzero.iterrows(), 1):
                contrib = (row["political_score"] / sum_pol * 100) if sum_pol > 0 else 0
                title = str(row.get("title", ""))[:60]
                print(f"  {rank:2}. [{row['source']}] \"{title}\" pol={row['political_score']:.0f} ({contrib:.1f}%)")

            # Top 10 economic contributors
            eco_nonzero = day_df[day_df["economic_score"] > 0].copy()
            eco_nonzero = eco_nonzero.sort_values("economic_score", ascending=False).head(10)
            print(f"\nTop 10 economic contributors (of {n_eco} with eco>0):")
            for rank, (_, row) in enumerate(eco_nonzero.iterrows(), 1):
                contrib = (row["economic_score"] / sum_eco * 100) if sum_eco > 0 else 0
                title = str(row.get("title", ""))[:60]
                print(f"  {rank:2}. [{row['source']}] \"{title}\" eco={row['economic_score']:.0f} ({contrib:.1f}%)")

            # Key events check
            print(f"\nKey events check:")
            titles_lower = day_df["title"].fillna("").str.lower()

            for event_name, pattern in [
                ("Petroperu",        "petroper[u\u00fa]"),
                ("Voto de confianza", r"voto de confianza"),
                ("Camisea/gas",      r"camisea|gasoducto|gas natural"),
            ]:
                mask = titles_lower.str.contains(pattern, regex=True, na=False)
                ev_df = day_df[mask]
                if len(ev_df) > 0:
                    pol_contrib = ev_df["political_score"].sum()
                    eco_contrib = ev_df["economic_score"].sum()
                    scores_pol = ev_df["political_score"].tolist()[:5]
                    scores_eco = ev_df["economic_score"].tolist()[:5]
                    pct_pol = (pol_contrib / sum_pol * 100) if sum_pol > 0 else 0
                    pct_eco = (eco_contrib / sum_eco * 100) if sum_eco > 0 else 0
                    print(f"  {event_name}: {len(ev_df)} articles, "
                          f"pol scores={[f'{s:.0f}' for s in scores_pol]} (+{pct_pol:.1f}% of raw_pol), "
                          f"eco scores={[f'{s:.0f}' for s in scores_eco]} (+{pct_eco:.1f}% of raw_eco)")
                else:
                    print(f"  {event_name}: 0 articles")


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
        "raw_intensity_pol", "raw_intensity_eco",
        "baseline_pol", "baseline_eco",
        "intensity_pol", "intensity_eco",
        "breadth_pol", "breadth_eco",
        "irp_smooth", "ire_smooth",
        "n_articles_pol_above_threshold", "n_articles_eco_above_threshold",
        "political_index", "economic_index",
        "political_smooth", "economic_smooth",
        "political_swp", "economic_swp",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources", "low_coverage",
    ])
