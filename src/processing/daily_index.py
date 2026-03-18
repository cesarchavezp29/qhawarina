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
# Feature flags — set False to follow GPR methodology (Caldara & Iacoviello)
# ---------------------------------------------------------------------------

USE_CLUSTERING = False      # TF-IDF deduplication: breaks GPR volume signal
USE_TOPK = False            # Top-K per-source filtering: breaks GPR count logic
USE_SOURCE_WEIGHTS = False  # Source credibility tiers: all sources equal in GPR


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
# V2: EPU-style severity-weighted proportion index
# ---------------------------------------------------------------------------

def build_daily_index_v2(
    articles_df: pd.DataFrame,
    start_date: str = "2025-07-01",
    smoothing_window: int = 7,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Build daily instability index using GPR methodology.

    Caldara & Iacoviello (2022, AER) adapted for AI-scored articles.

    Formula:
        index_t = (1/N_sources) × Σ_over_sources [ Σ(score_i^alpha) / N_articles_s ]

    Media coverage volume IS the signal:
    - No deduplication / clustering
    - No top-K filtering
    - All sources weighted equally
    - Zero-score articles remain in denominator (0^alpha=0, contribute nothing to numerator)

    Parameters
    ----------
    articles_df : DataFrame
        ALL articles (including irrelevant). Must have columns:
        published, source, and EITHER:
          - dual-score schema: political_score, economic_score (0-100 floats)
          - legacy schema: article_category, article_severity (str/int)
    start_date : str
        First date for the index (should be when coverage is stable).
    smoothing_window : int
        EWM span for smoothing (days). Use 1 for no smoothing.
    alpha : float
        Convex transformation exponent applied to each score.
        alpha=1.0 = linear (GPR baseline, default).
        alpha=1.3 amplifies high-severity events nonlinearly:
          score=80 → 80^1.3 ≈ 246  vs  score=10 → 10^1.3 ≈ 20.
        Configurable for sensitivity analysis — change here or pass explicitly.

    Returns
    -------
    DataFrame with columns:
        date,
        political_swp, political_index, political_smooth,
        economic_swp, economic_index, economic_smooth,
        n_articles_political, n_articles_economic, n_articles_total,
        n_sources, low_coverage
    """
    if articles_df.empty:
        logger.warning("No articles provided — returning empty index")
        return _empty_index_v2()

    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)

    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    if df.empty:
        logger.warning("No articles after start_date %s", start_date)
        return _empty_index_v2()

    # ── Schema detection ────────────────────────────────────────────────────
    use_dual_scores = (
        "political_score" in df.columns and "economic_score" in df.columns
    )
    if use_dual_scores:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)
        logger.info("Schema: dual-score (political_score / economic_score, 0-100)")
    else:
        logger.info("Schema: legacy (article_category / article_severity)")

    # ── Optional: Event clustering (USE_CLUSTERING=False by default) ────
    # Disabled: clustering breaks GPR volume signal. If 12 newspapers cover
    # Petroperú they should count as 12 articles, not 1 cluster.
    if USE_CLUSTERING and use_dual_scores and "title" in df.columns:
        df = _cluster_articles_by_day(df, threshold=0.55)
        logger.warning("USE_CLUSTERING=True: clustering applied — this breaks GPR volume logic")
    else:
        df["cluster_size"] = 1

    # ── Step 1: Per-source daily SWP = Σ(score^alpha) / N_articles ─────
    # GPR formula: all articles from source s on day t contribute score^alpha
    # to the numerator. Zero-score articles: 0^alpha=0, contribute nothing
    # to numerator but remain in denominator (N_articles). This is correct —
    # they are "non-matches" in GPR terminology.

    sources = sorted(df["source"].unique())
    logger.info("Sources (%d): %s", len(sources), sources)

    source_daily: dict = {}
    for src in sources:
        src_df = df[df["source"] == src]

        # Denominator: total articles per day from this source
        total = src_df.groupby("date").size().rename("total")

        if use_dual_scores:
            if USE_TOPK:
                # TOP-K DISABLED by default (USE_TOPK=False).
                # Preserved here for comparison / sensitivity analysis only.
                # Selecting only top K articles per source per day suppresses the
                # volume signal that is the entire point of GPR methodology.
                src_eco_w = SOURCE_ECO_WEIGHTS.get(src, _DEFAULT_SOURCE_WEIGHT) if USE_SOURCE_WEIGHTS else 1.0
                src_pol_w = SOURCE_POL_WEIGHTS.get(src, _DEFAULT_SOURCE_WEIGHT) if USE_SOURCE_WEIGHTS else 1.0
                pol_sev_by_day: dict = {}
                eco_sev_by_day: dict = {}
                for day, day_src_df in src_df.groupby("date"):
                    n_events = len(day_src_df)
                    pol_positive = day_src_df[day_src_df["political_score"] > 0].copy()
                    if not pol_positive.empty:
                        k_pol = min(20, max(5, n_events // 10))
                        pol_positive["_weighted_pol"] = pol_positive["political_score"] * src_pol_w
                        top_pol = pol_positive.nlargest(k_pol, "_weighted_pol")
                        pol_sev_by_day[day] = (top_pol["political_score"] ** alpha).sum()
                    else:
                        pol_sev_by_day[day] = 0.0
                    eco_positive = day_src_df[day_src_df["economic_score"] > 0].copy()
                    if not eco_positive.empty:
                        k_eco = min(20, max(5, n_events // 10))
                        eco_positive["_weighted_eco"] = eco_positive["economic_score"] * src_eco_w
                        top_eco = eco_positive.nlargest(k_eco, "_weighted_eco")
                        eco_sev_by_day[day] = (top_eco["economic_score"] ** alpha).sum()
                    else:
                        eco_sev_by_day[day] = 0.0
                pol_sev = pd.Series(pol_sev_by_day, name="pol_sev_sum")
                eco_sev = pd.Series(eco_sev_by_day, name="econ_sev_sum")
            else:
                # GPR (default): Σ(score^alpha) over ALL articles from this source.
                # score=0 → 0^alpha=0, contributes 0 to numerator.
                # score=80, alpha=1.3 → 80^1.3 ≈ 246 (high-severity amplified).
                pol_sev = src_df.groupby("date")["political_score"].apply(
                    lambda x: (x.fillna(0.0) ** alpha).sum()
                ).rename("pol_sev_sum")
                eco_sev = src_df.groupby("date")["economic_score"].apply(
                    lambda x: (x.fillna(0.0) ** alpha).sum()
                ).rename("econ_sev_sum")
        else:
            # Legacy schema: article_category + article_severity
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
        combined["source"] = src
        source_daily[src] = combined

    # ── Step 2: Source standardization — DISABLED (GPR uses raw SWPs) ──
    # In GPR methodology, sources contribute their raw SWP equally.
    # Standardization (dividing by source-level sigma) distorts the index:
    # it removes the absolute level information and makes a quiet day at
    # a normally-noisy source look equivalent to a crisis at a quiet source.
    # (Code preserved as comment for reference.)

    # ── Step 3: Equal-weight aggregation across sources ─────────────────
    # GPR formula: Z_t = (1/N_sources) × Σ_i(SWP_it)
    # Each source present on day t contributes its raw SWP equally.
    # This prevents high-volume sources from dominating (a source publishing
    # 168 articles vs one publishing 10 has the same index weight).

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
        sum_pol_swp = 0.0
        sum_eco_swp = 0.0
        raw_pol_swp = 0.0
        raw_eco_swp = 0.0
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
            sum_pol_swp += row_src["pol_swp"]
            sum_eco_swp += row_src["econ_swp"]
            raw_pol_swp += vol * row_src["pol_swp"]
            raw_eco_swp += vol * row_src["econ_swp"]

        if n_sources_today > 0:
            z_pol = sum_pol_swp / n_sources_today   # GPR equal-weight average
            z_econ = sum_eco_swp / n_sources_today
            swp_pol = raw_pol_swp / total_vol if total_vol > 0 else 0.0
            swp_econ = raw_eco_swp / total_vol if total_vol > 0 else 0.0
        else:
            z_pol = 0.0
            z_econ = 0.0
            swp_pol = 0.0
            swp_econ = 0.0

        # Count relevant articles
        day_all = df[df["date"] == day]
        if not day_all.empty:
            n_total = len(day_all)
            if use_dual_scores:
                n_pol = int((day_all["political_score"] > 0).sum())
                n_econ = int((day_all["economic_score"] > 0).sum())
            else:
                n_pol = len(day_all[day_all["article_category"].isin(["political", "both"])])
                n_econ = len(day_all[day_all["article_category"].isin(["economic", "both"])])

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

    # ── Step 3b: Flag and interpolate low-coverage days ─────────────────
    # Days with fewer than MIN_ARTICLES_PER_DAY total articles are noise
    # (typically weekends/holidays with sparse feeds). Interpolate their
    # Z scores from adjacent valid days so they don't spike the index.
    MIN_ARTICLES_PER_DAY = 25
    low_cov = result["n_articles_total"] < MIN_ARTICLES_PER_DAY
    n_low = int(low_cov.sum())
    if n_low > 0:
        logger.info(
            "Low-coverage interpolation: %d days with < %d articles → interpolating Z scores",
            n_low, MIN_ARTICLES_PER_DAY,
        )
        for dim in ["political", "economic"]:
            z_col = f"{dim}_z"
            result.loc[low_cov, z_col] = np.nan
        # Linear interpolation (fills gaps between valid days)
        result[["political_z", "economic_z"]] = (
            result[["political_z", "economic_z"]]
            .interpolate(method="linear", limit_direction="both")
        )

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

        # EWM smoothing: index_smooth_t = w*index_raw_t + (1-w)*index_smooth_{t-1}
        # ewm(span=7) → w ≈ 0.25, crises persist ~3-4 days.
        result[smooth_col] = result[idx_col].ewm(
            span=smoothing_window, adjust=False, min_periods=1,
        ).mean()

    # Output columns
    result["low_coverage"] = result["n_articles_total"] < MIN_ARTICLES_PER_DAY
    output_cols = [
        "date",
        "political_swp", "political_index", "political_smooth",
        "economic_swp", "economic_index", "economic_smooth",
        "n_articles_political", "n_articles_economic", "n_articles_total",
        "n_sources", "low_coverage",
    ]
    result = result[output_cols].copy()

    # Log summary
    logger.info(
        "GPR index (alpha=%.1f): %d days, %d sources, %d pol articles, %d eco articles",
        alpha, len(result), len(sources),
        result["n_articles_political"].sum(),
        result["n_articles_economic"].sum(),
    )
    logger.info(
        "  IRP: mean=%.1f median=%.1f min=%.1f max=%.1f",
        result["political_index"].mean(),
        result["political_index"].median(),
        result["political_index"].min(),
        result["political_index"].max(),
    )
    logger.info(
        "  IRE: mean=%.1f median=%.1f min=%.1f max=%.1f",
        result["economic_index"].mean(),
        result["economic_index"].median(),
        result["economic_index"].min(),
        result["economic_index"].max(),
    )

    return result


# ---------------------------------------------------------------------------
# Diagnostic output (spec section 5) — prints article-level stats per day
# ---------------------------------------------------------------------------

def print_diagnostic(
    articles_df: pd.DataFrame,
    index_df_10: pd.DataFrame,
    index_df_13: pd.DataFrame,
    date_strs: list[str],
    start_date: str = "2025-07-01",
) -> None:
    """Print diagnostic output for specified dates comparing α=1.0 vs α=1.3.

    Parameters
    ----------
    articles_df : classified articles DataFrame (political_score, economic_score, source, title)
    index_df_10 : index built with alpha=1.0
    index_df_13 : index built with alpha=1.3
    date_strs : list of date strings like ["2026-03-14", ..., "2026-03-18"]
    """
    df = articles_df.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date"] = df["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)
    start = pd.Timestamp(start_date)
    df = df[df["date"] >= start].copy()

    use_dual = "political_score" in df.columns and "economic_score" in df.columns
    if use_dual:
        df["political_score"] = pd.to_numeric(df["political_score"], errors="coerce").fillna(0.0)
        df["economic_score"] = pd.to_numeric(df["economic_score"], errors="coerce").fillna(0.0)

    idx10 = index_df_10.copy()
    idx13 = index_df_13.copy()
    idx10["date"] = pd.to_datetime(idx10["date"])
    idx13["date"] = pd.to_datetime(idx13["date"])
    idx10 = idx10.set_index("date")
    idx13 = idx13.set_index("date")

    for date_str in date_strs:
        day = pd.Timestamp(date_str)
        day_df = df[df["date"] == day]

        n_total = len(day_df)
        if use_dual:
            n_pol = int((day_df["political_score"] > 0).sum())
            n_eco = int((day_df["economic_score"] > 0).sum())
            sum_pol = day_df["political_score"].sum()
            sum_eco = day_df["economic_score"].sum()
        else:
            n_pol = n_eco = sum_pol = sum_eco = 0

        n_sources = int(day_df["source"].nunique()) if n_total > 0 else 0

        # Index values
        irp10 = idx10.loc[day, "political_index"] if day in idx10.index else float("nan")
        ire10 = idx10.loc[day, "economic_index"] if day in idx10.index else float("nan")
        irp13 = idx13.loc[day, "political_index"] if day in idx13.index else float("nan")
        ire13 = idx13.loc[day, "economic_index"] if day in idx13.index else float("nan")
        irp_sm = idx10.loc[day, "political_smooth"] if day in idx10.index else float("nan")
        ire_sm = idx10.loc[day, "economic_smooth"] if day in idx10.index else float("nan")

        print(f"\n{'='*60}")
        print(f"=== {day.strftime('%B %d, %Y')} ===")
        print(f"{'='*60}")
        print(f"Total articles:       {n_total}")
        print(f"Articles with pol > 0: {n_pol}")
        print(f"Articles with eco > 0: {n_eco}")
        print(f"Sum(pol_score):       {sum_pol:.0f}")
        print(f"Sum(eco_score):       {sum_eco:.0f}")
        print(f"N_sources active:     {n_sources}")
        print()
        print(f"IRP (a=1.0): {irp10:6.1f}    IRP (a=1.3): {irp13:6.1f}")
        print(f"IRE (a=1.0): {ire10:6.1f}    IRE (a=1.3): {ire13:6.1f}")
        print()
        print(f"IRP (smoothed a=1.0): {irp_sm:.1f}")
        print(f"IRE (smoothed a=1.0): {ire_sm:.1f}")

        if use_dual and n_total > 0:
            # Top 10 political contributors
            pol_nonzero = day_df[day_df["political_score"] > 0].copy()
            pol_nonzero = pol_nonzero.sort_values("political_score", ascending=False).head(10)
            print(f"\nTop 10 political contributors (of {n_pol}):")
            for rank, (_, row) in enumerate(pol_nonzero.iterrows(), 1):
                contrib = (row["political_score"] / sum_pol * 100) if sum_pol > 0 else 0
                title = str(row.get("title", ""))[:60]
                print(f"  {rank:2}. [{row['source']}] \"{title}\" pol={row['political_score']:.0f} ({contrib:.1f}%)")

            # Top 10 economic contributors
            eco_nonzero = day_df[day_df["economic_score"] > 0].copy()
            eco_nonzero = eco_nonzero.sort_values("economic_score", ascending=False).head(10)
            print(f"\nTop 10 economic contributors (of {n_eco}):")
            for rank, (_, row) in enumerate(eco_nonzero.iterrows(), 1):
                contrib = (row["economic_score"] / sum_eco * 100) if sum_eco > 0 else 0
                title = str(row.get("title", ""))[:60]
                print(f"  {rank:2}. [{row['source']}] \"{title}\" eco={row['economic_score']:.0f} ({contrib:.1f}%)")

            # Petroperú coverage
            title_col = day_df["title"].fillna("").str.lower() if "title" in day_df.columns else None
            if title_col is not None:
                pp_mask = title_col.str.contains(r"petroper[uú]|petroperu", regex=True, na=False)
                pp_df = day_df[pp_mask]
                if len(pp_df) > 0:
                    avg_eco = pp_df["economic_score"].mean()
                    pp_eco_total = pp_df["economic_score"].sum()
                    contrib = (pp_eco_total / sum_eco * 100) if sum_eco > 0 else 0
                    print(f"\nPetroperú coverage: {len(pp_df)} articles, avg eco={avg_eco:.0f}, total contribution={contrib:.1f}%")
                else:
                    print("\nPetroperú coverage: 0 articles")

    print(f"\n{'='*60}")
    print("OLD vs NEW index comparison (last 7 days of existing index):")
    print(f"{'='*60}")


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
