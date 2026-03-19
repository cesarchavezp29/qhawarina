"""Build the Daily RSS Instability Index — political + economic.

Fetches Peruvian news RSS feeds, classifies articles with Claude Haiku,
and builds separate daily political and economic instability indices.

Usage:
    python scripts/build_daily_index.py                    # full run
    python scripts/build_daily_index.py --backfill         # fetch historical archives + RSS
    python scripts/build_daily_index.py --skip-fetch       # use cached articles
    python scripts/build_daily_index.py --skip-claude      # use cached classifications
    python scripts/build_daily_index.py --force            # re-classify all articles
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import pandas as pd

from config.settings import (
    PROCESSED_DAILY_DIR,
    RAW_RSS_DIR,
    RSS_FEEDS_PATH,
)
from src.utils.io import save_parquet

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "daily_index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.daily_pipeline")


# ---------------------------------------------------------------------------
# AI-GPR methodology (Iacoviello & Tong 2026): Haiku classification only.
# NO keyword fallback. If Haiku is unavailable → pipeline FAILS.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build Daily RSS Instability Index")
    parser.add_argument(
        "--backfill", action="store_true",
        help="Fetch historical archives (Arc Publishing API) before RSS",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Use cached articles (skip RSS fetching)",
    )
    parser.add_argument(
        "--skip-claude", action="store_true",
        help="Use cached classifications (skip Claude API)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify all articles (ignore existing classifications)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="[LEGACY] Score exponent for SWP formula (ignored unless LEGACY_SWP=True).",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="[LEGACY] Set both beta_pol and beta_eco to the same value.",
    )
    parser.add_argument(
        "--beta-pol", type=float, default=0.5,
        help="Breadth exponent for IRP (default=0.5).",
    )
    parser.add_argument(
        "--beta-eco", type=float, default=0.3,
        help="Breadth exponent for IRE (default=0.3, lower because eco coverage concentrated).",
    )
    parser.add_argument(
        "--breadth-threshold", type=int, default=20,
        help="Minimum score for article to count toward breadth (default=20).",
    )
    parser.add_argument(
        "--baseline-window", type=int, default=90,
        help="Rolling window in days for intensity baseline (default=90).",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
        help="EMA persistence weight (default=0.3 → 70%% weight on today).",
    )
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Print diagnostic output for March 14-18 after building index.",
    )
    args = parser.parse_args()

    t0 = time.time()
    RAW_RSS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = RAW_RSS_DIR / "articles_cache.parquet"
    classified_path = RAW_RSS_DIR / "articles_classified.parquet"
    index_path = PROCESSED_DAILY_DIR / "daily_index.parquet"

    # Load config
    import yaml
    with open(RSS_FEEDS_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rss_config = config["rss"]

    # ── Step 0 (optional): Backfill historical archives ────────────────
    if args.backfill and not args.skip_fetch:
        logger.info("=" * 60)
        logger.info("STEP 0: Backfilling historical archives (Arc Publishing API)")
        from src.ingestion.archive_scraper import backfill_archives
        articles = backfill_archives(
            config_path=RSS_FEEDS_PATH,
            cache_path=cache_path,
            start_date=rss_config["index"]["start_date"],
        )
        logger.info("  After backfill: %d articles in cache", len(articles))

    # ── Step 1: Fetch RSS feeds ───────────────────────────────────────────
    logger.info("=" * 60)
    if args.skip_fetch:
        if cache_path.exists():
            logger.info("STEP 1: Loading cached articles")
            from src.ingestion.rss import load_article_cache
            articles = load_article_cache(cache_path)
        else:
            logger.warning("STEP 1: --skip-fetch but no cache found at %s", cache_path)
            articles = pd.DataFrame(
                columns=["url", "title", "summary", "published", "source", "feed_name", "url_hash"]
            )
    else:
        logger.info("STEP 1: Fetching RSS feeds")
        try:
            from src.ingestion.rss import fetch_and_merge
            articles = fetch_and_merge(RSS_FEEDS_PATH, cache_path)
        except ImportError as e:
            logger.error(
                "feedparser not installed. Install with: pip install feedparser\n%s", e
            )
            return 1

    logger.info("  Total articles: %d", len(articles))

    # ── Source whitelist — keep ONLY Peruvian sources ─────────────────────
    # Prevents foreign sources (Taiwanese, Arabic, Korean, Argentine, etc.)
    # from re-entering the pipeline on each nightly run.
    PERU_SOURCES = {
        "elcomercio", "gestion", "correo", "larepublica", "andina",
        "rpp", "peru21", "trome", "caretas", "atv", "canaln",
        "elbuho", "inforegion", "diariouno", "larazon", "panamericana",
    }
    if "source" in articles.columns:
        n_before = len(articles)
        articles = articles[articles["source"].isin(PERU_SOURCES)].copy()
        n_dropped = n_before - len(articles)
        if n_dropped > 0:
            logger.info("  Source whitelist: dropped %d foreign articles → %d Peru-only", n_dropped, len(articles))
    # ─────────────────────────────────────────────────────────────────────────

    if articles.empty:
        logger.warning("No articles available — writing empty index")
        from src.processing.daily_index import _empty_index
        save_parquet(_empty_index(), index_path)
        return 0

    # ── Step 2: Deduplicate ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Deduplicating articles")
    from src.ingestion.rss import deduplicate_articles
    articles = deduplicate_articles(articles)
    logger.info("  After dedup: %d articles", len(articles))

    # ── Step 2b: Restore existing dual scores ─────────────────────────────
    # Merge in political_score/economic_score from classified file so Step 3
    # only processes truly new (unscored) articles — not all 25k every run.
    if classified_path.exists() and not args.force:
        try:
            existing_cols = ["url_hash", "political_score", "economic_score"]
            existing = pd.read_parquet(classified_path)
            available = [c for c in existing_cols if c in existing.columns]
            if len(available) > 1:
                articles = articles.merge(existing[available], on="url_hash", how="left")
                n_scored = int(articles["political_score"].notna().sum()) if "political_score" in articles.columns else 0
                logger.info("  Restored %d existing dual scores", n_scored)
        except Exception as e:
            logger.warning("Could not restore scores: %s", e)

    # ── Step 3: Dual-classify with Claude Haiku (AI-GPR methodology) ────────
    logger.info("=" * 60)

    if args.skip_claude and classified_path.exists():
        logger.info("STEP 3: Loading cached classifications (--skip-claude)")
        articles = pd.read_parquet(classified_path)
    elif args.skip_claude:
        logger.error("STEP 3: --skip-claude but no classified parquet found. ABORTING.")
        return 1
    else:
        # ── PROTECTION A: Backup before any write ─────────────────────────
        if classified_path.exists():
            import shutil
            backup_path = classified_path.with_suffix(".BACKUP.parquet")
            shutil.copy2(classified_path, backup_path)
            logger.info("  Backup saved: %s", backup_path.name)

        # ── PROTECTION: Haiku-only — no API key = ABORT ───────────────────
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY not set. ABORTING — no fallback allowed.")
            return 1

        # ── PROTECTION B: Append-only — only classify articles with no scores ──
        logger.info("STEP 3: AI-GPR dual classification (Iacoviello & Tong 2026)")

        has_pol = "political_score" in articles.columns
        has_eco = "economic_score" in articles.columns

        if args.force or not (has_pol and has_eco):
            to_classify = articles
            logger.info("  Classifying all %d articles", len(to_classify))
        else:
            needs_mask = articles["political_score"].isna() | articles["economic_score"].isna()
            to_classify = articles[needs_mask]
            n_already = (~needs_mask).sum()
            logger.info("  Already scored: %d  |  New: %d", n_already, len(to_classify))

        if not to_classify.empty:
            from src.nlp.classifier import classify_articles_dual
            classified = None
            try:
                classified = classify_articles_dual(
                    to_classify,
                    batch_size=rss_config["classification"]["batch_size"],
                    model=rss_config["classification"]["model"],
                    delay=rss_config["classification"]["delay"],
                )
            except RuntimeError as e:
                # Classification failed — NEVER fill with 0.0.
                # Filling unscored articles with 0.0 is indistinguishable from
                # genuinely irrelevant articles and corrupts the index.
                # Leave unscored articles as null; the index builder handles nulls
                # by treating them as 0 only at aggregation time, not storage.
                logger.error("Classification FAILED: %s", e)
                logger.error("Parquet NOT updated. Keeping existing data.")
                logger.error(
                    "DO NOT fill with 0.0 — that corrupts append-only detection. "
                    "Re-run when API is available."
                )
                return 1

            if classified is not None:
                # ── PROTECTION C: Sanity check ─────────────────────────────────
                both_zero = (
                    (classified["political_score"].fillna(0) == 0) &
                    (classified["economic_score"].fillna(0) == 0)
                ).mean()
                if both_zero > 0.95 and len(classified) > 50:
                    logger.error(
                        "ABORT: %.0f%% of articles scored 0 on both indices. "
                        "Likely API failure. Parquet NOT updated.", both_zero * 100
                    )
                    return 1

                # Merge new scores back (append-only)
                if len(classified) < len(articles):
                    for col in ["political_score", "economic_score"]:
                        articles.loc[classified.index, col] = pd.to_numeric(
                            classified[col], errors="coerce"
                        ).values
                else:
                    articles = classified

                # Save classified articles
                save_parquet(articles, classified_path)
                logger.info("  Saved %d articles to %s", len(articles), classified_path.name)
        else:
            logger.info("  All articles already scored — skipping API call")

    # ── Step 4: Build daily index ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building daily index")

    from src.processing.daily_index import build_daily_index_v2, print_diagnostic

    # --beta overrides both beta_pol and beta_eco (legacy single-value mode)
    beta_pol = args.beta if args.beta is not None else args.beta_pol
    beta_eco = args.beta if args.beta is not None else args.beta_eco

    logger.info(
        "  Building with beta_pol=%.1f, beta_eco=%.1f, breadth_threshold=%d, baseline_window=%d, ema_alpha=%.2f",
        beta_pol, beta_eco, args.breadth_threshold, args.baseline_window, args.ema_alpha,
    )
    daily_index = build_daily_index_v2(
        articles,
        start_date=rss_config["index"]["start_date"],
        alpha=args.alpha,
        beta_pol=beta_pol,
        beta_eco=beta_eco,
        breadth_threshold=args.breadth_threshold,
        baseline_window=args.baseline_window,
        ema_alpha=args.ema_alpha,
    )

    save_parquet(daily_index, index_path)

    # ── Diagnostic output (spec section 5) ───────────────────────────────────
    if args.diagnostic or args.skip_claude:
        logger.info("Running diagnostic for March 14-18...")
        try:
            print_diagnostic(
                articles_df=articles,
                index_df_10=daily_index,
                date_strs=["2026-03-14", "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18"],
                start_date=rss_config["index"]["start_date"],
                breadth_threshold=args.breadth_threshold,
                beta_pol=beta_pol,
                beta_eco=beta_eco,
            )
        except Exception as e:
            logger.warning("Diagnostic failed: %s", e)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("DAILY RSS INDEX PIPELINE COMPLETE (%.1fs)", elapsed)
    logger.info("=" * 60)
    logger.info("  Articles total:        %d", len(articles))
    if "article_category" in articles.columns:
        n_pol = articles["article_category"].isin(["political", "both"]).sum()
        n_econ = articles["article_category"].isin(["economic", "both"]).sum()
        n_irr = (articles["article_category"] == "irrelevant").sum()
        logger.info("  Political articles:    %d", n_pol)
        logger.info("  Economic articles:     %d", n_econ)
        logger.info("  Irrelevant articles:   %d", n_irr)
    logger.info("  Daily index days:      %d", len(daily_index))
    if "political_index" in daily_index.columns:
        logger.info("  Political index mean:  %.1f", daily_index["political_index"].mean())
        logger.info("  Political index max:   %.1f", daily_index["political_index"].max())
    logger.info("  Output:                %s", index_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
