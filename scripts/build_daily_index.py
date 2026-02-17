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
# Keyword-based fallback classifier (no API key required)
# ---------------------------------------------------------------------------
_POLITICAL_KEYWORDS = [
    "congreso", "parlamento", "presidente", "boluarte", "premier", "gabinete",
    "ministro", "gobierno", "partido", "elecciones", "votación", "voto",
    "censura", "interpelación", "moción", "destitución", "renuncia", "golpe",
    "crisis política", "constitución", "referéndum", "fiscal", "corrupción",
    "investigación", "juicio", "tribunal", "poder judicial", "congresista",
    "ejecutivo", "legislativo", "democracia", "protesta", "marcha", "huelga",
    "paro", "bloqueo", "manifestación", "enfrentamiento", "represión",
    "estado de emergencia", "toque de queda", "fuerzas armadas", "policía",
    # English
    "president", "congress", "government", "minister", "cabinet", "election",
    "impeach", "resign", "protest", "strike", "coup", "corruption", "crisis",
]

_ECONOMIC_KEYWORDS = [
    "economía", "inflación", "pbi", "bcrp", "soles", "mef", "presupuesto",
    "inversión", "exportaciones", "importaciones", "inei", "pobreza",
    "empleo", "desempleo", "tipo de cambio", "dólar", "precio", "tarifas",
    "minería", "petróleo", "gas", "electricidad", "banca", "crédito",
    "deuda", "déficit", "superávit", "reservas", "aranceles", "aduana",
    "mercado", "bolsa", "lima stock", "cavali", "smv", "sunat", "tributario",
    # English
    "economy", "inflation", "gdp", "central bank", "exchange rate", "dollar",
    "trade", "investment", "mining", "fiscal", "budget", "debt", "tariff",
    "interest rate", "recession", "growth", "unemployment",
]

_HIGH_SEVERITY = [
    "muerto", "muerte", "fallecido", "víctima", "herido", "violencia",
    "golpe de estado", "autogolpe", "estado de emergencia", "toque de queda",
    "crisis", "colapso", "caída", "derrumbe", "escándalo", "corrupción grave",
    "detención", "arresto", "preso", "encarcelado", "fuga", "acusado",
    "dead", "killed", "violence", "emergency", "crisis", "collapse", "arrested",
]

_MED_SEVERITY = [
    "renuncia", "dimisión", "censura", "interpelación", "moción", "destitución",
    "investigación", "acusación", "denuncia", "escándalo", "protesta", "huelga",
    "bloqueo", "paro", "marcha", "manifestación", "enfrentamiento",
    "resign", "impeach", "investigate", "protest", "strike", "block",
]


def _keyword_classify(articles: pd.DataFrame, classified_path: Path) -> pd.DataFrame:
    """Classify articles using keywords. Merges with existing cached classifications."""
    articles = articles.copy()

    # Start from cached if available — only classify unclassified rows
    if classified_path.exists():
        cached = pd.read_parquet(classified_path)
        cached_hashes = set(cached["url_hash"].dropna())
    else:
        cached = pd.DataFrame()
        cached_hashes = set()

    needs_classification = ~articles["url_hash"].isin(cached_hashes)
    to_classify = articles[needs_classification].copy()
    logger.info("  %d articles need keyword classification (%d already cached)",
                needs_classification.sum(), (~needs_classification).sum())

    if not to_classify.empty:
        def classify_row(row):
            text = f"{row.get('title', '')} {row.get('summary', '')}".lower()

            is_political = any(kw in text for kw in _POLITICAL_KEYWORDS)
            is_economic = any(kw in text for kw in _ECONOMIC_KEYWORDS)

            if not is_political and not is_economic:
                return "irrelevant", 0, ""

            category = "political" if is_political else "economic"
            if is_political and is_economic:
                category = "political"  # political takes precedence

            # Severity: 1-5
            high = any(kw in text for kw in _HIGH_SEVERITY)
            med = any(kw in text for kw in _MED_SEVERITY)
            severity = 4 if high else (3 if med else 2)
            labels = {1: "minimal", 2: "low", 3: "moderate", 4: "high", 5: "critical"}

            return category, severity, labels[severity]

        results = to_classify.apply(classify_row, axis=1)
        to_classify["article_category"] = [r[0] for r in results]
        to_classify["article_severity"] = [r[1] for r in results]
        to_classify["article_severity_label"] = [r[2] for r in results]

        # Merge with cached
        if not cached.empty:
            combined = pd.concat([cached, to_classify], ignore_index=True)
            combined = combined.drop_duplicates(subset=["url_hash"], keep="last")
        else:
            combined = to_classify

        # Re-merge with full articles list to ensure all rows have classifications
        articles = articles.merge(
            combined[["url_hash", "article_category", "article_severity", "article_severity_label"]],
            on="url_hash", how="left"
        )
        articles["article_category"] = articles["article_category"].fillna("irrelevant")
        articles["article_severity"] = articles["article_severity"].fillna(0)
        articles["article_severity_label"] = articles["article_severity_label"].fillna("")
    else:
        # All cached — just merge
        articles = articles.merge(
            cached[["url_hash", "article_category", "article_severity", "article_severity_label"]],
            on="url_hash", how="left"
        )
        articles["article_category"] = articles["article_category"].fillna("irrelevant")
        articles["article_severity"] = articles["article_severity"].fillna(0)
        articles["article_severity_label"] = articles["article_severity_label"].fillna("")

    return articles


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

    # ── Step 3: Classify with Claude ──────────────────────────────────────
    logger.info("=" * 60)

    if args.skip_claude and classified_path.exists():
        logger.info("STEP 3: Loading cached classifications")
        articles = pd.read_parquet(classified_path)
    elif args.skip_claude:
        logger.warning("STEP 3: --skip-claude but no cached classifications found")
        logger.warning("  Skipping classification — index will be empty")
        articles["article_category"] = "irrelevant"
        articles["article_severity"] = 0
        articles["article_severity_label"] = ""
    else:
        # Check for API key
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not has_key:
            logger.warning("ANTHROPIC_API_KEY not set — using keyword classifier fallback")
            articles = _keyword_classify(articles, classified_path)
            articles.to_parquet(classified_path, index=False)
            logger.info("  Keyword-classified %d articles", len(articles))
        else:
            logger.info("STEP 3: Classifying articles with Claude API")

            if args.force or "article_category" not in articles.columns:
                # Classify all
                to_classify = articles
            else:
                # Only classify unclassified articles
                unclassified_mask = articles["article_category"].isna() | (
                    articles["article_category"] == ""
                )
                if unclassified_mask.any():
                    to_classify = articles[unclassified_mask]
                else:
                    to_classify = pd.DataFrame()  # all already classified
                    logger.info("  All articles already classified")

            if not to_classify.empty:
                try:
                    from src.nlp.classifier import classify_articles_batch
                    classified = classify_articles_batch(
                        to_classify,
                        batch_size=rss_config["classification"]["batch_size"],
                        model=rss_config["classification"]["model"],
                        delay=rss_config["classification"]["delay"],
                    )
                    # Merge classifications back
                    if len(classified) < len(articles):
                        for col in ["article_category", "article_severity", "article_severity_label"]:
                            articles.loc[classified.index, col] = classified[col]
                    else:
                        articles = classified
                except ImportError:
                    logger.error(
                        "anthropic not installed. Install with: pip install anthropic"
                    )
                    return 1

        # Save classified articles
        save_parquet(articles, classified_path)
        logger.info("  Saved classified articles to %s", classified_path.name)

    # ── Step 4: Build daily index ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building daily index")

    from src.processing.daily_index import build_daily_index
    daily_index = build_daily_index(
        articles,
        start_date=rss_config["index"]["start_date"],
        zscore_window=rss_config["index"]["zscore_window"],
        min_periods=rss_config["index"]["min_periods"],
    )

    save_parquet(daily_index, index_path)

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
    logger.info("  Output:                %s", index_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
