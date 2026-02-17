"""Backfill missing days in the RSS article cache using GDELT DOC API 2.0.

GDELT archives news from thousands of sources including Peruvian media
(El Comercio, RPP, La República, Gestión, etc.) going back years.

Usage:
    python scripts/backfill_gdelt.py                          # auto-detect gaps
    python scripts/backfill_gdelt.py --start 2026-02-09 --end 2026-02-12
    python scripts/backfill_gdelt.py --date 2026-02-10        # single day

GDELT DOC API 2.0:
    https://api.gdeltproject.org/api/v2/doc/doc
    Max 250 results per request, no auth required.
"""

import argparse
import hashlib
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "rss" / "articles_cache.parquet"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("nexus.gdelt_backfill")

# Search queries targeting Peru political + economic news
# Multiple queries per day to maximize coverage (250 results each)
QUERIES = [
    {
        "name": "politics",
        "q": 'peru (congreso OR boluarte OR gobierno OR ministro OR politica OR premier OR gabinete)',
        "sourcelang": "spa",
    },
    {
        "name": "economy",
        "q": 'peru (economia OR inflacion OR PBI OR BCRP OR soles OR MEF OR presupuesto OR inversion)',
        "sourcelang": "spa",
    },
    {
        "name": "social",
        "q": 'peru (huelga OR protesta OR conflicto OR mineria OR corrupcion OR fiscal OR juicio)',
        "sourcelang": "spa",
    },
    {
        "name": "english",
        "q": 'peru (government OR president OR congress OR economy OR crisis OR protest)',
        "sourcelang": "eng",
        "sourcecountry": "Peru",
    },
]


def _url_hash(url: str) -> str:
    parsed = urlparse(url)
    normalized = parsed.netloc.lower() + parsed.path.rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def gdelt_fetch(query: str, start_dt: datetime, end_dt: datetime,
                sourcelang: str = "spa", sourcecountry: str = None) -> list[dict]:
    """Fetch articles from GDELT DOC API for a given time window."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 250,
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
        "sourcelang": sourcelang,
    }
    if sourcecountry:
        params["sourcecountry"] = sourcecountry

    try:
        r = requests.get(GDELT_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", []) or []
        return articles
    except requests.exceptions.RequestException as e:
        logger.warning("GDELT request failed: %s", e)
        return []
    except Exception as e:
        logger.warning("GDELT parse error: %s", e)
        return []


def gdelt_to_cache_row(article: dict, query_name: str) -> dict | None:
    """Convert GDELT article dict to our cache schema."""
    url = article.get("url", "").strip()
    title = article.get("title", "").strip()
    if not url or not title:
        return None

    # Parse GDELT seendate: "20260209T123456Z"
    seen = article.get("seendate", "")
    try:
        published = pd.Timestamp(seen, tz="UTC")
    except Exception:
        published = pd.Timestamp.now(tz="UTC")

    domain = article.get("domain", urlparse(url).netloc)

    return {
        "url": url,
        "title": title,
        "summary": article.get("socialimage", ""),  # no body in artlist mode
        "published": published,
        "source": domain,
        "feed_name": f"gdelt_{query_name}",
        "url_hash": _url_hash(url),
    }


def fetch_day(day: date) -> pd.DataFrame:
    """Fetch all articles for a single day across all query profiles."""
    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0)
    end_dt = datetime(day.year, day.month, day.day, 23, 59, 59)

    rows = []
    for q_cfg in QUERIES:
        logger.info("  Query '%s' for %s...", q_cfg["name"], day)
        articles = gdelt_fetch(
            query=q_cfg["q"],
            start_dt=start_dt,
            end_dt=end_dt,
            sourcelang=q_cfg.get("sourcelang", "spa"),
            sourcecountry=q_cfg.get("sourcecountry"),
        )
        logger.info("    -> %d articles", len(articles))

        for art in articles:
            row = gdelt_to_cache_row(art, q_cfg["name"])
            if row:
                rows.append(row)

        time.sleep(1.5)  # GDELT rate limit courtesy

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Dedup within this day's batch by url_hash
    df = df.drop_duplicates(subset=["url_hash"], keep="first")
    return df


def detect_gaps(cache: pd.DataFrame, start: date, end: date) -> list[date]:
    """Find days in [start, end] with fewer than 3 articles in cache."""
    if cache.empty:
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    cache = cache.copy()
    cache["published"] = pd.to_datetime(cache["published"], utc=True)
    by_day = cache.groupby(cache["published"].dt.date).size()

    gaps = []
    current = start
    while current <= end:
        count = by_day.get(current, 0)
        if count < 3:
            gaps.append(current)
            logger.info("  Gap detected: %s (%d articles)", current, count)
        current += timedelta(days=1)
    return gaps


def load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=["url", "title", "summary", "published", "source", "feed_name", "url_hash"])


def save_cache(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)


def main():
    parser = argparse.ArgumentParser(description="Backfill RSS cache from GDELT")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--date", type=str, help="Single date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=14,
                        help="Look back N days for gaps (default: 14)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GDELT Backfill — Peru Political/Economic News")
    logger.info("=" * 60)

    # Determine date range
    if args.date:
        start = end = date.fromisoformat(args.date)
    elif args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    else:
        end = date.today()
        start = end - timedelta(days=args.days)

    logger.info("Date range: %s to %s", start, end)

    # Load existing cache
    cache = load_cache()
    logger.info("Existing cache: %d articles", len(cache))

    # Detect gaps
    logger.info("Detecting gaps...")
    gaps = detect_gaps(cache, start, end)

    if not gaps:
        logger.info("No gaps found — cache is complete for this period.")
        return 0

    logger.info("Found %d day(s) to backfill: %s", len(gaps), gaps)

    # Fetch each gap day
    new_rows = []
    for day in gaps:
        logger.info("Fetching %s...", day)
        day_df = fetch_day(day)
        if day_df.empty:
            logger.warning("  No articles found for %s", day)
        else:
            logger.info("  Found %d unique articles for %s", len(day_df), day)
            new_rows.append(day_df)

    if not new_rows:
        logger.info("No new articles fetched.")
        return 0

    # Merge with cache
    new_df = pd.concat(new_rows, ignore_index=True)
    logger.info("Total new articles: %d", len(new_df))

    combined = pd.concat([cache, new_df], ignore_index=True)

    # Global dedup by url_hash
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url_hash"], keep="first")
    logger.info("After dedup: %d -> %d", before, len(combined))

    # Sort by published date
    combined["published"] = pd.to_datetime(combined["published"], utc=True)
    combined = combined.sort_values("published").reset_index(drop=True)

    # Save
    save_cache(combined)
    logger.info("Saved %d articles to %s", len(combined), CACHE_PATH)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("  Days backfilled:  %d", len(gaps))
    logger.info("  New articles:     %d", len(new_df))
    logger.info("  Cache total:      %d", len(combined))
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
