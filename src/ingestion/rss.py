"""RSS feed fetcher for Peruvian news — daily instability index.

Fetches articles from El Comercio, Gestion, and La Republica RSS feeds.
Deduplicates by URL hash, cross-source path, and title similarity.
Caches articles in parquet for incremental updates.
"""

import hashlib
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import yaml

logger = logging.getLogger("nexus.ingestion.rss")

USER_AGENT = "NEXUS/1.0 (academic research)"


def _parse_feed(url: str, source: str, feed_name: str) -> list[dict]:
    """Parse a single RSS feed URL into article dicts."""
    import feedparser

    feed = feedparser.parse(url, agent=USER_AGENT)

    if feed.bozo and not feed.entries:
        logger.warning("Feed error for %s: %s", feed_name, feed.bozo_exception)
        return []

    articles = []
    for entry in feed.entries:
        # Parse published date
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                from time import mktime
                published = datetime.fromtimestamp(
                    mktime(entry.published_parsed), tz=timezone.utc
                )
            except (ValueError, OverflowError):
                pass

        if published is None:
            continue  # drop articles without dates

        title = getattr(entry, "title", "").strip()
        summary = getattr(entry, "summary", "").strip()
        link = getattr(entry, "link", "").strip()

        if not title or not link:
            continue

        # Use title as fallback text if no summary
        if not summary:
            summary = title

        articles.append({
            "url": link,
            "title": title,
            "summary": summary,
            "published": published,
            "source": source,
            "feed_name": feed_name,
        })

    logger.info("  %s: %d articles", feed_name, len(articles))
    return articles


def fetch_all_feeds(config_path: Path) -> pd.DataFrame:
    """Fetch all RSS feeds from config and return a DataFrame of articles.

    Parameters
    ----------
    config_path : Path to rss_feeds.yaml

    Returns
    -------
    DataFrame with columns: url, title, summary, published, source, feed_name
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feeds = config["rss"]["feeds"]
    all_articles = []

    logger.info("Fetching %d RSS feeds...", len(feeds))
    for feed_cfg in feeds:
        try:
            articles = _parse_feed(
                url=feed_cfg["url"],
                source=feed_cfg["source"],
                feed_name=feed_cfg["name"],
            )
            all_articles.extend(articles)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", feed_cfg["name"], e)

    if not all_articles:
        logger.warning("No articles fetched from any feed")
        return pd.DataFrame(
            columns=["url", "title", "summary", "published", "source", "feed_name"]
        )

    df = pd.DataFrame(all_articles)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    logger.info("Total articles fetched: %d", len(df))
    return df


def _url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL path (ignore query params)."""
    parsed = urlparse(url)
    normalized = parsed.netloc.lower() + parsed.path.rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _url_path(url: str) -> str:
    """Extract normalized URL path for cross-source matching."""
    parsed = urlparse(url)
    return parsed.path.rstrip("/").lower()


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """3-layer deduplication of articles.

    1. URL hash (exact same article URL)
    2. Cross-source URL path match (same path, different domain)
    3. Title similarity (SequenceMatcher > 0.70 within same day)
    """
    if df.empty:
        df["url_hash"] = pd.Series(dtype=str)
        return df

    df = df.copy()

    # Layer 1: URL hash dedup
    df["url_hash"] = df["url"].apply(_url_hash)
    before = len(df)
    df = df.drop_duplicates(subset=["url_hash"], keep="first")
    logger.info("  URL hash dedup: %d -> %d", before, len(df))

    # Layer 2: Cross-source path match
    df["_path"] = df["url"].apply(_url_path)
    before = len(df)
    df = df.drop_duplicates(subset=["_path"], keep="first")
    logger.info("  Cross-source path dedup: %d -> %d", before, len(df))
    df = df.drop(columns=["_path"])

    # Layer 3: Title similarity within same day
    # Only applied to recent articles (last 3 days) — older articles were
    # already deduped in prior runs and re-processing the full cache is O(n²).
    import datetime as _dt
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["_date"] = df["published"].dt.date
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=3)).date()
    recent = df[df["_date"] >= cutoff]
    keep = set(recent.index)

    for date_val, group in recent.groupby("_date"):
        indices = list(group.index)
        titles = group["title"].tolist()
        for i in range(len(indices)):
            if indices[i] not in keep:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] not in keep:
                    continue
                ratio = SequenceMatcher(None, titles[i], titles[j]).ratio()
                if ratio > 0.70:
                    keep.discard(indices[j])

    # Keep all older articles unchanged; filter only within the recent window
    old_indices = set(df[df["_date"] < cutoff].index)
    before = len(df)
    df = df.loc[df.index.isin(keep | old_indices)]
    logger.info("  Title similarity dedup: %d -> %d (checked last 3 days)", before, len(df))
    df = df.drop(columns=["_date"])

    return df.reset_index(drop=True)


def load_article_cache(cache_path: Path) -> pd.DataFrame:
    """Load previously fetched articles from parquet cache."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d cached articles from %s", len(df), cache_path.name)
        return df
    return pd.DataFrame(
        columns=["url", "title", "summary", "published", "source", "feed_name", "url_hash"]
    )


def save_article_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save articles to parquet cache."""
    from src.utils.io import save_parquet
    save_parquet(df, cache_path)
    logger.info("Saved %d articles to %s", len(df), cache_path.name)


def fetch_and_merge(config_path: Path, cache_path: Path) -> pd.DataFrame:
    """Fetch new articles, merge with cache, deduplicate, return full DataFrame."""
    cached = load_article_cache(cache_path)
    fresh = fetch_all_feeds(config_path)

    if fresh.empty and cached.empty:
        return cached

    # Add url_hash to fresh articles
    if not fresh.empty:
        fresh["url_hash"] = fresh["url"].apply(_url_hash)

    # Combine and deduplicate
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = deduplicate_articles(combined)

    # Save updated cache
    save_article_cache(combined, cache_path)

    return combined
