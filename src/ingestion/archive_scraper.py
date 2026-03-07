"""Historical article backfill via Arc Publishing API.

Fetches archived articles from El Comercio and Gestión using their public
Arc Publishing content API. Paginates through sections (politica, economia,
peru) to retrieve articles from 2025-01-01 to present.

La República is excluded from backfill (no public API); it accumulates
via RSS going forward.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.ingestion.rss import _url_hash

logger = logging.getLogger("nexus.ingestion.archive")

USER_AGENT = "NEXUS/1.0 (academic research)"
ARC_ENDPOINT = "/pf/api/v3/content/fetch/story-feed-by-section"


def _extract_summary(story: dict) -> str:
    """Extract summary from Arc story: subheadline or first text element."""
    sub = (story.get("subheadlines") or {}).get("basic", "")
    if sub:
        return sub.strip()

    for elem in story.get("content_elements", []):
        if elem.get("type") == "text" and elem.get("content"):
            return elem["content"].strip()

    return ""


def fetch_arc_section(
    domain: str,
    section: str,
    source: str,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    delay: float = 1.0,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Paginate Arc Publishing API for a single section.

    Parameters
    ----------
    domain : e.g. "elcomercio.pe"
    section : e.g. "politica"
    source : e.g. "elcomercio"
    start_date : ISO date string, stop paginating when articles are older
    end_date : ISO date string, skip articles newer than this
    delay : seconds between API calls
    batch_size : articles per page (max 100)

    Returns
    -------
    DataFrame with columns: url, title, summary, published, source, feed_name
    """
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = None
    if end_date:
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    url = f"https://{domain}{ARC_ENDPOINT}"
    feed_name = f"{source} - {section} (archive)"
    articles = []
    offset = 0
    total_pages = 0
    reached_start = False

    logger.info("  Fetching %s/%s from %s...", domain, section, start_date)

    while not reached_start:
        query = {
            "feedOffset": offset,
            "feedSize": batch_size,
            "section": f"/{section}",
        }
        params = {"query": str(query).replace("'", '"'), "_website": source}

        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("  API error at offset %d: %s", offset, e)
            break
        except ValueError:
            logger.warning("  Invalid JSON at offset %d", offset)
            break

        stories = data.get("content_elements", [])
        next_offset = data.get("next", offset + len(stories) if stories else None)

        if not stories:
            logger.info("  No more stories at offset %d", offset)
            break

        for story in stories:
            pub_str = story.get("first_publish_date", "")
            if not pub_str:
                continue

            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            if pub_dt < start_dt:
                reached_start = True
                break

            if end_dt and pub_dt > end_dt:
                continue

            title = (story.get("headlines") or {}).get("basic", "").strip()
            canonical = story.get("canonical_url", "")

            if not title or not canonical:
                continue

            # Build full URL if canonical is relative
            if canonical.startswith("/"):
                canonical = f"https://{domain}{canonical}"

            summary = _extract_summary(story)
            if not summary:
                summary = title

            articles.append({
                "url": canonical,
                "title": title,
                "summary": summary,
                "published": pub_dt,
                "source": source,
                "feed_name": feed_name,
            })

        total_pages += 1
        # Use API's `next` field for offset (API ignores feedSize, returns ~10/page)
        if next_offset is not None and next_offset > offset:
            offset = next_offset
        else:
            offset += len(stories)

        if total_pages % 100 == 0:
            logger.info("    %s/%s: %d articles so far (offset %d)",
                        domain, section, len(articles), offset)

        if not reached_start:
            time.sleep(delay)

    logger.info("  %s/%s: %d articles in %d pages",
                domain, section, len(articles), total_pages)
    return _to_dataframe(articles)


def _to_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Convert article list to DataFrame with proper dtypes."""
    if not articles:
        return pd.DataFrame(
            columns=["url", "title", "summary", "published", "source", "feed_name"]
        )
    df = pd.DataFrame(articles)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def fetch_arc_all_sections(
    domain: str,
    source: str,
    sections: list[str],
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    delay: float = 1.0,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Fetch all sections for one Arc Publishing domain."""
    dfs = []
    for section in sections:
        df = fetch_arc_section(
            domain=domain,
            section=section,
            source=source,
            start_date=start_date,
            end_date=end_date,
            delay=delay,
            batch_size=batch_size,
        )
        dfs.append(df)

    if not dfs:
        return _to_dataframe([])

    combined = pd.concat(dfs, ignore_index=True)
    # Dedup within domain (same article may appear in multiple sections)
    combined["url_hash"] = combined["url"].apply(_url_hash)
    combined = combined.drop_duplicates(subset=["url_hash"], keep="first")
    combined = combined.drop(columns=["url_hash"])
    logger.info("  %s total after cross-section dedup: %d", domain, len(combined))
    return combined


def backfill_archives(
    config_path: Path,
    cache_path: Path,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """Main entry: fetch El Comercio + Gestión archives, merge with cache, dedup.

    Parameters
    ----------
    config_path : Path to rss_feeds.yaml (with archives section)
    cache_path : Path to articles_cache.parquet
    start_date : ISO date, how far back to fetch

    Returns
    -------
    DataFrame of all articles (archive + existing cache), deduplicated
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    archive_cfg = config["rss"].get("archives", {})
    arc_domains = archive_cfg.get("arc_domains", [])
    delay = archive_cfg.get("delay", 1.0)
    batch_size = archive_cfg.get("batch_size", 100)

    if not arc_domains:
        logger.warning("No archive domains configured in rss_feeds.yaml")
        return _load_or_empty(cache_path)

    # Fetch archives from all Arc domains
    all_archive = []
    for domain_cfg in arc_domains:
        df = fetch_arc_all_sections(
            domain=domain_cfg["domain"],
            source=domain_cfg["source"],
            sections=domain_cfg["sections"],
            start_date=start_date,
            delay=delay,
            batch_size=batch_size,
        )
        all_archive.append(df)

    archive_df = pd.concat(all_archive, ignore_index=True) if all_archive else _to_dataframe([])
    logger.info("Archive fetch complete: %d articles from %d domains",
                len(archive_df), len(arc_domains))

    # Merge with existing cache
    from src.ingestion.rss import load_article_cache, save_article_cache, deduplicate_articles

    cached = load_article_cache(cache_path)

    if archive_df.empty and cached.empty:
        return cached

    # Add url_hash to archive articles
    if not archive_df.empty:
        archive_df["url_hash"] = archive_df["url"].apply(_url_hash)

    combined = pd.concat([cached, archive_df], ignore_index=True)
    combined = deduplicate_articles(combined)

    save_article_cache(combined, cache_path)
    logger.info("Cache updated: %d total articles", len(combined))

    return combined


def _load_or_empty(cache_path: Path) -> pd.DataFrame:
    """Load cache or return empty DataFrame."""
    from src.ingestion.rss import load_article_cache
    return load_article_cache(cache_path)
