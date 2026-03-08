"""Tests for the Daily RSS Instability Index pipeline.

Tests RSS fetching, deduplication, article classification, daily index
building, and Arc Publishing archive backfill with mocked APIs.
"""

import json
from datetime import datetime, timezone
from time import struct_time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Test RSS Fetching ─────────────────────────────────────────────────────


def _make_entry(title, summary, link, published_parsed):
    """Create a mock feedparser entry."""
    entry = MagicMock()
    entry.title = title
    entry.summary = summary
    entry.link = link
    entry.published_parsed = published_parsed
    return entry


def _make_feed(entries, bozo=False):
    """Create a mock feedparser result."""
    feed = MagicMock()
    feed.entries = entries
    feed.bozo = bozo
    feed.bozo_exception = None
    return feed


class TestRSSFetching:
    """Test RSS feed parsing and date extraction."""

    def test_parse_feed_basic(self):
        from src.ingestion.rss import _parse_feed

        ts = struct_time((2025, 2, 1, 12, 0, 0, 5, 32, 0))
        entries = [
            _make_entry(
                "Congreso aprueba ley",
                "El pleno aprobo nueva legislacion",
                "https://elcomercio.pe/politica/congreso-aprueba-ley",
                ts,
            ),
        ]
        mock_feed = _make_feed(entries)

        with patch("feedparser.parse", return_value=mock_feed):
            result = _parse_feed(
                "https://example.com/rss",
                source="elcomercio",
                feed_name="Test Feed",
            )

        assert len(result) == 1
        assert result[0]["title"] == "Congreso aprueba ley"
        assert result[0]["source"] == "elcomercio"
        assert result[0]["published"].year == 2025

    def test_parse_feed_no_date_dropped(self):
        from src.ingestion.rss import _parse_feed

        entries = [_make_entry("No date article", "text", "https://ex.com/a", None)]
        mock_feed = _make_feed(entries)

        with patch("feedparser.parse", return_value=mock_feed):
            result = _parse_feed("https://ex.com", "test", "Test")

        assert len(result) == 0

    def test_parse_feed_empty_on_error(self):
        from src.ingestion.rss import _parse_feed

        mock_feed = _make_feed([], bozo=True)
        with patch("feedparser.parse", return_value=mock_feed):
            result = _parse_feed("https://bad.com", "test", "Bad Feed")

        assert len(result) == 0

    def test_fetch_all_feeds(self, tmp_path):
        from src.ingestion.rss import fetch_all_feeds

        config = {
            "rss": {
                "feeds": [
                    {"name": "Test", "url": "https://ex.com/rss", "source": "test"},
                ],
                "classification": {"model": "test", "batch_size": 5, "delay": 0},
                "index": {"start_date": "2025-01-01", "zscore_window": 90, "min_periods": 7},
            }
        }
        config_path = tmp_path / "rss_feeds.yaml"
        import yaml
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        ts = struct_time((2025, 2, 1, 12, 0, 0, 5, 32, 0))
        entries = [_make_entry("Title", "Summary", "https://ex.com/a", ts)]
        mock_feed = _make_feed(entries)

        with patch("feedparser.parse", return_value=mock_feed):
            df = fetch_all_feeds(config_path)

        assert len(df) == 1
        assert "url" in df.columns
        assert "published" in df.columns


# ── Test Deduplication ────────────────────────────────────────────────────


class TestDeduplication:
    """Test 3-layer dedup: URL hash, cross-source path, title similarity."""

    def _make_articles(self, entries):
        """Create a DataFrame from list of (url, title, published) tuples."""
        records = []
        for url, title, published in entries:
            records.append({
                "url": url,
                "title": title,
                "summary": f"Summary for {title}",
                "published": pd.Timestamp(published, tz="UTC"),
                "source": url.split("/")[2].split(".")[0],
                "feed_name": "test",
            })
        return pd.DataFrame(records)

    def test_url_hash_dedup(self):
        from src.ingestion.rss import deduplicate_articles

        df = self._make_articles([
            ("https://ex.com/article-1", "Article one", "2025-02-01"),
            ("https://ex.com/article-1", "Article one copy", "2025-02-01"),
        ])
        result = deduplicate_articles(df)
        assert len(result) == 1

    def test_cross_source_path_dedup(self):
        from src.ingestion.rss import deduplicate_articles

        df = self._make_articles([
            ("https://elcomercio.pe/politica/noticia-123", "Noticia", "2025-02-01"),
            ("https://gestion.pe/politica/noticia-123", "Noticia copia", "2025-02-01"),
        ])
        result = deduplicate_articles(df)
        # Same path "/politica/noticia-123" → deduped
        assert len(result) == 1

    def test_title_similarity_dedup(self):
        from src.ingestion.rss import deduplicate_articles

        df = self._make_articles([
            ("https://a.com/1", "Congreso aprueba nueva ley de presupuesto 2025", "2025-02-01"),
            ("https://b.com/2", "Congreso aprueba nueva ley de presupuesto del 2025", "2025-02-01"),
        ])
        result = deduplicate_articles(df)
        assert len(result) == 1

    def test_different_titles_kept(self):
        from src.ingestion.rss import deduplicate_articles

        df = self._make_articles([
            ("https://a.com/1", "Congreso aprueba ley", "2025-02-01"),
            ("https://b.com/2", "Economia crece 3 porciento", "2025-02-01"),
        ])
        result = deduplicate_articles(df)
        assert len(result) == 2

    def test_empty_df_dedup(self):
        from src.ingestion.rss import deduplicate_articles

        df = pd.DataFrame(
            columns=["url", "title", "summary", "published", "source", "feed_name"]
        )
        result = deduplicate_articles(df)
        assert len(result) == 0
        assert "url_hash" in result.columns


# ── Test Article Classification ───────────────────────────────────────────


class TestArticleClassification:
    """Test Claude API classification of articles."""

    def _make_articles_df(self, n=3):
        records = []
        categories = ["political", "economic", "both"]
        for i in range(n):
            records.append({
                "url": f"https://ex.com/{i}",
                "title": f"Article {i}",
                "summary": f"Summary {i}",
                "published": pd.Timestamp("2025-02-01", tz="UTC") + pd.Timedelta(days=i),
                "source": "test",
                "feed_name": "test",
            })
        return pd.DataFrame(records)

    def test_classify_articles_batch(self):
        from src.nlp.classifier import classify_articles_batch

        df = self._make_articles_df(3)
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps([
            {"id": 1, "category": "political", "severity": 2},
            {"id": 2, "category": "economic", "severity": 1},
            {"id": 3, "category": "both", "severity": 3},
        ])

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = classify_articles_batch(df, client=mock_client, batch_size=10, delay=0)

        assert "article_category" in result.columns
        assert "article_severity" in result.columns
        assert "article_severity_label" in result.columns
        assert result.loc[0, "article_category"] == "political"
        assert result.loc[0, "article_severity"] == 2
        assert result.loc[1, "article_category"] == "economic"
        assert result.loc[2, "article_category"] == "both"
        assert result.loc[2, "article_severity"] == 3

    def test_classify_irrelevant_gets_zero_severity(self):
        from src.nlp.classifier import classify_articles_batch

        df = self._make_articles_df(1)
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps([
            {"id": 1, "category": "irrelevant", "severity": 1},
        ])

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = classify_articles_batch(df, client=mock_client, batch_size=10, delay=0)

        assert result.loc[0, "article_category"] == "irrelevant"
        assert result.loc[0, "article_severity"] == 0

    def test_classify_invalid_category_defaults_irrelevant(self):
        from src.nlp.classifier import classify_articles_batch

        df = self._make_articles_df(1)
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps([
            {"id": 1, "category": "sports", "severity": 2},
        ])

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = classify_articles_batch(df, client=mock_client, batch_size=10, delay=0)

        assert result.loc[0, "article_category"] == "irrelevant"
        assert result.loc[0, "article_severity"] == 0


# ── Test Daily Index ──────────────────────────────────────────────────────


class TestDailyIndex:
    """Test daily index aggregation and computation."""

    def _make_classified_df(self, rows):
        """Create classified articles DataFrame.

        rows: list of (date_str, category, severity) tuples.
        """
        records = []
        for i, (date, cat, sev) in enumerate(rows):
            records.append({
                "url": f"https://ex.com/{i}",
                "title": f"Article {i}",
                "summary": f"Summary {i}",
                "published": pd.Timestamp(date, tz="UTC"),
                "source": "test",
                "feed_name": "test",
                "article_category": cat,
                "article_severity": sev,
                "article_severity_label": {0: "", 1: "low", 2: "medium", 3: "high"}.get(sev, ""),
            })
        return pd.DataFrame(records)

    def test_basic_index_build(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "political", 2),
            ("2025-02-01", "economic", 1),
            ("2025-02-02", "both", 3),
            ("2025-02-03", "irrelevant", 0),
        ])

        result = build_daily_index(df, start_date="2025-02-01", zscore_window=30, min_periods=2)

        assert "political_score" in result.columns
        assert "economic_score" in result.columns
        assert "political_v2" in result.columns
        assert "economic_v2" in result.columns
        assert "n_articles_political" in result.columns
        assert len(result) == 2  # Feb 1-2 (irrelevant-only Feb 3 not in range)

    def test_political_score_calculation(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "political", 3),  # 3/3 = 1.0
            ("2025-02-01", "political", 2),  # 2/3 = 0.67
        ])

        result = build_daily_index(df, start_date="2025-02-01", zscore_window=30, min_periods=1)
        score = result.loc[result["date"] == pd.Timestamp("2025-02-01"), "political_score"].iloc[0]
        expected = (3 / 3 + 2 / 3) / 2  # mean(severity/3) = 0.833
        assert abs(score - expected) < 0.01

    def test_both_category_counted_in_both_indices(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "both", 2),
        ])

        result = build_daily_index(df, start_date="2025-02-01", zscore_window=30, min_periods=1)
        pol = result.loc[0, "political_score"]
        econ = result.loc[0, "economic_score"]
        assert pol > 0
        assert econ > 0
        assert abs(pol - econ) < 0.01  # same severity

    def test_irrelevant_excluded(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "irrelevant", 0),
        ])

        result = build_daily_index(df, start_date="2025-02-01")
        assert result.empty or len(result) == 0

    def test_missing_days_filled_with_zero(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "political", 2),
            ("2025-02-03", "political", 1),
        ])

        result = build_daily_index(df, start_date="2025-02-01", zscore_window=30, min_periods=1)
        feb2 = result[result["date"] == pd.Timestamp("2025-02-02")]
        assert len(feb2) == 1
        assert feb2.iloc[0]["political_score"] == 0.0

    def test_empty_articles_returns_empty_index(self):
        from src.processing.daily_index import build_daily_index

        df = pd.DataFrame(columns=[
            "published", "article_category", "article_severity",
        ])
        result = build_daily_index(df)
        assert result.empty

    def test_output_columns_complete(self):
        from src.processing.daily_index import build_daily_index

        df = self._make_classified_df([
            ("2025-02-01", "political", 2),
            ("2025-02-01", "economic", 1),
        ])
        result = build_daily_index(df, start_date="2025-02-01", zscore_window=30, min_periods=1)

        expected_cols = {
            "date",
            "political_score", "political_zscore", "political_level", "political_v2",
            "economic_score", "economic_zscore", "economic_level", "economic_v2",
            "n_articles_political", "n_articles_economic", "n_articles_total",
        }
        assert set(result.columns) == expected_cols


# ── Test Build Script (dry-run integration) ───────────────────────────────


class TestBuildScript:
    """Test the build script entry point with skip flags."""

    def test_script_importable(self):
        """Verify the script can be imported without errors."""
        import scripts.build_daily_index  # noqa: F401

    def test_skip_fetch_and_claude_with_no_cache(self, tmp_path, monkeypatch):
        """With --skip-fetch --skip-claude and no cache, should exit cleanly."""
        import importlib

        rss_dir = tmp_path / "rss"
        daily_dir = tmp_path / "daily"
        rss_dir.mkdir()
        daily_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "build_daily_index.py", "--skip-fetch", "--skip-claude",
        ])
        # Patch config.settings so re-import picks it up
        monkeypatch.setattr("config.settings.RAW_RSS_DIR", rss_dir)
        monkeypatch.setattr("config.settings.PROCESSED_DAILY_DIR", daily_dir)

        # Force re-import so the script picks up patched settings
        import scripts.build_daily_index as mod
        importlib.reload(mod)

        rc = mod.main()
        # Should succeed (empty index) or produce 0
        assert rc == 0


# ── Test Arc Publishing Archive Scraper ──────────────────────────────────


def _make_arc_response(stories, offset=0):
    """Create a mock Arc API response with content_elements."""
    return {
        "content_elements": stories,
        "next": offset + len(stories),
        "count": 10000,
    }


def _make_arc_story(title, pub_date, canonical, subheadline="", content_text=""):
    """Create a single Arc story dict."""
    story = {
        "headlines": {"basic": title},
        "first_publish_date": pub_date,
        "canonical_url": canonical,
    }
    if subheadline:
        story["subheadlines"] = {"basic": subheadline}
    if content_text:
        story["content_elements"] = [{"type": "text", "content": content_text}]
    return story


class TestArcAPI:
    """Test Arc Publishing API pagination and field extraction."""

    def test_fetch_single_page(self):
        from src.ingestion.archive_scraper import fetch_arc_section

        stories = [
            _make_arc_story(
                "Congreso debate reforma",
                "2025-06-15T10:00:00Z",
                "/politica/congreso-debate",
                subheadline="El pleno debatio la reforma",
            ),
            _make_arc_story(
                "Economia crece 3%",
                "2025-06-14T08:00:00Z",
                "/economia/economia-crece",
                content_text="El PBI aumento 3% en mayo.",
            ),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])  # empty page stops pagination

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        assert len(df) == 2
        assert df.iloc[0]["title"] == "Congreso debate reforma"
        assert df.iloc[0]["summary"] == "El pleno debatio la reforma"
        assert df.iloc[1]["summary"] == "El PBI aumento 3% en mayo."
        assert df.iloc[0]["source"] == "elcomercio"
        assert "archive" in df.iloc[0]["feed_name"]

    def test_pagination_stops_at_start_date(self):
        from src.ingestion.archive_scraper import fetch_arc_section

        # Page 1: articles from 2025
        page1_stories = [
            _make_arc_story("Recent", "2025-03-01T10:00:00Z", "/a/recent"),
            _make_arc_story("Old", "2024-12-15T10:00:00Z", "/a/old"),  # before start_date
        ]
        page1 = _make_arc_response(page1_stories)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=page1)

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        # Only the 2025 article should be kept
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Recent"

    def test_relative_urls_expanded(self):
        from src.ingestion.archive_scraper import fetch_arc_section

        stories = [
            _make_arc_story("Test", "2025-05-01T10:00:00Z", "/politica/test-article"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        assert df.iloc[0]["url"] == "https://elcomercio.pe/politica/test-article"

    def test_title_fallback_as_summary(self):
        from src.ingestion.archive_scraper import fetch_arc_section

        stories = [
            _make_arc_story("Titulo sin resumen", "2025-05-01T10:00:00Z", "/a/x"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        assert df.iloc[0]["summary"] == "Titulo sin resumen"

    def test_stories_without_date_skipped(self):
        from src.ingestion.archive_scraper import fetch_arc_section

        stories = [
            _make_arc_story("No date", "", "/a/nodate"),
            _make_arc_story("Has date", "2025-05-01T10:00:00Z", "/a/hasdate"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        assert len(df) == 1
        assert df.iloc[0]["title"] == "Has date"

    def test_api_error_returns_empty(self):
        from src.ingestion.archive_scraper import fetch_arc_section
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("503 Server Error")

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_section(
                    domain="elcomercio.pe",
                    section="politica",
                    source="elcomercio",
                    start_date="2025-01-01",
                    delay=0,
                )

        assert len(df) == 0

    def test_cross_section_dedup(self):
        from src.ingestion.archive_scraper import fetch_arc_all_sections

        stories = [
            _make_arc_story("Same article", "2025-05-01T10:00:00Z", "/politica/same"),
        ]
        page = _make_arc_response(stories)
        empty = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page, empty, page, empty])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                df = fetch_arc_all_sections(
                    domain="elcomercio.pe",
                    source="elcomercio",
                    sections=["politica", "economia"],
                    start_date="2025-01-01",
                    delay=0,
                )

        # Same article appears in both sections — should be deduped to 1
        assert len(df) == 1


class TestBackfill:
    """Test backfill_archives: merge with cache + dedup."""

    def _write_config(self, tmp_path):
        """Write a minimal rss_feeds.yaml with archive config."""
        config = {
            "rss": {
                "feeds": [],
                "archives": {
                    "arc_domains": [
                        {
                            "domain": "elcomercio.pe",
                            "source": "elcomercio",
                            "sections": ["politica"],
                        },
                    ],
                    "delay": 0,
                    "batch_size": 100,
                },
                "classification": {"model": "test", "batch_size": 5, "delay": 0},
                "index": {"start_date": "2025-01-01", "zscore_window": 90, "min_periods": 7},
            }
        }
        config_path = tmp_path / "rss_feeds.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        return config_path

    def test_backfill_creates_cache(self, tmp_path):
        from src.ingestion.archive_scraper import backfill_archives

        config_path = self._write_config(tmp_path)
        cache_path = tmp_path / "articles_cache.parquet"

        stories = [
            _make_arc_story("Article A", "2025-06-01T10:00:00Z", "/politica/a"),
            _make_arc_story("Article B", "2025-05-15T08:00:00Z", "/politica/b"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                result = backfill_archives(config_path, cache_path)

        assert len(result) == 2
        assert cache_path.exists()

        # Verify cache was saved
        cached = pd.read_parquet(cache_path)
        assert len(cached) == 2

    def test_backfill_merges_with_existing_cache(self, tmp_path):
        from src.ingestion.archive_scraper import backfill_archives
        from src.ingestion.rss import _url_hash

        config_path = self._write_config(tmp_path)
        cache_path = tmp_path / "articles_cache.parquet"

        # Pre-populate cache with one article
        existing = pd.DataFrame([{
            "url": "https://larepublica.pe/politica/existing",
            "title": "Existing RSS article",
            "summary": "From RSS feed",
            "published": pd.Timestamp("2025-06-10", tz="UTC"),
            "source": "larepublica",
            "feed_name": "La Republica - Politica",
            "url_hash": _url_hash("https://larepublica.pe/politica/existing"),
        }])
        existing.to_parquet(cache_path)

        # Archive returns 1 new article
        stories = [
            _make_arc_story("New archive", "2025-06-01T10:00:00Z", "/politica/new"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                result = backfill_archives(config_path, cache_path)

        # Should have both: existing RSS + new archive
        assert len(result) == 2
        sources = set(result["source"])
        assert "larepublica" in sources
        assert "elcomercio" in sources

    def test_backfill_dedup_archive_vs_cache(self, tmp_path):
        from src.ingestion.archive_scraper import backfill_archives
        from src.ingestion.rss import _url_hash

        config_path = self._write_config(tmp_path)
        cache_path = tmp_path / "articles_cache.parquet"

        # Cache already has an article from elcomercio
        existing = pd.DataFrame([{
            "url": "https://elcomercio.pe/politica/duplicate",
            "title": "Already cached",
            "summary": "From RSS",
            "published": pd.Timestamp("2025-06-01", tz="UTC"),
            "source": "elcomercio",
            "feed_name": "El Comercio - Politica",
            "url_hash": _url_hash("https://elcomercio.pe/politica/duplicate"),
        }])
        existing.to_parquet(cache_path)

        # Archive returns same article
        stories = [
            _make_arc_story("Already cached", "2025-06-01T10:00:00Z", "/politica/duplicate"),
        ]
        page1 = _make_arc_response(stories)
        page2 = _make_arc_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=[page1, page2])

        with patch("src.ingestion.archive_scraper.requests.get", return_value=mock_resp):
            with patch("src.ingestion.archive_scraper.time.sleep"):
                result = backfill_archives(config_path, cache_path)

        # Should be deduped to 1
        assert len(result) == 1

    def test_no_archive_config_returns_cache(self, tmp_path):
        from src.ingestion.archive_scraper import backfill_archives

        # Config with no archives section
        config = {
            "rss": {
                "feeds": [],
                "classification": {"model": "test", "batch_size": 5, "delay": 0},
                "index": {"start_date": "2025-01-01", "zscore_window": 90, "min_periods": 7},
            }
        }
        config_path = tmp_path / "rss_feeds.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        cache_path = tmp_path / "articles_cache.parquet"

        result = backfill_archives(config_path, cache_path)
        assert result.empty
