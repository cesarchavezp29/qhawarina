"""Tests for Qhawarina PDF instability reports."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_index(tmp_path):
    """Create a minimal daily index parquet."""
    dates = pd.date_range("2026-01-01", periods=31, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "political_score": [2.0 + i * 0.5 for i in range(31)],
        "political_zscore": [0.5 + i * 0.1 for i in range(31)],
        "political_level": [0.3 + i * 0.02 for i in range(31)],
        "political_v2": [0.3 + i * 0.02 for i in range(31)],
        "economic_score": [1.5 + i * 0.3 for i in range(31)],
        "economic_zscore": [0.3 + i * 0.08 for i in range(31)],
        "economic_level": [0.25 + i * 0.02 for i in range(31)],
        "economic_v2": [0.25 + i * 0.02 for i in range(31)],
        "n_articles_political": [3 + i for i in range(31)],
        "n_articles_economic": [2 + i for i in range(31)],
        "n_articles_total": [5 + 2 * i for i in range(31)],
    })
    path = tmp_path / "daily_index.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def sample_articles(tmp_path):
    """Create minimal classified articles parquet."""
    rows = []
    for day_offset in range(31):
        date = datetime(2026, 1, 1) + timedelta(days=day_offset)
        for j in range(3):
            rows.append({
                "url": f"https://test.com/pol-{day_offset}-{j}",
                "title": f"Political news {day_offset}-{j}: crisis institucional",
                "summary": "Summary text",
                "published": pd.Timestamp(date, tz="UTC"),
                "source": "elcomercio",
                "feed_name": "politica",
                "article_category": "political",
                "article_severity": (j % 3) + 1,
            })
        for j in range(2):
            rows.append({
                "url": f"https://test.com/econ-{day_offset}-{j}",
                "title": f"Economic news {day_offset}-{j}: mercado financiero",
                "summary": "Summary text",
                "published": pd.Timestamp(date, tz="UTC"),
                "source": "gestion",
                "feed_name": "economia",
                "article_category": "economic",
                "article_severity": (j % 2) + 1,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "articles_classified.parquet"
    df.to_parquet(path)
    return path


class TestDailyPDF:
    def test_generates_pdf(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_daily_pdf

        out = tmp_path / "daily.pdf"
        result = generate_daily_pdf(
            "2026-01-15", sample_index, sample_articles, out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000  # Non-trivial PDF

    def test_no_data_returns_path(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_daily_pdf

        out = tmp_path / "empty.pdf"
        result = generate_daily_pdf(
            "2020-01-01", sample_index, sample_articles, out,
        )
        assert result == out


class TestWeeklyPDF:
    def test_generates_pdf(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_weekly_pdf

        out = tmp_path / "weekly.pdf"
        generate_weekly_pdf(
            2026, 3, sample_index, sample_articles, out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_no_data_week(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_weekly_pdf

        out = tmp_path / "empty_week.pdf"
        generate_weekly_pdf(
            2020, 1, sample_index, sample_articles, out,
        )


class TestMonthlyPDF:
    def test_generates_pdf(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_monthly_pdf

        out = tmp_path / "monthly.pdf"
        generate_monthly_pdf(
            2026, 1, sample_index, sample_articles, out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_no_data_month(self, sample_index, sample_articles, tmp_path):
        from src.reporting.pdf_report import generate_monthly_pdf

        out = tmp_path / "empty_month.pdf"
        generate_monthly_pdf(
            2020, 6, sample_index, sample_articles, out,
        )


class TestScoreExplanation:
    def test_explanation_contains_articles(self, sample_articles):
        from src.reporting.pdf_report import _build_explanation, _load_articles, _get_period_articles

        articles = _load_articles(sample_articles)
        pol, econ = _get_period_articles(articles, "2026-01-15", "2026-01-15")

        explanation = _build_explanation(pol, econ)
        assert "Pol\u00edtico" in explanation
        assert "Econ\u00f3mico" in explanation
        assert "art." in explanation

    def test_empty_articles(self):
        from src.reporting.pdf_report import _build_explanation

        explanation = _build_explanation(pd.DataFrame(), pd.DataFrame())
        assert explanation == ""


class TestHelpers:
    def test_aggregate_row(self, sample_index):
        from src.reporting.pdf_report import _aggregate_row, _load_index

        idx = _load_index(sample_index)
        result = _aggregate_row(idx, "2026-01-01", "2026-01-07")

        assert result is not None
        assert result["n_days"] == 7
        assert result["political_score"] > 0
        assert result["economic_score"] > 0
        assert "political_score_max" in result
        assert "political_score_min" in result

    def test_aggregate_no_data(self, sample_index):
        from src.reporting.pdf_report import _aggregate_row, _load_index

        idx = _load_index(sample_index)
        result = _aggregate_row(idx, "2020-01-01", "2020-01-07")
        assert result is None
