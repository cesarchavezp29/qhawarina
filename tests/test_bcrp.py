"""Tests for the BCRP API client."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.bcrp import BCRPClient, _parse_value, _parse_response_data
from src.utils.dates import parse_bcrp_period, format_bcrp_date


# ── Date parsing tests ────────────────────────────────────────────────────────

class TestParseBCRPPeriod:
    def test_standard_monthly(self):
        assert parse_bcrp_period("Ene.2024") == date(2024, 1, 1)
        assert parse_bcrp_period("Feb.2024") == date(2024, 2, 1)
        assert parse_bcrp_period("Dic.2023") == date(2023, 12, 1)

    def test_all_months(self):
        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        for i, m in enumerate(months, 1):
            result = parse_bcrp_period(f"{m}.2020")
            assert result == date(2020, i, 1), f"Failed for {m}"

    def test_no_dot(self):
        assert parse_bcrp_period("Ene2024") == date(2024, 1, 1)

    def test_two_digit_year(self):
        assert parse_bcrp_period("Mar.07") == date(2007, 3, 1)

    def test_quarterly(self):
        assert parse_bcrp_period("T1.2024") == date(2024, 1, 1)
        assert parse_bcrp_period("T2.2024") == date(2024, 4, 1)
        assert parse_bcrp_period("T3.2024") == date(2024, 7, 1)
        assert parse_bcrp_period("T4.2024") == date(2024, 10, 1)

    def test_whitespace(self):
        assert parse_bcrp_period("  Ene.2024  ") == date(2024, 1, 1)

    def test_invalid(self):
        assert parse_bcrp_period("") is None
        assert parse_bcrp_period("Invalid") is None
        assert parse_bcrp_period("XYZ.2024") is None


class TestFormatBCRPDate:
    def test_basic(self):
        assert format_bcrp_date(2007, 1) == "2007-1"
        assert format_bcrp_date(2025, 12) == "2025-12"


# ── Value parsing tests ──────────────────────────────────────────────────────

class TestParseValue:
    def test_numeric_string(self):
        assert _parse_value("1234.56") == 1234.56

    def test_integer_string(self):
        assert _parse_value("100") == 100.0

    def test_negative(self):
        assert _parse_value("-3.5") == -3.5

    def test_nd_missing(self):
        assert _parse_value("n.d.") is None

    def test_empty_string(self):
        assert _parse_value("") is None

    def test_none(self):
        assert _parse_value(None) is None

    def test_dash(self):
        assert _parse_value("-") is None

    def test_whitespace(self):
        assert _parse_value("  1234.56  ") == 1234.56

    def test_thousands_comma(self):
        assert _parse_value("1,234.56") == 1234.56

    def test_numeric_types(self):
        assert _parse_value(42) == 42.0
        assert _parse_value(3.14) == 3.14


# ── Response parsing tests ────────────────────────────────────────────────────

class TestParseResponseData:
    SAMPLE_RESPONSE = {
        "config": {
            "series": [
                {"name": "PBI global"},
                {"name": "PBI agropecuario"},
            ]
        },
        "periods": [
            {"name": "Ene.2024", "values": ["3.5", "2.1"]},
            {"name": "Feb.2024", "values": ["4.0", "n.d."]},
            {"name": "Mar.2024", "values": ["3.8", "1.9"]},
        ],
    }

    def test_parses_correctly(self):
        df = _parse_response_data(self.SAMPLE_RESPONSE)
        assert len(df) == 6  # 3 periods × 2 series
        assert set(df.columns) == {"date", "series_name", "series_index", "value"}

    def test_series_names(self):
        df = _parse_response_data(self.SAMPLE_RESPONSE)
        assert set(df["series_name"].unique()) == {"PBI global", "PBI agropecuario"}

    def test_missing_values(self):
        df = _parse_response_data(self.SAMPLE_RESPONSE)
        # Feb.2024 PBI agropecuario should be NaN
        feb_agro = df[
            (df["date"] == pd.Timestamp("2024-02-01"))
            & (df["series_name"] == "PBI agropecuario")
        ]
        assert feb_agro["value"].isna().all()

    def test_dates_parsed(self):
        df = _parse_response_data(self.SAMPLE_RESPONSE)
        dates = df["date"].unique()
        assert pd.Timestamp("2024-01-01") in dates
        assert pd.Timestamp("2024-02-01") in dates
        assert pd.Timestamp("2024-03-01") in dates

    def test_empty_response(self):
        df = _parse_response_data({"config": {}, "periods": []})
        assert df.empty

    def test_no_periods_key(self):
        df = _parse_response_data({"config": {}})
        assert df.empty


# ── Client URL building tests ─────────────────────────────────────────────────

class TestBCRPClientURL:
    def test_build_url_single(self):
        client = BCRPClient()
        url = client._build_url(["PN01770AM"], "2007-1", "2025-12")
        assert url == (
            "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
            "PN01770AM/json/2007-1/2025-12/esp"
        )

    def test_build_url_multiple(self):
        client = BCRPClient()
        url = client._build_url(["PN01288PM", "PN01289PM"], "2007-1", "2025-12")
        assert "PN01288PM-PN01289PM" in url

    def test_build_url_lang(self):
        client = BCRPClient(lang="eng")
        url = client._build_url(["PN01770AM"], "2007-1", "2025-12")
        assert url.endswith("/eng")


# ── Integration test (requires network) ───────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("httpx", reason="httpx not installed"),
    reason="httpx not available",
)
class TestBCRPClientIntegration:
    """Integration tests that hit the real BCRP API.

    These are skipped by default. Run with:
        pytest tests/test_bcrp.py -m integration
    """

    @pytest.mark.integration
    def test_fetch_single_series(self):
        client = BCRPClient()
        df = client.fetch_series(
            ["PN01770AM"],
            start_year=2023, start_month=1,
            end_year=2023, end_month=6,
        )
        assert not df.empty
        assert "date" in df.columns
        assert "value" in df.columns

    @pytest.mark.integration
    def test_verify_series(self):
        client = BCRPClient()
        result = client.verify_series("PN01770AM")
        assert result["valid"] is True
        assert result["sample_count"] > 0
