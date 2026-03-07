"""Tests for Political Instability Index pipeline.

Tests cover:
1. Ground truth loading
2. Wikipedia API fetching
3. Event extraction from text
4. Cabinet timeline
5. Cabinet stability computation
6. Event matching
7. NLP classification (mocked)
8. Validation metrics
9. Index aggregation
10. Event study analysis
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Ground truth tests ──────────────────────────────────────────────────────

class TestGroundTruth:
    """Tests for loading and validating the ground truth dataset."""

    def test_load_ground_truth(self):
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        assert "events" in gt
        assert "high_impact" in gt
        assert "presidents" in gt
        assert len(gt["events"]) == 70

    def test_gt_event_columns(self):
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        required = ["event_id", "date", "event_type", "president_affected",
                     "event_description", "severity", "anticipated"]
        for col in required:
            assert col in gt["events"].columns, f"Missing column: {col}"

    def test_gt_severity_is_1_3(self):
        """Ground truth severity must be ordinal 1-3 (NOT rescaled)."""
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        severities = gt["events"]["severity"].unique()
        assert set(severities) <= {1, 2, 3}, f"Unexpected severities: {severities}"

    def test_gt_high_impact_events(self):
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        assert len(gt["high_impact"]) >= 13  # dataset has 14
        assert "event_window_start" in gt["high_impact"].columns
        assert "event_window_end" in gt["high_impact"].columns

    def test_gt_severity_distribution(self):
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        dist = gt["events"]["severity"].value_counts()
        # Per spec: 9 Low, 35 Medium, 26 High
        assert dist.get(1, 0) >= 8
        assert dist.get(2, 0) >= 30
        assert dist.get(3, 0) >= 20

    def test_gt_presidents_complete(self):
        from src.ingestion.political import load_ground_truth
        gt = load_ground_truth()
        assert len(gt["presidents"]) >= 10  # at least 10 presidents


# ── Wikipedia API tests ─────────────────────────────────────────────────────

class TestWikipediaAPI:
    """Tests for Wikipedia API fetching."""

    def test_fetch_single_page(self):
        """Can fetch a known Wikipedia page via API."""
        from src.ingestion.political import fetch_wikipedia_page
        text = fetch_wikipedia_page("Conflicto_de_Bagua", lang="es")
        assert len(text) > 100
        assert "bagua" in text.lower()

    def test_fetch_missing_page(self):
        """Missing page returns empty string."""
        from src.ingestion.political import fetch_wikipedia_page
        text = fetch_wikipedia_page("ThisPageDefinitelyDoesNotExist12345", lang="en")
        assert text == ""

    def test_load_sources_config(self):
        from src.ingestion.political import load_sources_config
        config = load_sources_config()
        assert "wikipedia" in config
        assert "en" in config["wikipedia"]
        assert "es" in config["wikipedia"]
        assert len(config["wikipedia"]["en"]) >= 15
        assert len(config["wikipedia"]["es"]) >= 15


# ── Event extraction tests ──────────────────────────────────────────────────

class TestEventExtraction:
    """Tests for extracting events from Wikipedia text."""

    SAMPLE_TEXT_ES = (
        "El 5 de junio de 2009, la policía y los pueblos indígenas amazónicos "
        "se enfrentaron en la ciudad de Bagua, departamento de Amazonas. "
        "El enfrentamiento dejó 33 muertos, incluyendo 23 policías y 10 civiles. "
        "El premier Yehude Simon renunció el 11 de julio de 2009 como consecuencia "
        "de la crisis. El presidente Alan García fue criticado por su manejo de la situación."
    )

    SAMPLE_TEXT_EN = (
        "On September 14, 2000, the vladivideo scandal broke when Canal N broadcast "
        "a video of Vladimiro Montesinos bribing opposition congressman Alberto Kouri. "
        "The crisis led to President Fujimori announcing new elections on September 16, 2000."
    )

    def test_extract_spanish_dates(self):
        from src.ingestion.political import extract_events_from_text
        events = extract_events_from_text(
            self.SAMPLE_TEXT_ES, source_title="Baguazo", lang="es"
        )
        assert len(events) >= 1
        dates = [e["date"] for e in events]
        assert any(d.year == 2009 and d.month == 6 for d in dates)

    def test_extract_english_dates(self):
        from src.ingestion.political import extract_events_from_text
        events = extract_events_from_text(
            self.SAMPLE_TEXT_EN, source_title="Alberto_Fujimori", lang="en"
        )
        assert len(events) >= 1
        dates = [e["date"] for e in events]
        assert any(d.year == 2000 and d.month == 9 for d in dates)

    def test_extract_has_required_fields(self):
        from src.ingestion.political import extract_events_from_text
        events = extract_events_from_text(
            self.SAMPLE_TEXT_ES, source_title="Baguazo", lang="es"
        )
        if events:
            e = events[0]
            assert "date" in e
            assert "event_description" in e
            assert "source" in e
            assert "source_page" in e

    def test_filters_non_political(self):
        """Non-political text should not produce events."""
        from src.ingestion.political import extract_events_from_text
        text = "El 15 de marzo de 2010, Lima celebró el festival gastronómico más grande del mundo."
        events = extract_events_from_text(text, source_title="test", lang="es")
        assert len(events) == 0

    def test_deduplication(self):
        from src.ingestion.political import deduplicate_events
        df = pd.DataFrame({
            "date": [pd.Timestamp("2009-06-05"), pd.Timestamp("2009-06-05")],
            "event_description": [
                "La policía y los pueblos indígenas se enfrentaron en Bagua",
                "La policía y los pueblos amazónicos se enfrentaron en Bagua"
            ],
            "source": ["wikipedia_api", "wikipedia_api"],
            "source_page": ["Baguazo", "Gobierno_Garcia"],
            "source_text_snippet": ["snippet1", "snippet2"],
        })
        result = deduplicate_events(df)
        assert len(result) == 1


# ── Cabinet timeline tests ──────────────────────────────────────────────────

class TestCabinetTimeline:
    """Tests for cabinet timeline extraction and stability."""

    def test_extract_cabinet_timeline(self):
        from src.ingestion.political import extract_cabinet_timeline
        timeline = extract_cabinet_timeline("")
        assert len(timeline) >= 30  # at least 30 premiers since 2001

    def test_cabinet_timeline_columns(self):
        from src.ingestion.political import extract_cabinet_timeline
        timeline = extract_cabinet_timeline("")
        required = ["premier_name", "start_date", "end_date", "president", "duration_days"]
        for col in required:
            assert col in timeline.columns, f"Missing column: {col}"

    def test_cabinet_timeline_sorted(self):
        from src.ingestion.political import extract_cabinet_timeline
        timeline = extract_cabinet_timeline("")
        dates = timeline["start_date"].tolist()
        assert dates == sorted(dates)

    def test_cabinet_timeline_known_premiers(self):
        """Verify some known premiers are in the timeline."""
        from src.ingestion.political import extract_cabinet_timeline
        timeline = extract_cabinet_timeline("")
        names = timeline["premier_name"].tolist()
        # Check a few known premiers
        assert any("Zavala" in n for n in names)  # Fernando Zavala (PPK)
        assert any("Bellido" in n for n in names)  # Guido Bellido (Castillo)
        assert any("Otárola" in n for n in names)  # Alberto Otárola (Boluarte)

    def test_cabinet_stability_monthly(self):
        from src.ingestion.political import extract_cabinet_timeline
        from src.processing.cabinet_stability import compute_cabinet_instability
        timeline = extract_cabinet_timeline("")
        result = compute_cabinet_instability(timeline, freq="M")
        assert "date" in result.columns
        assert "days_since_change" in result.columns
        assert "cabinet_zscore" in result.columns
        assert len(result) > 100  # at least 100 months

    def test_cabinet_stability_weekly(self):
        from src.ingestion.political import extract_cabinet_timeline
        from src.processing.cabinet_stability import compute_cabinet_instability
        timeline = extract_cabinet_timeline("")
        result = compute_cabinet_instability(timeline, freq="W-FRI")
        assert len(result) > 400  # at least 400 weeks

    def test_days_since_change_positive(self):
        from src.processing.cabinet_stability import compute_days_since_change
        timeline = pd.DataFrame({
            "start_date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        })
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="MS")
        days = compute_days_since_change(timeline, dates)
        assert (days.dropna() >= 0).all()


# ── NLP classifier tests (mocked) ──────────────────────────────────────────

class TestNLPClassifier:
    """Tests for NLP classification logic (mocked — no model download)."""

    def test_score_to_bin3_mapping(self):
        from src.nlp.classifier import SCORE_TO_BIN3
        assert SCORE_TO_BIN3[1] == 1
        assert SCORE_TO_BIN3[2] == 1
        assert SCORE_TO_BIN3[3] == 2
        assert SCORE_TO_BIN3[4] == 3
        assert SCORE_TO_BIN3[5] == 3

    def test_label_to_score_complete(self):
        from src.nlp.classifier import LABEL_TO_SCORE, CANDIDATE_LABELS
        for label in CANDIDATE_LABELS:
            assert label in LABEL_TO_SCORE

    def test_classify_event_mock(self):
        """Test classify_event with a mocked Claude API client."""
        from src.nlp.classifier import classify_event

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"score": 3, "label": "high"}')]
        mock_client.messages.create.return_value = mock_response

        result = classify_event(
            "Vacancia del presidente",
            date="2020-11-09",
            president="Vizcarra",
            client=mock_client,
        )
        assert result["severity_claude"] == 5  # bin3=3 maps to 5
        assert result["severity_claude_bin3"] == 3
        assert result["severity_claude_confidence"] == 1.0

    def test_monthly_event_score(self):
        from src.nlp.classifier import compute_monthly_event_score
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-05", "2020-01-15", "2020-02-10"]),
            "severity_claude": [3, 5, 2],
        })
        monthly = compute_monthly_event_score(df)
        assert monthly.iloc[0] == 5  # max of Jan
        assert monthly.iloc[1] == 2  # Feb


# ── Matching tests ──────────────────────────────────────────────────────────

class TestMatching:
    """Tests for event matching logic."""

    def test_exact_date_match(self):
        from src.ingestion.political import match_to_ground_truth
        gt = pd.DataFrame({
            "event_id": [1],
            "date": [pd.Timestamp("2020-11-09")],
            "event_description": ["Vizcarra removed by Congress"],
            "president_affected": ["Vizcarra"],
        })
        scraped = {
            "date": pd.Timestamp("2020-11-09"),
            "event_description": "Vizcarra was removed from office by Congress vote",
            "president_affected": "Vizcarra",
        }
        matches = match_to_ground_truth(scraped, gt)
        assert len(matches) >= 1
        best = matches[0]
        assert best["ground_truth_event_id"] == 1
        assert best["match_method"] == "exact_date"

    def test_fuzzy_date_match(self):
        from src.ingestion.political import match_to_ground_truth
        gt = pd.DataFrame({
            "event_id": [1],
            "date": [pd.Timestamp("2020-11-09")],
            "event_description": ["Vizcarra removed"],
            "president_affected": ["Vizcarra"],
        })
        scraped = {
            "date": pd.Timestamp("2020-11-10"),  # 1 day off
            "event_description": "Vizcarra was removed from office",
            "president_affected": "Vizcarra",
        }
        matches = match_to_ground_truth(scraped, gt)
        assert len(matches) >= 1

    def test_no_match(self):
        from src.ingestion.political import match_to_ground_truth
        gt = pd.DataFrame({
            "event_id": [1],
            "date": [pd.Timestamp("2020-11-09")],
            "event_description": ["Vizcarra removed"],
            "president_affected": ["Vizcarra"],
        })
        scraped = {
            "date": pd.Timestamp("2015-06-15"),  # way off
            "event_description": "Unrelated political event in Peru",
            "president_affected": "Humala",
        }
        matches = match_to_ground_truth(scraped, gt)
        assert len(matches) == 0

    def test_keyword_matching(self):
        """Keyword-based matching picks up events with distinctive terms."""
        from src.ingestion.political import match_to_ground_truth
        gt = pd.DataFrame({
            "event_id": [1],
            "date": [pd.Timestamp("2023-12-06")],
            "event_description": ["Fujimori released from Barbadillo prison"],
            "president_affected": ["Fujimori"],
        })
        scraped = {
            "date": pd.Timestamp("2023-12-05"),
            "event_description": "The court ordered the release of Alberto Fujimori",
            "president_affected": "Boluarte",
        }
        matches = match_to_ground_truth(scraped, gt)
        assert len(matches) >= 1

    def test_multi_pass_matching(self):
        """Multi-pass matching handles same-date GT clusters."""
        from src.ingestion.political import match_all_events
        scraped_df = pd.DataFrame({
            "date": pd.to_datetime(["2022-12-07"]),
            "event_description": ["Castillo was arrested after attempting a self-coup"],
            "source": ["wikipedia_api"],
            "source_page": ["Pedro_Castillo"],
            "source_text_snippet": ["Castillo arrested"],
        })
        gt = pd.DataFrame({
            "event_id": [32, 33, 34],
            "date": pd.to_datetime(["2022-12-07"] * 3),
            "event_description": [
                "Castillo attempts self-coup, dissolves Congress",
                "Congress removes Castillo for moral incapacity",
                "Castillo arrested en route to Mexican embassy",
            ],
            "president_affected": ["Castillo"] * 3,
            "severity": [3, 3, 3],
            "anticipated": [0, 0, 0],
        })
        result = match_all_events(scraped_df, gt)
        matched_gt = result["ground_truth_event_id"].dropna().nunique()
        assert matched_gt == 3  # all 3 GT events matched via pass1 + pass2 + pass3

    def test_president_attribution(self):
        from src.ingestion.political import get_president_for_date
        assert get_president_for_date(datetime(2020, 6, 15)) == "Vizcarra"
        assert get_president_for_date(datetime(2005, 3, 1)) == "Toledo"
        assert get_president_for_date(datetime(2022, 8, 1)) == "Castillo"


# ── Index aggregation tests ─────────────────────────────────────────────────

class TestIndexAggregation:
    """Tests for composite index building."""

    def _make_monthly_data(self, n=120):
        """Create synthetic monthly data for testing."""
        dates = pd.date_range("2010-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(42)
        return dates, rng

    def test_build_monthly_index(self):
        from src.processing.political_index import build_monthly_index
        dates, rng = self._make_monthly_data()

        events = pd.DataFrame({"date": dates, "events_score": rng.integers(0, 5, len(dates))})
        financial = pd.DataFrame({"date": dates, "financial_score": rng.normal(0, 1, len(dates))})
        cabinet = pd.DataFrame({"date": dates, "cabinet_zscore": rng.normal(0, 1, len(dates))})
        confidence = pd.DataFrame({"date": dates, "confidence_score": rng.normal(50, 10, len(dates))})

        result = build_monthly_index(events, financial, cabinet, confidence)
        assert "composite_index" in result.columns
        assert len(result) == 120
        assert result["composite_index"].notna().sum() > 50

    def test_build_weekly_index(self):
        from src.processing.political_index import build_weekly_index
        dates = pd.date_range("2010-01-01", periods=520, freq="W-FRI")
        rng = np.random.default_rng(42)

        events = pd.DataFrame({"date": dates, "events_score": rng.integers(0, 5, len(dates))})
        financial = pd.DataFrame({"date": dates, "financial_score": rng.normal(0, 1, len(dates))})
        cabinet = pd.DataFrame({"date": dates, "cabinet_zscore": rng.normal(0, 1, len(dates))})

        result = build_weekly_index(events, financial, cabinet)
        assert "weekly_index" in result.columns
        assert len(result) == 520

    def test_weights_sum_to_one(self):
        from src.processing.political_index import MONTHLY_WEIGHTS, WEEKLY_WEIGHTS
        assert abs(sum(MONTHLY_WEIGHTS.values()) - 1.0) < 0.001
        assert abs(sum(WEEKLY_WEIGHTS.values()) - 1.0) < 0.001

    def test_prepare_events_monthly(self):
        from src.processing.political_index import prepare_events_monthly
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-05", "2020-01-15", "2020-02-10"]),
            "severity_claude": [3, 5, 2],
        })
        monthly = prepare_events_monthly(df, warmup_months=0)
        assert len(monthly) == 2
        # Sum of weights: Jan = 3/5 + 5/5 = 1.6, Feb = 2/5 = 0.4
        assert abs(monthly.iloc[0]["events_score"] - 1.6) < 0.01
        assert abs(monthly.iloc[1]["events_score"] - 0.4) < 0.01
        # With warmup, should have more months
        monthly_warm = prepare_events_monthly(df, warmup_months=6)
        assert len(monthly_warm) == 8  # 6 warmup + 2 event months
        assert monthly_warm.iloc[0]["events_score"] == 0  # warmup zero
        assert abs(monthly_warm.iloc[-2]["events_score"] - 1.6) < 0.01


# ── Event study tests ───────────────────────────────────────────────────────

class TestEventStudy:
    """Tests for event study analysis."""

    def test_compute_event_window(self):
        from src.analysis.event_study import compute_event_window
        dates = pd.date_range("2020-11-01", "2020-11-20", freq="B")
        series = pd.Series(
            np.arange(len(dates), dtype=float),
            index=dates,
        )
        result = compute_event_window(series, pd.Timestamp("2020-11-09"), window_days=5)
        assert "change" in result
        assert not np.isnan(result["change"])
        assert result["change"] > 0  # upward trend

    def test_anticipated_effect(self):
        from src.analysis.event_study import test_anticipated_effect
        df = pd.DataFrame({
            "anticipated": [0, 0, 0, 1, 1, 1],
            "embi_change": [5.0, 4.0, 6.0, 1.0, 2.0, 1.5],
            "embi_abs_change": [5.0, 4.0, 6.0, 1.0, 2.0, 1.5],
        })
        result = test_anticipated_effect(df)
        assert result["test_possible"]
        assert result["surprise_larger"]  # surprise should have bigger effect


# ── Validation metrics tests ────────────────────────────────────────────────

class TestValidationMetrics:
    """Tests for validation metric computation."""

    def test_perfect_agreement(self):
        from src.nlp.validator import compute_validation_metrics
        df = pd.DataFrame({
            "severity_gt": [1, 2, 3, 1, 2, 3],
            "severity_claude_bin3": [1, 2, 3, 1, 2, 3],
            "severity_sonnet": [pd.NA] * 6,
            "severity_sonnet_bin3": [pd.NA] * 6,
        })
        metrics = compute_validation_metrics(df)
        assert metrics["claude_vs_gt_accuracy"] == 1.0
        assert metrics["claude_vs_gt_severe_errors"] == 0
        assert metrics["claude_vs_gt_kappa"] == 1.0

    def test_severe_errors_detected(self):
        from src.nlp.validator import compute_validation_metrics
        df = pd.DataFrame({
            "severity_gt": [3, 1],
            "severity_claude_bin3": [1, 3],  # both are severe errors
            "severity_sonnet": [pd.NA, pd.NA],
            "severity_sonnet_bin3": [pd.NA, pd.NA],
        })
        metrics = compute_validation_metrics(df)
        assert metrics["claude_vs_gt_severe_errors"] == 2


# ── Integration test (no network, no model) ─────────────────────────────────

class TestPipelineIntegration:
    """Integration test using GT data and mocked components."""

    def test_full_pipeline_structure(self):
        """Verify all modules can be imported and basic data flows work."""
        from src.ingestion.political import (
            load_ground_truth,
            extract_cabinet_timeline,
            get_president_for_date,
        )
        from src.processing.cabinet_stability import compute_cabinet_instability
        from src.processing.political_index import (
            build_monthly_index,
            prepare_events_monthly,
        )

        # Load GT
        gt = load_ground_truth()
        assert len(gt["events"]) == 70

        # Cabinet timeline
        cabinet = extract_cabinet_timeline("")
        assert len(cabinet) >= 30

        # Cabinet stability
        stability = compute_cabinet_instability(cabinet, freq="M")
        assert len(stability) > 100

        # Prepare events (use GT as events)
        events = gt["events"].copy()
        events["severity_claude"] = events["severity"].map({1: 1, 2: 3, 3: 5})
        events_monthly = prepare_events_monthly(events)

        # Build index
        n = len(events_monthly)
        financial = pd.DataFrame({"date": events_monthly["date"], "financial_score": 0.0})
        cabinet_m = stability[stability["date"].isin(events_monthly["date"])].copy()
        if len(cabinet_m) < n:
            cabinet_m = pd.DataFrame({
                "date": events_monthly["date"],
                "cabinet_zscore": 0.0,
            })
        confidence = pd.DataFrame({"date": events_monthly["date"], "confidence_score": 50.0})

        monthly_index = build_monthly_index(events_monthly, financial, cabinet_m, confidence)
        assert "composite_index" in monthly_index.columns
