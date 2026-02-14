"""Tests for the unified NEXUS orchestrator (update_nexus.py).

Validates that:
- All 11 steps are registered and callable
- Dry-run returns correct structure for each step
- VALID_STEPS matches STEPS dict keys
- run_all() handles --check correctly
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.update_nexus import (
    STEPS,
    VALID_STEPS,
    run_all,
    step_bcrp,
    step_regional_download,
    step_expanded,
    step_midagri,
    step_supermarket,
    step_gdp,
    step_inflation,
    step_enaho,
    step_panel,
    step_regional,
    step_political,
    step_daily_rss,
    step_viz,
    step_reports,
)


class TestStepRegistry:
    """Test that the step registry is complete and consistent."""

    def test_valid_steps_matches_steps_dict(self):
        assert VALID_STEPS == set(STEPS.keys())

    def test_all_14_steps_registered(self):
        assert len(STEPS) == 14

    def test_step_names(self):
        expected = {
            "bcrp", "regional_download", "expanded", "midagri", "supermarket",
            "gdp", "inflation", "enaho",
            "panel", "regional", "political", "daily_rss", "viz", "reports",
        }
        assert set(STEPS.keys()) == expected

    def test_each_step_has_description_and_callable(self):
        for name, (description, func) in STEPS.items():
            assert isinstance(description, str), f"{name} missing description"
            assert len(description) > 0, f"{name} has empty description"
            assert callable(func), f"{name} function not callable"

    def test_step_order(self):
        """Steps should be in logical order."""
        step_names = list(STEPS.keys())
        assert step_names.index("bcrp") < step_names.index("panel")
        assert step_names.index("regional_download") < step_names.index("regional")
        assert step_names.index("enaho") < step_names.index("viz")


class TestDryRun:
    """Test that each step's dry-run returns proper structure."""

    def test_step_bcrp_dry_run(self):
        result = step_bcrp(dry_run=True)
        assert "status" in result
        assert result["status"] == "dry_run"

    def test_step_regional_download_dry_run(self):
        result = step_regional_download(dry_run=True)
        assert result["status"] == "dry_run"
        assert "series_count" in result
        assert result["series_count"] > 0
        assert "categories" in result

    def test_step_expanded_dry_run(self):
        result = step_expanded(dry_run=True)
        assert result["status"] == "dry_run"
        assert "estimated_series" in result

    def test_step_gdp_dry_run(self):
        result = step_gdp(dry_run=True)
        assert result["status"] == "dry_run"
        assert "series_count" in result

    def test_step_inflation_dry_run(self):
        result = step_inflation(dry_run=True)
        assert result["status"] == "dry_run"

    def test_step_enaho_dry_run(self):
        result = step_enaho(dry_run=True)
        assert result["status"] == "dry_run"
        assert "missing_years" in result

    def test_step_panel_dry_run(self):
        result = step_panel(dry_run=True)
        assert result["status"] == "dry_run"
        assert "series_count" in result

    def test_step_regional_dry_run(self):
        result = step_regional(dry_run=True)
        assert result["status"] == "dry_run"
        assert "categories" in result

    def test_step_political_dry_run(self):
        result = step_political(dry_run=True)
        assert result["status"] == "dry_run"
        assert "has_api_key" in result
        assert "has_cached_pages" in result

    def test_step_viz_dry_run(self):
        result = step_viz(dry_run=True)
        assert result["status"] == "dry_run"

    def test_step_reports_dry_run(self):
        result = step_reports(dry_run=True)
        assert result["status"] == "dry_run"


class TestRunAll:
    """Test the run_all orchestrator in dry-run mode."""

    def test_run_all_dry_run(self):
        results = run_all(dry_run=True)
        assert len(results) == 14
        for name, result in results.items():
            assert "status" in result, f"{name} missing status"
            assert "elapsed_seconds" in result, f"{name} missing elapsed_seconds"

    def test_run_all_single_step(self):
        results = run_all(dry_run=True, only="viz")
        assert len(results) == 1
        assert "viz" in results

    def test_run_all_returns_all_step_names(self):
        results = run_all(dry_run=True)
        assert set(results.keys()) == VALID_STEPS
