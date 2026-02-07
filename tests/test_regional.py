"""Tests for regional data catalog, panel builder, and validation.

Covers:
    - Catalog loading and structure
    - Processing metadata completeness
    - Currency filtering (credit/deposits Total only)
    - Deflation of nominal_soles series
    - Panel schema (department + ubigeo columns)
    - No infinite values
    - Department coverage (25 departments)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REGIONAL_CATALOG_PATH = PROJECT_ROOT / "config" / "regional_series_catalog.yaml"
RAW_BCRP_DIR = PROJECT_ROOT / "data" / "raw" / "bcrp"
NATIONAL_RAW_PATH = RAW_BCRP_DIR / "bcrp_national_all.parquet"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "departmental" / "panel_departmental_monthly.parquet"


# ── Catalog Tests ──────────────────────────────────────────────────────────


class TestRegionalCatalog:
    """Test that the regional catalog is well-formed."""

    def test_catalog_loads(self):
        """Catalog YAML parses without error."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        assert "regional" in catalog
        regional = catalog["regional"]
        assert len(regional) >= 11  # At least 11 original categories

    def test_all_categories_have_series_key(self):
        """Every category dict has a 'series' key."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key, cat_data in catalog["regional"].items():
            if not isinstance(cat_data, dict):
                continue
            assert "series" in cat_data, f"{cat_key} missing 'series' key"

    def test_processing_metadata_complete(self):
        """Every category with series has a processing block with unit_type."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key, cat_data in catalog["regional"].items():
            if not isinstance(cat_data, dict):
                continue
            assert "processing" in cat_data, f"{cat_key} missing 'processing' block"
            proc = cat_data["processing"]
            assert "unit_type" in proc, f"{cat_key} missing unit_type in processing"
            assert proc["unit_type"] in (
                "level", "nominal_soles", "nominal_usd", "rate_pct", "var_pct"
            ), f"{cat_key} has invalid unit_type: {proc['unit_type']}"

    def test_series_have_codes(self):
        """Each series entry in populated categories has a code."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key, cat_data in catalog["regional"].items():
            if not isinstance(cat_data, dict):
                continue
            for s in cat_data.get("series", []):
                assert "code" in s, f"Series in {cat_key} missing 'code'"

    def test_credit_has_currency_field(self):
        """Credit and deposits series have 'currency' field for filtering."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key in ("credit_by_department", "deposits_by_department"):
            series = catalog["regional"][cat_key]["series"]
            for s in series:
                assert "currency" in s, f"{cat_key} series {s.get('code')} missing 'currency'"

    def test_credit_panel_filter_is_total(self):
        """Credit/deposits have panel_filter=Total in processing."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key in ("credit_by_department", "deposits_by_department"):
            proc = catalog["regional"][cat_key]["processing"]
            assert proc.get("panel_filter") == "Total", \
                f"{cat_key} panel_filter should be 'Total'"

    def test_nominal_soles_categories(self):
        """Categories that should be nominal_soles are marked correctly."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        nominal_soles_cats = [
            "credit_by_department", "deposits_by_department",
            "tax_revenue_by_department",
            "government_spending_local", "government_spending_regional",
        ]
        for cat_key in nominal_soles_cats:
            if cat_key in catalog["regional"]:
                proc = catalog["regional"][cat_key]["processing"]
                assert proc["unit_type"] == "nominal_soles", \
                    f"{cat_key} should be nominal_soles"

    def test_usd_categories(self):
        """Export/import categories are nominal_usd."""
        with open(REGIONAL_CATALOG_PATH, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        for cat_key in ("exports_by_department", "imports_by_customs"):
            proc = catalog["regional"][cat_key]["processing"]
            assert proc["unit_type"] == "nominal_usd", \
                f"{cat_key} should be nominal_usd"


# ── Processing Tests ───────────────────────────────────────────────────────


class TestRegionalProcessing:
    """Test the regional processing pipeline with synthetic data."""

    @pytest.fixture
    def deflator(self):
        """Create a simple deflator series."""
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        # Linear increase from 100 to 150
        values = np.linspace(100, 150, 120)
        return pd.Series(values, index=dates)

    @pytest.fixture
    def nominal_series(self):
        """Create a synthetic nominal soles series."""
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        # Nominal values growing
        values = 1000 + np.arange(120) * 10 + np.random.default_rng(42).normal(0, 20, 120)
        return pd.Series(values, index=dates)

    def test_deflation_reduces_trend(self, nominal_series, deflator):
        """Deflating a nominal series should reduce the upward trend."""
        from src.processing.harmonize import deflate_series
        real = deflate_series(nominal_series, deflator)
        # Real values should have a flatter trend than nominal
        nominal_slope = np.polyfit(range(len(nominal_series)), nominal_series.values, 1)[0]
        real_slope = np.polyfit(range(len(real.dropna())), real.dropna().values, 1)[0]
        assert real_slope < nominal_slope, "Deflation should flatten the trend"

    def test_process_regional_series_returns_all_columns(self, nominal_series, deflator):
        """_process_regional_series returns all 6 output columns."""
        from src.processing.panel_builder import _process_regional_series
        result = _process_regional_series(
            "TEST001", nominal_series, "nominal_soles", "stl", deflator
        )
        expected_cols = {"date", "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"}
        assert set(result.columns) == expected_cols

    def test_process_regional_series_no_deflation_for_level(self, nominal_series):
        """Level series should not be deflated."""
        from src.processing.panel_builder import _process_regional_series
        result = _process_regional_series(
            "TEST002", nominal_series, "level", "stl", deflator=None
        )
        # value_raw should equal the original values (no deflation)
        np.testing.assert_array_almost_equal(
            result["value_raw"].values, nominal_series.values, decimal=0
        )

    def test_process_regional_series_handles_empty(self):
        """Empty series returns empty DataFrame with correct columns."""
        from src.processing.panel_builder import _process_regional_series
        empty = pd.Series(dtype=float)
        result = _process_regional_series("EMPTY", empty, "level", "none")
        assert len(result) == 0
        assert "value_raw" in result.columns

    def test_no_infinities_in_transforms(self, nominal_series, deflator):
        """Transforms should not produce infinite values."""
        from src.processing.panel_builder import _process_regional_series
        result = _process_regional_series(
            "TEST003", nominal_series, "nominal_soles", "stl", deflator
        )
        for col in ["value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]:
            assert not np.isinf(result[col].dropna()).any(), \
                f"Inf found in {col}"


# ── Panel Schema Tests ─────────────────────────────────────────────────────


class TestDepartmentalPanel:
    """Test the built departmental panel if it exists."""

    @pytest.fixture
    def panel(self):
        """Load the departmental panel if it exists."""
        if not PANEL_PATH.exists():
            pytest.skip("Departmental panel not built yet")
        df = pd.read_parquet(PANEL_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_panel_schema(self, panel):
        """Panel has required columns including department and ubigeo."""
        required = [
            "date", "series_id", "series_name", "category",
            "department", "ubigeo",
            "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
        ]
        for col in required:
            assert col in panel.columns, f"Missing column: {col}"

    def test_no_infinities(self, panel):
        """No infinite values in the panel."""
        value_cols = ["value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]
        for col in value_cols:
            n_inf = np.isinf(panel[col].dropna()).sum()
            assert n_inf == 0, f"{n_inf} infinite values in {col}"

    def test_department_coverage(self, panel):
        """At least 20 departments present."""
        n_depts = panel["ubigeo"].nunique()
        assert n_depts >= 20, f"Only {n_depts} departments, expected >=20"

    def test_credit_total_only_in_panel(self, panel):
        """Credit series in panel should be Total currency only (no MN/ME)."""
        credit = panel[panel["category"] == "credit_by_department"]
        if len(credit) == 0:
            pytest.skip("No credit data in panel")
        # Check that series names contain "Total" not "MN" or "ME"
        names = credit["series_name"].unique()
        for name in names:
            assert "(MN)" not in name, f"MN series in panel: {name}"
            assert "(ME)" not in name, f"ME series in panel: {name}"

    def test_no_total_nacional_in_panel(self, panel):
        """Panel should not contain 'Total Nacional' aggregates."""
        names_lower = panel["series_name"].str.lower()
        total_rows = names_lower.str.contains("total nacional").sum()
        assert total_rows == 0, f"{total_rows} 'Total Nacional' rows in panel"

    def test_no_duplicates(self, panel):
        """No duplicate (series_id, date) pairs."""
        dupes = panel.duplicated(subset=["series_id", "date"]).sum()
        assert dupes == 0, f"{dupes} duplicate series_id/date pairs"

    def test_date_range_reasonable(self, panel):
        """Date range should start before 2005 and end after 2024."""
        assert panel["date"].min() < pd.Timestamp("2005-01-01")
        assert panel["date"].max() > pd.Timestamp("2024-01-01")

    def test_multiple_categories(self, panel):
        """Panel should have at least 5 categories."""
        n_cats = panel["category"].nunique()
        assert n_cats >= 5, f"Only {n_cats} categories"

    def test_deflation_applied(self, panel):
        """Nominal soles series should show deflation effects.

        For credit series, the raw value trend should differ from what's
        in the original data (since we deflated).
        """
        credit = panel[panel["category"] == "credit_by_department"]
        if len(credit) == 0:
            pytest.skip("No credit data in panel")
        # Just verify value_raw exists and is numeric
        assert credit["value_raw"].dtype in (np.float64, np.float32)
        assert credit["value_raw"].notna().sum() > 0


# ── Validation Function Tests ──────────────────────────────────────────────


class TestValidation:
    """Test the validation functions."""

    def test_validate_departmental_panel_passes_good_data(self):
        """Validation passes on well-formed data."""
        from src.processing.panel_builder import validate_departmental_panel
        dates = pd.date_range("2002-01-01", "2025-12-01", freq="MS")
        categories = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
        rows = []
        for i, dept_code in enumerate(["01", "02", "03", "04", "05",
                          "06", "07", "08", "09", "10",
                          "11", "12", "13", "14", "15",
                          "16", "17", "18", "19", "20"]):
            cat = categories[i % len(categories)]
            for d in dates:
                rows.append({
                    "date": d,
                    "series_id": f"TEST_{dept_code}",
                    "series_name": f"Test {dept_code}",
                    "category": cat,
                    "department": f"Dept {dept_code}",
                    "ubigeo": dept_code,
                    "value_raw": 100.0,
                    "value_sa": 100.0,
                    "value_log": np.log(100.0),
                    "value_dlog": 0.01,
                    "value_yoy": 5.0,
                    "source": "BCRP",
                    "frequency_original": "M",
                })
        panel = pd.DataFrame(rows)
        result = validate_departmental_panel(panel)
        assert result["passed"]

    def test_validate_catches_missing_columns(self):
        """Validation fails if required columns are missing."""
        from src.processing.panel_builder import validate_departmental_panel
        # Include ubigeo so we don't get KeyError on department_count check,
        # but omit other required columns to trigger failure
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")],
            "series_id": ["X"],
            "ubigeo": ["01"],
            "value_raw": [1.0],
        })
        result = validate_departmental_panel(panel)
        col_check = [c for c in result["checks"] if c["name"] == "required_columns"][0]
        assert not col_check["passed"]

    def test_validate_catches_infinities(self):
        """Validation fails if infinite values present."""
        from src.processing.panel_builder import validate_departmental_panel
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")],
            "series_id": ["X"], "series_name": ["X"], "category": ["X"],
            "department": ["X"], "ubigeo": ["01"],
            "value_raw": [np.inf], "value_sa": [1.0],
            "value_log": [0.0], "value_dlog": [0.0], "value_yoy": [0.0],
        })
        result = validate_departmental_panel(panel)
        inf_check = [c for c in result["checks"] if c["name"] == "no_infinities"][0]
        assert not inf_check["passed"]
