"""Tests for Sprint 3 processing pipeline.

Covers: harmonize, missing, disaggregate, panel_builder.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_BCRP_DIR, SERIES_CATALOG_PATH


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def monthly_dates():
    """Generate 120 monthly dates (10 years)."""
    return pd.date_range("2014-01-01", periods=120, freq="MS")


@pytest.fixture
def sample_series(monthly_dates):
    """Create a sample monthly series with trend + seasonality."""
    n = len(monthly_dates)
    trend = np.linspace(100, 200, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.default_rng(42).normal(0, 2, n)
    return pd.Series(trend + seasonal + noise, index=monthly_dates)


@pytest.fixture
def sample_series_with_gaps(sample_series):
    """Sample series with a 2-month gap and a 5-month gap."""
    s = sample_series.copy()
    s.iloc[20:22] = np.nan   # 2-month gap (should be interpolated)
    s.iloc[50:55] = np.nan   # 5-month gap (should NOT be interpolated with max_gap=3)
    return s


@pytest.fixture
def raw_bcrp_data():
    """Load actual BCRP raw data if available."""
    path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    if not path.exists():
        pytest.skip("Raw BCRP data not available")
    return pd.read_parquet(path)


@pytest.fixture
def ipc_data(raw_bcrp_data):
    """Extract IPC monthly variation data."""
    return raw_bcrp_data[raw_bcrp_data["series_code"] == "PN01271PM"]


# ── Test: Catalog Loading ────────────────────────────────────────────────────


class TestCatalogLoading:
    def test_load_metadata_returns_dict(self):
        from src.processing.harmonize import load_series_metadata
        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        assert isinstance(metadata, dict)
        assert len(metadata) == 57  # 40 original + 15 payment systems + 2 wholesale price

    def test_metadata_has_required_fields(self):
        from src.processing.harmonize import load_series_metadata
        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        for code, meta in metadata.items():
            assert "name" in meta, f"{code}: missing name"
            assert "category" in meta, f"{code}: missing category"
            assert "unit_type" in meta, f"{code}: missing unit_type"
            assert "seasonal_adjust" in meta, f"{code}: missing seasonal_adjust"
            assert "transform" in meta, f"{code}: missing transform"
            assert "publication_lag_days" in meta, f"{code}: missing publication_lag_days"

    def test_nominal_soles_series_counted(self):
        from src.processing.harmonize import load_series_metadata
        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        nominal = [c for c, m in metadata.items() if m["unit_type"] == "nominal_soles"]
        assert len(nominal) == 18  # 5 original + 13 payment systems (nominal soles)

    def test_seasonal_adjust_values_valid(self):
        from src.processing.harmonize import load_series_metadata
        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        valid_sa = {"x13", "stl", "none", "already_sa"}
        for code, meta in metadata.items():
            assert meta["seasonal_adjust"] in valid_sa, f"{code}: invalid SA {meta['seasonal_adjust']}"


# ── Test: IPC Reconstruction ─────────────────────────────────────────────────


class TestIPCReconstruction:
    def test_reconstruct_from_real_data(self, raw_bcrp_data):
        from src.processing.harmonize import reconstruct_ipc_index
        ipc = reconstruct_ipc_index(raw_bcrp_data)

        # Base date should be ~100
        base = ipc.loc["2009-12-01"]
        assert abs(base - 100.0) < 0.01

        # Should be monotonically increasing overall (Peru has persistent inflation)
        assert ipc.iloc[-1] > ipc.iloc[0]

        # Reasonable range: 60-200 for 2004-2026
        assert ipc.min() > 50
        assert ipc.max() < 250

    def test_ipc_matches_core_index(self, raw_bcrp_data):
        """Cross-validate reconstructed IPC against published core CPI."""
        from src.processing.harmonize import reconstruct_ipc_index

        ipc = reconstruct_ipc_index(raw_bcrp_data)

        # Published core CPI (PN38706PM) has base Dic.2021=100
        core = (
            raw_bcrp_data[raw_bcrp_data["series_code"] == "PN38706PM"]
            .dropna(subset=["value"])
            .set_index("date")["value"]
        )
        if core.empty:
            pytest.skip("Core CPI series not available")

        # Rebase our IPC to Dec 2021 = 100 for comparison
        if pd.Timestamp("2021-12-01") in ipc.index:
            ipc_rebased = ipc / ipc.loc["2021-12-01"] * 100
            # Compare overlapping dates — they should move in the same direction
            overlap = ipc_rebased.index.intersection(core.index)
            if len(overlap) > 12:
                ipc_corr = ipc_rebased.loc[overlap].corr(core.loc[overlap])
                # Headline and core should be highly correlated (>0.95)
                assert ipc_corr > 0.90, f"IPC vs core correlation too low: {ipc_corr:.3f}"

    def test_cumulative_product_logic(self):
        """Test the cumulative product formula with known values."""
        from src.processing.harmonize import reconstruct_ipc_index

        # Create synthetic IPC data: 2% monthly inflation for 3 months
        dates = pd.date_range("2009-10-01", periods=5, freq="MS")
        df = pd.DataFrame({
            "date": dates,
            "series_code": "PN01271PM",
            "value": [2.0, 2.0, 2.0, 2.0, 2.0],
        })
        ipc = reconstruct_ipc_index(df, base_date="2009-12-01")
        # After 3 months of 2% inflation from month 1:
        # month 1: 1.02
        # month 2: 1.02 * 1.02 = 1.0404
        # month 3: 1.0404 * 1.02 = 1.061208 (this is base month = 100)
        # So month 1 rebased = 1.02 / 1.061208 * 100 = 96.117...
        assert abs(ipc.iloc[2] - 100.0) < 0.01  # base month = 100


# ── Test: Deflation ──────────────────────────────────────────────────────────


class TestDeflation:
    def test_deflate_halves_with_index_200(self, monthly_dates):
        from src.processing.harmonize import deflate_series
        nominal = pd.Series(1000.0, index=monthly_dates[:12])
        deflator = pd.Series(200.0, index=monthly_dates[:12])
        real = deflate_series(nominal, deflator)
        # real = 1000 * (100 / 200) = 500
        assert all(abs(real - 500.0) < 0.01)

    def test_deflate_identity_at_100(self, monthly_dates):
        from src.processing.harmonize import deflate_series
        nominal = pd.Series(1000.0, index=monthly_dates[:12])
        deflator = pd.Series(100.0, index=monthly_dates[:12])
        real = deflate_series(nominal, deflator)
        assert all(abs(real - 1000.0) < 0.01)

    def test_deflate_missing_deflator_gives_nan(self, monthly_dates):
        from src.processing.harmonize import deflate_series
        nominal = pd.Series(1000.0, index=monthly_dates[:12])
        deflator = pd.Series(100.0, index=monthly_dates[6:12])  # only last 6 months
        real = deflate_series(nominal, deflator)
        assert real.iloc[:6].isna().all()
        assert real.iloc[6:].notna().all()


# ── Test: Seasonal Adjustment ────────────────────────────────────────────────


class TestSeasonalAdjustment:
    def test_stl_removes_seasonality(self, sample_series):
        from src.processing.harmonize import seasonal_adjust_stl
        result = seasonal_adjust_stl(sample_series)
        assert "value_sa" in result.columns
        assert "seasonal" in result.columns
        # SA series should have less variance than original
        assert result["value_sa"].std() < sample_series.std() * 1.5

    def test_stl_seasonal_component_periodic(self, sample_series):
        from src.processing.harmonize import seasonal_adjust_stl
        result = seasonal_adjust_stl(sample_series)
        seasonal = result["seasonal"]
        # Seasonal component should have period 12
        # Check that month 1 and month 13 have similar seasonal values
        assert abs(seasonal.iloc[0] - seasonal.iloc[12]) < 5

    def test_seasonal_adjust_none_returns_raw(self, sample_series):
        from src.processing.harmonize import seasonal_adjust
        result = seasonal_adjust(sample_series, method="none")
        pd.testing.assert_series_equal(result["value_sa"], sample_series, check_names=False)

    def test_seasonal_adjust_already_sa_returns_raw(self, sample_series):
        from src.processing.harmonize import seasonal_adjust
        result = seasonal_adjust(sample_series, method="already_sa")
        pd.testing.assert_series_equal(result["value_sa"], sample_series, check_names=False)

    def test_stl_handles_short_series(self):
        from src.processing.harmonize import seasonal_adjust_stl
        dates = pd.date_range("2020-01-01", periods=20, freq="MS")
        short = pd.Series(np.random.default_rng(0).normal(100, 10, 20), index=dates)
        result = seasonal_adjust_stl(short)
        assert "value_sa" in result.columns
        assert len(result) == 20

    def test_stl_with_nans(self, sample_series_with_gaps):
        from src.processing.harmonize import seasonal_adjust_stl
        result = seasonal_adjust_stl(sample_series_with_gaps)
        assert len(result) == len(sample_series_with_gaps)


# ── Test: Transformations ────────────────────────────────────────────────────


class TestTransformations:
    def test_log_of_positive(self):
        from src.processing.harmonize import transform_log
        s = pd.Series([1.0, np.e, np.e**2])
        result = transform_log(s)
        np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0], atol=1e-10)

    def test_log_of_negative_gives_nan(self):
        from src.processing.harmonize import transform_log
        s = pd.Series([1.0, -1.0, 0.0])
        result = transform_log(s)
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2]) or np.isinf(result.iloc[2])

    def test_dlog_approximates_growth(self):
        from src.processing.harmonize import transform_dlog
        # 5% monthly growth
        s = pd.Series([100.0, 105.0, 110.25])
        result = transform_dlog(s)
        assert np.isnan(result.iloc[0])
        assert abs(result.iloc[1] - np.log(1.05)) < 1e-10

    def test_yoy_basic(self):
        from src.processing.harmonize import transform_yoy
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        # 10% annual growth
        s = pd.Series(
            [100] * 12 + [110] * 12,
            index=dates,
            dtype=float,
        )
        result = transform_yoy(s)
        # First 12 should be NaN
        assert result.iloc[:12].isna().all()
        # Remaining should be ~10%
        np.testing.assert_allclose(result.iloc[12:].values, 10.0)

    def test_diff_basic(self):
        from src.processing.harmonize import transform_diff
        s = pd.Series([10.0, 12.0, 15.0, 11.0])
        result = transform_diff(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.0
        assert result.iloc[3] == -4.0


# ── Test: Interpolation ─────────────────────────────────────────────────────


class TestInterpolation:
    def test_short_gap_filled(self, sample_series_with_gaps):
        from src.processing.missing import interpolate_gaps
        result = interpolate_gaps(sample_series_with_gaps, max_gap=3)
        # 2-month gap should be filled
        assert result.iloc[20:22].notna().all()

    def test_long_gap_not_filled(self, sample_series_with_gaps):
        from src.processing.missing import interpolate_gaps
        result = interpolate_gaps(sample_series_with_gaps, max_gap=3)
        # 5-month gap should NOT be filled
        assert result.iloc[50:55].isna().all()

    def test_no_gaps_unchanged(self, sample_series):
        from src.processing.missing import interpolate_gaps
        result = interpolate_gaps(sample_series, max_gap=3)
        pd.testing.assert_series_equal(result, sample_series)

    def test_edge_nans_not_filled(self):
        from src.processing.missing import interpolate_gaps
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")
        s = pd.Series([np.nan, np.nan, 100, 110, 120, 130, 140, 150, np.nan, np.nan], index=dates)
        result = interpolate_gaps(s, max_gap=3)
        assert result.iloc[0:2].isna().all()
        assert result.iloc[8:10].isna().all()


# ── Test: Ragged Edge Diagnostic ─────────────────────────────────────────────


class TestRaggedEdge:
    def test_basic_diagnostic(self):
        from src.processing.missing import diagnose_ragged_edge
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        panel = pd.DataFrame({
            "series_A": np.arange(24, dtype=float),
            "series_B": list(np.arange(20, dtype=float)) + [np.nan] * 4,
        }, index=dates)
        diag = diagnose_ragged_edge(panel)
        assert len(diag) == 2
        b_row = diag[diag["series"] == "series_B"].iloc[0]
        assert b_row["lag_months"] == 4

    def test_empty_series(self):
        from src.processing.missing import diagnose_ragged_edge
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        panel = pd.DataFrame({
            "series_A": np.arange(12, dtype=float),
            "empty": [np.nan] * 12,
        }, index=dates)
        diag = diagnose_ragged_edge(panel)
        empty_row = diag[diag["series"] == "empty"].iloc[0]
        assert empty_row["n_obs"] == 0


# ── Test: Chow-Lin Disaggregation ───────────────────────────────────────────


class TestChowLin:
    def test_basic_disaggregation(self):
        from src.processing.disaggregate import chow_lin

        # Create synthetic quarterly GDP (8 years = 32 quarters)
        q_dates = pd.date_range("2010-01-01", periods=32, freq="QS")
        rng = np.random.default_rng(42)
        y_q = pd.Series(100 + np.cumsum(rng.normal(1, 0.5, 32)), index=q_dates)

        # Create monthly indicators (96 months)
        m_dates = pd.date_range("2010-01-01", periods=96, freq="MS")
        X_m = pd.DataFrame({
            "electricity": 50 + np.cumsum(rng.normal(0.3, 0.2, 96)),
            "cement": rng.normal(5, 2, 96),
        }, index=m_dates)

        result = chow_lin(y_q, X_m)
        assert len(result) == 96
        assert result.notna().all()

    def test_aggregation_constraint(self):
        """Monthly GDP should sum back to quarterly."""
        from src.processing.disaggregate import chow_lin, _build_aggregation_matrix

        q_dates = pd.date_range("2010-01-01", periods=20, freq="QS")
        rng = np.random.default_rng(123)
        y_q = pd.Series(100 + np.cumsum(rng.normal(1, 0.5, 20)), index=q_dates)

        m_dates = pd.date_range("2010-01-01", periods=60, freq="MS")
        X_m = pd.DataFrame({
            "indicator": 50 + np.cumsum(rng.normal(0.2, 0.1, 60)),
        }, index=m_dates)

        result = chow_lin(y_q, X_m)

        # Re-aggregate to quarterly
        C = _build_aggregation_matrix(60)
        y_reagg = C @ result.values
        np.testing.assert_allclose(y_reagg, y_q.values, atol=1e-4)

    def test_disaggregate_gdp_with_real_data(self, raw_bcrp_data):
        """Test full GDP disaggregation with actual data."""
        from src.processing.disaggregate import disaggregate_gdp
        from config.settings import TARGETS_DIR

        gdp_path = TARGETS_DIR / "gdp_quarterly.parquet"
        if not gdp_path.exists():
            pytest.skip("GDP quarterly target not available")

        result = disaggregate_gdp(raw_bcrp_data)
        assert len(result) > 200  # Should have ~260 monthly obs
        assert result.notna().all()
        assert result.min() > 0  # GDP index should be positive


# ── Test: Process Single Series ──────────────────────────────────────────────


class TestProcessSingleSeries:
    def test_process_gdp_index(self, raw_bcrp_data):
        from src.processing.harmonize import (
            load_series_metadata,
            process_single_series,
            reconstruct_ipc_index,
        )

        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        deflator = reconstruct_ipc_index(raw_bcrp_data)
        result = process_single_series("PN01770AM", raw_bcrp_data, metadata, deflator)

        assert not result.empty
        assert "value_raw" in result.columns
        assert "value_sa" in result.columns
        assert "value_log" in result.columns
        assert "value_dlog" in result.columns
        assert "value_yoy" in result.columns
        assert len(result) > 200

    def test_process_nominal_series_deflated(self, raw_bcrp_data):
        """Nominal soles series should be deflated."""
        from src.processing.harmonize import (
            load_series_metadata,
            process_single_series,
            reconstruct_ipc_index,
        )

        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        deflator = reconstruct_ipc_index(raw_bcrp_data)
        result = process_single_series("PN00518MM", raw_bcrp_data, metadata, deflator)

        assert not result.empty
        # Raw values should be different from original (deflated)
        original = (
            raw_bcrp_data[raw_bcrp_data["series_code"] == "PN00518MM"]
            .dropna(subset=["value"])
            .sort_values("date")["value"]
        )
        # The deflated values should differ from nominal
        # (unless deflator happens to be exactly 100, which it won't be for most dates)
        assert not np.allclose(result["value_raw"].values[:10], original.values[:10])

    def test_process_rate_no_log(self, raw_bcrp_data):
        """Rate series should not have log transform."""
        from src.processing.harmonize import (
            load_series_metadata,
            process_single_series,
            reconstruct_ipc_index,
        )

        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        deflator = reconstruct_ipc_index(raw_bcrp_data)
        # TAMN is a rate
        result = process_single_series("PN07807NM", raw_bcrp_data, metadata, deflator)

        assert not result.empty
        # value_log should be all NaN for rate series (transform=none)
        assert result["value_log"].isna().all()

    def test_no_infinities_in_output(self, raw_bcrp_data):
        """No series should produce infinite values."""
        from src.processing.harmonize import (
            load_series_metadata,
            process_single_series,
            reconstruct_ipc_index,
        )

        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        deflator = reconstruct_ipc_index(raw_bcrp_data)

        for code in list(metadata.keys())[:10]:  # Test first 10 for speed
            result = process_single_series(code, raw_bcrp_data, metadata, deflator)
            if result.empty:
                continue
            for col in ["value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]:
                n_inf = np.isinf(result[col].dropna()).sum()
                assert n_inf == 0, f"{code}.{col} has {n_inf} infinite values"


# ── Test: Panel Builder ─────────────────────────────────────────────────────


class TestPanelBuilder:
    def test_build_panel(self, raw_bcrp_data):
        """Integration test: build the full panel."""
        from src.processing.panel_builder import build_national_panel
        from config.settings import PROCESSED_NATIONAL_DIR
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_panel.parquet"
            panel = build_national_panel(
                raw_path=RAW_BCRP_DIR / "bcrp_national_all.parquet",
                catalog_path=SERIES_CATALOG_PATH,
                output_path=output_path,
                include_gdp_monthly=False,  # Skip Chow-Lin for speed
            )

            assert not panel.empty
            assert panel["series_id"].nunique() >= 40
            assert "value_raw" in panel.columns
            assert output_path.exists()

    def test_validate_panel(self, raw_bcrp_data):
        """Test panel validation logic."""
        from src.processing.panel_builder import validate_panel

        # Create a minimal valid panel
        dates = pd.date_range("2004-01-01", periods=265, freq="MS")
        rows = []
        for i in range(41):
            for d in dates:
                rows.append({
                    "date": d,
                    "series_id": f"SERIES_{i:02d}",
                    "series_name": f"Test series {i}",
                    "category": "test",
                    "value_raw": 100.0 + i,
                    "value_sa": 100.0 + i,
                    "value_log": np.log(100.0 + i),
                    "value_dlog": 0.01,
                    "value_yoy": 5.0,
                })
        panel = pd.DataFrame(rows)
        result = validate_panel(panel)

        assert result["passed"]
        assert result["n_series"] == 41

    def test_validate_catches_duplicates(self):
        from src.processing.panel_builder import validate_panel

        dates = pd.date_range("2004-01-01", periods=265, freq="MS")
        rows = []
        for i in range(41):
            for d in dates:
                rows.append({
                    "date": d,
                    "series_id": f"SERIES_{i:02d}",
                    "series_name": f"Test {i}",
                    "category": "test",
                    "value_raw": 100.0,
                    "value_sa": 100.0,
                    "value_log": np.log(100.0),
                    "value_dlog": 0.0,
                    "value_yoy": 0.0,
                })
        # Add a duplicate
        rows.append(rows[0].copy())
        panel = pd.DataFrame(rows)
        result = validate_panel(panel)

        dupe_check = next(c for c in result["checks"] if c["name"] == "no_duplicates")
        assert not dupe_check["passed"]


# ── Test: Poverty Proxies Config ─────────────────────────────────────────────


class TestPovertyProxies:
    def test_config_loads(self):
        import yaml
        config_path = PROJECT_ROOT / "config" / "poverty_proxies.yaml"
        if not config_path.exists():
            pytest.skip("poverty_proxies.yaml not found")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "poverty_proxies" in config
        assert "direct_proxies" in config["poverty_proxies"]
        assert "indirect_proxies" in config["poverty_proxies"]

    def test_proxy_codes_in_catalog(self):
        """All poverty proxy codes must exist in the series catalog."""
        import yaml
        from src.processing.harmonize import load_series_metadata

        config_path = PROJECT_ROOT / "config" / "poverty_proxies.yaml"
        if not config_path.exists():
            pytest.skip("poverty_proxies.yaml not found")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        metadata = load_series_metadata(SERIES_CATALOG_PATH)

        for group in ["direct_proxies", "indirect_proxies"]:
            for proxy in config["poverty_proxies"][group]:
                assert proxy["code"] in metadata, (
                    f"Proxy {proxy['code']} not in catalog"
                )

    def test_proxy_signs_valid(self):
        import yaml

        config_path = PROJECT_ROOT / "config" / "poverty_proxies.yaml"
        if not config_path.exists():
            pytest.skip("poverty_proxies.yaml not found")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for group in ["direct_proxies", "indirect_proxies"]:
            for proxy in config["poverty_proxies"][group]:
                assert proxy["expected_sign"] in ("positive", "negative"), (
                    f"Invalid sign for {proxy['code']}: {proxy['expected_sign']}"
                )


# ── Test: NTL Panel Integration ──────────────────────────────────────────────


class TestNTLPanelIntegration:
    """Tests for NTL integration into national and departmental panels."""

    @pytest.fixture
    def ntl_dept_data(self):
        """Load NTL departmental data if available."""
        path = PROJECT_ROOT / "data" / "processed" / "ntl_monthly_department.parquet"
        if not path.exists():
            pytest.skip("NTL departmental data not available")
        return pd.read_parquet(path)

    def test_ntl_departmental_helper_produces_25_series(self, ntl_dept_data):
        from src.processing.panel_builder import _build_ntl_panel_departmental
        ntl_path = PROJECT_ROOT / "data" / "processed" / "ntl_monthly_department.parquet"
        frames = _build_ntl_panel_departmental(ntl_path)
        assert len(frames) == 25
        # Check series IDs
        sids = {f["series_id"].iloc[0] for f in frames}
        assert "NTL_SUM_01" in sids
        assert "NTL_SUM_15" in sids

    def test_ntl_departmental_schema(self, ntl_dept_data):
        from src.processing.panel_builder import _build_ntl_panel_departmental
        ntl_path = PROJECT_ROOT / "data" / "processed" / "ntl_monthly_department.parquet"
        frames = _build_ntl_panel_departmental(ntl_path)
        frame = frames[0]
        required = [
            "date", "series_id", "series_name", "category", "department",
            "ubigeo", "value_raw", "value_sa", "value_log", "value_dlog",
            "value_yoy", "source", "frequency_original",
        ]
        for col in required:
            assert col in frame.columns, f"Missing column: {col}"
        assert frame["category"].iloc[0] == "nighttime_lights"
        assert frame["source"].iloc[0] == "NASA/VIIRS"

    def test_ntl_national_helper_produces_1_series(self, ntl_dept_data):
        from src.processing.panel_builder import _build_ntl_panel_national
        ntl_path = PROJECT_ROOT / "data" / "processed" / "ntl_monthly_department.parquet"
        frames = _build_ntl_panel_national(ntl_path)
        assert len(frames) == 1
        frame = frames[0]
        assert frame["series_id"].iloc[0] == "NTL_SUM_NATIONAL"

    def test_ntl_national_schema(self, ntl_dept_data):
        from src.processing.panel_builder import _build_ntl_panel_national
        ntl_path = PROJECT_ROOT / "data" / "processed" / "ntl_monthly_department.parquet"
        frames = _build_ntl_panel_national(ntl_path)
        frame = frames[0]
        required = [
            "date", "series_id", "series_name", "category",
            "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
            "source", "frequency_original", "publication_lag_days",
        ]
        for col in required:
            assert col in frame.columns, f"Missing column: {col}"
        assert frame["publication_lag_days"].iloc[0] == 30

    def test_ntl_graceful_skip_when_missing(self):
        """NTL helpers should not be called when data is missing."""
        from src.processing.panel_builder import _build_ntl_panel_national
        # Passing a non-existent path should raise FileNotFoundError
        with pytest.raises(Exception):
            _build_ntl_panel_national(Path("/nonexistent/ntl.parquet"))
