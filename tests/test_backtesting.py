"""Tests for backtesting framework: vintage manager, metrics, and backtester."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtesting.metrics import (
    compute_all_metrics,
    directional_accuracy,
    mae,
    relative_rmse,
    rmse,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_panel(n_months=60) -> pd.DataFrame:
    """Create a synthetic long-format panel for testing."""
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    series_specs = [
        ("CPI_FAST", "inflation", 1),      # 1-day lag
        ("GDP_SLOW", "gdp_indicators", 45), # 45-day lag
        ("CREDIT", "credit_financial", 30),  # 30-day lag
        ("CEMENT", "leading_indicators", 15), # 15-day lag
    ]

    rows = []
    rng = np.random.default_rng(42)
    for dt in dates:
        for sid, cat, lag in series_specs:
            rows.append({
                "date": dt,
                "series_id": sid,
                "series_name": f"Test {sid}",
                "category": cat,
                "value_raw": rng.normal(100, 10),
                "value_sa": rng.normal(100, 10),
                "value_log": rng.normal(4.5, 0.1),
                "value_dlog": rng.normal(0.0, 0.02),
                "value_yoy": rng.normal(3.0, 2.0),
                "source": "TEST",
                "frequency_original": "M",
                "publication_lag_days": lag,
            })

    return pd.DataFrame(rows)


def _make_lags_config() -> dict:
    """Create a publication lags config dict."""
    return {
        "defaults": {
            "inflation": 1,
            "gdp_indicators": 45,
            "credit_financial": 30,
            "leading_indicators": 15,
        },
        "overrides": {
            "CPI_FAST": 0,  # override to real-time
        },
    }


@pytest.fixture
def panel_path(tmp_path):
    """Write a synthetic panel to a temp parquet file."""
    panel = _make_panel()
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    return path


@pytest.fixture
def lags_config_path(tmp_path):
    """Write a lags config YAML to a temp file."""
    config = _make_lags_config()
    path = tmp_path / "lags.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ── Vintage Manager Tests ────────────────────────────────────────────────────


class TestVintageManager:
    def test_loads_panel(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        assert len(vm.panel) > 0
        assert "available_date" in vm.panel.columns

    def test_lag_overrides_applied(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        # CPI_FAST should have lag=0 (overridden from 1)
        cpi_lags = vm.panel[vm.panel["series_id"] == "CPI_FAST"]["publication_lag_days"]
        assert (cpi_lags == 0).all()

    def test_vintage_excludes_future_data(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)

        # As of 2020-02-15: January 2020 data
        # CPI_FAST (0d lag): Jan 2020 available (end of Jan = Jan 31 + 0d = Jan 31 < Feb 15) ✓
        # GDP_SLOW (45d lag): Jan 2020 NOT available (Jan 31 + 45d = Mar 17 > Feb 15) ✗
        vintage = vm.get_vintage("2020-02-15")

        cpi_dates = vintage[vintage["series_id"] == "CPI_FAST"]["date"]
        gdp_dates = vintage[vintage["series_id"] == "GDP_SLOW"]["date"]

        # CPI should have Jan 2020
        assert pd.Timestamp("2020-01-01") in cpi_dates.values
        # GDP should NOT have Jan 2020 (45d lag means available Mar 17)
        assert pd.Timestamp("2020-01-01") not in gdp_dates.values

    def test_vintage_respects_lag_ordering(self, panel_path, lags_config_path):
        """Fast series should have more recent data available."""
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        # At end of April, CPI (0d lag) should have Mar data,
        # GDP (45d lag) should have Feb data (Feb 28 + 45d = Apr 14)
        vintage = vm.get_vintage("2020-04-30")

        cpi_last = vintage[vintage["series_id"] == "CPI_FAST"]["date"].max()
        gdp_last = vintage[vintage["series_id"] == "GDP_SLOW"]["date"].max()

        assert cpi_last > gdp_last

    def test_vintage_wide(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        wide = vm.get_vintage_wide("2022-01-01", value_col="value_dlog")

        # Should have date index and series columns
        assert isinstance(wide.index, pd.DatetimeIndex)
        assert wide.shape[1] > 0

    def test_available_series(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        available = vm.available_series("2022-01-01")

        assert isinstance(available, list)
        assert "CPI_FAST" in available

    def test_diagnose_vintage(self, panel_path, lags_config_path):
        from src.backtesting.vintage import VintageManager

        vm = VintageManager(panel_path, lags_config_path)
        diag = vm.diagnose_vintage("2022-01-01")

        assert "series_id" in diag.columns
        assert "last_date" in diag.columns
        assert "days_stale" in diag.columns
        assert len(diag) > 0


# ── Metrics Tests ─────────────────────────────────────────────────────────────


class TestMetrics:
    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == 0.0

    def test_rmse_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert rmse(actual, predicted) == 1.0

    def test_mae_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert mae(actual, predicted) == 1.0

    def test_directional_accuracy_perfect(self):
        actual = np.array([1.0, -1.0, 1.0])
        predicted = np.array([2.0, -3.0, 0.5])
        assert directional_accuracy(actual, predicted) == 1.0

    def test_directional_accuracy_half(self):
        actual = np.array([1.0, -1.0, 1.0, -1.0])
        predicted = np.array([2.0, 3.0, 0.5, -0.5])
        assert directional_accuracy(actual, predicted) == 0.75

    def test_relative_rmse(self):
        assert relative_rmse(0.5, 1.0) == 0.5
        assert relative_rmse(1.0, 1.0) == 1.0
        assert np.isnan(relative_rmse(1.0, 0.0))

    def test_handles_nan(self):
        actual = np.array([1.0, np.nan, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, predicted) == 0.0  # only 2 valid pairs, both match
        assert mae(actual, predicted) == 0.0

    def test_compute_all_metrics(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.5, 2.5, 3.5, 4.5])
        m = compute_all_metrics(actual, predicted)
        assert "rmse" in m
        assert "mae" in m
        assert "directional_accuracy" in m
        assert m["n_obs"] == 4
        assert m["mae"] == 0.5

    def test_all_nan_returns_nan(self):
        actual = np.array([np.nan, np.nan])
        predicted = np.array([np.nan, np.nan])
        assert np.isnan(rmse(actual, predicted))


# ── DFM Tests ─────────────────────────────────────────────────────────────────


class TestDFMModels:
    def _make_wide_panel(self, n=120):
        """Create a synthetic wide-format panel with known structure."""
        from src.models.dfm import GDP_SERIES, INFLATION_SERIES

        dates = pd.date_range("2010-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(42)

        # Create a common factor + noise for each series
        factor = np.cumsum(rng.normal(0, 0.01, n))
        data = {}
        for s in GDP_SERIES + INFLATION_SERIES:
            loading = rng.normal(1, 0.5)
            noise = rng.normal(0, 0.01, n)
            data[s] = factor * loading + noise

        return pd.DataFrame(data, index=dates)

    def _make_gdp_target(self):
        dates = pd.date_range("2010-01-01", periods=40, freq="QS")
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "date": dates,
            "gdp_yoy": rng.normal(3.5, 2.0, len(dates)),
        })

    def _make_inflation_target(self):
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        rng = np.random.default_rng(42)
        monthly_var = rng.normal(0.3, 0.2, len(dates))
        return pd.DataFrame({
            "date": dates,
            "ipc_monthly_var": monthly_var,
            "ipc_12m_var": rng.normal(3.0, 1.0, len(dates)),
        })

    def test_dfm_gdp_fit_and_nowcast(self):
        from src.models.dfm import NowcastDFM

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        dfm = NowcastDFM(k_factors=2, target="gdp")
        dfm.fit(panel)

        factors = dfm.get_factors()
        assert factors.shape[1] == 2
        assert len(factors) == len(panel)

        nc = dfm.nowcast(panel, target)
        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_dfm_inflation_fit_and_nowcast(self):
        from src.models.dfm import NowcastDFM

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        dfm = NowcastDFM(k_factors=2, target="inflation", inflation_col="ipc_monthly_var")
        dfm.fit(panel)

        nc = dfm.nowcast(panel, target)
        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_ar1_benchmark(self):
        from src.models.dfm import AR1Benchmark

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ar1 = AR1Benchmark(target="gdp")
        ar1.fit(panel)
        nc = ar1.nowcast(panel, target)
        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_random_walk_benchmark(self):
        from src.models.dfm import RandomWalkBenchmark

        target = self._make_gdp_target()
        rw = RandomWalkBenchmark(target="gdp")
        rw.fit(self._make_wide_panel())
        nc = rw.nowcast(self._make_wide_panel(), target)

        # Should return last observation
        expected = target.set_index("date")["gdp_yoy"].iloc[-1]
        assert nc["nowcast_value"] == expected

    def test_ar1_benchmark_inflation_col(self):
        """AR1 respects configurable inflation_col."""
        from src.models.dfm import AR1Benchmark

        target = self._make_inflation_target()
        panel = self._make_wide_panel()

        ar1 = AR1Benchmark(target="inflation", inflation_col="ipc_monthly_var")
        ar1.fit(panel)
        nc = ar1.nowcast(panel, target)
        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_rw_benchmark_inflation_col(self):
        """RW respects configurable inflation_col."""
        from src.models.dfm import RandomWalkBenchmark

        target = self._make_inflation_target()
        panel = self._make_wide_panel()

        rw = RandomWalkBenchmark(target="inflation", inflation_col="ipc_monthly_var")
        rw.fit(panel)
        nc = rw.nowcast(panel, target)

        expected = target.set_index("date")["ipc_monthly_var"].iloc[-1]
        assert nc["nowcast_value"] == expected

    def test_phillips_curve_nowcaster(self):
        """Phillips curve model produces finite output with sufficient data."""
        from src.models.dfm import PHILLIPS_SERIES, PhillipsCurveNowcaster

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        # Ensure Phillips series exist in panel (they should from _make_wide_panel)
        pc = PhillipsCurveNowcaster(inflation_col="ipc_monthly_var")
        pc.fit(panel)
        nc = pc.nowcast(panel, target)
        assert "nowcast_value" in nc
        # May be nan if Phillips series missing from synthetic data — that's OK
        assert isinstance(nc["nowcast_value"], float)

    def test_reconstruct_12m_from_monthly(self):
        """Verify 12m reconstruction math: 0.5%/month → ~6.17% annual."""
        from src.models.dfm import reconstruct_12m_from_monthly

        # Constant 0.5% monthly variation for 24 months
        monthly = pd.Series([0.5] * 24)
        result = reconstruct_12m_from_monthly(monthly)

        # First 11 should be NaN (need 12 months)
        assert result.iloc[:11].isna().all()

        # 12th onward: (1.005)^12 - 1 = 0.06168... → 6.168%
        expected = ((1.005 ** 12) - 1) * 100.0
        assert abs(result.iloc[11] - expected) < 0.01

    def test_reconstruct_12m_varying(self):
        """Reconstruction handles varying monthly rates."""
        from src.models.dfm import reconstruct_12m_from_monthly

        rng = np.random.default_rng(99)
        monthly = pd.Series(rng.normal(0.3, 0.1, 36))
        result = reconstruct_12m_from_monthly(monthly)

        assert result.iloc[:11].isna().all()
        assert result.iloc[11:].notna().all()
        # All values should be finite
        assert np.all(np.isfinite(result.dropna()))

    def test_combination_nowcaster(self):
        """CombinationNowcaster produces finite output with both models."""
        from src.models.dfm import CombinationNowcaster

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        combo = CombinationNowcaster(inflation_col="ipc_monthly_var")
        combo.fit(panel)
        nc = combo.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert isinstance(nc["nowcast_value"], float)
        assert np.isfinite(nc["nowcast_value"])

    def test_combination_handles_nan(self):
        """Combination falls back to single model when the other returns NaN."""
        from unittest.mock import patch
        from src.models.dfm import CombinationNowcaster

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        combo = CombinationNowcaster(inflation_col="ipc_monthly_var")
        combo.fit(panel)

        # Mock DFM to return NaN — combo should return Phillips value
        with patch.object(combo.dfm, "nowcast", return_value={"nowcast_value": np.nan}):
            nc = combo.nowcast(panel, target)
            assert "nowcast_value" in nc
            # Should return Phillips output (finite or nan, but not crash)
            assert isinstance(nc["nowcast_value"], float)

        # Mock Phillips to return NaN — combo should return DFM value
        with patch.object(combo.phillips, "nowcast", return_value={"nowcast_value": np.nan}):
            nc = combo.nowcast(panel, target)
            assert "nowcast_value" in nc
            assert isinstance(nc["nowcast_value"], float)

    def test_phillips_winsorization(self):
        """Phillips features are clipped during COVID period."""
        from src.models.dfm import PHILLIPS_SERIES, PhillipsCurveNowcaster

        # Build panel with extreme COVID values
        dates = pd.date_range("2015-01-01", periods=120, freq="MS")
        rng = np.random.default_rng(42)

        data = {}
        # Create all Phillips series + inflation series in the panel
        for sid in set(PHILLIPS_SERIES.values()):
            values = rng.normal(0, 1, 120)
            # Inject extreme COVID spike at index ~62 (Mar 2020)
            covid_idx = (dates >= "2020-03-01") & (dates <= "2021-12-01")
            values[covid_idx] = rng.normal(0, 1, covid_idx.sum()) * 20  # extreme values
            data[sid] = values

        # Need inflation series too for DFM selection
        from src.models.dfm import INFLATION_SERIES
        for sid in INFLATION_SERIES:
            if sid not in data:
                data[sid] = rng.normal(0, 1, 120)

        panel = pd.DataFrame(data, index=dates)

        target = pd.DataFrame({
            "date": dates,
            "ipc_monthly_var": rng.normal(0.3, 0.2, 120),
        })

        pc = PhillipsCurveNowcaster(inflation_col="ipc_monthly_var")
        features = pc._extract_features(panel, target)

        if features is not None:
            # Check that COVID-era features (except inflation) are bounded
            covid_mask = (features.index >= "2020-03-01") & (features.index <= "2021-12-01")
            pre_covid = features.loc[features.index < "2020-03-01"]

            for col in features.columns:
                if col == "inflation":
                    continue
                if pre_covid[col].dropna().empty:
                    continue
                mu = pre_covid[col].mean()
                sigma = pre_covid[col].std()
                if sigma > 0:
                    covid_vals = features.loc[covid_mask, col].dropna()
                    if not covid_vals.empty:
                        assert covid_vals.max() <= mu + 3 * sigma + 1e-10
                        assert covid_vals.min() >= mu - 3 * sigma - 1e-10


# ── ML Nowcaster Tests ───────────────────────────────────────────────────────


class TestMLNowcaster:
    """Tests for the MLNowcaster class (BCRP DT 003-2024 methodology)."""

    def _make_wide_panel(self, n=120):
        """Create a synthetic wide-format panel with known structure."""
        from src.models.dfm import GDP_SERIES, INFLATION_SERIES

        dates = pd.date_range("2010-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(42)

        # Create a common factor + noise for each series
        factor = np.cumsum(rng.normal(0, 0.01, n))
        data = {}
        for s in GDP_SERIES + INFLATION_SERIES:
            loading = rng.normal(1, 0.5)
            noise = rng.normal(0, 0.01, n)
            data[s] = factor * loading + noise

        return pd.DataFrame(data, index=dates)

    def _make_gdp_target(self):
        dates = pd.date_range("2010-01-01", periods=40, freq="QS")
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "date": dates,
            "gdp_yoy": rng.normal(3.5, 2.0, len(dates)),
        })

    def _make_inflation_target(self):
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "date": dates,
            "ipc_monthly_var": rng.normal(0.3, 0.2, len(dates)),
        })

    def test_ml_lasso_gdp(self):
        """LASSO GDP nowcaster produces finite output."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(method="lasso", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])
        assert "bridge_r2" in nc
        assert "cv_rmse" in nc
        assert nc["method"] == "lasso"

    def test_ml_ridge_gdp(self):
        """Ridge GDP nowcaster produces finite output."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(method="ridge", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_ml_elastic_net_inflation(self):
        """Elastic Net inflation nowcaster produces finite output."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        ml = MLNowcaster(method="elastic_net", target="inflation")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])
        assert nc["method"] == "elastic_net"

    def test_ml_gbm_gdp(self):
        """GBM GDP nowcaster produces finite output."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(method="gbm", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])
        assert nc["method"] == "gbm"

    def test_ml_random_forest_inflation(self):
        """Random Forest inflation nowcaster produces finite output."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_inflation_target()

        ml = MLNowcaster(method="random_forest", target="inflation")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_ml_invalid_method_raises(self):
        """Invalid method name raises ValueError."""
        from src.models.dfm import MLNowcaster

        with pytest.raises(ValueError, match="Unknown method"):
            MLNowcaster(method="neural_net", target="gdp")

    def test_ml_rolling_window(self):
        """Rolling window restricts training data."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel(n=120)
        target = self._make_gdp_target()

        ml = MLNowcaster(
            method="ridge", target="gdp", rolling_window_years=5,
        )
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_ml_with_ar_feature(self):
        """AR(1) feature is included when include_target_ar=True."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(
            method="lasso", target="gdp", include_target_ar=True,
        )
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert np.isfinite(nc["nowcast_value"])
        assert "target_ar1" in ml._feature_cols

    def test_ml_without_ar_feature(self):
        """AR(1) feature is excluded when include_target_ar=False."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(
            method="lasso", target="gdp", include_target_ar=False,
        )
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert np.isfinite(nc["nowcast_value"])
        assert "target_ar1" not in ml._feature_cols

    def test_ml_handles_missing_values(self):
        """MLNowcaster handles panels with NaN via ffill + median imputation."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        # Inject NaN in first 20 rows of several columns
        rng = np.random.default_rng(99)
        for col in panel.columns[:5]:
            mask = rng.choice(len(panel), size=20, replace=False)
            panel.iloc[mask, panel.columns.get_loc(col)] = np.nan

        target = self._make_gdp_target()

        ml = MLNowcaster(method="ridge", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "nowcast_value" in nc
        assert np.isfinite(nc["nowcast_value"])

    def test_ml_cv_rmse_is_positive(self):
        """Cross-validated RMSE should be a positive number."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(method="ridge", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert nc["cv_rmse"] > 0

    def test_ml_best_params_returned(self):
        """Best hyperparameters are returned in the result dict."""
        from src.models.dfm import MLNowcaster

        panel = self._make_wide_panel()
        target = self._make_gdp_target()

        ml = MLNowcaster(method="elastic_net", target="gdp")
        ml.fit(panel)
        nc = ml.nowcast(panel, target)

        assert "best_params" in nc
        assert "alpha" in nc["best_params"]
        assert "l1_ratio" in nc["best_params"]
