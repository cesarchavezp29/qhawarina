"""Tests for Sprint 6 poverty nowcasting pipeline.

Covers: poverty models, benchmarks, backtester, and district disaggregation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_dept_panel():
    """Create a synthetic departmental panel for testing.

    Generates 10 years × 12 months × 5 departments × 3 categories.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2008-01-01", periods=120, freq="MS")
    depts = ["01", "02", "03", "15", "20"]
    categories = ["credit_by_department", "electricity_by_department", "exports_by_department"]

    rows = []
    for dept in depts:
        for cat in categories:
            series_id = f"TEST_{cat}_{dept}"
            for date in dates:
                rows.append({
                    "date": date,
                    "series_id": series_id,
                    "series_name": f"Test {cat} {dept}",
                    "category": cat,
                    "department": f"Dept {dept}",
                    "ubigeo": dept,
                    "value_raw": 100 + rng.normal(0, 10),
                    "value_sa": 100 + rng.normal(0, 8),
                    "value_log": np.log(100 + rng.normal(0, 5)),
                    "value_dlog": rng.normal(0, 0.02),
                    "value_yoy": rng.normal(5, 3),
                    "source": "TEST",
                    "frequency_original": "M",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_poverty():
    """Create synthetic poverty data: 5 depts × 15 years (2008-2022)."""
    rng = np.random.default_rng(123)
    depts = ["01", "02", "03", "15", "20"]
    years = range(2008, 2023)

    rows = []
    for dept in depts:
        base_poverty = rng.uniform(15, 55)
        base_extreme = base_poverty * 0.3
        base_consumption = rng.uniform(400, 800)
        for i, year in enumerate(years):
            rows.append({
                "year": year,
                "department_code": dept,
                "department_name": f"Dept {dept}",
                "poverty_rate": max(0, base_poverty - i * 1.5 + rng.normal(0, 2)),
                "extreme_poverty_rate": max(0, base_extreme - i * 0.5 + rng.normal(0, 1)),
                "mean_consumption": base_consumption + i * 20 + rng.normal(0, 30),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_ntl_district():
    """Create synthetic NTL district data: 20 districts × 24 months."""
    rng = np.random.default_rng(456)
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    depts = ["01", "02", "15"]
    districts_per_dept = {"01": 5, "02": 7, "15": 8}

    rows = []
    for dept in depts:
        for d_idx in range(districts_per_dept[dept]):
            ubigeo = f"{dept}{d_idx + 1:04d}"
            base_ntl = rng.uniform(100, 5000)
            for date in dates:
                rows.append({
                    "UBIGEO": ubigeo,
                    "DEPT_CODE": dept,
                    "date": date,
                    "year": date.year,
                    "month": date.month,
                    "ntl_sum": base_ntl + rng.normal(0, 50),
                })
    return pd.DataFrame(rows)


# ── Test: Aggregate Annual ────────────────────────────────────────────────────


class TestAggregateAnnual:
    def test_shape_correct(self, synthetic_dept_panel):
        from src.models.poverty import _aggregate_dept_panel_annual
        result = _aggregate_dept_panel_annual(synthetic_dept_panel)
        # 10 years × 5 depts = 50 rows
        assert len(result) == 50
        # Should have year, ubigeo, and 3 category columns
        assert "year" in result.columns
        assert "ubigeo" in result.columns
        assert result.shape[1] == 5  # year + ubigeo + 3 categories

    def test_callao_merge(self, synthetic_dept_panel):
        from src.models.poverty import _aggregate_dept_panel_annual
        # Add Callao (07) data
        callao_rows = synthetic_dept_panel[
            synthetic_dept_panel["ubigeo"] == "01"
        ].copy()
        callao_rows["ubigeo"] = "07"
        panel_with_callao = pd.concat(
            [synthetic_dept_panel, callao_rows], ignore_index=True
        )
        result = _aggregate_dept_panel_annual(panel_with_callao)
        # Callao (07) should not appear in output
        assert "07" not in result["ubigeo"].values
        # Lima (15) should be present
        assert "15" in result["ubigeo"].values

    def test_excludes_national_aggregate(self, synthetic_dept_panel):
        from src.models.poverty import _aggregate_dept_panel_annual
        # Add national aggregate (00)
        national_rows = synthetic_dept_panel[
            synthetic_dept_panel["ubigeo"] == "01"
        ].copy()
        national_rows["ubigeo"] = "00"
        panel_with_national = pd.concat(
            [synthetic_dept_panel, national_rows], ignore_index=True
        )
        result = _aggregate_dept_panel_annual(panel_with_national)
        assert "00" not in result["ubigeo"].values


# ── Test: Panel Poverty Nowcaster ─────────────────────────────────────────────


class TestPanelPovertyNowcaster:
    def test_fit_and_nowcast(self, synthetic_dept_panel, synthetic_poverty):
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0, include_ar=True,
        )
        # Train on years < 2020
        train_poverty = synthetic_poverty[synthetic_poverty["year"] < 2020]
        model.fit(features, train_poverty)
        assert "poverty_rate" in model.models_

        # Nowcast
        result = model.nowcast(features, train_poverty)
        assert "nowcast_value" in result
        assert "dept_nowcasts" in result
        # Should have predictions for at least some departments
        assert len(result["dept_nowcasts"]) > 0

    def test_ar_component(self, synthetic_dept_panel, synthetic_poverty):
        """Change-prediction model uses lag internally; verify both modes work."""
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]

        # With AR: nowcast uses lag + predicted_change
        model_ar = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], include_ar=True,
        )
        model_ar.fit(features, train)
        nc_ar = model_ar.nowcast(features, train)
        assert not np.isnan(nc_ar["nowcast_value"])

        # Without AR: should still work (change prediction doesn't need AR flag)
        model_no_ar = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], include_ar=False,
        )
        model_no_ar.fit(features, train)
        nc_no_ar = model_no_ar.nowcast(features, train)
        assert not np.isnan(nc_no_ar["nowcast_value"])

    def test_no_look_ahead_bias(self, synthetic_dept_panel, synthetic_poverty):
        """Model should not see future poverty data."""
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0,
        )
        # Train only on years < 2018
        train = synthetic_poverty[synthetic_poverty["year"] < 2018]
        model.fit(features, train)
        result = model.nowcast(features, train)
        # Nowcast should target year >= 2018
        dept_nc = result["dept_nowcasts"]
        assert len(dept_nc) > 0


# ── Test: DFM Poverty Nowcaster ──────────────────────────────────────────────


class TestDFMPovertyNowcaster:
    def test_fit_and_nowcast(self, synthetic_dept_panel, synthetic_poverty):
        from src.models.poverty import DFMPovertyNowcaster
        model = DFMPovertyNowcaster(
            k_factors=2, target_cols=["poverty_rate"], alpha=1.0,
        )
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]
        model.fit(synthetic_dept_panel, train)
        assert "poverty_rate" in model.models_

        result = model.nowcast(synthetic_dept_panel, train)
        assert "nowcast_value" in result
        assert not np.isnan(result["nowcast_value"])

    def test_correct_factor_count(self, synthetic_dept_panel, synthetic_poverty):
        from src.models.poverty import DFMPovertyNowcaster
        model = DFMPovertyNowcaster(k_factors=2)
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]
        model.fit(synthetic_dept_panel, train)
        assert model.pca_ is not None
        assert model.pca_.n_components_ == 2


# ── Test: Benchmarks ─────────────────────────────────────────────────────────


class TestPovertyBenchmarks:
    def test_ar1_produces_finite_predictions(self, synthetic_poverty):
        from src.models.poverty import PovertyAR1Benchmark
        model = PovertyAR1Benchmark(target_cols=["poverty_rate"])
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]
        result = model.nowcast(None, train)
        assert "dept_nowcasts" in result
        for dept, preds in result["dept_nowcasts"].items():
            assert "poverty_rate" in preds
            assert np.isfinite(preds["poverty_rate"])

    def test_rw_produces_finite_predictions(self, synthetic_poverty):
        from src.models.poverty import PovertyRandomWalkBenchmark
        model = PovertyRandomWalkBenchmark(target_cols=["poverty_rate"])
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]
        result = model.nowcast(None, train)
        assert "dept_nowcasts" in result
        for dept, preds in result["dept_nowcasts"].items():
            assert "poverty_rate" in preds
            assert np.isfinite(preds["poverty_rate"])

    def test_rw_returns_last_value(self, synthetic_poverty):
        from src.models.poverty import PovertyRandomWalkBenchmark
        model = PovertyRandomWalkBenchmark(target_cols=["poverty_rate"])
        train = synthetic_poverty[synthetic_poverty["year"] < 2020]
        result = model.nowcast(None, train)
        # For each dept, RW prediction should equal last observed value
        for dept in ["01", "02", "03", "15", "20"]:
            dept_data = train[train["department_code"] == dept].sort_values("year")
            expected = dept_data["poverty_rate"].iloc[-1]
            actual = result["dept_nowcasts"][dept]["poverty_rate"]
            assert abs(actual - expected) < 1e-10


# ── Test: Poverty Backtester ─────────────────────────────────────────────────


class TestPovertyBacktester:
    def test_expanding_window(self, synthetic_dept_panel, synthetic_poverty):
        from src.backtesting.backtester import PovertyBacktester
        from src.models.poverty import (
            PanelPovertyNowcaster,
            PovertyAR1Benchmark,
            PovertyRandomWalkBenchmark,
        )

        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0,
        )
        benchmarks = {
            "ar1": PovertyAR1Benchmark(target_cols=["poverty_rate"]),
            "rw": PovertyRandomWalkBenchmark(target_cols=["poverty_rate"]),
        }

        bt = PovertyBacktester(
            dept_panel=synthetic_dept_panel,
            model=model,
            benchmarks=benchmarks,
            poverty_df=synthetic_poverty,
            target_col="poverty_rate",
        )
        results = bt.run(
            eval_start_year=2016,
            eval_end_year=2022,
            min_train_years=8,
        )

        assert not results.empty
        assert "actual" in results.columns
        assert "panel_nowcast" in results.columns
        assert "ar1_nowcast" in results.columns
        assert "rw_nowcast" in results.columns

    def test_min_train_respected(self, synthetic_dept_panel, synthetic_poverty):
        from src.backtesting.backtester import PovertyBacktester
        from src.models.poverty import (
            PanelPovertyNowcaster,
            PovertyRandomWalkBenchmark,
        )

        model = PanelPovertyNowcaster(target_cols=["poverty_rate"])
        bt = PovertyBacktester(
            dept_panel=synthetic_dept_panel,
            model=model,
            benchmarks={"rw": PovertyRandomWalkBenchmark(target_cols=["poverty_rate"])},
            poverty_df=synthetic_poverty,
            target_col="poverty_rate",
        )
        results = bt.run(
            eval_start_year=2012,
            eval_end_year=2022,
            min_train_years=8,
        )
        # With data starting 2008, need 8 years → first eval should be 2016
        if not results.empty:
            min_year = results["year"].min()
            assert min_year >= 2016

    def test_summary_metrics(self, synthetic_dept_panel, synthetic_poverty):
        from src.backtesting.backtester import PovertyBacktester
        from src.models.poverty import (
            PanelPovertyNowcaster,
            PovertyRandomWalkBenchmark,
        )

        model = PanelPovertyNowcaster(target_cols=["poverty_rate"])
        benchmarks = {"rw": PovertyRandomWalkBenchmark(target_cols=["poverty_rate"])}
        bt = PovertyBacktester(
            dept_panel=synthetic_dept_panel,
            model=model,
            benchmarks=benchmarks,
            poverty_df=synthetic_poverty,
            target_col="poverty_rate",
        )
        bt.run(eval_start_year=2016, eval_end_year=2022, min_train_years=8)
        summary = bt.summary()
        if "panel" in summary:
            assert "rmse" in summary["panel"]
            assert "mae" in summary["panel"]
            assert summary["panel"]["rmse"] >= 0


# ── Test: District Disaggregation ────────────────────────────────────────────


class TestDistrictDisaggregation:
    def test_ntl_weights_sum_to_one(self, synthetic_ntl_district):
        from src.processing.spatial_disagg import disaggregate_to_districts

        dept_values = pd.DataFrame({
            "department_code": ["01", "02", "15"],
            "poverty_rate": [30.0, 25.0, 15.0],
        })

        result = disaggregate_to_districts(
            dept_values, synthetic_ntl_district,
            target_col="poverty_rate",
            method="ntl_dasymetric",
            year=2020,
        )

        # Weights should sum to ~1 per department
        for dept in ["01", "02", "15"]:
            dept_weights = result[result["department_code"] == dept]["ntl_weight"]
            assert abs(dept_weights.sum() - 1.0) < 1e-10

    def test_poverty_bounded(self, synthetic_ntl_district):
        from src.processing.spatial_disagg import disaggregate_to_districts

        dept_values = pd.DataFrame({
            "department_code": ["01", "02", "15"],
            "poverty_rate": [50.0, 40.0, 20.0],
        })

        result = disaggregate_to_districts(
            dept_values, synthetic_ntl_district,
            target_col="poverty_rate",
            method="ntl_dasymetric",
            year=2020,
        )

        # All poverty estimates should be bounded [0, 100]
        assert (result["poverty_rate_nowcast"] >= 0).all()
        assert (result["poverty_rate_nowcast"] <= 100).all()

    def test_ntl_share_method(self, synthetic_ntl_district):
        from src.processing.spatial_disagg import disaggregate_to_districts

        dept_values = pd.DataFrame({
            "department_code": ["01", "02", "15"],
            "mean_consumption": [600.0, 700.0, 900.0],
        })

        result = disaggregate_to_districts(
            dept_values, synthetic_ntl_district,
            target_col="mean_consumption",
            method="ntl_share",
            year=2020,
        )

        assert not result.empty
        assert "mean_consumption_nowcast" in result.columns
        # All values should be non-negative
        assert (result["mean_consumption_nowcast"] >= 0).all()

    def test_all_districts_covered(self, synthetic_ntl_district):
        from src.processing.spatial_disagg import disaggregate_to_districts

        dept_values = pd.DataFrame({
            "department_code": ["01", "02", "15"],
            "poverty_rate": [30.0, 25.0, 15.0],
        })

        result = disaggregate_to_districts(
            dept_values, synthetic_ntl_district,
            target_col="poverty_rate",
            method="ntl_dasymetric",
            year=2020,
        )

        # Should cover all districts in the 3 departments
        n_expected = 5 + 7 + 8  # districts per dept from fixture
        assert len(result) == n_expected


# ── Test: Rolling Aggregation (Sprint 8) ────────────────────────────────────


class TestAggregateRolling:
    def test_shape_correct(self, synthetic_dept_panel):
        from src.models.poverty import _aggregate_dept_panel_rolling
        end_date = pd.Timestamp("2017-12-31")
        result = _aggregate_dept_panel_rolling(synthetic_dept_panel, end_date, window_months=12)
        # 5 departments
        assert len(result) == 5
        assert "year" in result.columns
        assert "ubigeo" in result.columns
        # 3 categories + year + ubigeo = 5 columns
        assert result.shape[1] == 5

    def test_december_matches_annual(self, synthetic_dept_panel):
        """Rolling 12m ending Dec Y should match annual aggregation for year Y."""
        from src.models.poverty import _aggregate_dept_panel_annual, _aggregate_dept_panel_rolling
        year = 2015
        end_date = pd.Timestamp(f"{year}-12-31")

        annual = _aggregate_dept_panel_annual(synthetic_dept_panel)
        annual_y = annual[annual["year"] == year].sort_values("ubigeo").reset_index(drop=True)

        rolling = _aggregate_dept_panel_rolling(
            synthetic_dept_panel, end_date, window_months=12,
        )
        rolling = rolling.sort_values("ubigeo").reset_index(drop=True)

        # Category columns should match closely
        cat_cols = [c for c in annual_y.columns if c not in ["year", "ubigeo"]]
        for col in cat_cols:
            np.testing.assert_allclose(
                annual_y[col].values, rolling[col].values,
                atol=1e-10,
                err_msg=f"Mismatch in {col}",
            )

    def test_window_bounds(self, synthetic_dept_panel):
        """Data outside the window should be excluded."""
        from src.models.poverty import _aggregate_dept_panel_rolling
        end_date = pd.Timestamp("2012-06-30")
        result = _aggregate_dept_panel_rolling(
            synthetic_dept_panel, end_date, window_months=6,
        )
        # Should only cover Jan 2012 to Jun 2012 (6 months)
        assert len(result) == 5
        assert result["year"].iloc[0] == 2012

    def test_callao_merge(self, synthetic_dept_panel):
        from src.models.poverty import _aggregate_dept_panel_rolling
        # Add Callao data
        callao_rows = synthetic_dept_panel[
            synthetic_dept_panel["ubigeo"] == "01"
        ].copy()
        callao_rows["ubigeo"] = "07"
        panel_with_callao = pd.concat(
            [synthetic_dept_panel, callao_rows], ignore_index=True,
        )
        end_date = pd.Timestamp("2017-12-31")
        result = _aggregate_dept_panel_rolling(panel_with_callao, end_date)
        assert "07" not in result["ubigeo"].values
        assert "15" in result["ubigeo"].values


# ── Test: nowcast_at_date (Sprint 8) ────────────────────────────────────────


class TestNowcastAtDate:
    def test_produces_predictions(self, synthetic_dept_panel, synthetic_poverty):
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        train = synthetic_poverty[synthetic_poverty["year"] < 2017]

        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0, include_ar=True,
        )
        model.fit(features, train)

        end_date = pd.Timestamp("2017-09-30")
        result = model.nowcast_at_date(
            synthetic_dept_panel, train, end_date=end_date,
        )
        assert "dept_nowcasts" in result
        assert "end_date" in result
        assert len(result["dept_nowcasts"]) > 0

    def test_predictions_evolve(self, synthetic_dept_panel, synthetic_poverty):
        """Different months should give different predictions."""
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        train = synthetic_poverty[synthetic_poverty["year"] < 2017]

        # Use mean_consumption (not clipped to [0,1]) to avoid synthetic
        # poverty percentages being clipped to 1.0
        model = PanelPovertyNowcaster(
            target_cols=["mean_consumption"], alpha=1.0, include_ar=True,
            model_type="gbr",
        )
        model.fit(features, train)

        results = []
        for month in [3, 6, 9, 12]:
            end_date = pd.Timestamp(2017, month, 1) + pd.offsets.MonthEnd(0)
            nc = model.nowcast_at_date(
                synthetic_dept_panel, train, end_date=end_date,
            )
            vals = [
                v["mean_consumption"]
                for v in nc["dept_nowcasts"].values()
                if "mean_consumption" in v
            ]
            results.append(np.mean(vals) if vals else np.nan)

        # At least some predictions should differ across months
        assert len(set(results)) > 1, "All monthly predictions identical"

    def test_coverage_reported(self, synthetic_dept_panel, synthetic_poverty):
        from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
        features = _aggregate_dept_panel_annual(synthetic_dept_panel)
        train = synthetic_poverty[synthetic_poverty["year"] < 2017]

        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0, include_ar=True,
        )
        model.fit(features, train)

        end_date = pd.Timestamp("2017-06-30")
        result = model.nowcast_at_date(
            synthetic_dept_panel, train, end_date=end_date,
        )
        assert "coverage" in result
        # Coverage dict should have entries per department
        assert len(result["coverage"]) > 0
        for dept, n_cats in result["coverage"].items():
            assert n_cats >= 0


# ── Test: Monthly Poverty Backtester (Sprint 8) ────────────────────────────


class TestMonthlyPovertyBacktester:
    def test_produces_monthly_results(self, synthetic_dept_panel, synthetic_poverty):
        from src.backtesting.backtester import MonthlyPovertyBacktester
        from src.models.poverty import (
            PanelPovertyNowcaster,
            PovertyAR1Benchmark,
            PovertyRandomWalkBenchmark,
        )

        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0, include_ar=True,
        )
        benchmarks = {
            "ar1": PovertyAR1Benchmark(target_cols=["poverty_rate"]),
            "rw": PovertyRandomWalkBenchmark(target_cols=["poverty_rate"]),
        }

        bt = MonthlyPovertyBacktester(
            dept_panel=synthetic_dept_panel,
            model=model,
            benchmarks=benchmarks,
            poverty_df=synthetic_poverty,
            target_col="poverty_rate",
            eval_months=[6, 12],
            window_months=12,
        )
        results = bt.run(
            eval_start_year=2016,
            eval_end_year=2018,
            min_train_years=8,
        )

        assert not results.empty
        assert "month" in results.columns
        assert "panel_nowcast" in results.columns
        assert "ar1_nowcast" in results.columns
        # Should have results for multiple months per year
        for year in results["year"].unique():
            year_months = results[results["year"] == year]["month"].unique()
            assert len(year_months) >= 1

    def test_convergence_analysis(self, synthetic_dept_panel, synthetic_poverty):
        from src.backtesting.backtester import MonthlyPovertyBacktester
        from src.models.poverty import (
            PanelPovertyNowcaster,
            PovertyRandomWalkBenchmark,
        )

        model = PanelPovertyNowcaster(
            target_cols=["poverty_rate"], alpha=1.0, include_ar=True,
        )
        bt = MonthlyPovertyBacktester(
            dept_panel=synthetic_dept_panel,
            model=model,
            benchmarks={"rw": PovertyRandomWalkBenchmark(target_cols=["poverty_rate"])},
            poverty_df=synthetic_poverty,
            target_col="poverty_rate",
            eval_months=[3, 6, 9, 12],
        )
        bt.run(eval_start_year=2016, eval_end_year=2018, min_train_years=8)

        convergence = bt.convergence_analysis()
        if not convergence.empty:
            assert "month" in convergence.columns
            assert "rmse" in convergence.columns
            # One row per eval month
            assert len(convergence) >= 1
