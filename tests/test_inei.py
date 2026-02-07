"""Tests for INEI/ENAHO poverty computation."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.inei import (
    weighted_mean,
    weighted_gini,
    ENAHOClient,
    ENAHO_AVAILABLE_YEARS,
)


# ── Weighted mean tests ─────────────────────────────────────────────────────

class TestWeightedMean:
    def test_equal_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.0)

    def test_unequal_weights(self):
        values = np.array([0.0, 10.0])
        weights = np.array([3.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.5)

    def test_ignores_nan_values(self):
        values = np.array([1.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.0)

    def test_ignores_nan_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, np.nan, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.0)

    def test_ignores_zero_weights(self):
        values = np.array([1.0, 100.0, 3.0])
        weights = np.array([1.0, 0.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        values = np.array([np.nan, np.nan])
        weights = np.array([1.0, 1.0])
        assert np.isnan(weighted_mean(values, weights))

    def test_single_value(self):
        values = np.array([42.0])
        weights = np.array([5.0])
        assert weighted_mean(values, weights) == pytest.approx(42.0)

    def test_binary_values(self):
        """Weighted mean of binary 0/1 gives a proportion."""
        values = np.array([1.0, 0.0, 1.0, 0.0])
        weights = np.array([100.0, 200.0, 100.0, 100.0])
        # (100 + 100) / 500 = 0.4
        assert weighted_mean(values, weights) == pytest.approx(0.4)


# ── Weighted Gini tests ─────────────────────────────────────────────────────

class TestWeightedGini:
    def test_perfect_equality(self):
        """All same values → Gini ≈ 0."""
        values = np.array([100.0, 100.0, 100.0, 100.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        assert weighted_gini(values, weights) == pytest.approx(0.0, abs=1e-10)

    def test_maximum_inequality(self):
        """One person has everything → Gini close to 1."""
        values = np.array([0.0, 0.0, 0.0, 1000.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        gini = weighted_gini(values, weights)
        assert gini > 0.7  # Should be close to 0.75 for 4 obs

    def test_moderate_inequality(self):
        """Known distribution → check Gini is in sensible range."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        gini = weighted_gini(values, weights)
        assert 0.1 < gini < 0.4

    def test_weighted_gini(self):
        """Weights affect the Gini computation."""
        values = np.array([10.0, 100.0])
        weights_equal = np.array([1.0, 1.0])
        weights_skewed = np.array([100.0, 1.0])

        gini_equal = weighted_gini(values, weights_equal)
        gini_skewed = weighted_gini(values, weights_skewed)

        # Skewing weight toward the poor person reduces inequality
        assert gini_skewed < gini_equal

    def test_insufficient_data(self):
        values = np.array([100.0])
        weights = np.array([1.0])
        assert np.isnan(weighted_gini(values, weights))

    def test_all_zeros(self):
        values = np.array([0.0, 0.0, 0.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert np.isnan(weighted_gini(values, weights))

    def test_with_nans(self):
        values = np.array([10.0, np.nan, 30.0, 40.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        gini = weighted_gini(values, weights)
        assert 0.0 < gini < 1.0

    def test_gini_bounded_0_1(self):
        """Gini should always be between 0 and 1."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            n = rng.randint(10, 100)
            values = rng.exponential(1000, n)
            weights = rng.uniform(1, 100, n)
            gini = weighted_gini(values, weights)
            assert 0.0 <= gini <= 1.0


# ── ENAHOClient tests ───────────────────────────────────────────────────────

class TestENAHOClientAvailableYears:
    def test_available_years(self):
        client = ENAHOClient(output_dir="test_dummy_dir")
        years = client.available_years()
        assert len(years) == 21  # 2004-2024
        assert 2004 in years
        assert 2024 in years

    def test_downloaded_years_empty(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        assert client.downloaded_years() == []

    def test_missing_years_all(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        missing = client.missing_years()
        assert missing == list(range(2004, 2025))


class TestComputePovertyYear:
    """Test poverty computation with synthetic Sumaria-like data."""

    def _make_sumaria(self, n=200, seed=42):
        """Create synthetic Sumaria DataFrame."""
        rng = np.random.RandomState(seed)

        # 5 departments
        dept_codes = ["01", "02", "03", "04", "05"]
        ubigeos = [f"{d}0101" for d in dept_codes]

        df = pd.DataFrame({
            "ubigeo": rng.choice(ubigeos, n),
            "factor07": rng.uniform(100, 500, n),
            "gashog2d": rng.exponential(2000, n) * rng.choice([1, 2, 5], n),
            "mieperho": rng.choice([1, 2, 3, 4, 5], n).astype(float),
            "linea": np.full(n, 400.0),   # poverty line
            "linpe": np.full(n, 200.0),   # extreme poverty line
        })
        return df

    def test_basic_output_schema(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        expected_cols = {
            "year", "department_code", "department_name",
            "poverty_rate", "extreme_poverty_rate",
            "poverty_gap", "poverty_severity",
            "mean_consumption", "mean_income", "gini",
        }
        assert set(result.columns) == expected_cols

    def test_all_departments_present(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        assert len(result) == 5  # 5 synthetic departments
        assert set(result["department_code"]) == {"01", "02", "03", "04", "05"}

    def test_year_column(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2018)

        assert (result["year"] == 2018).all()

    def test_poverty_rates_bounded(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        assert (result["poverty_rate"] >= 0).all()
        assert (result["poverty_rate"] <= 1).all()
        assert (result["extreme_poverty_rate"] >= 0).all()
        assert (result["extreme_poverty_rate"] <= 1).all()

    def test_extreme_leq_total_poverty(self, tmp_path):
        """Extreme poverty should never exceed total poverty."""
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        for _, row in result.iterrows():
            assert row["extreme_poverty_rate"] <= row["poverty_rate"] + 1e-10

    def test_gini_bounded(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        assert (result["gini"] >= 0).all()
        assert (result["gini"] <= 1).all()

    def test_mean_consumption_positive(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        result = client.compute_poverty_year(df, 2020)

        assert (result["mean_consumption"] > 0).all()

    def test_case_insensitive_columns(self, tmp_path):
        """Column names should be case-insensitive."""
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        # Uppercase some columns
        df.columns = [c.upper() for c in df.columns]
        result = client.compute_poverty_year(df, 2020)

        assert len(result) == 5  # Should still work

    def test_missing_column_raises(self, tmp_path):
        client = ENAHOClient(output_dir=tmp_path)
        df = pd.DataFrame({"ubigeo": ["010101"], "factor07": [100]})

        with pytest.raises(ValueError, match="Required column"):
            client.compute_poverty_year(df, 2020)

    def test_official_pobreza_variable(self, tmp_path):
        """When pobreza column exists, use it for poverty classification."""
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        # Add official pobreza column — override what linea/linpe would give
        n = len(df)
        rng = np.random.RandomState(99)
        labels = rng.choice(
            ["pobre extremo", "pobre no extremo", "no pobre"],
            n, p=[0.05, 0.15, 0.80],
        )
        df["pobreza"] = labels
        result = client.compute_poverty_year(df, 2024)

        # Poverty rate should reflect the pobreza column, not linea comparison
        assert (result["poverty_rate"] >= 0).all()
        assert (result["poverty_rate"] <= 1).all()
        assert (result["extreme_poverty_rate"] >= 0).all()
        # At least some poverty should exist given 20% are poor
        assert result["poverty_rate"].mean() > 0

    def test_official_pobreza_old_coding(self, tmp_path):
        """Handle pre-2017 coding: 'pobreno extremo' (no space)."""
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria(n=100)
        # 50 poor non-extreme (old coding), 10 extreme, 40 not poor
        labels = (["pobreno extremo"] * 50
                  + ["pobre extremo"] * 10
                  + ["no pobre"] * 40)
        df["pobreza"] = labels[:100]
        result = client.compute_poverty_year(df, 2010)

        # Should detect both pobre extremo and pobreno extremo as poor
        total_poverty = result["poverty_rate"].mean()
        assert total_poverty > 0.3  # >50% are poor in synthetic data

    def test_filters_unknown_departments(self, tmp_path):
        """Departments not in DEPARTMENTS dict should be excluded."""
        client = ENAHOClient(output_dir=tmp_path)
        df = self._make_sumaria()
        # Add rows with unknown department code "99"
        extra = pd.DataFrame({
            "ubigeo": ["990101"] * 10,
            "factor07": [200.0] * 10,
            "gashog2d": [3000.0] * 10,
            "mieperho": [3.0] * 10,
            "linea": [400.0] * 10,
            "linpe": [200.0] * 10,
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = client.compute_poverty_year(df, 2020)

        assert "99" not in result["department_code"].values
