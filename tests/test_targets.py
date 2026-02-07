"""Tests for target variable compilation (GDP + inflation)."""

from datetime import date

import pandas as pd
import pytest

from src.ingestion.targets import (
    GDP_CODE_MAP,
    pivot_gdp_to_target,
    compile_inflation_targets,
    load_quarterly_codes,
)


# ── GDP target pivoting ─────────────────────────────────────────────────────

class TestPivotGDPToTarget:
    """Test pivot_gdp_to_target with synthetic data."""

    def _make_long_df(self):
        """Create synthetic long-format GDP data (like BCRP API output)."""
        rows = []
        dates = [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1)]
        for d in dates:
            for code, col in GDP_CODE_MAP.items():
                rows.append({
                    "date": pd.Timestamp(d),
                    "series_code": code,
                    "value": 3.5 if "yoy" in col else 150.0,
                    "series_name": f"Test {col}",
                })
        return pd.DataFrame(rows)

    def test_basic_pivot(self):
        df = self._make_long_df()
        result = pivot_gdp_to_target(df)

        assert len(result) == 3  # 3 quarters
        assert "date" in result.columns
        assert "gdp_yoy" in result.columns
        assert "gdp_index" in result.columns

    def test_all_columns_present(self):
        df = self._make_long_df()
        result = pivot_gdp_to_target(df)

        expected_cols = {"date"} | set(GDP_CODE_MAP.values())
        assert set(result.columns) == expected_cols

    def test_values_correct(self):
        df = self._make_long_df()
        result = pivot_gdp_to_target(df)

        # All yoy values should be 3.5, index should be 150.0
        assert (result["gdp_yoy"] == 3.5).all()
        assert (result["gdp_index"] == 150.0).all()

    def test_empty_input(self):
        result = pivot_gdp_to_target(pd.DataFrame())
        assert result.empty

    def test_missing_series(self):
        """If some series are missing, their columns should be None."""
        rows = [
            {"date": pd.Timestamp("2024-01-01"), "series_code": "PN02507AQ", "value": 5.0},
            {"date": pd.Timestamp("2024-01-01"), "series_code": "PN02516AQ", "value": 155.0},
        ]
        df = pd.DataFrame(rows)
        result = pivot_gdp_to_target(df)

        assert len(result) == 1
        assert result["gdp_yoy"].iloc[0] == 5.0
        assert result["gdp_index"].iloc[0] == 155.0
        # Missing sectors should be None/NaN
        assert pd.isna(result["pesca_yoy"].iloc[0])

    def test_sorted_by_date(self):
        rows = [
            {"date": pd.Timestamp("2024-07-01"), "series_code": "PN02507AQ", "value": 4.0},
            {"date": pd.Timestamp("2024-01-01"), "series_code": "PN02507AQ", "value": 3.0},
            {"date": pd.Timestamp("2024-04-01"), "series_code": "PN02507AQ", "value": 3.5},
        ]
        df = pd.DataFrame(rows)
        result = pivot_gdp_to_target(df)

        dates = result["date"].tolist()
        assert dates == sorted(dates)


class TestLoadQuarterlyCodes:
    def test_loads_codes(self, tmp_path):
        """Test loading quarterly codes from a catalog YAML."""
        catalog_content = """
quarterly:
  gdp_quarterly:
    frequency: quarterly
    series:
      - code: PN02507AQ
        name: "PBI global"
      - code: PN02516AQ
        name: "PBI indice"
"""
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.write_text(catalog_content, encoding="utf-8")

        codes = load_quarterly_codes(catalog_path)
        assert codes == ["PN02507AQ", "PN02516AQ"]


class TestCompileInflation:
    """Test inflation target compilation with mock source data."""

    def test_compile_creates_output(self, tmp_path):
        """Test that compile_inflation_targets reshapes data correctly."""
        # Create mock inflation source data
        rows = []
        for month in range(1, 4):
            for code, name in [
                ("PN01271PM", "IPC var mensual"),
                ("PN01273PM", "IPC var 12m"),
                ("PN38706PM", "IPC core"),
            ]:
                rows.append({
                    "date": pd.Timestamp(f"2024-{month:02d}-01"),
                    "series_code": code,
                    "series_name": name,
                    "series_index": 0,
                    "value": 0.5 * month,
                    "category": "inflation",
                    "source": "BCRP",
                    "frequency_original": "M",
                })

        source_df = pd.DataFrame(rows)

        # Save as mock source
        source_path = tmp_path / "bcrp_national_inflation.parquet"
        source_df.to_parquet(source_path, index=False)

        # Monkey-patch the source and output paths
        import src.ingestion.targets as targets_mod
        original_raw = targets_mod.RAW_BCRP_DIR
        targets_mod.RAW_BCRP_DIR = tmp_path

        try:
            result = targets_mod.compile_inflation_targets(output_dir=tmp_path)
        finally:
            targets_mod.RAW_BCRP_DIR = original_raw

        assert result["status"] == "updated"
        assert result["rows"] == 3

        # Read back and verify
        output_path = tmp_path / "inflation_monthly.parquet"
        assert output_path.exists()
        out_df = pd.read_parquet(output_path)
        assert "ipc_monthly_var" in out_df.columns
        assert "ipc_12m_var" in out_df.columns
        assert "ipc_core_index" in out_df.columns
        assert len(out_df) == 3
