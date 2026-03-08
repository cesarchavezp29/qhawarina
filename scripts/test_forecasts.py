"""Quick test of DFM forecast functionality."""

import logging
import sys
from pathlib import Path

import pandas as pd

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_NATIONAL_DIR, TARGETS_DIR, CONFIG_DIR
from src.backtesting.vintage import VintageManager
from src.models.dfm import NowcastDFM


def test_gdp_forecast():
    """Test GDP forecast (6 quarters ahead)."""
    print("=" * 60)
    print("GDP FORECAST TEST")
    print("=" * 60)

    # Load data
    vm = VintageManager(
        PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet",
        CONFIG_DIR / "publication_lags.yaml"
    )
    gdp_df = pd.read_parquet(TARGETS_DIR / "gdp_quarterly.parquet")
    gdp_df["date"] = pd.to_datetime(gdp_df["date"])

    # Get panel as of latest date
    panel_wide = vm._pivot_auto(vm.panel)

    # Create and fit model
    model = NowcastDFM(
        k_factors=3, factor_order=1, target="gdp",
        bridge_method="ridge", bridge_alpha=1.0,
        rolling_window_years=7
    )

    model.fit(panel_wide)

    # Generate nowcast
    nowcast = model.nowcast(panel_wide, gdp_df)
    print(f"\nLatest nowcast: {nowcast['nowcast_value']:.2f}% (R²={nowcast['bridge_r2']:.3f})")

    # Generate forecasts
    print("\nCalling model.forecast...")
    print(f"Model has forecast method: {hasattr(model, 'forecast')}")
    try:
        forecasts = model.forecast(panel_wide, gdp_df, horizons=6)
        print(f"Forecast returned: {type(forecasts)}")
        print(f"\nForecasts (6 quarters ahead):")
        print(forecasts.to_string(index=False))
    except Exception as e:
        print(f"\n[ERROR] Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        forecasts = pd.DataFrame()

    print("\n[OK] GDP forecast test passed!")
    return forecasts


def test_inflation_forecast():
    """Test inflation forecast (6 months ahead)."""
    print("\n" + "=" * 60)
    print("INFLATION FORECAST TEST")
    print("=" * 60)

    # Load data
    vm = VintageManager(
        PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet",
        CONFIG_DIR / "publication_lags.yaml"
    )
    inf_df = pd.read_parquet(TARGETS_DIR / "inflation_monthly.parquet")
    inf_df["date"] = pd.to_datetime(inf_df["date"])

    # Get panel as of latest date
    panel_wide = vm._pivot_auto(vm.panel)

    # Create and fit model
    model = NowcastDFM(
        k_factors=2, factor_order=1, target="inflation",
        inflation_col="ipc_3m_ma",
        include_factor_lags=1,
        include_target_ar=True
    )

    model.fit(panel_wide)

    # Generate nowcast
    nowcast = model.nowcast(panel_wide, inf_df)
    print(f"\nLatest nowcast: {nowcast['nowcast_value']:.3f}% (R²={nowcast['bridge_r2']:.3f})")

    # Generate forecasts
    try:
        forecasts = model.forecast(panel_wide, inf_df, horizons=6)
        print(f"\nForecasts (6 months ahead):")
        print(forecasts.to_string(index=False))
    except Exception as e:
        print(f"\n[ERROR] Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        forecasts = pd.DataFrame()

    print("\n[OK] Inflation forecast test passed!")
    return forecasts


if __name__ == "__main__":
    try:
        gdp_fc = test_gdp_forecast()
        inf_fc = test_inflation_forecast()
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
