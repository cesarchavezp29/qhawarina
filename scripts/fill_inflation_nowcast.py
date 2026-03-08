"""Generate missing inflation nowcast for 2026-01."""

import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_NATIONAL_DIR, CONFIG_DIR, RESULTS_DIR
from src.backtesting.vintage import VintageManager
from scripts.generate_nowcast import nowcast_inflation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fill_inflation")

def main():
    # Load existing backtest
    backtest_path = RESULTS_DIR / "backtest_inflation.parquet"
    backtest = pd.read_parquet(backtest_path)
    backtest["target_period"] = pd.to_datetime(backtest["target_period"])

    logger.info(f"Existing backtest: {len(backtest)} periods, last = {backtest['target_period'].max():%Y-%m}")

    # Initialize VintageManager
    vm = VintageManager(
        PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet",
        CONFIG_DIR / "publication_lags.yaml"
    )

    # Generate nowcast for 2026-01
    target_month = pd.Timestamp("2026-01-01")
    logger.info(f"\nGenerating nowcast for {target_month:%Y-%m}...")

    # Generate nowcast as of end of that month
    as_of = target_month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    result = nowcast_inflation(vm, as_of)

    new_row = {
        "target_period": target_month,
        "dfm_nowcast": result["nowcast_value"],
        "bridge_r2": result.get("bridge_r2", None),
    }

    logger.info(f"  → {result['nowcast_value']:.3f}%")

    # Append to backtest
    new_df = pd.DataFrame([new_row])
    updated = pd.concat([backtest, new_df], ignore_index=True)
    updated = updated.sort_values("target_period").drop_duplicates("target_period", keep="last")

    logger.info(f"\nUpdated backtest: {len(updated)} periods (was {len(backtest)})")
    updated.to_parquet(backtest_path, index=False)

    logger.info(f"✓ Saved to {backtest_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
