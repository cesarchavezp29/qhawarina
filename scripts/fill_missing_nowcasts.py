"""Generate missing nowcasts for 2025-Q1, Q2, Q3 to fill the gap."""

import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_NATIONAL_DIR, CONFIG_DIR, RESULTS_DIR
from src.backtesting.vintage import VintageManager
from scripts.generate_nowcast import nowcast_gdp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fill_nowcasts")

def main():
    # Load existing backtest
    backtest_path = RESULTS_DIR / "backtest_gdp.parquet"
    backtest = pd.read_parquet(backtest_path)
    backtest["target_period"] = pd.to_datetime(backtest["target_period"])

    logger.info(f"Existing backtest: {len(backtest)} periods, last = {backtest['target_period'].max()}")

    # Initialize VintageManager
    vm = VintageManager(
        PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet",
        CONFIG_DIR / "publication_lags.yaml"
    )

    # Generate nowcasts for missing quarters
    missing_quarters = [
        pd.Timestamp("2025-01-01"),  # Q1
        pd.Timestamp("2025-04-01"),  # Q2
        pd.Timestamp("2025-07-01"),  # Q3
    ]

    new_rows = []
    for target_quarter in missing_quarters:
        logger.info(f"\nGenerating nowcast for {target_quarter.year}-Q{(target_quarter.month-1)//3+1}...")

        # Generate nowcast as of end of that quarter
        as_of = target_quarter + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        result = nowcast_gdp(vm, as_of)

        new_rows.append({
            "target_period": target_quarter,
            "dfm_nowcast": result["nowcast_value"],
            "bridge_r2": result.get("bridge_r2", None),
        })

        logger.info(f"  → {result['nowcast_value']:.2f}%")

    # Append to backtest
    new_df = pd.DataFrame(new_rows)
    updated = pd.concat([backtest, new_df], ignore_index=True)
    updated = updated.sort_values("target_period").drop_duplicates("target_period", keep="last")

    logger.info(f"\nUpdated backtest: {len(updated)} periods (was {len(backtest)})")
    updated.to_parquet(backtest_path, index=False)

    logger.info(f"✓ Saved to {backtest_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
