"""Generate monthly poverty estimates using NTL + CPI temporal disaggregation."""

import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import TARGETS_DIR, RESULTS_DIR, PROCESSED_DIR
from src.processing.temporal_disaggregation import poverty_monthly_disagg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("poverty_monthly")

def main():
    # Load annual poverty (including 2025 nowcast)
    poverty_annual = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")

    # Add 2025 nowcast
    nowcast_2025 = pd.read_parquet(RESULTS_DIR / "poverty_nowcast_2025.parquet")
    nowcast_2025 = nowcast_2025.rename(columns={"ubigeo": "department_code"})
    nowcast_2025["poverty_rate"] = nowcast_2025["poverty_rate_nowcast"]
    nowcast_2025 = nowcast_2025[["year", "department_code", "poverty_rate"]]

    # Combine
    poverty_full = pd.concat([
        poverty_annual[["year", "department_code", "poverty_rate"]],
        nowcast_2025
    ], ignore_index=True)

    # National average
    poverty_national = poverty_full.groupby("year")["poverty_rate"].mean().reset_index()

    logger.info(f"Annual poverty: {len(poverty_national)} years, {poverty_national.year.min()}-{poverty_national.year.max()}")

    # Load NTL monthly (national)
    panel_national = pd.read_parquet(PROCESSED_DIR / "national" / "panel_national_monthly.parquet")
    ntl_monthly = panel_national[panel_national["series_id"] == "NTL_SUM_NATIONAL"].copy()
    ntl_monthly = ntl_monthly[["date", "value_log"]].rename(columns={"value_log": "value"})

    # Load CPI monthly
    cpi_monthly = pd.read_parquet(TARGETS_DIR / "inflation_monthly.parquet")

    logger.info(f"NTL monthly: {len(ntl_monthly)} months")
    logger.info(f"CPI monthly: {len(cpi_monthly)} months")

    # Generate monthly estimates with 3-month MA
    logger.info("\nGenerating monthly poverty estimates (NTL + CPI, 3M-MA)...")
    poverty_monthly = poverty_monthly_disagg(
        poverty_national,
        ntl_monthly,
        cpi_monthly,
        method="chow-lin",
        smoothing=3
    )

    logger.info(f"\nMonthly poverty generated:")
    logger.info(f"  Rows: {len(poverty_monthly)}")
    logger.info(f"  Months: {poverty_monthly['date'].min()} to {poverty_monthly['date'].max()}")

    logger.info(f"\nLast 6 months (smoothed):")
    for _, row in poverty_monthly.tail(6).iterrows():
        logger.info(f"  {row['date'].strftime('%Y-%m')}: {row['poverty_rate_smooth']*100:.1f}%")

    # Save
    output_file = RESULTS_DIR / "poverty_monthly_national.parquet"
    poverty_monthly.to_parquet(output_file, index=False)

    logger.info(f"\n✓ Saved to {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
