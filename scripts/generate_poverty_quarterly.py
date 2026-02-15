"""Generate quarterly poverty estimates using temporal disaggregation."""

import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import TARGETS_DIR, RESULTS_DIR
from src.processing.temporal_disaggregation import poverty_quarterly_disagg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("poverty_quarterly")

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

    logger.info(f"Annual poverty: {len(poverty_full)} rows, years {poverty_full.year.min()}-{poverty_full.year.max()}")

    # Load quarterly indicators
    gdp_quarterly = pd.read_parquet(TARGETS_DIR / "gdp_quarterly.parquet")
    cpi_monthly = pd.read_parquet(TARGETS_DIR / "inflation_monthly.parquet")

    logger.info(f"GDP quarterly: {len(gdp_quarterly)} quarters")
    logger.info(f"CPI monthly: {len(cpi_monthly)} months")

    # Generate quarterly estimates (Chow-Lin)
    logger.info("\nGenerating quarterly poverty estimates (Chow-Lin)...")
    poverty_quarterly = poverty_quarterly_disagg(
        poverty_full,
        gdp_quarterly[["date", "gdp_yoy"]],
        cpi_monthly[["date", "ipc_monthly_var"]],
        method="chow-lin"
    )

    logger.info(f"\nQuarterly poverty generated:")
    logger.info(f"  Rows: {len(poverty_quarterly)}")
    logger.info(f"  Departments: {poverty_quarterly['department_code'].nunique()}")
    logger.info(f"  Quarters: {poverty_quarterly['date'].min()} to {poverty_quarterly['date'].max()}")

    # National average
    national_quarterly = poverty_quarterly.groupby("date")["poverty_rate_quarterly"].mean().reset_index()
    national_quarterly = national_quarterly.rename(columns={"poverty_rate_quarterly": "poverty_rate"})

    logger.info(f"\nNational quarterly poverty (last 4 quarters):")
    for _, row in national_quarterly.tail(4).iterrows():
        q = (row['date'].month - 1) // 3 + 1
        logger.info(f"  {row['date'].year}-Q{q}: {row['poverty_rate']*100:.1f}%")

    # Save
    output_dept = RESULTS_DIR / "poverty_quarterly_departmental.parquet"
    output_nat = RESULTS_DIR / "poverty_quarterly_national.parquet"

    poverty_quarterly.to_parquet(output_dept, index=False)
    national_quarterly.to_parquet(output_nat, index=False)

    logger.info(f"\n✓ Saved departmental to {output_dept}")
    logger.info(f"✓ Saved national to {output_nat}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
