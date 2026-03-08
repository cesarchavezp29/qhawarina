"""Generate poverty nowcast for 2025."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DEPARTMENTAL_DIR, TARGETS_DIR, RESULTS_DIR
from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
from src.backtesting.backtester import PovertyBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("poverty_2025")

def main():
    # Load data
    panel = pd.read_parquet(PROCESSED_DEPARTMENTAL_DIR / "panel_departmental_monthly.parquet")
    targets = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")

    logger.info(f"Panel: {len(panel)} rows")
    logger.info(f"Targets: {len(targets)} rows, years {targets.year.min()}-{targets.year.max()}")

    # Create model
    model = PanelPovertyNowcaster(
        target_cols=["poverty_rate"],
        model_type="gbr",
        exclude_covid=True,
        include_ar=True
    )

    # Use backtester to run nowcast for 2025
    backtester = PovertyBacktester(
        dept_panel=panel,
        model=model,
        benchmarks={},
        poverty_df=targets,
        target_col="poverty_rate",
        value_col="value_yoy"
    )

    # Manually run for 2025
    logger.info("Generating nowcast for 2025...")
    target_year = 2025

    # Train on all available data (2004-2024)
    train_poverty = targets[targets.year < target_year].copy()

    # Aggregate panel to annual features (panel available through mid-2026)
    cutoff = pd.Timestamp(f"{target_year + 1}-07-01")
    panel_available = panel[panel["date"] < cutoff].copy()
    panel_available["date"] = pd.to_datetime(panel_available["date"])

    dept_features = _aggregate_dept_panel_annual(panel_available, value_col="value_yoy")
    logger.info(f"Aggregated panel: {len(dept_features)} dept-years")

    # Fit model
    model.fit(dept_features, train_poverty)

    # Generate nowcast (automatically detects 2025 as nowcast year)
    result = model.nowcast(dept_features, train_poverty)

    # Extract department nowcasts
    dept_nowcasts = result.get("dept_nowcasts", {})
    nowcast_rows = []
    for dept, values in dept_nowcasts.items():
        nowcast_rows.append({
            "ubigeo": dept,
            "poverty_rate_nowcast": values.get("poverty_rate", np.nan)
        })

    nowcast_2025 = pd.DataFrame(nowcast_rows)

    logger.info(f"\nPoverty Nowcast 2025:")
    logger.info(f"  Departments: {len(nowcast_2025)}")
    logger.info(f"  National avg: {nowcast_2025['poverty_rate_nowcast'].mean():.1f}%")

    logger.info(f"\n  Top 5 highest poverty:")
    top5 = nowcast_2025.nlargest(5, 'poverty_rate_nowcast')
    for _, row in top5.iterrows():
        logger.info(f"    Dept {row['ubigeo']}: {row['poverty_rate_nowcast']:.1f}%")

    # Save
    nowcast_2025['year'] = target_year
    output_path = RESULTS_DIR / "poverty_nowcast_2025.parquet"
    nowcast_2025.to_parquet(output_path, index=False)

    logger.info(f"\n✓ Saved to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
