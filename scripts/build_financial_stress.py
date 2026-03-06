"""Generate financial stress index and add to national panel.

This script:
1. Loads the national panel
2. Computes financial stress score (FX vol + credit spread + reserves)
3. Adds it as a new series to the panel
4. Saves updated panel
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processing.financial_component import build_financial_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_financial_stress")

PANEL_PATH = project_root / "data/processed/national/panel_national_monthly.parquet"
OUTPUT_PATH = PANEL_PATH  # Overwrite in place


def main():
    logger.info("Loading national panel from %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    logger.info("Loaded %d rows, %d series", len(panel), panel.series_id.nunique())

    # Generate financial stress score
    logger.info("Computing financial stress score...")
    stress_df = build_financial_score(panel, freq="M", zscore_window=60)

    if stress_df.empty or stress_df["financial_score"].isna().all():
        logger.error("Financial stress score could not be computed")
        return 1

    logger.info(
        "Financial stress: %d months, range [%.3f, %.3f]",
        len(stress_df),
        stress_df["financial_score"].min(),
        stress_df["financial_score"].max(),
    )

    # Convert to long format matching panel schema
    stress_long = pd.DataFrame({
        "date": stress_df["date"],
        "series_id": "FINANCIAL_STRESS_INDEX",
        "series_name": "Financial stress index (z-score composite)",
        "category": "financial",
        "value_raw": stress_df["financial_score"],
        "value_sa": stress_df["financial_score"],  # No seasonal adjustment needed
        "value_log": stress_df["financial_score"],  # Already transformed
        "value_dlog": stress_df["financial_score"].diff(),
        "value_yoy": stress_df["financial_score"].diff(12),  # diff, not pct_change — z-score crosses zero
        "source": "COMPUTED",
        "frequency_original": "M",
        "publication_lag_days": 0,  # Computed from contemporaneous data
    })

    # Remove any existing financial stress series
    panel_clean = panel[panel.series_id != "FINANCIAL_STRESS_INDEX"].copy()

    # Append new series
    panel_updated = pd.concat([panel_clean, stress_long], ignore_index=True)

    logger.info(
        "Updated panel: %d rows (was %d), %d series (was %d)",
        len(panel_updated),
        len(panel),
        panel_updated.series_id.nunique(),
        panel.series_id.nunique(),
    )

    # Save
    logger.info("Saving updated panel to %s", OUTPUT_PATH)
    panel_updated.to_parquet(OUTPUT_PATH, index=False)

    logger.info("✓ Financial stress index added to panel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
