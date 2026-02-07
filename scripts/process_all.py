"""Reproducible processing pipeline — runs the full Sprint 3 pipeline.

This script executes the complete data processing sequence:
    1. Reconstruct CPI deflator from monthly variation
    2. Process all 40 BCRP series (deflation, SA, transforms)
    3. Run Chow-Lin GDP disaggregation
    4. Build national monthly panel
    5. Validate output

Designed for reproducibility: running this script from a fresh data download
should produce identical results.

Usage:
    python scripts/process_all.py              # full processing
    python scripts/process_all.py --validate   # validate only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_NATIONAL_DIR,
    RAW_BCRP_DIR,
    SERIES_CATALOG_PATH,
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "process_all.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.process_all")


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS reproducible processing pipeline"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Only validate the existing panel",
    )
    args = parser.parse_args()

    raw_path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    output_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"

    if args.validate:
        return validate_only(output_path)

    return run_pipeline(raw_path, output_path)


def run_pipeline(raw_path: Path, output_path: Path) -> int:
    """Execute the full processing pipeline."""
    from src.processing.panel_builder import build_national_panel, validate_panel

    if not raw_path.exists():
        logger.error("Raw data not found: %s", raw_path)
        logger.info("Run 'python scripts/update_nexus.py --only bcrp' first.")
        return 1

    logger.info("=" * 60)
    logger.info("NEXUS Processing Pipeline")
    logger.info("=" * 60)
    logger.info("Input: %s", raw_path)
    logger.info("Output: %s", output_path)

    t0 = time.time()

    try:
        panel = build_national_panel(
            raw_path=raw_path,
            catalog_path=SERIES_CATALOG_PATH,
            output_path=output_path,
            include_gdp_monthly=True,
        )
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    elapsed = time.time() - t0

    # Validate
    result = validate_panel(panel)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("  Rows: %d", result["n_rows"])
    logger.info("  Series: %d", result["n_series"])
    logger.info("  Date range: %s", result["date_range"])
    logger.info("  Validation: %s", "PASSED" if result["passed"] else "FAILED")
    for check in result["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        logger.info("    [%s] %s: %s", status, check["name"], check["detail"])
    logger.info("=" * 60)

    return 0 if result["passed"] else 1


def validate_only(panel_path: Path) -> int:
    """Validate existing panel."""
    import pandas as pd
    from src.processing.panel_builder import validate_panel

    if not panel_path.exists():
        logger.error("Panel not found: %s", panel_path)
        return 1

    panel = pd.read_parquet(panel_path)
    result = validate_panel(panel)

    logger.info("Validation: %s", "PASSED" if result["passed"] else "FAILED")
    for check in result["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        logger.info("  [%s] %s: %s", status, check["name"], check["detail"])

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
