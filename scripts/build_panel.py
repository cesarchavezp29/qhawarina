"""Master panel construction script.

Builds the national monthly panel from raw BCRP data, applying:
  - Deflation (nominal soles → real via reconstructed IPC)
  - Seasonal adjustment (STL/X-13)
  - Transformations (log, dlog, yoy)
  - Chow-Lin monthly GDP disaggregation

Usage:
    python scripts/build_panel.py                # full build
    python scripts/build_panel.py --dry-run      # preview what would happen
    python scripts/build_panel.py --validate     # validate existing panel
    python scripts/build_panel.py --no-gdp       # skip Chow-Lin GDP
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
        logging.FileHandler(LOG_DIR / "build_panel.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.build_panel")


def main():
    parser = argparse.ArgumentParser(description="NEXUS panel builder")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be built without writing files",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate an existing panel instead of building",
    )
    parser.add_argument(
        "--no-gdp", action="store_true",
        help="Skip Chow-Lin GDP disaggregation",
    )
    args = parser.parse_args()

    raw_path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    output_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"

    if args.validate:
        return _validate(output_path)

    if args.dry_run:
        return _dry_run(raw_path)

    return _build(raw_path, output_path, include_gdp=not args.no_gdp)


def _dry_run(raw_path: Path) -> int:
    """Preview panel build without writing."""
    import pandas as pd
    from src.processing.harmonize import load_series_metadata

    if not raw_path.exists():
        logger.error("Raw data not found: %s", raw_path)
        return 1

    metadata = load_series_metadata(SERIES_CATALOG_PATH)
    raw_df = pd.read_parquet(raw_path)
    codes_in_data = set(raw_df["series_code"].unique())
    codes_in_catalog = set(metadata.keys())

    logger.info("DRY RUN — Panel Build Preview")
    logger.info("  Raw data: %s (%d rows)", raw_path, len(raw_df))
    logger.info("  Series in catalog: %d", len(codes_in_catalog))
    logger.info("  Series in data: %d", len(codes_in_data))
    logger.info("  Matched: %d", len(codes_in_catalog & codes_in_data))
    logger.info("  Missing from data: %s", codes_in_catalog - codes_in_data)

    # Count by processing type
    sa_count = sum(1 for m in metadata.values() if m["seasonal_adjust"] in ("stl", "x13"))
    deflate_count = sum(1 for m in metadata.values() if m["unit_type"] == "nominal_soles")
    logger.info("  Seasonal adjustment needed: %d series", sa_count)
    logger.info("  Deflation needed: %d series", deflate_count)

    return 0


def _build(raw_path: Path, output_path: Path, include_gdp: bool = True) -> int:
    """Build the national panel."""
    from src.processing.panel_builder import build_national_panel, validate_panel

    if not raw_path.exists():
        logger.error("Raw data not found: %s", raw_path)
        return 1

    t0 = time.time()
    try:
        panel = build_national_panel(
            raw_path=raw_path,
            catalog_path=SERIES_CATALOG_PATH,
            output_path=output_path,
            include_gdp_monthly=include_gdp,
        )
    except Exception:
        logger.exception("Panel build failed")
        return 1

    elapsed = time.time() - t0

    # Validate
    result = validate_panel(panel)
    logger.info("Validation: %s", "PASSED" if result["passed"] else "FAILED")
    for check in result["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        logger.info("  [%s] %s: %s", status, check["name"], check["detail"])

    logger.info("Panel built in %.1fs", elapsed)
    return 0 if result["passed"] else 1


def _validate(panel_path: Path) -> int:
    """Validate an existing panel."""
    import pandas as pd
    from src.processing.panel_builder import validate_panel

    if not panel_path.exists():
        logger.error("Panel not found: %s", panel_path)
        logger.info("Run 'python scripts/build_panel.py' to build it first.")
        return 1

    panel = pd.read_parquet(panel_path)
    result = validate_panel(panel)

    logger.info("Panel validation: %s", "PASSED" if result["passed"] else "FAILED")
    logger.info("  Rows: %d", result["n_rows"])
    logger.info("  Series: %d", result["n_series"])
    logger.info("  Date range: %s", result["date_range"])
    for check in result["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        logger.info("  [%s] %s: %s", status, check["name"], check["detail"])

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
