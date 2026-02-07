"""Build the departmental monthly panel from regional BCRP data.

Processes all downloaded regional series through the harmonization pipeline
(deflation, seasonal adjustment, transformations) and stacks them into a
long-format panel suitable for spatial nowcasting models.

Usage:
    python scripts/build_regional_panel.py              # full build
    python scripts/build_regional_panel.py --validate   # check existing panel
    python scripts/build_regional_panel.py --dry-run    # preview without building
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DEPARTMENTAL_DIR,
    RAW_BCRP_DIR,
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "build_regional_panel.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.build_regional_panel")

REGIONAL_CATALOG = PROJECT_ROOT / "config" / "regional_series_catalog.yaml"
NATIONAL_RAW = RAW_BCRP_DIR / "bcrp_national_all.parquet"
OUTPUT_PATH = PROCESSED_DEPARTMENTAL_DIR / "panel_departmental_monthly.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Build departmental monthly panel from regional BCRP data"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing panel without rebuilding",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview categories and series counts without building",
    )
    args = parser.parse_args()

    if args.validate:
        return validate_existing()

    if args.dry_run:
        return dry_run()

    return build()


def dry_run() -> int:
    """Preview what would be built."""
    import yaml
    from src.utils.io import list_parquet_files

    with open(REGIONAL_CATALOG, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    regional = catalog.get("regional", {})
    total_series = 0

    print("=" * 70)
    print("DRY RUN — Departmental Panel Builder")
    print("=" * 70)

    for cat_key, cat_data in regional.items():
        if not isinstance(cat_data, dict):
            continue
        series = cat_data.get("series", [])
        processing = cat_data.get("processing", {})
        unit_type = processing.get("unit_type", "level")
        panel_filter = processing.get("panel_filter", None)

        # Count effective series (after filter)
        if panel_filter:
            effective = len([s for s in series if s.get("currency") == panel_filter and s.get("department")])
        else:
            effective = len([s for s in series if s.get("department") or s.get("ubigeo")])

        total_series += effective
        status = "OK" if effective > 0 else "EMPTY"
        print(f"  {cat_key:<40s} {effective:>3d} series  (unit={unit_type})  [{status}]")

    print("-" * 70)
    print(f"  TOTAL: {total_series} departmental series")

    # Check data files
    existing = list_parquet_files(RAW_BCRP_DIR)
    regional_files = [f for f in existing if "regional" in f.name]
    print(f"\n  Available parquet files: {len(regional_files)}")
    for f in regional_files:
        import pandas as pd
        df = pd.read_parquet(f)
        print(f"    {f.name}: {len(df):,} rows, {df['series_code'].nunique()} series")

    print(f"\n  National raw data: {'EXISTS' if NATIONAL_RAW.exists() else 'MISSING'}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 70)

    return 0


def build() -> int:
    """Build the departmental panel."""
    from src.processing.panel_builder import (
        build_departmental_panel,
        validate_departmental_panel,
    )

    if not NATIONAL_RAW.exists():
        logger.error("National raw data not found: %s", NATIONAL_RAW)
        logger.error("Run 'python scripts/update_nexus.py --only bcrp' first")
        return 1

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("BUILDING DEPARTMENTAL PANEL")
    logger.info("=" * 60)

    panel = build_departmental_panel(
        regional_data_dir=RAW_BCRP_DIR,
        regional_catalog_path=REGIONAL_CATALOG,
        national_raw_path=NATIONAL_RAW,
        output_path=OUTPUT_PATH,
    )

    elapsed = time.time() - t0
    logger.info("Build completed in %.1f seconds", elapsed)

    # Validate
    result = validate_departmental_panel(panel)
    print_validation(result)

    return 0 if result["passed"] else 1


def validate_existing() -> int:
    """Validate an existing panel file."""
    import pandas as pd
    from src.processing.panel_builder import validate_departmental_panel

    if not OUTPUT_PATH.exists():
        logger.error("Panel file not found: %s", OUTPUT_PATH)
        logger.error("Run 'python scripts/build_regional_panel.py' first")
        return 1

    panel = pd.read_parquet(OUTPUT_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    result = validate_departmental_panel(panel)
    print_validation(result)

    # Additional summary
    print("\n--- Panel Summary ---")
    print(f"  Rows: {len(panel):,}")
    print(f"  Series: {panel['series_id'].nunique()}")
    print(f"  Departments: {panel['ubigeo'].nunique()}")
    print(f"  Categories: {panel['category'].nunique()}")
    print(f"  Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    print("\n--- By Category ---")
    for cat in sorted(panel["category"].unique()):
        sub = panel[panel["category"] == cat]
        print(f"  {cat:<40s} {sub['series_id'].nunique():>3d} series, "
              f"{sub['ubigeo'].nunique():>2d} depts, "
              f"{len(sub):>6,d} rows")

    print("\n--- By Department ---")
    for ubigeo in sorted(panel["ubigeo"].unique()):
        sub = panel[panel["ubigeo"] == ubigeo]
        dept = sub["department"].iloc[0] if len(sub) > 0 else "?"
        print(f"  {ubigeo} {dept:<20s} {sub['series_id'].nunique():>3d} series, "
              f"{len(sub):>6,d} rows")

    return 0 if result["passed"] else 1


def print_validation(result: dict):
    """Print validation results."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for check in result["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['name']}: {check['detail']}")

    overall = "PASSED" if result["passed"] else "FAILED"
    print(f"\nOverall: {overall}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
