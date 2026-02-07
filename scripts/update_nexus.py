"""Unified NEXUS updater — single entry point for all data operations.

Replaces update_all.py with expanded step coverage:
    1. bcrp     — Incremental BCRP monthly series update
    2. gdp      — Quarterly GDP targets download
    3. inflation — Inflation target compilation
    4. enaho    — ENAHO poverty download + computation
    5. panel    — Rebuild national monthly panel (processing pipeline)
    6. viz      — Regenerate all maps and charts

Each step runs independently — a failure in one does not block others.

Usage:
    python scripts/update_nexus.py                     # update everything
    python scripts/update_nexus.py --only bcrp         # only BCRP data
    python scripts/update_nexus.py --only panel        # only rebuild panel
    python scripts/update_nexus.py --only viz          # only regenerate visualizations
    python scripts/update_nexus.py --check             # dry-run for all steps
    python scripts/update_nexus.py --force             # force re-download all

Exit codes:
    0 = all steps succeeded
    1 = one or more steps failed
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "update_nexus.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("nexus.update")


VALID_STEPS = {"bcrp", "gdp", "inflation", "enaho", "panel", "viz"}


def step_bcrp(dry_run: bool = False, force: bool = False) -> dict:
    """Step 1: Update BCRP monthly series."""
    from scripts.update_bcrp import update_bcrp
    force_months = 3 if force else 0
    return update_bcrp(dry_run=dry_run, force_months=force_months)


def step_gdp(dry_run: bool = False, force: bool = False) -> dict:
    """Step 2: Download quarterly GDP targets."""
    from src.ingestion.targets import update_quarterly_gdp
    if dry_run:
        from src.ingestion.targets import GDP_CODE_MAP
        logger.info("DRY RUN: Would download %d quarterly GDP series", len(GDP_CODE_MAP))
        return {"status": "dry_run", "series_count": len(GDP_CODE_MAP)}
    return update_quarterly_gdp(force=force)


def step_inflation(dry_run: bool = False, force: bool = False) -> dict:
    """Step 3: Compile inflation targets from existing data."""
    from src.ingestion.targets import compile_inflation_targets
    if dry_run:
        logger.info("DRY RUN: Would reshape inflation data into target schema")
        return {"status": "dry_run"}
    return compile_inflation_targets()


def step_enaho(dry_run: bool = False, force: bool = False) -> dict:
    """Step 4: Download ENAHO and compute poverty targets."""
    from src.ingestion.inei import update_enaho
    if dry_run:
        from src.ingestion.inei import ENAHOClient
        client = ENAHOClient()
        missing = client.missing_years()
        logger.info(
            "DRY RUN: %d ENAHO years to download: %s",
            len(missing), missing,
        )
        return {"status": "dry_run", "missing_years": missing}
    return update_enaho(force=force)


def step_panel(dry_run: bool = False, force: bool = False) -> dict:
    """Step 5: Rebuild national monthly panel."""
    from config.settings import PROCESSED_NATIONAL_DIR, RAW_BCRP_DIR, SERIES_CATALOG_PATH

    raw_path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    output_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"

    if dry_run:
        import pandas as pd
        from src.processing.harmonize import load_series_metadata
        metadata = load_series_metadata(SERIES_CATALOG_PATH)
        logger.info("DRY RUN: Would process %d series into panel", len(metadata))
        return {"status": "dry_run", "series_count": len(metadata)}

    from src.processing.panel_builder import build_national_panel, validate_panel
    panel = build_national_panel(
        raw_path=raw_path,
        catalog_path=SERIES_CATALOG_PATH,
        output_path=output_path,
        include_gdp_monthly=True,
    )
    result = validate_panel(panel)
    return {
        "status": "ok" if result["passed"] else "validation_failed",
        "n_rows": result["n_rows"],
        "n_series": result["n_series"],
    }


def step_viz(dry_run: bool = False, force: bool = False) -> dict:
    """Step 6: Regenerate maps and charts."""
    if dry_run:
        logger.info("DRY RUN: Would regenerate 9 map/chart outputs")
        return {"status": "dry_run"}

    from config.settings import TARGETS_DIR
    maps_dir = TARGETS_DIR / "maps"
    if not maps_dir.exists():
        logger.warning("Maps directory does not exist: %s", maps_dir)
        return {"status": "skipped", "reason": "No maps directory"}

    # Import and run map generation
    try:
        from src.visualization.maps import generate_all_maps
        from src.visualization.charts import generate_all_charts
        generate_all_maps()
        generate_all_charts()
        return {"status": "ok"}
    except ImportError:
        logger.warning("Visualization modules not fully available; skipping")
        return {"status": "skipped", "reason": "Import error"}
    except Exception as e:
        logger.error("Visualization failed: %s", e)
        return {"status": "error", "error": str(e)}


STEPS = {
    "bcrp": ("BCRP monthly series", step_bcrp),
    "gdp": ("Quarterly GDP targets", step_gdp),
    "inflation": ("Inflation targets", step_inflation),
    "enaho": ("ENAHO poverty targets", step_enaho),
    "panel": ("National monthly panel", step_panel),
    "viz": ("Maps and charts", step_viz),
}


def run_all(
    dry_run: bool = False,
    force: bool = False,
    only: str | None = None,
) -> dict:
    """Run all (or selected) update steps."""
    if only:
        step_names = [only]
    else:
        step_names = list(STEPS.keys())

    results = {}
    failures = 0

    for name in step_names:
        description, func = STEPS[name]
        logger.info("=" * 60)
        logger.info("STEP: %s", description)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            result = func(dry_run=dry_run, force=force)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            results[name] = result
            logger.info(
                "STEP %s completed in %.1fs: %s",
                name, elapsed, result.get("status", "ok"),
            )
        except Exception as e:
            elapsed = time.time() - t0
            logger.exception("STEP %s FAILED after %.1fs", name, elapsed)
            results[name] = {
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
            }
            failures += 1

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: %d/%d steps completed", len(step_names) - failures, len(step_names))
    for name, result in results.items():
        status = result.get("status", "unknown")
        elapsed = result.get("elapsed_seconds", 0)
        mark = "OK" if status not in ("error", "validation_failed") else "FAIL"
        logger.info("  [%s] %-12s: %-12s (%.1fs)", mark, name, status, elapsed)
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS unified data updater"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Dry run — show what would be updated without changes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download/rebuild all data",
    )
    parser.add_argument(
        "--only", choices=sorted(VALID_STEPS),
        help="Run only the specified step",
    )
    args = parser.parse_args()

    dry_run = args.check

    logger.info(
        "NEXUS Update started at %s (dry_run=%s, force=%s, only=%s)",
        datetime.now().isoformat(), dry_run, args.force, args.only,
    )

    try:
        results = run_all(
            dry_run=dry_run,
            force=args.force,
            only=args.only,
        )
        has_errors = any(r.get("status") == "error" for r in results.values())
        return 1 if has_errors else 0
    except Exception:
        logger.exception("Update failed with unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
