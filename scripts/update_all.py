"""Unified NEXUS data updater — runs all ingestion steps.

Orchestrates:
    1. BCRP monthly series (incremental update)
    2. BCRP quarterly GDP targets (full re-download)
    3. Inflation target compilation (reshape existing data)
    4. ENAHO poverty targets (download missing years + compute)

Each step runs independently — a failure in one does not block others.

Usage:
    python scripts/update_all.py                    # Run all steps
    python scripts/update_all.py --dry-run          # Show what would happen
    python scripts/update_all.py --force             # Force re-download
    python scripts/update_all.py --only bcrp         # Run only BCRP monthly
    python scripts/update_all.py --only gdp          # Run only quarterly GDP
    python scripts/update_all.py --only inflation    # Run only inflation
    python scripts/update_all.py --only enaho        # Run only ENAHO

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
            LOG_DIR / "update_all.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("nexus.update_all")

VALID_STEPS = {"bcrp", "gdp", "inflation", "enaho"}


def step_bcrp(dry_run: bool = False, force: bool = False) -> dict:
    """Step 1: Update BCRP monthly series."""
    from scripts.update_bcrp import update_bcrp
    force_months = 3 if force else 0
    result = update_bcrp(dry_run=dry_run, force_months=force_months)

    # Also update quarterly GDP levels (run every time — fast, quarterly series)
    if not dry_run:
        from scripts.update_bcrp import update_gdp_levels
        levels_result = update_gdp_levels()
        result["gdp_levels"] = levels_result

    return result


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
    import pandas as pd

    if dry_run:
        logger.info("DRY RUN: Would reshape inflation data into target schema")
        return {"status": "dry_run"}

    result = compile_inflation_targets()

    # compile_inflation_targets() drops derived columns — rebuild ipc_3m_ma
    target_path = PROJECT_ROOT / "data" / "targets" / "inflation_monthly.parquet"
    if target_path.exists():
        df = pd.read_parquet(target_path).sort_values("date")
        if "ipc_3m_ma" not in df.columns:
            df["ipc_3m_ma"] = df["ipc_monthly_var"].rolling(3, min_periods=1).mean()
            df.to_parquet(target_path, index=False)
            logger.info("Rebuilt ipc_3m_ma in inflation_monthly.parquet")

    return result


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


STEPS = {
    "bcrp": ("BCRP monthly series", step_bcrp),
    "gdp": ("Quarterly GDP targets", step_gdp),
    "inflation": ("Inflation targets", step_inflation),
    "enaho": ("ENAHO poverty targets", step_enaho),
}


def run_all(
    dry_run: bool = False,
    force: bool = False,
    only: str | None = None,
) -> dict:
    """Run all (or selected) update steps.

    Returns dict mapping step name → result dict.
    """
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
            results[name] = {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 1)}
            failures += 1

    logger.info("=" * 60)
    logger.info("SUMMARY: %d/%d steps completed", len(step_names) - failures, len(step_names))
    for name, result in results.items():
        status = result.get("status", "unknown")
        elapsed = result.get("elapsed_seconds", 0)
        logger.info("  %-12s: %-12s (%.1fs)", name, status, elapsed)
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS unified data updater"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be updated without downloading",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download all data",
    )
    parser.add_argument(
        "--only", choices=sorted(VALID_STEPS),
        help="Run only the specified step",
    )
    args = parser.parse_args()

    logger.info(
        "NEXUS Update started at %s (dry_run=%s, force=%s, only=%s)",
        datetime.now().isoformat(), args.dry_run, args.force, args.only,
    )

    try:
        results = run_all(
            dry_run=args.dry_run,
            force=args.force,
            only=args.only,
        )
        # Return 1 if any step had an error
        has_errors = any(r.get("status") == "error" for r in results.values())
        return 1 if has_errors else 0
    except Exception:
        logger.exception("Update failed with unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
