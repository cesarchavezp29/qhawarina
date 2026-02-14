"""Unified NEXUS updater — single entry point for all data operations.

Full pipeline (12 steps):
     1. bcrp              — National BCRP monthly series download (55 series)
     2. regional_download — Regional BCRP departmental download (549 series)
     3. expanded          — Inflation components + sectoral production (~90 series)
     4. midagri           — MIDAGRI wholesale food price bulletins (download + extract)
     5. gdp               — Quarterly GDP targets download
     6. inflation         — Inflation target compilation
     7. enaho             — ENAHO poverty download + computation
     8. panel             — Rebuild national monthly panel
     9. regional          — Build departmental monthly panel
    10. political         — Political instability index
    11. daily_rss         — Daily RSS instability index (political + economic)
    12. viz               — Regenerate all maps and charts

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
import os
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


VALID_STEPS = {
    "bcrp", "regional_download", "expanded", "midagri", "supermarket",
    "gdp", "inflation", "enaho",
    "panel", "regional", "political", "daily_rss", "viz", "reports",
}


# ── Step 1: National BCRP download ──────────────────────────────────────────

def step_bcrp(dry_run: bool = False, force: bool = False) -> dict:
    """Step 1: Update BCRP monthly series."""
    from scripts.update_bcrp import update_bcrp
    force_months = 3 if force else 0
    return update_bcrp(dry_run=dry_run, force_months=force_months)


# ── Step 2: Regional BCRP download ──────────────────────────────────────────

def step_regional_download(dry_run: bool = False, force: bool = False) -> dict:
    """Step 2: Download regional/departmental BCRP series."""
    import yaml

    catalog_path = PROJECT_ROOT / "config" / "regional_series_catalog.yaml"
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    regional = catalog.get("regional", {})
    n_cats = sum(1 for v in regional.values() if isinstance(v, dict) and v.get("series"))
    n_series = sum(
        len(v.get("series", []))
        for v in regional.values()
        if isinstance(v, dict)
    )

    if dry_run:
        logger.info(
            "DRY RUN: Would download %d series across %d categories",
            n_series, n_cats,
        )
        return {"status": "dry_run", "series_count": n_series, "categories": n_cats}

    # Use sys.argv manipulation to call the argparse-based script
    saved_argv = sys.argv
    try:
        sys.argv = ["download_regional_bcrp.py"]
        from scripts.download_regional_bcrp import main as regional_main
        rc = regional_main()
        return {
            "status": "ok" if rc == 0 else "error",
            "series_count": n_series,
            "categories": n_cats,
        }
    finally:
        sys.argv = saved_argv


# ── Step 3: Expanded BCRP (inflation components + sectoral) ─────────────────

def step_expanded(dry_run: bool = False, force: bool = False) -> dict:
    """Step 3: Download inflation components + sectoral production."""
    if dry_run:
        logger.info("DRY RUN: Would download ~90 inflation/sectoral series (--download-only)")
        return {"status": "dry_run", "estimated_series": 90}

    saved_argv = sys.argv
    try:
        sys.argv = ["download_expanded_bcrp.py", "--download-only"]
        from scripts.download_expanded_bcrp import main as expanded_main
        rc = expanded_main()
        return {"status": "ok" if rc == 0 else "error"}
    finally:
        sys.argv = saved_argv


# ── Step 4: MIDAGRI wholesale prices ──────────────────────────────────────

def step_midagri(dry_run: bool = False, force: bool = False) -> dict:
    """Step 4: Download + extract MIDAGRI wholesale food + poultry bulletins."""
    from src.ingestion.midagri import (
        MidagriBulletinScraper,
        MidagriPDFParser,
        MidagriAggregator,
        PoultryBulletinScraper,
        PoultryPDFParser,
        PoultryAggregator,
    )

    if dry_run:
        logger.info("DRY RUN: Would download new MIDAGRI bulletins + extract prices")
        return {"status": "dry_run"}

    # --- Produce (GMML wholesale market) ---
    scraper = MidagriBulletinScraper()
    new_pdfs = scraper.scrape_new_bulletins(max_pages=10, force=force)
    n_downloaded = sum(1 for e in new_pdfs if e["status"] == "downloaded")

    parser = MidagriPDFParser()
    daily_df = parser.parse_all_bulletins(force=force)

    monthly_rows = 0
    if not daily_df.empty:
        agg = MidagriAggregator()
        monthly_df = agg.build_monthly_series(daily_df)
        monthly_rows = len(monthly_df)

    # --- Poultry (chicken & egg) ---
    poultry_scraper = PoultryBulletinScraper()
    poultry_new = poultry_scraper.scrape_new_bulletins(max_pages=5, force=force)
    n_poultry_downloaded = sum(1 for e in poultry_new if e["status"] == "downloaded")

    poultry_parser = PoultryPDFParser()
    poultry_daily = poultry_parser.parse_all_bulletins(force=force)

    poultry_monthly_rows = 0
    if not poultry_daily.empty:
        poultry_agg = PoultryAggregator()
        poultry_monthly = poultry_agg.build_monthly_series(poultry_daily)
        poultry_monthly_rows = len(poultry_monthly)

    return {
        "status": "ok",
        "new_pdfs": n_downloaded,
        "daily_records": len(daily_df) if not daily_df.empty else 0,
        "monthly_rows": monthly_rows,
        "poultry_new_pdfs": n_poultry_downloaded,
        "poultry_daily_records": len(poultry_daily) if not poultry_daily.empty else 0,
        "poultry_monthly_rows": poultry_monthly_rows,
    }


# ── Step 4b: Supermarket prices ──────────────────────────────────────────────

def step_supermarket(dry_run: bool = False, force: bool = False) -> dict:
    """Step 4b: Scrape supermarket prices and build price index."""
    from src.ingestion.supermarket import (
        PriceIndexBuilder,
        SupermarketScraper,
    )

    if dry_run:
        scraper = SupermarketScraper()
        snapshots = scraper.list_available_snapshots()
        logger.info(
            "DRY RUN: Would scrape 3 stores, %d existing snapshots",
            len(snapshots),
        )
        return {"status": "dry_run", "existing_snapshots": len(snapshots)}

    scraper = SupermarketScraper()

    # Scrape all stores for today
    df = scraper.scrape_all_stores()
    n_products = len(df)

    if not df.empty:
        scraper.save_daily_snapshot(df)

    # Build index if we have >= 2 snapshots
    builder = PriceIndexBuilder()
    snapshots = scraper.list_available_snapshots()
    monthly_rows = 0

    if len(snapshots) >= 2:
        daily_index = builder.build_daily_index()
        if not daily_index.empty:
            monthly = builder.build_monthly_index(daily_index)
            if not monthly.empty:
                builder.save_monthly_index(monthly)
                monthly_rows = len(monthly)

    return {
        "status": "ok",
        "products_scraped": n_products,
        "total_snapshots": len(snapshots),
        "monthly_rows": monthly_rows,
    }


# ── Step 5: GDP targets ─────────────────────────────────────────────────────

def step_gdp(dry_run: bool = False, force: bool = False) -> dict:
    """Step 4: Download quarterly GDP targets."""
    from src.ingestion.targets import update_quarterly_gdp
    if dry_run:
        from src.ingestion.targets import GDP_CODE_MAP
        logger.info("DRY RUN: Would download %d quarterly GDP series", len(GDP_CODE_MAP))
        return {"status": "dry_run", "series_count": len(GDP_CODE_MAP)}
    return update_quarterly_gdp(force=force)


# ── Step 5: Inflation targets ───────────────────────────────────────────────

def step_inflation(dry_run: bool = False, force: bool = False) -> dict:
    """Step 5: Compile inflation targets from existing data."""
    from src.ingestion.targets import compile_inflation_targets
    if dry_run:
        logger.info("DRY RUN: Would reshape inflation data into target schema")
        return {"status": "dry_run"}
    return compile_inflation_targets()


# ── Step 6: ENAHO poverty ───────────────────────────────────────────────────

def step_enaho(dry_run: bool = False, force: bool = False) -> dict:
    """Step 6: Download ENAHO and compute poverty targets."""
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


# ── Step 7: National panel ──────────────────────────────────────────────────

def step_panel(dry_run: bool = False, force: bool = False) -> dict:
    """Step 7: Rebuild national monthly panel."""
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


# ── Step 8: Departmental panel ──────────────────────────────────────────────

def step_regional(dry_run: bool = False, force: bool = False) -> dict:
    """Step 8: Build departmental panel from regional BCRP data."""
    from config.settings import PROCESSED_DEPARTMENTAL_DIR, RAW_BCRP_DIR

    regional_catalog = PROJECT_ROOT / "config" / "regional_series_catalog.yaml"
    national_raw = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    output_path = PROCESSED_DEPARTMENTAL_DIR / "panel_departmental_monthly.parquet"

    if dry_run:
        import yaml
        with open(regional_catalog, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        regional = catalog.get("regional", {})
        n_cats = sum(1 for v in regional.values() if isinstance(v, dict) and v.get("series"))
        logger.info("DRY RUN: Would build departmental panel from %d categories", n_cats)
        return {"status": "dry_run", "categories": n_cats}

    from src.processing.panel_builder import (
        build_departmental_panel,
        validate_departmental_panel,
    )
    panel = build_departmental_panel(
        regional_data_dir=RAW_BCRP_DIR,
        regional_catalog_path=regional_catalog,
        national_raw_path=national_raw,
        output_path=output_path,
    )
    result = validate_departmental_panel(panel)
    return {
        "status": "ok" if result["passed"] else "validation_failed",
        "n_rows": result["n_rows"],
        "n_series": result["n_series"],
        "n_departments": result["n_departments"],
    }


# ── Step 9: Political instability index ─────────────────────────────────────

def step_political(dry_run: bool = False, force: bool = False) -> dict:
    """Step 9: Build political instability index."""
    from config.settings import RAW_POLITICAL_DIR, PROCESSED_POLITICAL_DIR

    if dry_run:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        cache_exists = (RAW_POLITICAL_DIR / "wikipedia_pages_cache.parquet").exists()
        logger.info(
            "DRY RUN: Would build political index "
            "(API key=%s, cached pages=%s)",
            "yes" if has_key else "no",
            "yes" if cache_exists else "no",
        )
        return {
            "status": "dry_run",
            "has_api_key": has_key,
            "has_cached_pages": cache_exists,
        }

    # Build argv for the political index script
    argv = ["build_political_index.py"]

    # Check for cached Wikipedia pages
    cache_path = RAW_POLITICAL_DIR / "wikipedia_pages_cache.parquet"
    if cache_path.exists():
        argv.append("--skip-fetch")
        logger.info("Using cached Wikipedia pages")

    # Check for Anthropic API key
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_key:
        # Also check if anthropic package is even available
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning(
                "anthropic package not installed — political index will use GT fallback. "
                "Install with: pip install anthropic"
            )
            argv.append("--skip-claude")
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set — political index will use GT fallback scores. "
                "Set the key in .env or environment to enable Claude classification."
            )
            argv.append("--skip-claude")

    saved_argv = sys.argv
    try:
        sys.argv = argv
        from scripts.build_political_index import main as political_main
        rc = political_main()
        return {"status": "ok" if rc == 0 else "error"}
    finally:
        sys.argv = saved_argv


# ── Step 10: Daily RSS instability index ────────────────────────────

def step_daily_rss(dry_run: bool = False, force: bool = False) -> dict:
    """Step 10: Build daily RSS political + economic instability index."""
    from config.settings import RAW_RSS_DIR, PROCESSED_DAILY_DIR

    if dry_run:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        cache_exists = (RAW_RSS_DIR / "articles_classified.parquet").exists()
        logger.info(
            "DRY RUN: Would build daily RSS index "
            "(API key=%s, cached articles=%s)",
            "yes" if has_key else "no",
            "yes" if cache_exists else "no",
        )
        return {
            "status": "dry_run",
            "has_api_key": has_key,
            "has_cached_articles": cache_exists,
        }

    # Check feedparser
    try:
        import feedparser  # noqa: F401
    except ImportError:
        logger.warning(
            "feedparser not installed — skipping daily RSS step. "
            "Install with: pip install feedparser"
        )
        return {"status": "skipped", "reason": "feedparser not installed"}

    argv = ["build_daily_index.py"]

    # Check for cached articles
    cache_path = RAW_RSS_DIR / "articles_cache.parquet"
    if cache_path.exists():
        argv.append("--skip-fetch")
        logger.info("Using cached RSS articles")

    # Check for Anthropic API key
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_key:
        classified_path = RAW_RSS_DIR / "articles_classified.parquet"
        if classified_path.exists():
            argv.append("--skip-claude")
            logger.info("No API key — using cached classifications")
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set and no cached classifications — "
                "daily index will be empty"
            )
            argv.append("--skip-claude")

    if force:
        argv.append("--force")

    saved_argv = sys.argv
    try:
        sys.argv = argv
        from scripts.build_daily_index import main as daily_main
        rc = daily_main()
        return {"status": "ok" if rc == 0 else "error"}
    finally:
        sys.argv = saved_argv


# ── Step 11: Visualization ──────────────────────────────────────────────────

def step_viz(dry_run: bool = False, force: bool = False) -> dict:
    """Step 10: Regenerate maps and charts."""
    if dry_run:
        logger.info("DRY RUN: Would regenerate maps + charts")
        return {"status": "dry_run"}

    # Check geopandas availability
    try:
        import geopandas  # noqa: F401
    except ImportError:
        logger.warning(
            "geopandas not installed — skipping visualization step. "
            "Install with: pip install geopandas"
        )
        return {"status": "skipped", "reason": "geopandas not installed"}

    # Check geodata availability
    from src.ingestion.geo import check_geodata_available, geodata_download_instructions
    if not check_geodata_available():
        logger.warning(
            "Geodata files missing — skipping visualization step.\n%s",
            geodata_download_instructions(),
        )
        return {"status": "skipped", "reason": "geodata files missing"}

    # Generate maps and charts
    try:
        from src.visualization.maps import generate_all_maps
        from src.visualization.charts import generate_all_charts

        map_paths = generate_all_maps()
        chart_paths = generate_all_charts()

        total = len(map_paths) + len(chart_paths)
        return {
            "status": "ok",
            "maps_generated": len(map_paths),
            "charts_generated": len(chart_paths),
            "total_files": total,
        }
    except Exception as e:
        logger.error("Visualization failed: %s", e)
        return {"status": "error", "error": str(e)}


# ── Step 12: PDF reports ───────────────────────────────────────────────────

def step_reports(dry_run: bool = False, force: bool = False) -> dict:
    """Step 12: Generate daily PDF report (+ weekly on Mondays, monthly on 1st)."""
    from config.settings import PROCESSED_DAILY_DIR, RAW_RSS_DIR

    index_path = PROCESSED_DAILY_DIR / "daily_index.parquet"
    classified_path = RAW_RSS_DIR / "articles_classified.parquet"

    if not index_path.exists() or not classified_path.exists():
        logger.warning("Daily index or classified articles not found — skipping reports")
        return {"status": "skipped", "reason": "missing data"}

    if dry_run:
        logger.info("DRY RUN: Would generate daily PDF report")
        return {"status": "dry_run"}

    from src.reporting.pdf_report import (
        generate_daily_pdf,
        generate_weekly_pdf,
        generate_monthly_pdf,
    )

    logo_path = PROJECT_ROOT / "assets" / "logo.png"
    report_dir = PROCESSED_DAILY_DIR / "reports"
    today = datetime.now().date()
    generated = []

    # Always generate yesterday's daily report (data is more likely complete)
    yesterday = (today - __import__("datetime").timedelta(days=1)).isoformat()
    daily_out = report_dir / "daily" / f"report_{yesterday}.pdf"
    if not daily_out.exists() or force:
        generate_daily_pdf(yesterday, index_path, classified_path, daily_out, logo_path)
        generated.append(f"daily:{yesterday}")

    # Weekly report on Mondays (for previous week)
    if today.weekday() == 0 or force:
        prev_week = today - __import__("datetime").timedelta(days=7)
        yr, wk, _ = prev_week.isocalendar()
        weekly_out = report_dir / "weekly" / f"report_{yr}-W{wk:02d}.pdf"
        if not weekly_out.exists() or force:
            generate_weekly_pdf(yr, wk, index_path, classified_path, weekly_out, logo_path)
            generated.append(f"weekly:{yr}-W{wk:02d}")

    # Monthly report on 1st (for previous month)
    if today.day == 1 or force:
        prev_month = today.replace(day=1) - __import__("datetime").timedelta(days=1)
        monthly_out = report_dir / "monthly" / f"report_{prev_month.year}-{prev_month.month:02d}.pdf"
        if not monthly_out.exists() or force:
            generate_monthly_pdf(
                prev_month.year, prev_month.month,
                index_path, classified_path, monthly_out, logo_path,
            )
            generated.append(f"monthly:{prev_month.year}-{prev_month.month:02d}")

    return {"status": "ok", "generated": generated}


# ── Step registry ───────────────────────────────────────────────────────────

STEPS = {
    "bcrp": ("National BCRP series", step_bcrp),
    "regional_download": ("Regional BCRP download", step_regional_download),
    "expanded": ("Inflation components + sectoral", step_expanded),
    "midagri": ("MIDAGRI wholesale prices", step_midagri),
    "supermarket": ("Supermarket prices (BPP)", step_supermarket),
    "gdp": ("Quarterly GDP targets", step_gdp),
    "inflation": ("Inflation targets", step_inflation),
    "enaho": ("ENAHO poverty targets", step_enaho),
    "panel": ("National monthly panel", step_panel),
    "regional": ("Departmental monthly panel", step_regional),
    "political": ("Political instability index", step_political),
    "daily_rss": ("Daily RSS instability index", step_daily_rss),
    "viz": ("Maps and charts", step_viz),
    "reports": ("PDF instability reports", step_reports),
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
        logger.info("  [%s] %-20s: %-20s (%.1fs)", mark, name, status, elapsed)
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
