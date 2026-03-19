"""
=============================================================================
QHAWARINA DAILY PIPELINE — run_daily_pipeline.py
=============================================================================

FULL DOCUMENTATION: D:/nexus/nexus/PIPELINE.md  ← READ THIS FIRST

Quick reference:
  - Runs daily at 21:00 Lima (PET = UTC-5) via Windows Task Scheduler
  - Total runtime: ~30 minutes
  - Log: logs/daily_pipeline_YYYY-MM-DD.log

STEPS (in order):
  1. Supermarket scrape  — Plaza Vea, Metro, Wong (~42k products)
  2. Political index     — RSS feeds + Claude Haiku → IRP + IRE indices
  3. Export web data     — All JSON/CSV for website (export_web_data.py)
  4. Sync to web         — copies exports/ → D:/qhawarina/public/assets/data/
  5. Monthly poverty     — runs on 15th only (GBR nowcast model)
  6. Git push            — disabled by default, use --push flag

KEY OUTPUT: data/processed/daily_instability/daily_index.parquet
  ↑ This is the IRP/IRE source of truth. Updated daily.
  ↑ Do NOT use data/processed/daily/daily_index.parquet (legacy, stuck 2026-03-13)

IRP/IRE SCALE:
  0-50   = quiet   |  50-100 = moderate  |  100-180 = elevated
  180-250 = high   |  250+   = very high (major political shock)

USAGE:
    python scripts/run_daily_pipeline.py                   # standard (no push)
    python scripts/run_daily_pipeline.py --push            # + git commit/push
    python scripts/run_daily_pipeline.py --skip-scrape     # skip supermarket
    python scripts/run_daily_pipeline.py --skip-political  # skip RSS/Haiku
    python scripts/run_daily_pipeline.py --no-push         # (alias for default)

KNOWN BUGS (fixed):
  - .iloc[-1] IRP/IRE partial-day UTC bleed → fixed in export_web_data.py
    (now uses last row with n_articles_total >= 100)

=============================================================================
"""

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"daily_pipeline_{date.today().isoformat()}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.daily")

PYTHON = sys.executable
SCRIPTS = PROJECT_ROOT / "scripts"
WEB_DIR = Path("D:/qhawarina")


def run(label: str, cmd: list, cwd=PROJECT_ROOT) -> bool:
    logger.info("── %s ──────────────────────────────────", label)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
    if result.returncode != 0:
        logger.error("FAILED (%d): %s", result.returncode, label)
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-20:]:
                logger.error("  %s", line)
        return False
    return True


def step_scrape_supermarket() -> bool:
    """Step 1: Scrape supermarket prices and save snapshot."""
    script = f"""
import sys, pandas as pd
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.ingestion.supermarket import SupermarketScraper
from datetime import date
today = date.today()
scraper = SupermarketScraper()
df = scraper.scrape_all_stores(today)
if not df.empty:
    scraper.save_daily_snapshot(df, today)
    by_store = df.groupby('store')['product_name'].count()
    print(f'Saved {{len(df)}} products for {{today}}')
    print(by_store.to_string())
else:
    print('WARNING: scraper returned empty dataframe')
"""
    return run(
        "Supermarket scrape",
        [PYTHON, "-c", script],
    )


def step_political_index() -> bool:
    """Step 2: Rebuild political instability index."""
    return run(
        "Political index",
        [PYTHON, str(SCRIPTS / "build_daily_index.py")],
    )


def step_export() -> bool:
    """Step 3: Export all web data (includes price index rebuild)."""
    return run(
        "Export web data",
        [PYTHON, str(SCRIPTS / "export_web_data.py")],
    )


def step_sync() -> bool:
    """Step 4: Sync exports to qhawarina web project."""
    return run(
        "Sync to web",
        [PYTHON, str(SCRIPTS / "sync_web_data.py")],
    )


def step_monthly_poverty() -> bool:
    """
    Monthly poverty nowcast — runs on the 15th of each month.

    Sequence:
      1. regional_download — fetch latest departmental BCRP series
      2. regional          — rebuild panel_departmental_monthly.parquet
      3. build_poverty_nowcast.py  — rebuild vintage_panel + monthly_indicators
      4. generate_poverty_nowcast.py — produce poverty_nowcast.json + copy to web

    Returns True even if some sub-steps warn (non-fatal), False only on hard failure.
    """
    today_dt = datetime.now()
    if today_dt.day != 15:
        logger.info("Monthly poverty: skipping (today is the %dth, runs on the 15th)", today_dt.day)
        return True  # not an error — just not time yet

    logger.info("=" * 60)
    logger.info("MONTHLY POVERTY NOWCAST — %s", today_dt.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    # Step 1: Fetch latest regional BCRP data
    ok1 = run(
        "Regional BCRP download",
        [PYTHON, str(SCRIPTS / "update_nexus.py"), "--only", "regional_download"],
    )
    if not ok1:
        logger.warning("Regional download failed — poverty nowcast will use existing data")

    # Step 2: Rebuild departmental monthly panel
    ok2 = run(
        "Regional panel rebuild",
        [PYTHON, str(SCRIPTS / "update_nexus.py"), "--only", "regional"],
    )
    if not ok2:
        logger.warning("Regional panel rebuild failed — poverty nowcast may be stale")

    # Step 3: Rebuild vintage panel + monthly indicators
    ok3 = run(
        "Build poverty vintage panel",
        [PYTHON, str(SCRIPTS / "build_poverty_nowcast.py")],
    )
    if not ok3:
        logger.error("build_poverty_nowcast.py FAILED — aborting nowcast generation")
        return False

    # Step 4: Generate nowcast + export JSON to web
    ok4 = run(
        "Generate poverty nowcast",
        [PYTHON, str(SCRIPTS / "generate_poverty_nowcast.py")],
    )
    if not ok4:
        logger.error("generate_poverty_nowcast.py FAILED")
        return False

    logger.info("Monthly poverty nowcast complete.")
    return True


def step_git_push(today: date, include_poverty: bool = False) -> bool:
    """Step 5: Git commit + push in qhawarina repo."""
    if include_poverty:
        msg = f"Daily update {today.isoformat()}: prices + political index + poverty nowcast"
    else:
        msg = f"Daily update {today.isoformat()}: prices + political index"

    ok = run("Git add", ["git", "add", "public/assets/data/"], cwd=WEB_DIR)
    if not ok:
        return False

    # Check if there's anything to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        cwd=WEB_DIR, capture_output=True, text=True
    )
    if not result.stdout.strip():
        logger.info("  No changes to commit (data unchanged)")
        return True

    ok = run(
        "Git commit",
        ["git", "commit", "-m", msg],
        cwd=WEB_DIR,
    )
    if not ok:
        return False

    return run("Git push", ["git", "push"], cwd=WEB_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-political", action="store_true")
    parser.add_argument("--no-push", action="store_true")  # kept for backward compat
    parser.add_argument("--push", action="store_true", help="Push to GitHub after export (off by default)")
    args = parser.parse_args()

    today = date.today()
    logger.info("=" * 60)
    logger.info("QHAWARINA DAILY PIPELINE — %s", today.isoformat())
    logger.info("=" * 60)

    results = {}

    if not args.skip_scrape:
        results["scrape"] = step_scrape_supermarket()
    else:
        logger.info("Skipping supermarket scrape (--skip-scrape)")

    if not args.skip_political:
        results["political"] = step_political_index()
    else:
        logger.info("Skipping political index (--skip-political)")

    results["export"] = step_export()
    results["sync"] = step_sync()

    poverty_ran = False
    poverty_result = step_monthly_poverty()  # has its own day-15 guard; returns True on non-15th
    if datetime.now().day == 15:
        results["poverty"] = poverty_result
        poverty_ran = poverty_result

    if args.push:
        results["push"] = step_git_push(today, include_poverty=poverty_ran)
    else:
        logger.info("Skipping git push (auto-push disabled — run with --push to deploy)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %s", datetime.now().strftime("%H:%M:%S"))
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %-12s %s", step, status)
    logger.info("Log: %s", log_file)
    logger.info("=" * 60)

    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
