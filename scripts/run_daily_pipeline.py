"""
Daily automated pipeline for Qhawarina.pe

Runs every day at 08:30 PET (13:30 UTC) via Windows Task Scheduler.

Steps:
  1. Scrape supermarket prices (Plaza Vea, Metro, Wong)
  2. Build political instability index (RSS + GPT-4o)
  3. Export all web data (nowcasts, political, poverty, price index)
  4. Sync exports to qhawarina web project
  5. Git commit + push

Usage:
    python scripts/run_daily_pipeline.py
    python scripts/run_daily_pipeline.py --skip-scrape   # skip supermarket
    python scripts/run_daily_pipeline.py --skip-political
    python scripts/run_daily_pipeline.py --no-push
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


def step_git_push(today: date) -> bool:
    """Step 5: Git commit + push in qhawarina repo."""
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
    parser.add_argument("--no-push", action="store_true")
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

    if not args.no_push:
        results["push"] = step_git_push(today)
    else:
        logger.info("Skipping git push (--no-push)")

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
