"""Incremental BCRP data updater.

Checks the latest date in existing data and downloads only new observations.
Designed to run on a schedule (daily/weekly) via Windows Task Scheduler or cron.

Usage:
    python scripts/update_bcrp.py              # Update with new data
    python scripts/update_bcrp.py --dry-run    # Show what would be updated
    python scripts/update_bcrp.py --force      # Re-download last 3 months

Exit codes:
    0 = success (data updated or already up to date)
    1 = error during update
"""

import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import SERIES_CATALOG_PATH, RAW_BCRP_DIR
from src.ingestion.bcrp import BCRPClient, load_series_codes
from src.utils.io import save_parquet

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "update_bcrp.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)


def get_latest_date(parquet_path: Path) -> date | None:
    """Get the latest date from existing parquet data."""
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path, columns=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def update_bcrp(dry_run: bool = False, force_months: int = 0) -> dict:
    """Download new BCRP data since last update.

    Args:
        dry_run: If True, only report what would be updated.
        force_months: Re-download this many months back (0 = only new).

    Returns:
        Dict with update statistics.
    """
    combined_path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    latest = get_latest_date(combined_path)
    today = date.today()

    if latest is None:
        logger.info("No existing data found. Run download_all.py first for initial load.")
        return {"status": "no_data", "new_rows": 0}

    # Calculate start date for update
    if force_months > 0:
        # Go back N months from latest
        start_year = latest.year
        start_month = latest.month - force_months
        while start_month < 1:
            start_month += 12
            start_year -= 1
    else:
        # Start from the month after latest
        start_year = latest.year
        start_month = latest.month + 1
        if start_month > 12:
            start_month = 1
            start_year += 1

    end_year = today.year
    end_month = today.month

    # Check if we're already up to date
    if (start_year, start_month) > (end_year, end_month) and force_months == 0:
        logger.info(
            "Data already up to date (latest: %s, today: %s).",
            latest, today,
        )
        return {"status": "up_to_date", "new_rows": 0, "latest": str(latest)}

    logger.info(
        "Updating BCRP data: %s to %s-%02d (latest existing: %s)",
        f"{start_year}-{start_month:02d}", end_year, end_month, latest,
    )

    if dry_run:
        codes = load_series_codes(SERIES_CATALOG_PATH, section="national")
        logger.info(
            "DRY RUN: Would download %d series for %s to %s-%02d",
            len(codes), f"{start_year}-{start_month:02d}", end_year, end_month,
        )
        return {
            "status": "dry_run",
            "series_count": len(codes),
            "start": f"{start_year}-{start_month:02d}",
            "end": f"{end_year}-{end_month:02d}",
        }

    # Load existing data FIRST (before any downloads that might overwrite)
    existing_df = pd.read_parquet(combined_path)
    existing_df["date"] = pd.to_datetime(existing_df["date"])

    # Download new data using fetch_series (does NOT overwrite files)
    codes = load_series_codes(SERIES_CATALOG_PATH, section="national")
    client = BCRPClient(request_delay=1.5)
    new_df = client.fetch_series(
        codes,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
    )

    if new_df.empty:
        logger.info("No new data returned from BCRP API.")
        return {"status": "no_new_data", "new_rows": 0}

    # Enrich new data with metadata from existing
    code_to_cat = existing_df.drop_duplicates("series_code").set_index(
        "series_code"
    )["category"].to_dict()
    new_df["category"] = new_df["series_code"].map(code_to_cat)
    new_df["source"] = "BCRP"
    new_df["frequency_original"] = "M"
    new_df["date"] = pd.to_datetime(new_df["date"])

    # Remove duplicates: keep new data for overlapping dates
    existing_keys = set(
        zip(existing_df["date"].dt.date, existing_df["series_code"])
    )
    new_keys = set(
        zip(new_df["date"].dt.date, new_df["series_code"])
    )
    overlap = existing_keys & new_keys

    if overlap:
        logger.info("Updating %d existing observations with fresh data.", len(overlap))
        # Remove old rows that will be replaced
        mask = existing_df.apply(
            lambda r: (r["date"].date(), r["series_code"]) not in overlap,
            axis=1,
        )
        existing_df = existing_df[mask]

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.sort_values(["series_code", "date"]).reset_index(drop=True)

    # Save updated combined file
    save_parquet(combined, combined_path)

    # Also update per-category files
    for category in combined["category"].dropna().unique():
        cat_df = combined[combined["category"] == category]
        cat_path = RAW_BCRP_DIR / f"bcrp_national_{category}.parquet"
        save_parquet(cat_df, cat_path)

    new_latest = combined["date"].max().date()
    truly_new = len(new_keys - existing_keys)

    logger.info(
        "Update complete: %d new observations, %d updated. "
        "Total: %d rows. Latest: %s",
        truly_new, len(overlap), len(combined), new_latest,
    )

    return {
        "status": "updated",
        "new_rows": truly_new,
        "updated_rows": len(overlap),
        "total_rows": len(combined),
        "latest": str(new_latest),
        "previous_latest": str(latest),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Incremental BCRP data updater"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be updated without downloading",
    )
    parser.add_argument(
        "--force", type=int, default=0, metavar="MONTHS",
        help="Re-download last N months (overwrite existing)",
    )
    args = parser.parse_args()

    try:
        result = update_bcrp(dry_run=args.dry_run, force_months=args.force)
        logger.info("Result: %s", result)
        return 0
    except Exception:
        logger.exception("Update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
