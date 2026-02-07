"""Master download script for NEXUS data.

Usage:
    python scripts/download_all.py [--verify] [--start-year 2004] [--end-year 2025]

Steps:
    1. Verify all BCRP series codes (optional, with --verify)
    2. Download all national monthly series from BCRP
    3. Download departmental series (when available)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SERIES_CATALOG_PATH,
    RAW_BCRP_DIR,
    DEFAULT_START_YEAR,
    DEFAULT_START_MONTH,
    DEFAULT_END_YEAR,
    DEFAULT_END_MONTH,
)
from src.ingestion.bcrp import BCRPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download NEXUS data sources")
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify all BCRP series codes before downloading",
    )
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help=f"Start year (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
        help=f"End year (default: {DEFAULT_END_YEAR})",
    )
    parser.add_argument(
        "--start-month", type=int, default=DEFAULT_START_MONTH,
        help=f"Start month (default: {DEFAULT_START_MONTH})",
    )
    parser.add_argument(
        "--end-month", type=int, default=DEFAULT_END_MONTH,
        help=f"End month (default: {DEFAULT_END_MONTH})",
    )
    args = parser.parse_args()

    client = BCRPClient()

    # ── Step 1: Verify series codes ───────────────────────────────────────────
    if args.verify:
        logger.info("=== Verifying BCRP series codes ===")
        results = client.verify_all_series(SERIES_CATALOG_PATH)
        valid = results["valid"].sum()
        total = len(results)
        logger.info(
            "Verification complete: %d/%d series valid", valid, total
        )
        if valid < total:
            logger.warning("Invalid series:")
            for _, row in results[~results["valid"]].iterrows():
                logger.warning(
                    "  %s (%s): %s", row["code"], row["expected_name"], row["error"]
                )

    # ── Step 2: Download national series ──────────────────────────────────────
    logger.info("=== Downloading BCRP national series ===")
    df = client.download_national_series(
        catalog_path=SERIES_CATALOG_PATH,
        output_dir=RAW_BCRP_DIR,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
    )
    logger.info(
        "Download complete: %d rows, %d unique series",
        len(df),
        df["series_code"].nunique() if not df.empty else 0,
    )

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    if not df.empty:
        logger.info("=== Summary ===")
        logger.info("Date range: %s to %s", df["date"].min(), df["date"].max())
        logger.info("Series downloaded:")
        for code in sorted(df["series_code"].dropna().unique()):
            series_df = df[df["series_code"] == code]
            non_null = series_df["value"].notna().sum()
            name = series_df["series_name"].iloc[0]
            logger.info("  %s: %s (%d observations)", code, name, non_null)


if __name__ == "__main__":
    main()
