"""Download MIDAGRI poultry price bulletins + extract daily prices.

Downloads daily PDF bulletins from the gob.pe poultry collection,
extracts chicken and egg wholesale prices, aggregates to monthly series,
exports clean Excel files, and optionally removes PDFs after extraction.

Data source: MIDAGRI - Boletín diario de comercialización y precio de aves
Coverage: Feb 2024 to present, ~20 daily PDFs per month (business days)

Usage:
    python scripts/download_poultry_bulletins.py              # full pipeline
    python scripts/download_poultry_bulletins.py --since 2025-01  # only recent months
    python scripts/download_poultry_bulletins.py --parse-only     # skip download
    python scripts/download_poultry_bulletins.py --keep-pdfs      # don't delete PDFs
    python scripts/download_poultry_bulletins.py --force           # re-download everything
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MIDAGRI_POULTRY_BULLETINS_DIR, PROCESSED_NATIONAL_DIR
from src.ingestion.midagri import (
    PoultryAggregator,
    PoultryBulletinScraper,
    PoultryPDFParser,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nexus.download_poultry")


def main():
    parser = argparse.ArgumentParser(description="Download MIDAGRI poultry bulletins")
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only download months >= this YYYY-MM (e.g. 2025-01)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=5,
        help="Max collection pages to scan (25 months each, default 5)",
    )
    parser.add_argument(
        "--parse-only", action="store_true",
        help="Skip downloading, only parse existing PDFs",
    )
    parser.add_argument(
        "--keep-pdfs", action="store_true",
        help="Keep PDF files after extraction (default: delete them)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-parse everything",
    )
    args = parser.parse_args()

    # Step 1: Download PDFs
    if not args.parse_only:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading poultry bulletins from gob.pe")
        scraper = PoultryBulletinScraper()
        new_entries = scraper.scrape_new_bulletins(
            max_pages=args.max_pages,
            since_date=args.since,
            force=args.force,
        )
        n_ok = sum(1 for e in new_entries if e["status"] == "downloaded")
        logger.info("Downloaded %d new PDFs", n_ok)
    else:
        logger.info("Skipping download (--parse-only)")

    # Step 2: Parse PDFs
    logger.info("=" * 60)
    logger.info("STEP 2: Parsing poultry bulletin PDFs")
    pdf_parser = PoultryPDFParser()
    daily_df = pdf_parser.parse_all_bulletins(force=args.force)

    if daily_df.empty:
        logger.warning("No daily poultry prices extracted!")
        return 1

    logger.info(
        "Daily prices: %d records, date range %s to %s",
        len(daily_df),
        daily_df["date"].min().strftime("%Y-%m-%d"),
        daily_df["date"].max().strftime("%Y-%m-%d"),
    )

    # Summary of extracted fields
    for col in ["chicken_wholesale", "chicken_farm", "chicken_retail", "egg"]:
        if col in daily_df.columns:
            n = daily_df[col].notna().sum()
            if n > 0:
                avg = daily_df[col].mean()
                logger.info("  %s: %d obs, avg=S/%.2f/kg", col, n, avg)

    # Step 3: Monthly aggregation
    logger.info("=" * 60)
    logger.info("STEP 3: Aggregating to monthly series")
    agg = PoultryAggregator()
    monthly_df = agg.build_monthly_series(daily_df)

    if not monthly_df.empty:
        logger.info("Monthly series: %d rows, %d series",
                     len(monthly_df), monthly_df["series_id"].nunique())
        for sid in monthly_df["series_id"].unique():
            sub = monthly_df[monthly_df["series_id"] == sid]
            logger.info("  %s: %d months", sid, len(sub))

    # Step 4: Export clean Excel
    logger.info("=" * 60)
    logger.info("STEP 4: Exporting clean Excel file")
    excel_path = PROCESSED_NATIONAL_DIR / "midagri_poultry_prices.xlsx"
    PoultryAggregator.export_to_excel(daily_df, monthly_df, excel_path)

    # Step 5: Delete PDFs (unless --keep-pdfs)
    if not args.keep_pdfs:
        logger.info("=" * 60)
        logger.info("STEP 5: Cleaning up PDF files")
        pdf_dir = MIDAGRI_POULTRY_BULLETINS_DIR
        if pdf_dir.exists():
            pdf_count = len(list(pdf_dir.rglob("*.pdf")))
            shutil.rmtree(pdf_dir)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            logger.info("  Deleted %d PDFs from %s", pdf_count, pdf_dir)
        else:
            logger.info("  No PDF directory to clean")
    else:
        logger.info("STEP 5: Keeping PDFs (--keep-pdfs)")

    # Summary
    logger.info("=" * 60)
    logger.info("POULTRY PIPELINE COMPLETE")
    logger.info("  Daily records:  %d", len(daily_df))
    logger.info("  Monthly series: %d rows", len(monthly_df) if not monthly_df.empty else 0)
    logger.info("  Excel output:   %s", excel_path)
    logger.info("  Parquet output: %s", agg.output_path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
