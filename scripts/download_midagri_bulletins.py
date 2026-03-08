"""Download and extract MIDAGRI wholesale food price bulletins.

Scrapes the MIDAGRI bulletin collection for Lima wholesale market
(Gran Mercado Mayorista de Lima) price bulletins, extracts structured
price data via pdfplumber, and aggregates into monthly series.

Usage:
    # First time: download + extract
    python scripts/download_midagri_bulletins.py --extract --max-pages 10

    # Daily/weekly incremental update
    python scripts/download_midagri_bulletins.py --extract

    # Just re-aggregate from existing extracted data
    python scripts/download_midagri_bulletins.py --aggregate-only

    # Dry run
    python scripts/download_midagri_bulletins.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.midagri")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract MIDAGRI wholesale bulletins"
    )
    parser.add_argument(
        "--max-pages", type=int, default=10,
        help="Max collection pages to fetch (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List new bulletins without downloading",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Also parse PDFs after downloading",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip download, just re-aggregate from extracted data",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only process from this month onwards (YYYY-MM)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download/re-parse everything",
    )
    parser.add_argument(
        "--excel", type=str, default=None,
        help="Export clean data to Excel file (path)",
    )
    parser.add_argument(
        "--drop-pdfs", action="store_true",
        help="Delete all PDF files after extraction (keeps parquet + Excel)",
    )
    args = parser.parse_args()

    from src.ingestion.midagri import (
        MidagriBulletinScraper,
        MidagriPDFParser,
        MidagriAggregator,
    )

    logger.info("=" * 60)
    logger.info("MIDAGRI WHOLESALE BULLETIN PIPELINE")
    logger.info("=" * 60)

    # Phase 1: Download
    if not args.aggregate_only:
        scraper = MidagriBulletinScraper()

        if args.dry_run:
            logger.info("DRY RUN: counting new bulletins...")
            count = scraper.count_new_bulletins(max_pages=args.max_pages)
            logger.info("New bulletins available: %d", count)
            return 0

        new_pdfs = scraper.scrape_new_bulletins(
            max_pages=args.max_pages,
            since_date=args.since,
            force=args.force,
        )
        n_ok = sum(1 for e in new_pdfs if e["status"] == "downloaded")
        logger.info("Download complete: %d new PDFs", n_ok)

        if not args.extract:
            return 0

    # Phase 2: Extract
    logger.info("=" * 60)
    logger.info("PHASE 2: PDF EXTRACTION")
    logger.info("=" * 60)

    parser_obj = MidagriPDFParser()
    daily_df = parser_obj.parse_all_bulletins(force=args.force)

    if daily_df.empty:
        logger.warning("No daily prices extracted")
        return 1

    logger.info(
        "Extracted: %d daily records, %d unique dates, %d products",
        len(daily_df),
        daily_df["date"].nunique(),
        daily_df["product"].nunique(),
    )

    # Phase 3: Aggregate
    logger.info("=" * 60)
    logger.info("PHASE 3: MONTHLY AGGREGATION")
    logger.info("=" * 60)

    agg = MidagriAggregator()
    monthly_df = agg.build_monthly_series(daily_df)

    if monthly_df.empty:
        logger.warning("No monthly series generated")
        return 1

    logger.info(
        "Monthly series: %d rows, %d series, %s to %s",
        len(monthly_df),
        monthly_df["series_id"].nunique(),
        monthly_df["date"].min().strftime("%Y-%m"),
        monthly_df["date"].max().strftime("%Y-%m"),
    )

    # Phase 4: Excel export (optional)
    if args.excel:
        from pathlib import Path as P
        excel_path = P(args.excel)
        logger.info("=" * 60)
        logger.info("PHASE 4: EXCEL EXPORT")
        logger.info("=" * 60)
        agg.export_to_excel(daily_df, monthly_df, excel_path)

    # Phase 5: Drop PDFs (optional)
    if args.drop_pdfs:
        import shutil
        from config.settings import MIDAGRI_BULLETINS_DIR
        if MIDAGRI_BULLETINS_DIR.exists():
            pdf_count = len(list(MIDAGRI_BULLETINS_DIR.rglob("*.pdf")))
            shutil.rmtree(MIDAGRI_BULLETINS_DIR)
            logger.info("Deleted %d PDF files from %s", pdf_count, MIDAGRI_BULLETINS_DIR)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
