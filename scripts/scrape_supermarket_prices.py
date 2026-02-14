"""CLI for supermarket price scraping and index construction.

Usage:
    # Scrape a single store
    python scripts/scrape_supermarket_prices.py --store plazavea

    # Scrape all stores
    python scripts/scrape_supermarket_prices.py --store all

    # Build price index from existing snapshots
    python scripts/scrape_supermarket_prices.py --build-index

    # Scrape and build index in one go
    python scripts/scrape_supermarket_prices.py --store all --build-index

    # List available snapshots
    python scripts/scrape_supermarket_prices.py --list-snapshots
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 console support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("nexus.scrape_supermarket")


def main():
    parser = argparse.ArgumentParser(
        description="Supermarket price scraper (BPP for Peru)"
    )
    parser.add_argument(
        "--store",
        choices=["plazavea", "metro", "wong", "all"],
        help="Store to scrape (or 'all' for all stores)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for scraping (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build price index from existing snapshots",
    )
    parser.add_argument(
        "--list-snapshots",
        action="store_true",
        help="List available snapshot dates",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for index construction (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for index construction (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    from src.ingestion.supermarket import (
        STORES,
        PriceIndexBuilder,
        SupermarketScraper,
    )

    scraper = SupermarketScraper()

    # List snapshots
    if args.list_snapshots:
        snapshots = scraper.list_available_snapshots()
        if not snapshots:
            print("No snapshots available.")
        else:
            print(f"Available snapshots ({len(snapshots)}):")
            for d in snapshots:
                df = scraper.load_snapshot(d)
                stores = df["store"].unique().tolist() if not df.empty else []
                products = len(df)
                print(f"  {d.isoformat()}  stores={stores}  products={products}")
        return 0

    # Scrape
    if args.store:
        target_date = date.fromisoformat(args.date) if args.date else date.today()

        if args.store == "all":
            df = scraper.scrape_all_stores(target_date)
        else:
            df = scraper.scrape_store(args.store, target_date)

        if df.empty:
            logger.warning("No products scraped")
            return 1

        path = scraper.save_daily_snapshot(df, target_date)
        logger.info("Saved %d products to %s", len(df), path)

        # Summary by store
        for store in df["store"].unique():
            store_df = df[df["store"] == store]
            logger.info(
                "  %s: %d products, %d categories, price range S/%.2f - S/%.2f",
                store,
                len(store_df),
                store_df["category_l1"].nunique(),
                store_df["price"].min(),
                store_df["price"].max(),
            )

    # Build index
    if args.build_index:
        builder = PriceIndexBuilder()

        start = date.fromisoformat(args.start_date) if args.start_date else None
        end = date.fromisoformat(args.end_date) if args.end_date else None

        logger.info("Building daily price index...")
        daily_index = builder.build_daily_index(start_date=start, end_date=end)

        if daily_index.empty:
            logger.error("Could not build daily index (need >= 2 snapshots)")
            return 1

        logger.info("Daily index: %d days", len(daily_index))

        # Show latest values
        latest = daily_index.iloc[-1]
        index_cols = [c for c in daily_index.columns if c.startswith("index_")]
        logger.info("Latest index values:")
        for col in sorted(index_cols):
            logger.info("  %s = %.2f", col, latest[col])

        # Build monthly
        logger.info("Building monthly index...")
        monthly = builder.build_monthly_index(daily_index)

        if not monthly.empty:
            path = builder.save_monthly_index(monthly)
            logger.info("Monthly index: %d months, saved to %s", len(monthly), path)

            # Show latest month
            latest_m = monthly.iloc[-1]
            var_cols = [c for c in monthly.columns if c.startswith("var_")]
            logger.info("Latest monthly variations:")
            for col in sorted(var_cols):
                val = latest_m.get(col, float("nan"))
                if not pd.isna(val):
                    logger.info("  %s = %.2f%%", col, val)

    if not args.store and not args.build_index and not args.list_snapshots:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
