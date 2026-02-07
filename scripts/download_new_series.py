"""Download only the NEW BCRP series and merge with existing data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from pathlib import Path
from src.ingestion.bcrp import BCRPClient
from src.utils.io import save_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# New series codes grouped by category
NEW_SERIES = {
    "commodity_prices": [
        "PN01652XM", "PN01654XM", "PN01660XM", "PN01655XM", "PN01657XM",
    ],
    "food_prices": [
        "PN39445PM", "PN01383PM",
    ],
    "trade": [
        "PN38923BM",
    ],
    "monetary": [
        "PN01013MM", "PN00178MM",
    ],
    "employment": [
        "PN38063GM", "PN31879GM",
    ],
    "confidence": [
        "PD37981AM", "PD12912AM",
    ],
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "bcrp"
COMBINED_PATH = OUTPUT_DIR / "bcrp_national_all.parquet"


def main():
    client = BCRPClient(request_delay=1.5)

    all_new_frames = []

    for category, codes in NEW_SERIES.items():
        logger.info("Downloading %s (%d series)...", category, len(codes))

        df = client.fetch_series(codes, 2004, 1, 2026, 2)

        if df.empty:
            logger.warning("No data for category %s", category)
            continue

        df["category"] = category
        df["source"] = "BCRP"
        df["frequency_original"] = "M"

        # Save per-category file
        cat_path = OUTPUT_DIR / f"bcrp_national_{category}.parquet"
        save_parquet(df, cat_path)
        logger.info("  Saved %s: %d rows", cat_path.name, len(df))

        all_new_frames.append(df)

    if not all_new_frames:
        logger.error("No new data downloaded!")
        return

    new_df = pd.concat(all_new_frames, ignore_index=True)
    logger.info("Total new data: %d rows across %d series",
                len(new_df), new_df["series_code"].nunique())

    # Merge with existing data
    if COMBINED_PATH.exists():
        existing_df = pd.read_parquet(COMBINED_PATH)
        logger.info("Existing data: %d rows across %d series",
                    len(existing_df), existing_df["series_code"].nunique())

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Deduplicate
        combined = combined.drop_duplicates(
            subset=["date", "series_code"], keep="last"
        )
        combined = combined.sort_values(["series_code", "date"]).reset_index(drop=True)
    else:
        combined = new_df

    save_parquet(combined, COMBINED_PATH)
    logger.info("Saved combined data: %d rows across %d series",
                len(combined), combined["series_code"].nunique())

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  New series downloaded: {new_df['series_code'].nunique()}")
    print(f"  New observations: {len(new_df)}")
    print(f"  Total combined rows: {len(combined)}")
    print(f"  Total series: {combined['series_code'].nunique()}")
    print(f"\nSeries by category:")
    for cat in sorted(combined["category"].unique()):
        cat_df = combined[combined["category"] == cat]
        codes = cat_df["series_code"].nunique()
        rows = len(cat_df)
        print(f"  {cat}: {codes} series, {rows} rows")


if __name__ == "__main__":
    main()
