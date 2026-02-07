"""Download all regional/departmental BCRP series from the catalog.

Reads config/regional_series_catalog.yaml and downloads all series
using BCRPClient. Saves data in batches by category to parquet files.

Categories:
  - Financial: credit, deposits, pension affiliates (by department)
  - Production: electricity, mining copper, mining gold (by department)
  - Public sector: tax revenue, local gov spending, regional gov spending
  - Trade: exports, imports (by department/customs)

Usage:
    python scripts/download_regional_bcrp.py                # download all
    python scripts/download_regional_bcrp.py --only credit   # only credit
    python scripts/download_regional_bcrp.py --only electricity,tax
    python scripts/download_regional_bcrp.py --dry-run        # count only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.ingestion.bcrp import BCRPClient
from src.utils.io import save_parquet, ensure_dir
from config.settings import RAW_BCRP_DIR

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "download_regional.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.regional_download")

# Category → output file mapping
CATEGORY_FILES = {
    "credit_by_department": "bcrp_regional_credit.parquet",
    "deposits_by_department": "bcrp_regional_deposits.parquet",
    "pension_affiliates_by_department": "bcrp_regional_pension.parquet",
    "electricity_by_department": "bcrp_regional_electricity.parquet",
    "mining_copper_by_department": "bcrp_regional_mining_copper.parquet",
    "mining_gold_by_department": "bcrp_regional_mining_gold.parquet",
    "tax_revenue_by_department": "bcrp_regional_tax.parquet",
    "government_spending_local": "bcrp_regional_spending_local.parquet",
    "government_spending_regional": "bcrp_regional_spending_regional.parquet",
    "exports_by_department": "bcrp_regional_exports.parquet",
    "imports_by_customs": "bcrp_regional_imports.parquet",
}

# Short aliases for --only flag
ALIASES = {
    "credit": "credit_by_department",
    "deposits": "deposits_by_department",
    "pension": "pension_affiliates_by_department",
    "electricity": "electricity_by_department",
    "copper": "mining_copper_by_department",
    "gold": "mining_gold_by_department",
    "tax": "tax_revenue_by_department",
    "spending_local": "government_spending_local",
    "spending_regional": "government_spending_regional",
    "exports": "exports_by_department",
    "imports": "imports_by_customs",
}


def load_catalog() -> dict:
    """Load the regional series catalog."""
    catalog_path = PROJECT_ROOT / "config" / "regional_series_catalog.yaml"
    with open(catalog_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_category(
    category_key: str,
    category_data: dict,
    output_path: Path,
    client: BCRPClient,
) -> pd.DataFrame:
    """Download all series in a catalog category."""
    series_list = category_data.get("series", [])
    if not series_list:
        logger.warning("No series in category %s", category_key)
        return pd.DataFrame()

    codes = [s["code"] for s in series_list]
    code_meta = {s["code"]: s for s in series_list}

    logger.info(
        "Downloading %d series for [%s]...",
        len(codes), category_key,
    )

    # Exports only go to 2022
    end_year = 2026
    end_month = 2
    if "exports" in category_key:
        end_year = 2022
        end_month = 12

    df = client.fetch_series(
        codes,
        start_year=2001,
        start_month=1,
        end_year=end_year,
        end_month=end_month,
    )

    if df.empty:
        logger.warning("No data returned for %s", category_key)
        return df

    # Enrich with metadata from catalog
    def get_meta(code, field):
        return code_meta.get(code, {}).get(field, "")

    df["category"] = category_key
    df["department"] = df["series_code"].map(lambda c: get_meta(c, "department"))
    df["ubigeo"] = df["series_code"].map(lambda c: get_meta(c, "ubigeo"))
    df["source"] = "BCRP"
    df["frequency_original"] = "M"

    # Save
    ensure_dir(output_path.parent)
    save_parquet(df, output_path)
    logger.info(
        "  Saved %s → %s (%d rows, %d series, %s to %s)",
        category_key, output_path.name,
        len(df), df["series_code"].nunique(),
        df["date"].min().strftime("%Y-%m") if "date" in df.columns else "?",
        df["date"].max().strftime("%Y-%m") if "date" in df.columns else "?",
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download regional/departmental BCRP series"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated category keys or aliases to download",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count series, don't download",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    regional = catalog.get("regional", {})

    # Determine which categories to download
    if args.only:
        requested = [s.strip() for s in args.only.split(",")]
        categories = []
        for r in requested:
            resolved = ALIASES.get(r, r)
            if resolved in regional:
                categories.append(resolved)
            else:
                logger.error("Unknown category: %s", r)
                logger.info("Available: %s", ", ".join(
                    list(ALIASES.keys()) + list(regional.keys())
                ))
                return 1
    else:
        categories = [k for k in regional.keys() if isinstance(regional[k], dict)]

    # Count total series
    total_series = 0
    for cat_key in categories:
        cat_data = regional[cat_key]
        n = len(cat_data.get("series", []))
        total_series += n
        logger.info("  %s: %d series", cat_key, n)

    logger.info("=" * 60)
    logger.info("TOTAL: %d series across %d categories", total_series, len(categories))
    logger.info("Estimated time: %.0f minutes (at 1.5s/series)",
                total_series * 1.5 / 60)
    logger.info("=" * 60)

    if args.dry_run:
        return 0

    # Download
    client = BCRPClient(request_delay=1.5)
    t0 = time.time()
    results = {}

    for i, cat_key in enumerate(categories):
        cat_data = regional[cat_key]
        output_file = CATEGORY_FILES.get(cat_key, f"bcrp_regional_{cat_key}.parquet")
        output_path = RAW_BCRP_DIR / output_file

        logger.info("")
        logger.info("[%d/%d] Category: %s", i + 1, len(categories), cat_key)

        df = download_category(cat_key, cat_data, output_path, client)
        results[cat_key] = {
            "rows": len(df),
            "series": df["series_code"].nunique() if not df.empty else 0,
            "file": output_file,
        }

    # Summary
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("REGIONAL DOWNLOAD COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)
    total_rows = 0
    total_downloaded = 0
    for cat_key, info in results.items():
        logger.info("  %-35s %5d rows, %3d series → %s",
                     cat_key, info["rows"], info["series"], info["file"])
        total_rows += info["rows"]
        total_downloaded += info["series"]
    logger.info("  %-35s %5d rows, %3d series", "TOTAL", total_rows, total_downloaded)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
