"""Target variable compilation for NEXUS backtesting.

Produces:
    - data/targets/gdp_quarterly.parquet  — quarterly GDP from BCRP API
    - data/targets/inflation_monthly.parquet — monthly inflation from existing BCRP data
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from config.settings import (
    RAW_BCRP_DIR, SERIES_CATALOG_PATH, TARGETS_DIR,
    DEFAULT_START_YEAR, DEFAULT_START_MONTH,
)
from src.ingestion.bcrp import BCRPClient
from src.utils.dates import format_bcrp_date
from src.utils.io import save_parquet

logger = logging.getLogger(__name__)

# ── Quarterly GDP series codes → column names ────────────────────────────────

GDP_CODE_MAP = {
    "PN02507AQ": "gdp_yoy",
    "PN02499AQ": "agropecuario_yoy",
    "PN02500AQ": "pesca_yoy",
    "PN02501AQ": "mineria_yoy",
    "PN02502AQ": "manufactura_yoy",
    "PN02503AQ": "electricidad_yoy",
    "PN02504AQ": "construccion_yoy",
    "PN02505AQ": "comercio_yoy",
    "PN02506AQ": "servicios_yoy",
    "PN02516AQ": "gdp_index",
}


def load_quarterly_codes(catalog_path: Path = SERIES_CATALOG_PATH) -> list[str]:
    """Load quarterly GDP series codes from the catalog."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    codes = []
    quarterly = catalog.get("quarterly", {})
    for category in quarterly.values():
        if isinstance(category, dict):
            for entry in category.get("series", []):
                codes.append(entry["code"])
    return codes


def pivot_gdp_to_target(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format BCRP data into the GDP target schema.

    Input: DataFrame with columns [date, series_code, value, ...]
    Output: DataFrame with columns [date, gdp_yoy, gdp_index, ..sector_yoy..]
    """
    if df.empty:
        return pd.DataFrame()

    # Map series_code to column name
    df = df.copy()
    df["column"] = df["series_code"].map(GDP_CODE_MAP)
    df = df.dropna(subset=["column"])

    # Pivot: one row per date, columns are the target variables
    pivoted = df.pivot_table(
        index="date", columns="column", values="value", aggfunc="first"
    ).reset_index()

    # Ensure all expected columns exist
    for col in GDP_CODE_MAP.values():
        if col not in pivoted.columns:
            pivoted[col] = None

    # Order columns
    col_order = ["date"] + list(GDP_CODE_MAP.values())
    pivoted = pivoted[[c for c in col_order if c in pivoted.columns]]
    pivoted = pivoted.sort_values("date").reset_index(drop=True)

    return pivoted


def update_quarterly_gdp(
    force: bool = False,
    output_dir: Path = TARGETS_DIR,
) -> dict:
    """Download quarterly GDP from BCRP API and save as target parquet.

    Returns dict with status info.
    """
    output_path = output_dir / "gdp_quarterly.parquet"
    raw_path = RAW_BCRP_DIR / "bcrp_quarterly_gdp.parquet"
    today = date.today()

    codes = list(GDP_CODE_MAP.keys())
    logger.info("Downloading %d quarterly GDP series from BCRP...", len(codes))

    client = BCRPClient(request_delay=1.5)
    df = client.fetch_series(
        codes,
        start_year=DEFAULT_START_YEAR,
        start_month=DEFAULT_START_MONTH,
        end_year=today.year,
        end_month=today.month,
    )

    if df.empty:
        logger.warning("No quarterly GDP data returned from BCRP.")
        return {"status": "no_data", "rows": 0}

    # Save raw long-format
    save_parquet(df, raw_path)
    logger.info("Saved raw quarterly GDP data to %s (%d rows)", raw_path, len(df))

    # Pivot to target schema
    target_df = pivot_gdp_to_target(df)
    save_parquet(target_df, output_path)
    logger.info("Saved GDP target to %s (%d quarters)", output_path, len(target_df))

    return {
        "status": "updated",
        "rows": len(target_df),
        "latest_date": str(target_df["date"].max()),
        "output_path": str(output_path),
    }


def compile_inflation_targets(
    output_dir: Path = TARGETS_DIR,
) -> dict:
    """Reshape existing BCRP inflation parquet into target schema.

    No API call needed — reads from data/raw/bcrp/bcrp_national_inflation.parquet.

    Output schema: [date, ipc_monthly_var, ipc_12m_var, ipc_core_index]
    """
    source_path = RAW_BCRP_DIR / "bcrp_national_inflation.parquet"
    output_path = output_dir / "inflation_monthly.parquet"

    if not source_path.exists():
        logger.warning("Inflation source not found at %s", source_path)
        return {"status": "no_source", "rows": 0}

    df = pd.read_parquet(source_path)

    if df.empty:
        logger.warning("Inflation source is empty.")
        return {"status": "empty_source", "rows": 0}

    # Map series codes to target column names
    code_to_col = {
        "PN01271PM": "ipc_monthly_var",
        "PN01273PM": "ipc_12m_var",
        "PN38706PM": "ipc_core_index",
    }

    df = df.copy()
    df["column"] = df["series_code"].map(code_to_col)
    df = df.dropna(subset=["column"])

    pivoted = df.pivot_table(
        index="date", columns="column", values="value", aggfunc="first"
    ).reset_index()

    # Ensure all columns exist
    for col in code_to_col.values():
        if col not in pivoted.columns:
            pivoted[col] = None

    col_order = ["date", "ipc_monthly_var", "ipc_12m_var", "ipc_core_index"]
    pivoted = pivoted[[c for c in col_order if c in pivoted.columns]]
    pivoted = pivoted.sort_values("date").reset_index(drop=True)

    save_parquet(pivoted, output_path)
    logger.info(
        "Saved inflation target to %s (%d months)", output_path, len(pivoted)
    )

    return {
        "status": "updated",
        "rows": len(pivoted),
        "latest_date": str(pivoted["date"].max()),
        "output_path": str(output_path),
    }
