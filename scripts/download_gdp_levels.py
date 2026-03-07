"""Download BCRP quarterly GDP level series (constant 2007 soles) and compute
sector contributions to YoY growth.

Accounting identity:
    contribution_i,t = (GDP_i,t - GDP_i,t-4) / GDP_total,t-4 * 100
    Sum of sector contributions == total YoY growth (up to rounding)

Usage:
    python scripts/download_gdp_levels.py
    python scripts/download_gdp_levels.py --verify-only   # skip download, just check file
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

from src.ingestion.bcrp import BCRPClient
from src.utils.io import save_parquet, ensure_dir
from config.settings import RAW_BCRP_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nexus.gdp_levels")

# ── Series definitions ────────────────────────────────────────────────────────

SECTOR_CODES = {
    "PN37684AQ": "Agropecuario",
    "PN37685AQ": "Pesca",
    "PN37686AQ": "Minería e Hidrocarburos",
    "PN37687AQ": "Manufactura",
    "PN37688AQ": "Electricidad y Agua",
    "PN37689AQ": "Construcción",
    "PN37690AQ": "Comercio",
    "PN37691AQ": "Servicios",
    "PN37692AQ": "PBI Global",          # total
    "PN37693AQ": "Sectores Primarios",  # memo
    "PN37694AQ": "Sectores No Primarios",  # memo
}

# Sectors that contribute to GDP (excludes memo items and total)
CONTRIB_SECTORS = [
    "PN37684AQ",  # Agropecuario
    "PN37685AQ",  # Pesca
    "PN37686AQ",  # Minería e Hidrocarburos
    "PN37687AQ",  # Manufactura
    "PN37688AQ",  # Electricidad y Agua
    "PN37689AQ",  # Construcción
    "PN37690AQ",  # Comercio
    "PN37691AQ",  # Servicios
]
TOTAL_CODE = "PN37692AQ"

OUT_LEVELS   = RAW_BCRP_DIR / "bcrp_quarterly_gdp_levels.parquet"
OUT_CONTRIB  = RAW_BCRP_DIR / "bcrp_quarterly_gdp_contributions.parquet"


# ── Download ──────────────────────────────────────────────────────────────────

def download_levels() -> pd.DataFrame:
    """Fetch all GDP level series from BCRP (1980 Q1 → present)."""
    client = BCRPClient()
    codes = list(SECTOR_CODES.keys())

    logger.info("Fetching %d GDP level series from BCRP (1980–2025)...", len(codes))
    # BCRPClient.fetch_series uses format_bcrp_date(year, month) → "1980-1"
    # BCRP accepts month-based dates for quarterly series (month 1/4/7/10 or any month)
    df = client.fetch_series(
        codes=codes,
        start_year=1980, start_month=1,
        end_year=2026,   end_month=3,
    )

    if df.empty:
        raise RuntimeError("BCRP returned no data for GDP level series.")

    # Attach human-readable sector name
    df["sector"] = df["series_code"].map(SECTOR_CODES)

    logger.info("Downloaded %d rows, %d series.", len(df), df["series_code"].nunique())
    for code, name in SECTOR_CODES.items():
        n = (df["series_code"] == code).sum()
        logger.info("  %s  %s — %d periods", code, name, n)

    save_parquet(df, OUT_LEVELS)
    logger.info("Saved → %s", OUT_LEVELS)
    return df


# ── Contribution computation ──────────────────────────────────────────────────

def compute_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sectoral contributions to YoY GDP growth.

    contribution_i,t = (GDP_i,t - GDP_i,t-4) / GDP_total,t-4 * 100

    The accounting identity guarantees:
        sum(contribution_i) == (GDP_total,t - GDP_total,t-4) / GDP_total,t-4 * 100
    """
    # Pivot: rows = date, cols = series_code
    wide = df.pivot_table(index="date", columns="series_code", values="value")
    wide = wide.sort_index()

    if TOTAL_CODE not in wide.columns:
        raise ValueError(f"Total GDP series {TOTAL_CODE} not found in data.")

    # YoY denominator: total GDP one year ago
    gdp_total_lag4 = wide[TOTAL_CODE].shift(4)

    contrib_rows = []
    for code in CONTRIB_SECTORS:
        if code not in wide.columns:
            logger.warning("Sector %s not in data — skipping.", code)
            continue
        name = SECTOR_CODES[code]
        gdp_i      = wide[code]
        gdp_i_lag4 = wide[code].shift(4)
        contribution = (gdp_i - gdp_i_lag4) / gdp_total_lag4 * 100
        s = contribution.rename(name)
        contrib_rows.append(s)

    contrib_df = pd.concat(contrib_rows, axis=1)

    # Total YoY growth (for identity check)
    contrib_df["_total_yoy"] = (wide[TOTAL_CODE] - gdp_total_lag4) / gdp_total_lag4 * 100
    contrib_df["_contrib_sum"] = contrib_df[list(SECTOR_CODES[c] for c in CONTRIB_SECTORS if c in wide.columns)].sum(axis=1)
    contrib_df["_residual"] = contrib_df["_total_yoy"] - contrib_df["_contrib_sum"]

    contrib_df = contrib_df.reset_index()
    return contrib_df


def verify_identity(contrib_df: pd.DataFrame) -> None:
    """Check that sector contributions sum to total YoY growth."""
    recent = contrib_df.dropna(subset=["_total_yoy", "_contrib_sum"]).tail(20)
    if recent.empty:
        logger.warning("Not enough data to verify identity.")
        return

    max_residual = recent["_residual"].abs().max()
    mean_residual = recent["_residual"].abs().mean()

    logger.info("=== Identity check (last 20 quarters) ===")
    logger.info("  Max |residual| = %.4f pp", max_residual)
    logger.info("  Mean |residual| = %.4f pp", mean_residual)

    if max_residual < 0.1:
        logger.info("  PASS — contributions sum to total YoY growth within 0.1 pp")
    else:
        logger.warning("  WARN — max residual %.3f pp (may be due to memo items or rounding)", max_residual)

    # Print last 8 quarters for human inspection
    print("\n=== Last 8 quarters: contributions vs total ===")
    cols = ["date", "_total_yoy", "_contrib_sum", "_residual"]
    print(recent.tail(8)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def save_contributions(contrib_df: pd.DataFrame) -> None:
    """Save contribution table (dropping internal check columns)."""
    drop_cols = [c for c in ["_total_yoy", "_contrib_sum", "_residual"] if c in contrib_df.columns]
    out = contrib_df.drop(columns=drop_cols)
    out.to_parquet(OUT_CONTRIB, index=False)
    logger.info("Saved contributions → %s", OUT_CONTRIB)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip download; recompute and verify from existing file")
    args = parser.parse_args()

    ensure_dir(RAW_BCRP_DIR)

    if args.verify_only:
        if not OUT_LEVELS.exists():
            logger.error("Level file not found: %s", OUT_LEVELS)
            sys.exit(1)
        df = pd.read_parquet(OUT_LEVELS)
        logger.info("Loaded %d rows from %s", len(df), OUT_LEVELS)
    else:
        df = download_levels()

    contrib_df = compute_contributions(df)
    verify_identity(contrib_df)
    save_contributions(contrib_df)

    # Quick summary of latest quarter
    latest = contrib_df.dropna(subset=["_total_yoy"]).iloc[-1]
    d = latest['date']
    quarter_label = f"{d.year}-Q{(d.month-1)//3+1}" if hasattr(d, 'month') else str(d)
    print(f"\n=== Latest quarter: {quarter_label} ===")
    sector_names = [SECTOR_CODES[c] for c in CONTRIB_SECTORS if SECTOR_CODES[c] in contrib_df.columns]
    for name in sector_names:
        val = latest.get(name, float("nan"))
        bar = "█" * int(abs(val) * 2) if not np.isnan(val) else ""
        sign = "+" if val >= 0 else ""
        print(f"  {name:<30} {sign}{val:.2f} pp  {bar}")
    print(f"  {'TOTAL YoY':<30} {latest['_total_yoy']:+.2f}%")
    print(f"  {'Sum of contributions':<30} {latest['_contrib_sum']:+.2f} pp")


if __name__ == "__main__":
    main()
