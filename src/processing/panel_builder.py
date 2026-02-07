"""Build national and departmental monthly panels.

This module assembles all processed BCRP series into a long-format panel
suitable for nowcasting models. The panel is the primary input for Sprint 4.

Output Schema (long format)
============================
    date               : datetime64 — first of month
    series_id          : str — BCRP series code (e.g., PN01770AM)
    series_name        : str — human-readable name
    category           : str — thematic category (gdp_indicators, inflation, etc.)
    value_raw          : float64 — original or deflated level
    value_sa           : float64 — seasonally adjusted
    value_log          : float64 — log(value_sa)
    value_dlog         : float64 — log-difference (monthly return)
    value_yoy          : float64 — year-on-year growth (%)
    source             : str — 'BCRP'
    frequency_original : str — 'M' (monthly)
    publication_lag_days : int — approximate reporting delay
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.harmonize import (
    load_series_metadata,
    process_single_series,
    reconstruct_ipc_index,
)
from src.utils.io import save_parquet

logger = logging.getLogger("nexus.panel_builder")


def build_national_panel(
    raw_path: Path,
    catalog_path: Path,
    output_path: Path,
    include_gdp_monthly: bool = True,
) -> pd.DataFrame:
    """Build the national monthly panel from raw BCRP data.

    Processes all 40 national series through deflation, seasonal adjustment,
    and transformation, then stacks them into a long-format panel.

    Parameters
    ----------
    raw_path : Path
        Path to bcrp_national_all.parquet.
    catalog_path : Path
        Path to series_catalog.yaml.
    output_path : Path
        Where to save the resulting panel (.parquet).
    include_gdp_monthly : bool
        If True, also run Chow-Lin disaggregation and include monthly GDP.

    Returns
    -------
    pd.DataFrame
        Long-format panel with the output schema.
    """
    logger.info("Building national panel...")

    # Load raw data and metadata
    raw_df = pd.read_parquet(raw_path)
    metadata = load_series_metadata(catalog_path)

    # Reconstruct IPC deflator
    deflator = reconstruct_ipc_index(raw_df)
    logger.info("IPC deflator reconstructed: %d months", len(deflator))

    # Process each series
    all_frames = []
    sa_success = 0
    sa_total = 0

    for code, meta in metadata.items():
        try:
            processed = process_single_series(code, raw_df, metadata, deflator)
            if processed.empty:
                continue

            # Add metadata columns
            processed["series_id"] = code
            processed["series_name"] = meta["name"]
            processed["category"] = meta["category"]
            processed["source"] = "BCRP"
            processed["frequency_original"] = "M"
            processed["publication_lag_days"] = meta["publication_lag_days"]

            all_frames.append(processed)

            if meta["seasonal_adjust"] in ("stl", "x13"):
                sa_total += 1
                # Check if SA was actually applied (not just raw passthrough)
                if not np.allclose(
                    processed["value_raw"].dropna().values,
                    processed["value_sa"].dropna().values,
                    equal_nan=True,
                ):
                    sa_success += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", code, e)

    # Optionally add monthly GDP from Chow-Lin
    if include_gdp_monthly:
        try:
            from src.processing.disaggregate import disaggregate_gdp
            gdp_monthly = disaggregate_gdp(raw_df)

            gdp_frame = pd.DataFrame({
                "date": gdp_monthly.index,
                "value_raw": gdp_monthly.values,
                "value_sa": gdp_monthly.values,
                "value_log": np.log(gdp_monthly.values),
                "value_dlog": np.log(gdp_monthly).diff().values,
                "value_yoy": ((gdp_monthly / gdp_monthly.shift(12)) - 1).values * 100,
                "series_id": "GDP_MONTHLY_CL",
                "series_name": "PBI mensual (Chow-Lin, índice 2007=100)",
                "category": "gdp_indicators",
                "source": "BCRP/Chow-Lin",
                "frequency_original": "Q→M",
                "publication_lag_days": 90,
            })
            all_frames.append(gdp_frame)
            logger.info("Chow-Lin monthly GDP added: %d observations", len(gdp_frame))
        except Exception as e:
            logger.warning("Chow-Lin GDP disaggregation failed: %s", e)

    if not all_frames:
        raise ValueError("No series were successfully processed")

    panel = pd.concat(all_frames, ignore_index=True)

    # Reorder columns
    col_order = [
        "date", "series_id", "series_name", "category",
        "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
        "source", "frequency_original", "publication_lag_days",
    ]
    panel = panel[col_order]

    # Sort
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    # Save
    save_parquet(panel, output_path)
    logger.info(
        "National panel saved: %s (%d rows, %d series, %d months)",
        output_path, len(panel), panel["series_id"].nunique(),
        panel["date"].nunique(),
    )

    if sa_total > 0:
        logger.info(
            "Seasonal adjustment: %d/%d series adjusted (%.0f%%)",
            sa_success, sa_total, sa_success / sa_total * 100,
        )

    return panel


def validate_panel(panel: pd.DataFrame) -> dict:
    """Validate the structure and quality of a built panel.

    Checks:
    - Required columns present
    - No infinite values
    - Date range plausibility
    - Minimum series count
    - No duplicate (series_id, date) pairs

    Parameters
    ----------
    panel : pd.DataFrame
        Panel to validate.

    Returns
    -------
    dict
        Validation results with 'passed' bool and 'checks' detail list.
    """
    checks = []

    # Required columns
    required_cols = [
        "date", "series_id", "series_name", "category",
        "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
    ]
    missing_cols = [c for c in required_cols if c not in panel.columns]
    checks.append({
        "name": "required_columns",
        "passed": len(missing_cols) == 0,
        "detail": f"Missing: {missing_cols}" if missing_cols else "All present",
    })

    # No infinities
    value_cols = ["value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]
    n_inf = sum(np.isinf(panel[c].dropna()).sum() for c in value_cols if c in panel.columns)
    checks.append({
        "name": "no_infinities",
        "passed": n_inf == 0,
        "detail": f"{n_inf} infinite values" if n_inf > 0 else "None",
    })

    # Date range
    date_min = panel["date"].min()
    date_max = panel["date"].max()
    date_ok = (
        pd.Timestamp("2003-01-01") < date_min < pd.Timestamp("2005-01-01")
        and date_max > pd.Timestamp("2024-01-01")
    )
    checks.append({
        "name": "date_range",
        "passed": date_ok,
        "detail": f"{date_min} to {date_max}",
    })

    # Series count
    n_series = panel["series_id"].nunique()
    checks.append({
        "name": "series_count",
        "passed": n_series >= 40,
        "detail": f"{n_series} series",
    })

    # No duplicates
    dupes = panel.duplicated(subset=["series_id", "date"]).sum()
    checks.append({
        "name": "no_duplicates",
        "passed": dupes == 0,
        "detail": f"{dupes} duplicates" if dupes > 0 else "None",
    })

    # value_raw completeness
    raw_na_pct = panel["value_raw"].isna().mean() * 100
    checks.append({
        "name": "value_raw_completeness",
        "passed": raw_na_pct < 5.0,
        "detail": f"{raw_na_pct:.1f}% missing",
    })

    all_passed = all(c["passed"] for c in checks)

    return {
        "passed": all_passed,
        "n_rows": len(panel),
        "n_series": n_series,
        "date_range": f"{date_min} to {date_max}",
        "checks": checks,
    }


def build_departmental_panel(
    poverty_path: Path,
    catalog_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build a departmental panel (stub — to be implemented in Sprint 5).

    Will combine departmental BCRP series with ENAHO poverty data to create
    a department × time panel for spatial nowcasting.

    Parameters
    ----------
    poverty_path : Path
        Path to poverty_departmental.parquet.
    catalog_path : Path
        Path to series_catalog.yaml.
    output_path : Path
        Output path for the departmental panel.

    Returns
    -------
    pd.DataFrame
        Departmental panel (currently just poverty data reshaped).
    """
    logger.info("Departmental panel: loading poverty data (stub)")
    poverty = pd.read_parquet(poverty_path)

    # For now, just save the poverty data in panel format
    # Sprint 5 will add departmental BCRP series
    save_parquet(poverty, output_path)
    logger.info("Departmental panel saved (stub): %s", output_path)

    return poverty
