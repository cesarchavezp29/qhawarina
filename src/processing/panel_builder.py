"""Build national and departmental monthly panels.

This module assembles all processed BCRP series into a long-format panel
suitable for nowcasting models. The panel is the primary input for Sprint 4.

National Output Schema (long format)
======================================
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

Departmental Output Schema (long format)
==========================================
    date               : datetime64 — first of month
    series_id          : str — BCRP series code (e.g., RD13560DM)
    series_name        : str — human-readable name
    category           : str — thematic category (credit_by_department, etc.)
    department         : str — department name (Amazonas, Lima, etc.)
    ubigeo             : str — 2-digit department code ('01'-'25')
    value_raw          : float64 — original or deflated level
    value_sa           : float64 — seasonally adjusted
    value_log          : float64 — log(value_sa)
    value_dlog         : float64 — log-difference (monthly return)
    value_yoy          : float64 — year-on-year growth (%)
    source             : str — 'BCRP'
    frequency_original : str — 'M' (monthly)
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.processing.harmonize import (
    deflate_series,
    load_series_metadata,
    process_single_series,
    reconstruct_ipc_index,
    seasonal_adjust,
    transform_dlog,
    transform_log,
    transform_yoy,
)
from src.processing.missing import interpolate_gaps
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


def load_regional_catalog(catalog_path: Path) -> dict:
    """Load regional series catalog with processing metadata.

    Returns
    -------
    dict
        Keyed by category name, each with 'processing' and 'series' entries.
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    return catalog.get("regional", {})


def _process_regional_series(
    code: str,
    values: pd.Series,
    unit_type: str,
    sa_method: str,
    deflator: pd.Series | None = None,
) -> pd.DataFrame:
    """Process a single regional series through deflation, SA, and transforms.

    Mirrors process_single_series from harmonize.py but operates on
    pre-extracted values rather than raw_df lookup.

    Parameters
    ----------
    code : str
        Series code for logging.
    values : pd.Series
        Time series with DatetimeIndex.
    unit_type : str
        One of 'nominal_soles', 'nominal_usd', 'level', etc.
    sa_method : str
        Seasonal adjustment method ('stl', 'x13', 'none').
    deflator : pd.Series or None
        IPC deflator index (required if unit_type == 'nominal_soles').

    Returns
    -------
    pd.DataFrame
        Columns: [date, value_raw, value_sa, value_log, value_dlog, value_yoy]
    """
    if values.empty or values.dropna().empty:
        return pd.DataFrame(
            columns=["date", "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]
        )

    # 1. Deflate nominal soles series
    if unit_type == "nominal_soles" and deflator is not None:
        values = deflate_series(values, deflator)

    # 2. Drop leading NaN block (e.g. pre-2004 after deflation by IPC that
    #    starts in 2004).  STL can choke on long leading NaN runs during
    #    batch processing, so we trim them, process the valid range, then
    #    pad the NaN prefix back onto the result.
    full_index = values.index
    first_valid = values.first_valid_index()
    pre_gap_len = 0
    if first_valid is not None and first_valid != values.index[0]:
        pre_gap_len = values.index.get_loc(first_valid)
        values = values.loc[first_valid:]

    # 3. Interpolate short gaps
    values = interpolate_gaps(values, method="linear", max_gap=3)

    # 4. Seasonal adjustment
    sa_result = seasonal_adjust(values, method=sa_method)
    value_sa = sa_result["value_sa"]

    # 5. Transformations
    if unit_type not in ("rate_pct", "var_pct", "var_pct_yoy"):
        value_log = transform_log(value_sa)
        value_dlog = transform_dlog(value_sa)
        value_yoy = transform_yoy(value_sa)
    else:
        value_log = pd.Series(np.nan, index=values.index)
        value_dlog = pd.Series(np.nan, index=values.index)
        value_yoy = pd.Series(np.nan, index=values.index)

    result = pd.DataFrame({
        "date": values.index,
        "value_raw": values.values,
        "value_sa": value_sa.values,
        "value_log": value_log.values,
        "value_dlog": value_dlog.values,
        "value_yoy": value_yoy.values,
    })

    # Pad leading NaN rows back so every series spans the full original range
    if pre_gap_len > 0:
        pre_dates = full_index[:pre_gap_len]
        pre_df = pd.DataFrame({
            "date": pre_dates,
            "value_raw": np.nan,
            "value_sa": np.nan,
            "value_log": np.nan,
            "value_dlog": np.nan,
            "value_yoy": np.nan,
        })
        result = pd.concat([pre_df, result], ignore_index=True)

    return result


def build_departmental_panel(
    regional_data_dir: Path,
    regional_catalog_path: Path,
    national_raw_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build the departmental monthly panel from regional BCRP data.

    Processes all regional series through deflation, seasonal adjustment,
    and transformation, filtering to department-level Total-currency series.

    Pipeline per category:
        1. Load raw parquet
        2. Filter: credit/deposits use 'Total' currency only
        3. Exclude 'Total Nacional' aggregates
        4. Deflate nominal_soles series using national IPC
        5. Seasonal adjust (STL, robust=True, period=12)
        6. Compute transforms: value_log, value_dlog, value_yoy
        7. Stack into long-format panel

    Parameters
    ----------
    regional_data_dir : Path
        Directory containing bcrp_regional_*.parquet files.
    regional_catalog_path : Path
        Path to regional_series_catalog.yaml.
    national_raw_path : Path
        Path to bcrp_national_all.parquet (for IPC deflator).
    output_path : Path
        Where to save the resulting panel.

    Returns
    -------
    pd.DataFrame
        Long-format departmental panel.
    """
    from scripts.download_regional_bcrp import CATEGORY_FILES

    logger.info("Building departmental panel...")

    # Load regional catalog
    regional_catalog = load_regional_catalog(regional_catalog_path)

    # Reconstruct IPC deflator from national data
    national_raw = pd.read_parquet(national_raw_path)
    deflator = reconstruct_ipc_index(national_raw)
    logger.info("IPC deflator: %d months [%.1f, %.1f]",
                len(deflator), deflator.min(), deflator.max())

    all_frames = []
    sa_success = 0
    sa_total = 0
    skipped_categories = []

    for cat_key, cat_data in regional_catalog.items():
        if not isinstance(cat_data, dict):
            continue

        series_list = cat_data.get("series", [])
        if not series_list:
            skipped_categories.append(cat_key)
            continue

        processing = cat_data.get("processing", {})
        unit_type = processing.get("unit_type", "level")
        sa_method = processing.get("seasonal_adjust", "stl")
        panel_filter = processing.get("panel_filter", None)

        # Find parquet file
        parquet_name = CATEGORY_FILES.get(cat_key)
        if not parquet_name:
            parquet_name = f"bcrp_regional_{cat_key}.parquet"
        parquet_path = regional_data_dir / parquet_name

        if not parquet_path.exists():
            logger.warning("Missing data file for %s: %s", cat_key, parquet_path)
            skipped_categories.append(cat_key)
            continue

        logger.info("Processing category: %s (%s)", cat_key, parquet_path.name)
        raw_df = pd.read_parquet(parquet_path)
        raw_df["date"] = pd.to_datetime(raw_df["date"])

        # Build lookup for series metadata
        code_meta = {s["code"]: s for s in series_list}

        # Filter: credit/deposits use Total currency only (from catalog metadata)
        if panel_filter:
            before = raw_df["series_code"].nunique()
            total_codes = {
                s["code"] for s in series_list
                if s.get("currency") == panel_filter
            }
            if total_codes:
                raw_df = raw_df[raw_df["series_code"].isin(total_codes)]
                after = raw_df["series_code"].nunique()
                logger.info("  Filtered %s currency: %d → %d series",
                            panel_filter, before, after)

        # Exclude 'Total Nacional' aggregates (no department)
        dept_codes = {
            s["code"] for s in series_list
            if s.get("department") or s.get("ubigeo")
        }
        raw_df = raw_df[raw_df["series_code"].isin(dept_codes)]

        if raw_df.empty:
            logger.warning("  No departmental data after filtering for %s", cat_key)
            continue

        # Process each series
        for code in raw_df["series_code"].unique():
            meta = code_meta.get(code, {})
            series_data = (
                raw_df[raw_df["series_code"] == code]
                .dropna(subset=["value"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
            )
            if series_data.empty:
                continue

            values = series_data.set_index("date")["value"]

            try:
                processed = _process_regional_series(
                    code, values, unit_type, sa_method,
                    deflator=deflator if unit_type == "nominal_soles" else None,
                )
                if processed.empty:
                    continue

                # Add metadata
                processed["series_id"] = code
                processed["series_name"] = meta.get("name", code)
                processed["category"] = cat_key
                processed["department"] = meta.get("department", "")
                processed["ubigeo"] = meta.get("ubigeo", "")
                processed["source"] = "BCRP"
                processed["frequency_original"] = "M"

                all_frames.append(processed)

                if sa_method in ("stl", "x13"):
                    sa_total += 1
                    if not np.allclose(
                        processed["value_raw"].dropna().values,
                        processed["value_sa"].dropna().values,
                        equal_nan=True,
                    ):
                        sa_success += 1

            except Exception as e:
                logger.error("  Failed to process %s: %s", code, e)

    if skipped_categories:
        logger.info("Skipped categories (no data): %s", skipped_categories)

    if not all_frames:
        raise ValueError("No regional series were successfully processed")

    panel = pd.concat(all_frames, ignore_index=True)

    # Reorder columns
    col_order = [
        "date", "series_id", "series_name", "category",
        "department", "ubigeo",
        "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
        "source", "frequency_original",
    ]
    panel = panel[col_order]

    # Sort
    panel = panel.sort_values(["category", "ubigeo", "series_id", "date"]).reset_index(drop=True)

    # Replace inf with NaN
    value_cols = ["value_raw", "value_sa", "value_log", "value_dlog", "value_yoy"]
    for c in value_cols:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)

    # Save
    save_parquet(panel, output_path)

    n_series = panel["series_id"].nunique()
    n_depts = panel["ubigeo"].nunique()
    n_months = panel["date"].nunique()
    n_cats = panel["category"].nunique()

    logger.info(
        "Departmental panel saved: %s (%d rows, %d series, %d depts, %d months, %d categories)",
        output_path, len(panel), n_series, n_depts, n_months, n_cats,
    )

    if sa_total > 0:
        logger.info(
            "Seasonal adjustment: %d/%d series adjusted (%.0f%%)",
            sa_success, sa_total, sa_success / sa_total * 100,
        )

    return panel


def validate_departmental_panel(panel: pd.DataFrame) -> dict:
    """Validate the departmental panel structure and quality.

    Parameters
    ----------
    panel : pd.DataFrame
        Departmental panel to validate.

    Returns
    -------
    dict
        Validation results.
    """
    checks = []

    # Required columns
    required_cols = [
        "date", "series_id", "series_name", "category",
        "department", "ubigeo",
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

    # Date range (regional data starts from 2001)
    date_min = panel["date"].min()
    date_max = panel["date"].max()
    date_ok = (
        date_min < pd.Timestamp("2005-01-01")
        and date_max > pd.Timestamp("2024-01-01")
    )
    checks.append({
        "name": "date_range",
        "passed": date_ok,
        "detail": f"{date_min} to {date_max}",
    })

    # Department count (25 departments including Callao)
    n_depts = panel["ubigeo"].nunique() if "ubigeo" in panel.columns else 0
    checks.append({
        "name": "department_count",
        "passed": n_depts >= 20,
        "detail": f"{n_depts} departments",
    })

    # No duplicates
    dupes = panel.duplicated(subset=["series_id", "date"]).sum()
    checks.append({
        "name": "no_duplicates",
        "passed": dupes == 0,
        "detail": f"{dupes} duplicates" if dupes > 0 else "None",
    })

    # Category count
    n_cats = panel["category"].nunique() if "category" in panel.columns else 0
    checks.append({
        "name": "category_count",
        "passed": n_cats >= 5,
        "detail": f"{n_cats} categories",
    })

    # value_raw completeness
    raw_na_pct = panel["value_raw"].isna().mean() * 100
    checks.append({
        "name": "value_raw_completeness",
        "passed": raw_na_pct < 10.0,
        "detail": f"{raw_na_pct:.1f}% missing",
    })

    all_passed = all(c["passed"] for c in checks)

    return {
        "passed": all_passed,
        "n_rows": len(panel),
        "n_series": panel["series_id"].nunique(),
        "n_departments": n_depts,
        "n_categories": n_cats,
        "date_range": f"{date_min} to {date_max}",
        "checks": checks,
    }
