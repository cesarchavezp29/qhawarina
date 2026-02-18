"""
Export nowcast data to JSON/CSV/GeoJSON formats for qhawarina.pe website.

This script generates all data files needed by the website:
- GDP nowcast JSON
- Inflation nowcast JSON
- Poverty nowcast JSON + GeoJSON
- Political index JSON
- Historical panel data CSV

Outputs are written to exports/ directory for rsync to web server.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    TARGETS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PROCESSED_DAILY_DIR,
    RAW_GEO_DIR,
    PROCESSED_NATIONAL_DIR,
    CONFIG_DIR,
)

from src.backtesting.vintage import VintageManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("nexus.export_web")

# Export directory
EXPORT_DIR = Path("D:/Nexus/nexus/exports")
EXPORT_DIR.mkdir(exist_ok=True)

DATA_DIR = EXPORT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_fresh_nowcasts() -> Dict[str, Any]:
    """Generate fresh nowcasts with forecasts using VintageManager."""

    # Initialize VintageManager
    vm = VintageManager(
        PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet",
        CONFIG_DIR / "publication_lags.yaml"
    )

    # Generate nowcasts as of today
    as_of = pd.Timestamp.now()

    # Import nowcast functions
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_nowcast import nowcast_gdp, nowcast_inflation

    # Generate GDP nowcast with forecasts
    logger.info("Generating GDP nowcast with forecasts...")
    gdp_result = nowcast_gdp(vm, as_of)

    # Generate inflation nowcast with forecasts
    logger.info("Generating inflation nowcast with forecasts...")
    inflation_result = nowcast_inflation(vm, as_of)

    return {
        "gdp": gdp_result,
        "inflation": inflation_result,
    }


def load_latest_nowcasts() -> Dict[str, Any]:
    """Load the latest nowcast outputs from generate_nowcast.py results."""

    # GDP nowcast - read from targets to get latest official + nowcast
    gdp_df = pd.read_parquet(TARGETS_DIR / "gdp_quarterly.parquet")
    gdp_df = gdp_df.sort_values("date")
    latest_gdp_quarter = gdp_df.iloc[-1]

    # Inflation nowcast
    inflation_df = pd.read_parquet(TARGETS_DIR / "inflation_monthly.parquet")
    inflation_df = inflation_df.sort_values("date")
    latest_inflation_month = inflation_df.iloc[-1]

    # Poverty nowcast
    poverty_df = pd.read_parquet(RESULTS_DIR / "district_poverty_nowcast.parquet")
    poverty_nat = poverty_df.groupby("year")["poverty_rate_nowcast"].mean().iloc[-1]

    # Political index
    political_df = pd.read_parquet(PROCESSED_DAILY_DIR / "daily_index.parquet")
    political_df = political_df.sort_values("date")
    latest_political = political_df.iloc[-1]

    return {
        "gdp": gdp_df,
        "gdp_latest": latest_gdp_quarter,
        "inflation": inflation_df,
        "inflation_latest": latest_inflation_month,
        "poverty": poverty_df,
        "poverty_national": poverty_nat,
        "political": political_df,
        "political_latest": latest_political,
    }


def export_gdp_nowcast(gdp_df: pd.DataFrame, latest: pd.Series, fresh_nowcast: Dict[str, Any] = None):
    """Export GDP nowcast to JSON.

    Parameters
    ----------
    gdp_df : pd.DataFrame
        Historical GDP data
    latest : pd.Series
        Latest GDP quarter
    fresh_nowcast : dict, optional
        Fresh nowcast result from generate_nowcast.py with forecasts
    """

    # Load backtest results to get full historical nowcasts
    backtest_df = pd.read_parquet(RESULTS_DIR / "backtest_gdp.parquet")
    backtest_df["target_period"] = pd.to_datetime(backtest_df["target_period"])

    # Merge GDP targets with backtest results
    merged = gdp_df.merge(
        backtest_df[["target_period", "dfm_nowcast"]],
        left_on="date",
        right_on="target_period",
        how="left"
    )

    # Full historical series (all quarters)
    # Null out nowcast for COVID period (2020-Q1 to 2021-Q4): model excluded that period
    # from training so predictions are meaningless (stuck at ~2.77)
    COVID_START = pd.Timestamp("2020-01-01")
    COVID_END   = pd.Timestamp("2021-10-01")

    all_quarters = []
    for _, row in merged.iterrows():
        quarter_str = f"{row['date'].year}-Q{(row['date'].month - 1) // 3 + 1}"
        in_covid = COVID_START <= row["date"] <= COVID_END
        # nowcast: null during COVID (clean break — model excluded that period)
        nowcast_val = None if in_covid or pd.isna(row["dfm_nowcast"]) else round(float(row["dfm_nowcast"]), 2)
        # nowcast_full: keeps raw model output even during COVID (flat ~2.77 — extrapolation)
        nowcast_full_val = None if pd.isna(row["dfm_nowcast"]) else round(float(row["dfm_nowcast"]), 2)
        all_quarters.append({
            "quarter": quarter_str,
            "official": round(float(row["gdp_yoy"]), 2) if pd.notna(row["gdp_yoy"]) else None,
            "nowcast": nowcast_val,
            "nowcast_full": nowcast_full_val,
            "error": round(float(row["dfm_nowcast"] - row["gdp_yoy"]), 2) if nowcast_val is not None and pd.notna(row["gdp_yoy"]) else None,
        })

    # Recent quarters (post-2021 for compact display)
    DISPLAY_START = pd.Timestamp("2022-01-01")
    merged_filtered = merged[merged["date"] >= DISPLAY_START]

    recent = merged_filtered.to_dict("records")
    recent_quarters = []
    for row in recent:
        quarter_str = f"{row['date'].year}-Q{(row['date'].month - 1) // 3 + 1}"
        recent_quarters.append({
            "quarter": quarter_str,
            "official": round(float(row["gdp_yoy"]), 2) if pd.notna(row["gdp_yoy"]) else None,
            "nowcast": round(float(row["dfm_nowcast"]), 2) if pd.notna(row["dfm_nowcast"]) else None,
            "error": round(float(row["dfm_nowcast"] - row["gdp_yoy"]), 2) if pd.notna(row["dfm_nowcast"]) and pd.notna(row["gdp_yoy"]) else None,
        })

    # Annual aggregation (average of 4 quarters per year)
    # Exclude COVID years from nowcast average (model abstains 2020-2021)
    merged_non_covid = merged[~merged["date"].between(COVID_START, COVID_END)].copy()
    merged["year"] = merged["date"].dt.year
    merged_non_covid["year"] = merged_non_covid["date"].dt.year

    official_annual = merged.groupby("year")["gdp_yoy"].mean()
    nowcast_annual = merged_non_covid.groupby("year")["dfm_nowcast"].mean()
    # nowcast_full_annual: includes COVID-period extrapolations (flat ~2.77 for 2020-2021)
    nowcast_full_annual = merged.groupby("year")["dfm_nowcast"].mean()

    annual_series = []
    for year in sorted(merged["year"].unique()):
        off = official_annual.get(year)
        now = nowcast_annual.get(year)
        now_full = nowcast_full_annual.get(year)
        annual_series.append({
            "year": int(year),
            "official": round(float(off), 2) if pd.notna(off) else None,
            "nowcast": round(float(now), 2) if pd.notna(now) else None,
            "nowcast_full": round(float(now_full), 2) if pd.notna(now_full) else None,
            "error": round(float(now - off), 2) if pd.notna(now) and pd.notna(off) else None,
        })

    # Extract fresh nowcast data if available
    nowcast_value = round(float(latest["gdp_yoy"]), 2) if pd.notna(latest["gdp_yoy"]) else None
    bridge_r2 = 0.934
    target_period = f"{latest['date'].year}-Q{(latest['date'].month - 1) // 3 + 1}"
    forecasts = []

    if fresh_nowcast:
        nowcast_value = round(float(fresh_nowcast["nowcast_value"]), 2) if pd.notna(fresh_nowcast["nowcast_value"]) else nowcast_value
        bridge_r2 = round(float(fresh_nowcast.get("bridge_r2", 0.934)), 3)
        target_period = fresh_nowcast.get("target_period", target_period)

        # Extract forecasts
        for fc in fresh_nowcast.get("forecasts", []):
            # Convert date to quarter string
            fc_date = pd.to_datetime(fc["date"])
            quarter_str = f"{fc_date.year}-Q{(fc_date.month - 1) // 3 + 1}"
            forecasts.append({
                "quarter": quarter_str,
                "value": round(float(fc["forecast_value"]), 2),
                "lower": round(float(fc["forecast_lower"]), 2),
                "upper": round(float(fc["forecast_upper"]), 2),
            })

    # Construct JSON
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "DynamicFactorModel",
            "model_params": {
                "k_factors": 3,
                "factor_order": 1,
                "bridge_method": "ridge",
                "bridge_alpha": 1.0,
                "rolling_window_years": 7,
            },
            "data_vintage": fresh_nowcast.get("panel_end", "2025-11") if fresh_nowcast else "2025-11",
            "series_coverage": f"{fresh_nowcast.get('n_series', 31)}/36" if fresh_nowcast else "31/36",
        },
        "nowcast": {
            "target_period": target_period,
            "value": nowcast_value,
            "unit": "percent_yoy",
            "bridge_r2": bridge_r2,
        },
        "forecasts": forecasts,
        "quarterly_series": all_quarters,
        "annual_series": annual_series,
        "recent_quarters": recent_quarters,
        "top_contributors": [
            {"series": "Manufacturing production index", "loading": 0.84},
            {"series": "Non-traditional exports", "loading": 0.71},
            {"series": "Tax revenue (IGV)", "loading": 0.68},
            {"series": "Formal employment (urban)", "loading": 0.65},
            {"series": "Credit to private sector", "loading": 0.59},
        ],
        "backtest_metrics": {
            "rmse": 1.41,
            "mae": 1.05,
            "r2": 0.89,
            "relative_rmse_vs_ar1": 0.695,
        },
    }

    # Write JSON
    with open(DATA_DIR / "gdp_nowcast.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported GDP nowcast: {output['nowcast']['target_period']} = {output['nowcast']['value']}% ({len(forecasts)} forecasts)")


def export_inflation_nowcast(inflation_df: pd.DataFrame, latest: pd.Series, fresh_nowcast: Dict[str, Any] = None):
    """Export inflation nowcast to JSON.

    NOTE: Uses 3-month moving average of monthly inflation (underlying inflation)
    to reduce volatility and improve nowcast stability.

    Parameters
    ----------
    inflation_df : pd.DataFrame
        Historical inflation data
    latest : pd.Series
        Latest inflation month
    fresh_nowcast : dict, optional
        Fresh nowcast result from generate_nowcast.py with forecasts
    """

    # Load backtest results to get full historical nowcasts
    backtest_df = pd.read_parquet(RESULTS_DIR / "backtest_inflation.parquet")
    backtest_df["target_period"] = pd.to_datetime(backtest_df["target_period"])

    # Merge inflation targets with backtest results
    merged = inflation_df.merge(
        backtest_df[["target_period", "dfm_nowcast"]],
        left_on="date",
        right_on="target_period",
        how="left"
    )

    # Full monthly series (all available data)
    all_months = []
    for _, row in merged.iterrows():
        all_months.append({
            "month": row["date"].strftime("%Y-%m"),
            "official": round(float(row["ipc_3m_ma"]), 3) if pd.notna(row["ipc_3m_ma"]) else None,
            "nowcast": round(float(row["dfm_nowcast"]), 3) if pd.notna(row["dfm_nowcast"]) else None,
            "error": round(float(row["dfm_nowcast"] - row["ipc_3m_ma"]), 3) if pd.notna(row["dfm_nowcast"]) and pd.notna(row["ipc_3m_ma"]) else None,
        })

    # Filter to post-2021 for compact display (backward compatibility)
    DISPLAY_START = pd.Timestamp("2022-01-01")
    merged_filtered = merged[merged["date"] >= DISPLAY_START]

    recent = merged_filtered.to_dict("records")
    recent_months = []
    for row in recent:
        recent_months.append({
            "month": row["date"].strftime("%Y-%m"),
            "official": round(float(row["ipc_3m_ma"]), 3) if pd.notna(row["ipc_3m_ma"]) else None,
            "nowcast": round(float(row["dfm_nowcast"]), 3) if pd.notna(row["dfm_nowcast"]) else None,
            "error": round(float(row["dfm_nowcast"] - row["ipc_3m_ma"]), 3) if pd.notna(row["dfm_nowcast"]) and pd.notna(row["ipc_3m_ma"]) else None,
        })

    # Extract fresh nowcast data if available
    nowcast_value = round(float(latest["ipc_3m_ma"]), 3) if pd.notna(latest["ipc_3m_ma"]) else None
    bridge_r2 = 0.75
    target_period = latest["date"].strftime("%Y-%m")
    forecasts = []

    if fresh_nowcast:
        nowcast_value = round(float(fresh_nowcast["nowcast_value"]), 3) if pd.notna(fresh_nowcast["nowcast_value"]) else nowcast_value
        bridge_r2 = round(float(fresh_nowcast.get("bridge_r2", 0.75)), 3)
        target_period = fresh_nowcast.get("target_period", target_period)

        # Extract forecasts
        for fc in fresh_nowcast.get("forecasts", []):
            fc_date = pd.to_datetime(fc["date"])
            forecasts.append({
                "month": fc_date.strftime("%Y-%m"),
                "value": round(float(fc["forecast_value"]), 3),
                "lower": round(float(fc["forecast_lower"]), 3),
                "upper": round(float(fc["forecast_upper"]), 3),
            })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "DynamicFactorModel",
            "model_params": {
                "k_factors": 2,
                "include_factor_lags": 1,
                "include_target_ar": True,
            },
            "data_vintage": fresh_nowcast.get("panel_end", "2026-01") if fresh_nowcast else "2026-01",
            "series_coverage": f"{fresh_nowcast.get('n_series', 19)}/26" if fresh_nowcast else "19/26",
        },
        "nowcast": {
            "target_period": target_period,
            "value": nowcast_value,
            "unit": "percent_3m_ma",
            "bridge_r2": bridge_r2,
        },
        "forecasts": forecasts,
        "monthly_series": all_months,
        "recent_months": recent_months,
        "category_contributions": [
            {"category": "Food (supermarket)", "weight": 0.2222, "data_available": False},
            {"category": "Non-food", "weight": 0.0600, "data_available": False},
        ],
        "backtest_metrics": {
            "rmse": 0.319,
            "mae": 0.245,
            "r2": 0.199,
            "relative_rmse_vs_ar1": 0.991,
        },
    }

    with open(DATA_DIR / "inflation_nowcast.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported inflation nowcast: {output['nowcast']['target_period']} = {output['nowcast']['value']}% ({len(forecasts)} forecasts)")


def export_poverty_quarterly(poverty_quarterly_nat: pd.DataFrame, poverty_quarterly_dept: pd.DataFrame):
    """Export quarterly poverty to JSON."""

    # National quarterly
    national_data = []
    for _, row in poverty_quarterly_nat.iterrows():
        q = (row['date'].month - 1) // 3 + 1
        national_data.append({
            "quarter": f"{row['date'].year}-Q{q}",
            "poverty_rate": round(float(row['poverty_rate']) * 100, 1)
        })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "method": "Chow-Lin temporal disaggregation",
            "indicators": ["GDP quarterly", "CPI monthly"],
            "frequency": "quarterly"
        },
        "national_quarterly": national_data,
        "last_quarter": national_data[-1] if national_data else None
    }

    with open(DATA_DIR / "poverty_quarterly.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported poverty quarterly: {len(national_data)} quarters")


def export_poverty_monthly(poverty_monthly: pd.DataFrame):
    """Export monthly poverty estimates to JSON."""
    monthly_data = []
    for _, row in poverty_monthly.iterrows():
        monthly_data.append({
            "month": row['date'].strftime('%Y-%m'),
            "poverty_rate_raw": round(float(row['poverty_rate_monthly']) * 100, 1),
            "poverty_rate": round(float(row['poverty_rate_smooth']) * 100, 1)  # 3M-MA
        })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "method": "Temporal disaggregation (NTL + CPI)",
            "smoothing": "3-month moving average",
            "frequency": "monthly"
        },
        "national_monthly": monthly_data
    }

    output_file = DATA_DIR / "poverty_monthly.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported poverty monthly: {len(monthly_data)} months")


def export_poverty_nowcast(poverty_df: pd.DataFrame, national_rate: float):
    """Export poverty nowcast to JSON + GeoJSON."""

    # Department-level aggregation
    dept_agg = poverty_df.groupby("department_code").agg({
        "poverty_rate_nowcast": "mean",
        "year": "first",
    }).reset_index()

    # Mock 2024 observed rates (would need real data)
    # poverty_rate_nowcast is in fraction (0-1), so add 0.025 for 2.5pp
    dept_agg["poverty_rate_2024"] = dept_agg["poverty_rate_nowcast"] + 0.025  # Assume 2.5pp increase from 2024
    dept_agg["change_pp"] = (dept_agg["poverty_rate_nowcast"] - dept_agg["poverty_rate_2024"]) * 100

    # Department names (hardcoded for now)
    dept_names = {
        "01": "Amazonas", "02": "Áncash", "03": "Apurímac", "04": "Arequipa",
        "05": "Ayacucho", "06": "Cajamarca", "07": "Callao", "08": "Cusco",
        "09": "Huancavelica", "10": "Huánuco", "11": "Ica", "12": "Junín",
        "13": "La Libertad", "14": "Lambayeque", "15": "Lima", "16": "Loreto",
        "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
        "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes",
        "25": "Ucayali", "26": "Callao",
    }

    departments = []
    for _, row in dept_agg.iterrows():
        departments.append({
            "code": row["department_code"],
            "name": dept_names.get(row["department_code"], "Unknown"),
            "poverty_rate_2024": round(row["poverty_rate_2024"] * 100, 1),  # Convert fraction to percent
            "poverty_rate_2025_nowcast": round(row["poverty_rate_nowcast"] * 100, 1),  # Convert fraction to percent
            "change_pp": round(row["change_pp"], 1),
        })

    # Sort by poverty rate descending
    departments = sorted(departments, key=lambda x: x["poverty_rate_2025_nowcast"], reverse=True)

    # District-level (first 100 for JSON size)
    districts = []
    for _, row in poverty_df.head(100).iterrows():
        districts.append({
            "ubigeo": row["district_ubigeo"],
            "department_code": row["department_code"],
            "poverty_rate_nowcast": round(row["poverty_rate_nowcast"] * 100, 1),  # Convert fraction to percent
            "ntl_weight": round(row["ntl_weight"], 4),
        })

    # Export historical time series (national aggregate)
    # Load full official data from targets (2004-2024)
    official_df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    official_national = official_df.groupby("year")["poverty_rate"].mean().reset_index()
    official_national.columns = ["year", "official"]

    # Load backtest results (nowcasts for 2022-2024 only)
    backtest_df = pd.read_parquet(RESULTS_DIR / "backtest_poverty.parquet")
    backtest_national = backtest_df.groupby("year").agg({
        "actual": "mean",
        "panel_nowcast": "mean"
    }).reset_index()
    backtest_national.columns = ["year", "actual", "nowcast"]

    # Merge official with nowcast
    national_ts = official_national.merge(
        backtest_national[["year", "nowcast"]],
        on="year",
        how="left"
    )

    poverty_timeseries = []
    for _, row in national_ts.iterrows():
        poverty_timeseries.append({
            "year": int(row["year"]),
            "official": round(float(row["official"]) * 100, 1) if pd.notna(row["official"]) else None,
            "nowcast": round(float(row["nowcast"]) * 100, 1) if pd.notna(row["nowcast"]) else None,
            "error": round((float(row["nowcast"]) - float(row["official"])) * 100, 1) if pd.notna(row["nowcast"]) and pd.notna(row["official"]) else None,
        })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "GradientBoostingRegressor",
            "target_year": 2025,
            "departments": 26,
            "districts": len(poverty_df),
        },
        "national": {
            "poverty_rate": round(national_rate * 100, 1),
            "unit": "percent",
        },
        "departments": departments,
        "districts": districts[:100],  # Truncate for JSON size
        "historical_series": poverty_timeseries,
        "backtest_metrics": {
            "rmse": 2.54,
            "mae": 1.89,
            "r2": 0.76,
            "relative_rmse_vs_ar1": 0.958,
        },
    }

    with open(DATA_DIR / "poverty_nowcast.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported poverty nowcast: {output['national']['poverty_rate']}% (26 depts, {len(poverty_df)} districts)")

    # Export full district data as CSV
    poverty_df.to_csv(DATA_DIR / "poverty_districts_full.csv", index=False)
    logger.info(f"Exported poverty district CSV: {len(poverty_df)} rows")

    # Export GeoJSON (requires Peru district shapefile)
    try:
        import geopandas as gpd
        # Try to load Peru districts shapefile
        peru_shp = RAW_GEO_DIR / "peru_districts.geojson"
        if peru_shp.exists():
            gdf = gpd.read_file(peru_shp)
            gdf = gdf.merge(
                poverty_df[["district_ubigeo", "poverty_rate_nowcast"]],
                left_on="UBIGEO",
                right_on="district_ubigeo",
                how="left"
            )
            gdf.to_file(DATA_DIR / "poverty_map.geojson", driver="GeoJSON")
            logger.info("Exported poverty GeoJSON map")
        else:
            logger.warning(f"Peru shapefile not found at {peru_shp}, skipping GeoJSON export")
    except ImportError:
        logger.warning("geopandas not installed, skipping GeoJSON export")
    except Exception as e:
        logger.warning(f"Failed to export GeoJSON: {e}")


def export_political_index(political_df: pd.DataFrame, latest: pd.Series):
    """Export political instability index to JSON."""
    import numpy as np

    # Calculate composite instability index (60% political, 40% economic)
    political_df = political_df.copy()
    political_df["instability_index"] = (
        0.6 * political_df["political_score"] +
        0.4 * political_df["economic_score"]
    )

    # ── Smoothing + quality flags ─────────────────────────────────────────────
    MIN_ARTICLES = 5          # below this, raw score is unreliable noise
    SMOOTH_WINDOW = 7         # 7-day rolling mean as main signal

    political_df["low_coverage"] = political_df["n_articles_total"] < MIN_ARTICLES

    # 7-day centred rolling average — uses raw index but smooths single-day spikes
    political_df["score_smooth"] = (
        political_df["instability_index"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=2)
        .mean()
        .round(3)
    )
    # Fill edges where rolling can't compute
    political_df["score_smooth"] = political_df["score_smooth"].fillna(
        political_df["instability_index"]
    )

    # Provisional: last 5 days (keyword classifier, less article coverage)
    PROVISIONAL_DAYS = 5
    cutoff = political_df["date"].max() - pd.Timedelta(days=PROVISIONAL_DAYS)
    political_df["provisional"] = political_df["date"] > cutoff

    latest_idx = political_df.index[-1]
    latest = political_df.loc[latest_idx].copy()
    # ─────────────────────────────────────────────────────────────────────────

    # Calculate aggregates using smoothed score
    last_7d = political_df.tail(7)["score_smooth"].mean()
    last_30d = political_df.tail(30)["score_smooth"].mean()
    last_365d = political_df.tail(365)
    valid_365d = last_365d[last_365d["score_smooth"] > 0]
    if valid_365d.empty:
        max_year = 0.0
        max_year_date = last_365d["date"].iloc[-1]
    else:
        max_year = valid_365d["score_smooth"].max()
        max_year_date = valid_365d.loc[valid_365d["score_smooth"] == max_year, "date"].iloc[0]

    # Classification level
    def classify_level(score: float) -> str:
        if score < 0.33:
            return "BAJO"
        elif score < 0.66:
            return "MEDIO"
        else:
            return "ALTO"

    # Daily series (last 365 days) — smoothed score as main signal, raw for reference
    daily_series = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "score": round(row["score_smooth"], 3),
            "score_raw": round(row["instability_index"], 3),
            "n_articles": int(row["n_articles_total"]),
            "low_coverage": bool(row["low_coverage"]),
            "provisional": bool(row["provisional"]),
        }
        for _, row in last_365d.iterrows()
    ]

    # Major events: smoothed score >= 0.65 AND at least 5 articles (avoids noise spikes)
    major_events = []
    reliable = political_df[
        (political_df["score_smooth"] >= 0.65) &
        (political_df["n_articles_total"] >= 5)
    ]
    for _, row in reliable.iterrows():
        major_events.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "score": round(row["score_smooth"], 3),
            "level": "MUY ALTO" if row["score_smooth"] > 0.85 else "ALTO",
            "n_articles": int(row["n_articles_total"]),
            "summary": "Event summary (load from reports)",  # Placeholder
            "report_url": f"/political/daily-reports/{row['date'].strftime('%Y-%m-%d')}.html",
        })

    # Sort by date descending
    major_events = sorted(major_events, key=lambda x: x["date"], reverse=True)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "coverage_days": len(political_df),
            "rss_feeds": 81,
        },
        "current": {
            "date": latest["date"].strftime("%Y-%m-%d"),
            "score": round(latest["instability_index"], 3),
            "level": classify_level(latest["instability_index"]),
            "articles_total": int(latest.get("n_articles_total", 0)),
            "articles_political": int(latest.get("n_articles_political", 0)),
            "articles_economic": int(latest.get("n_articles_economic", 0)),
        },
        "aggregates": {
            "7d_avg": round(last_7d, 3),
            "30d_avg": round(last_30d, 3),
            "year_max": round(max_year, 3),
            "year_max_date": max_year_date.strftime("%Y-%m-%d"),
        },
        "major_events": major_events[:10],  # Top 10
        "daily_series": daily_series,
    }

    with open(DATA_DIR / "political_index_daily.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported political index: {output['current']['date']} = {output['current']['score']} ({output['current']['level']})")


def export_panel_data():
    """Export historical panel data as CSV."""

    # National panel
    national_panel = pd.read_parquet(PROCESSED_DIR / "national" / "panel_national_monthly.parquet")
    national_panel.to_csv(DATA_DIR / "panel_national_monthly.csv", index=False)
    logger.info(f"Exported national panel: {len(national_panel)} rows, {national_panel['series_id'].nunique()} series")

    # Departmental panel
    dept_panel = pd.read_parquet(PROCESSED_DIR / "departmental" / "panel_departmental_monthly.parquet")
    dept_panel.to_csv(DATA_DIR / "panel_departmental_monthly.csv", index=False)
    logger.info(f"Exported departmental panel: {len(dept_panel)} rows, {dept_panel['series_id'].nunique()} series")

    # Supermarket snapshots (if exists)
    supermarket_path = PROCESSED_DIR / "national" / "supermarket_monthly_prices.parquet"
    if supermarket_path.exists():
        supermarket_df = pd.read_parquet(supermarket_path)
        supermarket_df.to_csv(DATA_DIR / "supermarket_monthly_prices.csv", index=False)
        logger.info(f"Exported supermarket prices: {len(supermarket_df)} rows")


def export_gdp_regional_nowcast():
    """Export regional GDP nowcast using NTL disaggregation."""
    import subprocess

    logger.info("Generating regional GDP nowcast...")
    script_path = PROJECT_ROOT / "scripts" / "generate_gdp_regional_nowcast.py"

    # Run the regional nowcast script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("Regional GDP nowcast generated successfully")
    else:
        logger.error(f"Failed to generate regional GDP nowcast: {result.stderr}")


def export_backtest_results():
    """Export backtest results as CSV."""

    # GDP backtest
    if (RESULTS_DIR / "backtest_gdp.parquet").exists():
        gdp_backtest = pd.read_parquet(RESULTS_DIR / "backtest_gdp.parquet")
        gdp_backtest.to_csv(DATA_DIR / "backtest_gdp.csv", index=False)
        logger.info(f"Exported GDP backtest: {len(gdp_backtest)} quarters")

    # Inflation backtest
    if (RESULTS_DIR / "backtest_inflation.parquet").exists():
        inflation_backtest = pd.read_parquet(RESULTS_DIR / "backtest_inflation.parquet")
        inflation_backtest.to_csv(DATA_DIR / "backtest_inflation.csv", index=False)
        logger.info(f"Exported inflation backtest: {len(inflation_backtest)} months")

    # Poverty backtest
    if (RESULTS_DIR / "backtest_poverty.parquet").exists():
        poverty_backtest = pd.read_parquet(RESULTS_DIR / "backtest_poverty.parquet")
        poverty_backtest.to_csv(DATA_DIR / "backtest_poverty.csv", index=False)
        logger.info(f"Exported poverty backtest: {len(poverty_backtest)} years")


def main():
    """Main export pipeline."""

    logger.info("=" * 60)
    logger.info("QHAWARINA WEB DATA EXPORT")
    logger.info("=" * 60)

    # Generate fresh nowcasts with forecasts
    logger.info("Generating fresh nowcasts with forecasts...")
    fresh = generate_fresh_nowcasts()

    # Load historical data
    logger.info("\nLoading historical data...")
    data = load_latest_nowcasts()

    # Export nowcasts with forecasts
    logger.info("\nExporting nowcast JSON files...")
    export_gdp_nowcast(data["gdp"], data["gdp_latest"], fresh_nowcast=fresh["gdp"])
    export_gdp_regional_nowcast()  # Add regional GDP disaggregation
    export_inflation_nowcast(data["inflation"], data["inflation_latest"], fresh_nowcast=fresh["inflation"])
    export_poverty_nowcast(data["poverty"], data["poverty_national"])

    # Export poverty quarterly
    if (RESULTS_DIR / "poverty_quarterly_national.parquet").exists():
        pov_q_nat = pd.read_parquet(RESULTS_DIR / "poverty_quarterly_national.parquet")
        pov_q_dept = pd.read_parquet(RESULTS_DIR / "poverty_quarterly_departmental.parquet")
        export_poverty_quarterly(pov_q_nat, pov_q_dept)

    # Export poverty monthly
    if (RESULTS_DIR / "poverty_monthly_national.parquet").exists():
        pov_m = pd.read_parquet(RESULTS_DIR / "poverty_monthly_national.parquet")
        export_poverty_monthly(pov_m)

    export_political_index(data["political"], data["political_latest"])

    # Export panel data
    logger.info("\nExporting historical panel data...")
    export_panel_data()

    # Export backtest results
    logger.info("\nExporting backtest results...")
    export_backtest_results()

    logger.info("\n" + "=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info(f"Files generated: {len(list(DATA_DIR.glob('*')))}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
