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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so ANTHROPIC_API_KEY is available for Haiku justification
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.settings import (
    TARGETS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PROCESSED_DAILY_DIR,
    RAW_GEO_DIR,
    RAW_BCRP_DIR,
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
EXPORT_DIR = PROJECT_ROOT / "exports"
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

    # Compute forecast vs actual track record (last 8 non-COVID quarters with both values)
    forecast_vs_actual = [
        {"quarter": q["quarter"], "official": q["official"], "nowcast": q["nowcast"], "error": q["error"]}
        for q in all_quarters
        if q["official"] is not None and q["nowcast"] is not None
    ][-8:]

    # Last official GDP quarter (before nowcast target)
    last_official_quarters = [q for q in all_quarters if q["official"] is not None]
    last_official_gdp = last_official_quarters[-1] if last_official_quarters else None

    # Subcomponent contributions from fresh nowcast (if available)
    subcomponent_contributions = fresh_nowcast.get("contributions", []) if fresh_nowcast else []

    n_indicators = fresh_nowcast.get("n_series", 31) if fresh_nowcast else 31

    # Construct JSON
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "methodology": "Dynamic Factor Model with Ridge bridge equation (Baker 2023, Aruoba 2020)",
            "model": "DynamicFactorModel",
            "model_params": {
                "k_factors": 3,
                "factor_order": 1,
                "bridge_method": "ridge",
                "bridge_alpha": 1.0,
                "rolling_window_years": 7,
                "exclude_covid": True,
            },
            "n_indicators": n_indicators,
            "training_window": "2007-2024 (rolling 7-year window, excl. COVID 2020-2021)",
            "last_official_gdp": last_official_gdp,
            "data_vintage": fresh_nowcast.get("panel_end", "2025-11") if fresh_nowcast else "2025-11",
            "series_coverage": f"{n_indicators}/36",
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
        "forecast_vs_actual": forecast_vs_actual,
        "subcomponent_contributions": subcomponent_contributions,
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

    # Use real 2024 official poverty rates from INEI (via poverty_departmental.parquet)
    official_2024 = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    official_2024 = official_2024[official_2024["year"] == 2024][["department_code", "poverty_rate"]]
    official_2024 = official_2024.rename(columns={"poverty_rate": "poverty_rate_2024"})
    dept_agg = dept_agg.merge(official_2024, on="department_code", how="left")
    # Fallback: where 2024 data is missing, approximate from nowcast (e.g. Callao=Lima)
    dept_agg["poverty_rate_2024"] = dept_agg["poverty_rate_2024"].fillna(
        dept_agg["poverty_rate_nowcast"] + 0.02
    )
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

    MODEL_RMSE_PP = 2.54   # GBR model RMSE in percentage points
    CI_HALF = 1.96 * MODEL_RMSE_PP  # ±4.98pp at 95%

    departments = []
    for _, row in dept_agg.iterrows():
        est_pp = round(row["poverty_rate_nowcast"] * 100, 1)
        departments.append({
            "code": row["department_code"],
            "name": dept_names.get(row["department_code"], "Unknown"),
            "poverty_rate_2024": round(row["poverty_rate_2024"] * 100, 1),
            "poverty_rate_2025_proyeccion": est_pp,
            "lower_bound": round(max(0.0, est_pp - CI_HALF), 1),
            "upper_bound": round(min(100.0, est_pp + CI_HALF), 1),
            "change_pp": round(row["change_pp"], 1),
        })

    # Sort by poverty rate descending
    departments = sorted(departments, key=lambda x: x["poverty_rate_2025_proyeccion"], reverse=True)

    # District-level (all districts for choropleth)
    districts = []
    for _, row in poverty_df.iterrows():
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

    nat_pp = round(national_rate * 100, 1)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "tipo_estimacion": "proyección",
            "methodology_note": (
                "Proyección basada en correlaciones históricas entre indicadores "
                "macroeconómicos mensuales y tasas de pobreza anuales ENAHO (2004-2024). "
                "No es una encuesta; los intervalos de confianza reflejan el error histórico del modelo."
            ),
            "model_type": "GradientBoostingRegressor",
            "training_window": "2004-2024 (excl. COVID 2020-2021)",
            "last_official_enaho_year": 2024,
            "target_year": 2025,
            "departments": 26,
            "districts": len(poverty_df),
        },
        "national": {
            "poverty_rate": nat_pp,
            "lower_bound": round(max(0.0, nat_pp - CI_HALF), 1),
            "upper_bound": round(min(100.0, nat_pp + CI_HALF), 1),
            "rmse_pp": MODEL_RMSE_PP,
            "unit": "percent",
        },
        "departments": departments,
        "districts": districts,
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


def _haiku_justification(
    today_arts: "pd.DataFrame",
    prr_raw: float,
    prr_7d: float,
    level: str,
    date_str: str,
) -> dict:
    """Call Claude Haiku to justify today's PRR score.

    Returns dict with keys:
        justification : str  — 3-4 sentence analysis in Spanish
        top_drivers   : list — top 5 articles [{title, source, category, severity}]
    """
    _INT_TO_FLOAT = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.9}
    result = {"justification": None, "top_drivers": []}

    if today_arts.empty:
        return result

    # Build top drivers list (sorted by severity desc)
    drivers_df = today_arts.copy()
    drivers_df["sev_float"] = drivers_df["article_severity"].map(_INT_TO_FLOAT).fillna(0.0)
    drivers_df = drivers_df.sort_values("sev_float", ascending=False).head(5)

    top_drivers = [
        {
            "title": str(r["title"])[:120],
            "source": str(r.get("source", "")),
            "category": str(r["article_category"]),
            "severity": round(float(r["sev_float"]), 2),
        }
        for _, r in drivers_df.iterrows()
    ]
    result["top_drivers"] = top_drivers

    n_pol = int((today_arts["article_category"].isin(["political", "both"])).sum())
    n_econ = int((today_arts["article_category"].isin(["economic", "both"])).sum())
    n_total = len(today_arts)
    mult = f"{prr_raw / 100:.1f}"
    level_labels = {
        "MINIMO": "mínimo", "BAJO": "bajo", "MODERADO": "moderado",
        "ELEVADO": "elevado", "ALTO": "alto", "CRITICO": "crítico",
    }
    level_es = level_labels.get(level, level.lower())

    drivers_text = "\n".join(
        f"- [{d['severity']:.1f}] [{d['category']}] {d['title']} ({d['source']})"
        for d in top_drivers
    )

    prompt = (
        f"Eres el analista de riesgo político de Qhawarina. "
        f"Hoy ({date_str}) analizaste {n_total} artículos de prensa peruana.\n\n"
        f"Resultado: PRR = {prr_raw:.0f} ({mult}× el promedio histórico). "
        f"Nivel: {level_es}.\n"
        f"Artículos políticos: {n_pol} | Artículos económicos: {n_econ}\n\n"
        f"Los {len(top_drivers)} artículos de mayor severidad hoy:\n{drivers_text}\n\n"
        f"Escribe un análisis breve (3-4 oraciones) en español que explique:\n"
        f"1. Por qué el riesgo político está en este nivel hoy\n"
        f"2. Qué eventos específicos lo impulsan\n"
        f"3. Si la tendencia es de escalamiento o de normalización\n\n"
        f"No uses viñetas. Escribe en prosa fluida, tono analítico profesional. "
        f"Máximo 80 palabras."
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        result["justification"] = msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Haiku justification failed: %s", e)

    return result


def _haiku_driver_phrase(articles_text: str, date_str: str) -> str | None:
    """Legacy: get a short driver phrase. Kept for major_events summary."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"Dadas estas noticias políticas del día {date_str} en Perú, "
            f"resume el factor principal de inestabilidad en exactamente una frase "
            f"de 5-10 palabras en español. Solo la frase, sin explicación.\n\n"
            f"{articles_text}"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Haiku driver_phrase failed: %s", e)
        return None


def export_political_index(political_df: pd.DataFrame, latest: pd.Series):
    """Export political instability index to JSON. Supports v1 and v2 schema."""
    import numpy as np

    political_df = political_df.copy()

    # ── Load articles cache (AI-GPR computation + Haiku summaries) ───────────
    from config.settings import RAW_RSS_DIR
    articles_cache_path = RAW_RSS_DIR / "articles_classified.parquet"
    articles_cache = None
    if articles_cache_path.exists():
        try:
            articles_cache = pd.read_parquet(articles_cache_path)
            articles_cache["published"] = pd.to_datetime(articles_cache["published"], utc=True)
            articles_cache["date"] = articles_cache["published"].dt.date
        except Exception as e:
            logger.warning("Could not load articles cache: %s", e)

    # ── AI-GPR dual index (Iacoviello & Tong 2026) ───────────────────────────
    # Normalized formula (Iacoviello & Tong eq. 1, footnote 4):
    #   IRP_t = (Σ political_score_i / N_articles_t) / S_bar_political × 100
    #   IRE_t = (Σ economic_score_i / N_articles_t) / S_bar_economic × 100
    #
    # Where N_articles_t = TOTAL articles on day t (including irrelevant ones).
    # This eliminates volume bias: a day with 150 articles and 15 political ones
    # scores the same as a day with 50 articles and 5 political ones (same proportion
    # and same average intensity). Makes the index stationary by construction.

    if articles_cache is not None and len(articles_cache) > 0:
        art = articles_cache.copy()
        art["day"] = pd.to_datetime(art["date"]).dt.normalize()

        # LEVEL 3 SAFETY NET: exclude GDELT and non-Peru aggregator feeds.
        # Direct Peru feeds (elcomercio, gestion, larepublica, andina, rpp, correo)
        # are the only valid signal sources. GDELT adds ~12k global noise articles.
        if "feed_name" in art.columns:
            n_before = len(art)
            art = art[~art["feed_name"].str.contains("gdelt", case=False, na=False)]
            n_excluded = n_before - len(art)
            if n_excluded > 0:
                logger.info("Excluded %d GDELT articles from index computation", n_excluded)

        # Total articles per day (denominator for normalization — ALL articles, including score=0)
        n_total_s = art.groupby("day").size().rename("n_total")

        has_dual = "political_score" in art.columns and "economic_score" in art.columns
        if has_dual:
            # New dual-score schema (AI-GPR: IRP/IRE)
            pol_sum_s = art.groupby("day")["political_score"].sum().fillna(0).rename("pol_sum")
            eco_sum_s = art.groupby("day")["economic_score"].sum().fillna(0).rename("eco_sum")
        else:
            # Legacy fallback: map article_severity to scores
            _INT_TO_FLOAT = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.9}
            art["sev_float"] = art["article_severity"].map(_INT_TO_FLOAT).fillna(0.0)
            pol_sum_s = (
                art[art["article_category"].isin(["political", "both"])]
                .groupby("day")["sev_float"].sum() * 60
            ).rename("pol_sum")
            eco_sum_s = (
                art[art["article_category"].isin(["economic", "both"])]
                .groupby("day")["sev_float"].sum() * 40
            ).rename("eco_sum")

        daily_sums = pd.concat([n_total_s, pol_sum_s, eco_sum_s], axis=1).fillna(0).reset_index()
        daily_sums.columns = ["date", "n_total", "pol_sum", "eco_sum"]

        # Normalize by total articles per day (Iacoviello & Tong footnote 4)
        daily_sums["pol_norm"] = daily_sums["pol_sum"] / daily_sums["n_total"].clip(lower=1)
        daily_sums["eco_norm"] = daily_sums["eco_sum"] / daily_sums["n_total"].clip(lower=1)

        # Baseline: 2025 calendar year (freeze after 30+ days available)
        ref = daily_sums[pd.to_datetime(daily_sums["date"]).dt.year == 2025]
        if len(ref) >= 30:
            s_bar_pol = ref["pol_norm"].mean() or 1.0
            s_bar_eco = ref["eco_norm"].mean() or 1.0
        else:
            s_bar_pol = daily_sums["pol_norm"].mean() or 1.0
            s_bar_eco = daily_sums["eco_norm"].mean() or 1.0

        daily_sums["irp"] = (daily_sums["pol_norm"] / s_bar_pol) * 100.0
        daily_sums["ire"] = (daily_sums["eco_norm"] / s_bar_eco) * 100.0

        today_vals = daily_sums.iloc[-1] if not daily_sums.empty else None
        if today_vals is not None:
            print(f"[AI-GPR] IRP today = {today_vals['irp']:.1f}  IRE today = {today_vals['ire']:.1f}")

        political_df = political_df.merge(
            daily_sums[["date", "irp", "ire", "n_total"]], on="date", how="left"
        )
        political_df["irp"] = political_df["irp"].fillna(0.0)
        political_df["ire"] = political_df["ire"].fillna(0.0)
        # Use clean article count (no GDELT) for display — overwrite old parquet's n_articles_total
        political_df["n_articles_total"] = political_df["n_total"].fillna(0).astype(int)
        # Keep instability_index as political IRP for backward compat with smoothing below
        political_df["instability_index"] = political_df["irp"]
    else:
        political_df["irp"] = 0.0
        political_df["ire"] = 0.0
        political_df["instability_index"] = 0.0

    # ── Smoothing (7-day rolling, centered) for both indices ─────────────────
    MIN_ARTICLES = 5
    SMOOTH_WINDOW = 7

    political_df["low_coverage"] = political_df["n_articles_total"] < MIN_ARTICLES

    political_df["irp_7d"] = (
        political_df["irp"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=2)
        .mean().round(1).fillna(political_df["irp"])
    )
    political_df["ire_7d"] = (
        political_df["ire"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=2)
        .mean().round(1).fillna(political_df["ire"])
    )
    # Backward compat
    political_df["instability_index"] = political_df["irp"]
    political_df["score_smooth"] = political_df["irp_7d"]

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

    # Level classification (applied to both IRP and IRE)
    def classify_level(score: float) -> str:
        if score < 50:   return "MINIMO"
        if score < 100:  return "BAJO"
        if score < 150:  return "MODERADO"
        if score < 200:  return "ELEVADO"
        if score < 300:  return "ALTO"
        return "CRITICO"

    # Daily series (last 365 days) — dual indices
    daily_series = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "political_raw": round(row["irp"], 1),
            "political_7d":  round(row["irp_7d"], 1),
            "economic_raw":  round(row["ire"], 1),
            "economic_7d":   round(row["ire_7d"], 1),
            # legacy fields (kept for backward compat)
            "score":     round(row["irp_7d"], 1),
            "prr":       round(row["irp"], 1),
            "prr_7d":    round(row["irp_7d"], 1),
            "n_articles": int(row["n_articles_total"]),
            "low_coverage": bool(row["low_coverage"]),
            "provisional":  bool(row["provisional"]),
        }
        for _, row in last_365d.iterrows()
    ]

    # Major events: PRR >= 120 (ELEVADO+) AND at least 5 articles (avoids noise spikes)
    major_events = []
    reliable = political_df[
        (political_df["score_smooth"] >= 120) &
        (political_df["n_articles_total"] >= 5)
    ]
    for _, row in reliable.iterrows():
        event_date = row["date"].date() if hasattr(row["date"], "date") else row["date"]
        event_summary = None
        if articles_cache is not None:
            day_arts = articles_cache[
                (articles_cache["date"] == event_date) &
                (articles_cache["political_score"] > 0)
            ].head(3)
            if not day_arts.empty:
                arts_text = "\n".join(
                    f"- {r['title']}" for _, r in day_arts.iterrows()
                )
                event_summary = _haiku_driver_phrase(arts_text, str(event_date))
        major_events.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "score": round(row["score_smooth"], 1),
            "level": classify_level(row["score_smooth"]),
            "n_articles": int(row["n_articles_total"]),
            "summary": event_summary or "Alta inestabilidad política registrada.",
            "report_url": f"/political/daily-reports/{row['date'].strftime('%Y-%m-%d')}.html",
        })

    # Sort by date descending
    major_events = sorted(major_events, key=lambda x: x["date"], reverse=True)

    # ── Daily FX series (same period as political index) ─────────────────────
    import httpx, re as _re

    _MONTH_MAP = {
        "Ene": "01", "Feb": "02", "Mar": "03", "Abr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Ago": "08",
        "Set": "09", "Sep": "09", "Oct": "10", "Nov": "11", "Dic": "12",
    }

    def _parse_bcrp_daily_date(name: str) -> str | None:
        """Parse '02.Ene.25' → '2025-01-02'."""
        m = _re.match(r"(\d{2})\.([A-Za-z]{3})\.(\d{2})$", name)
        if not m:
            return None
        day, mon, yr = m.group(1), m.group(2).capitalize(), m.group(3)
        mon_num = _MONTH_MAP.get(mon)
        if not mon_num:
            return None
        year = f"20{yr}"
        return f"{year}-{mon_num}-{day}"

    daily_fx_series = []
    try:
        pol_start = political_df["date"].min().strftime("%Y-%m-%d")
        pol_end   = political_df["date"].max().strftime("%Y-%m-%d")
        url = (
            f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
            f"PD04638PD/json/{pol_start}/{pol_end}/esp"
        )
        resp = httpx.get(url, timeout=20, follow_redirects=True)
        if resp.status_code == 200:
            for period in resp.json().get("periods", []):
                iso = _parse_bcrp_daily_date(period["name"])
                vals = period.get("values", [])
                if iso and vals and vals[0] not in ("", "n.d."):
                    daily_fx_series.append({
                        "date": iso,
                        "fx": round(float(vals[0]), 4),
                    })
    except Exception as _e:
        logger.warning("Could not fetch daily FX from BCRP: %s", _e)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Monthly series: political avg + economic avg + FX yoy ────────────────
    political_df["month"] = political_df["date"].dt.to_period("M")
    monthly_pol = (
        political_df.groupby("month").agg(
            score_smooth=("score_smooth", "mean"),
            ire_7d=("ire_7d", "mean"),
            n_articles_total=("n_articles_total", "sum"),
        ).reset_index()
    )
    monthly_pol["month_str"] = monthly_pol["month"].dt.strftime("%Y-%m")

    # Load FX yoy from panel (has full history → real YoY for all months)
    panel_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"
    try:
        panel_df = pd.read_parquet(panel_path, columns=["date", "series_id", "value_raw", "value_yoy"])
        fx_df = panel_df[panel_df["series_id"] == "PN01246PM"][["date", "value_raw", "value_yoy"]].copy()
        fx_df["month_str"] = pd.to_datetime(fx_df["date"]).dt.strftime("%Y-%m")
        fx_df = fx_df.rename(columns={"value_raw": "fx_level", "value_yoy": "fx_yoy"})
        merged_monthly = monthly_pol.merge(fx_df[["month_str", "fx_level", "fx_yoy"]], on="month_str", how="left")
    except Exception:
        merged_monthly = monthly_pol.copy()
        merged_monthly["fx_level"] = None
        merged_monthly["fx_yoy"] = None

    monthly_series = []
    for _, row in merged_monthly.iterrows():
        fx_lv = float(row["fx_level"]) if pd.notna(row.get("fx_level")) else None
        fx_yy = float(row["fx_yoy"]) if pd.notna(row.get("fx_yoy")) else None
        monthly_series.append({
            "month": row["month_str"],
            "political_avg": round(float(row["score_smooth"]), 1),
            "economic_avg": round(float(row["ire_7d"]), 1),
            "n_articles": int(row["n_articles_total"]),
            "fx_level": round(fx_lv, 4) if fx_lv is not None else None,
            "fx_yoy": round(fx_yy, 2) if fx_yy is not None else None,
        })
    # ─────────────────────────────────────────────────────────────────────────

    # ── Haiku dual justification for today ───────────────────────────────────
    irp_today  = float(latest.get("irp", 0))
    ire_today  = float(latest.get("ire", 0))
    irp_7d     = float(latest.get("irp_7d", irp_today))
    ire_7d     = float(latest.get("ire_7d", ire_today))

    political_justification = None
    economic_justification  = None
    top_political_drivers: list = []
    top_economic_drivers:  list = []
    today_all = pd.DataFrame()  # initialized here so it's always in scope

    if articles_cache is not None:
        today_date = latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]
        today_all  = articles_cache[articles_cache["date"] == today_date]
        has_dual   = "political_score" in today_all.columns and "economic_score" in today_all.columns

        if has_dual and (irp_today > 0 or ire_today > 0):
            try:
                client = __import__("anthropic").Anthropic()

                # Top 5 political articles
                top_pol = today_all.nlargest(5, "political_score")[
                    ["title", "source", "political_score"]
                ]
                if not top_pol.empty and irp_today > 0:
                    drivers_text = "\n".join(
                        f"- [{int(r.political_score)}] {r.title} ({r.source})"
                        for _, r in top_pol.iterrows() if r.political_score > 0
                    )
                    if drivers_text:
                        msg = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=150,
                            temperature=0,
                            messages=[{"role": "user", "content":
                                f"Eres el analista de riesgo político de Qhawarina. "
                                f"Hoy el Índice de Riesgo Político = {irp_today:.0f} "
                                f"({irp_today/100:.1f}× el promedio).\n"
                                f"Los 5 artículos con mayor puntaje político:\n{drivers_text}\n\n"
                                f"Escribe 2-3 oraciones explicando por qué el riesgo político "
                                f"está en este nivel. Máximo 60 palabras. Solo el texto."
                            }]
                        )
                        political_justification = msg.content[0].text.strip()
                    top_political_drivers = [
                        {"title": r.title, "source": r.source, "score": int(r.political_score)}
                        for _, r in top_pol.iterrows() if r.political_score > 0
                    ]

                # Top 5 economic articles
                top_eco = today_all.nlargest(5, "economic_score")[
                    ["title", "source", "economic_score"]
                ]
                if not top_eco.empty and ire_today > 0:
                    drivers_text = "\n".join(
                        f"- [{int(r.economic_score)}] {r.title} ({r.source})"
                        for _, r in top_eco.iterrows() if r.economic_score > 0
                    )
                    if drivers_text:
                        msg = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=150,
                            temperature=0,
                            messages=[{"role": "user", "content":
                                f"Eres el analista de riesgo económico de Qhawarina. "
                                f"Hoy el Índice de Riesgo Económico = {ire_today:.0f} "
                                f"({ire_today/100:.1f}× el promedio).\n"
                                f"Los 5 artículos con mayor puntaje económico:\n{drivers_text}\n\n"
                                f"Escribe 2-3 oraciones explicando por qué el riesgo económico "
                                f"está en este nivel. Máximo 60 palabras. Solo el texto."
                            }]
                        )
                        economic_justification = msg.content[0].text.strip()
                    top_economic_drivers = [
                        {"title": r.title, "source": r.source, "score": int(r.economic_score)}
                        for _, r in top_eco.iterrows() if r.economic_score > 0
                    ]
            except Exception as e:
                logger.warning("Haiku justification failed: %s", e)
    # ─────────────────────────────────────────────────────────────────────────

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "coverage_days": len(political_df),
            "rss_feeds": 11,
            "rss_sources": 6,
            "methodology": "AI-GPR dual index (Iacoviello & Tong 2026): IRP + IRE, mean=100 over 2025 baseline",
        },
        "current": {
            "date": latest["date"].strftime("%Y-%m-%d"),
            # Dual indices
            "political_raw":  round(irp_today, 1),
            "political_7d":   round(irp_7d, 1),
            "political_level": classify_level(irp_7d),
            "political_multiplier": round(irp_7d / 100, 2),
            "economic_raw":   round(ire_today, 1),
            "economic_7d":    round(ire_7d, 1),
            "economic_level": classify_level(ire_7d),
            "economic_multiplier": round(ire_7d / 100, 2),
            # Legacy fields (for homepage card backward compat)
            "score":   round(irp_7d, 1),
            "prr_7d":  round(irp_7d, 1),
            "prr_raw": round(irp_today, 1),
            "level":   classify_level(irp_7d),
            "articles_total":               int(latest.get("n_articles_total", 0)),
            "articles_political_relevant":  int(today_all["political_score"].gt(0).sum() if not today_all.empty and "political_score" in today_all.columns else 0),
            "articles_economic_relevant":   int(today_all["economic_score"].gt(0).sum() if not today_all.empty and "economic_score" in today_all.columns else 0),
            "political_justification": political_justification,
            "economic_justification":  economic_justification,
            "top_political_drivers":   top_political_drivers,
            "top_economic_drivers":    top_economic_drivers,
            # Legacy
            "justification": political_justification,
            "top_drivers":   top_political_drivers,
        },
        "aggregates": {
            "political_7d_avg":  round(political_df.tail(7)["irp_7d"].mean(), 1),
            "political_30d_avg": round(political_df.tail(30)["irp_7d"].mean(), 1),
            "economic_7d_avg":   round(political_df.tail(7)["ire_7d"].mean(), 1),
            "economic_30d_avg":  round(political_df.tail(30)["ire_7d"].mean(), 1),
            "year_max": round(max_year, 1),
            "year_max_date": max_year_date.strftime("%Y-%m-%d"),
        },
        "major_events": major_events[:10],
        "daily_series": daily_series,
        "daily_fx_series": daily_fx_series,
        "monthly_series": monthly_series,
    }

    # Append guard: refuse to overwrite if new data would shrink the series by >10%
    existing_path = DATA_DIR / "political_index_daily.json"
    if existing_path.exists():
        try:
            with open(existing_path, "r", encoding="utf-8") as f_ex:
                existing_json = json.load(f_ex)
            existing_count = len(existing_json.get("daily_series", []))
            new_count = len(daily_series)
            if new_count < existing_count * 0.9:
                raise ValueError(
                    f"Append guard triggered: new daily_series has {new_count} entries "
                    f"but existing has {existing_count} (threshold: {int(existing_count * 0.9)}). "
                    "Refusing to overwrite. Check data pipeline integrity."
                )
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted existing JSON — allow overwrite

    with open(DATA_DIR / "political_index_daily.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported political index: {output['current']['date']} = {output['current']['score']} ({output['current']['level']})")


def export_bcrp_financial_markets():
    """Export BCRP FX interventions and financial market data to JSON.

    Fetches daily BCRP series (Jan 2020 → today):
      PD04659MD — Compras netas spot (Mill. USD)
      PD04660MD — Swaps netos (Mill. USD)
      PD04638PD — TC PEN/USD interbancario
      PD31893DD — Bono soberano 10a S/ (yield %)
      PD31894DD — Bono soberano 10a USD (yield %)
      PD38026MD — BVL Índice General
      PD12301MD — Tasa referencia BCRP (%)
    """
    import httpx
    import re as _re

    # IMPORTANT: BCRP API returns series sorted alphabetically by code,
    # NOT in the order specified in the URL. SERIES_KEYS must match
    # the alphabetically sorted order of SERIES_CODES.
    # Sorted: PD04638PD, PD04659MD, PD04660MD, PD12301MD, PD31893DD, PD31894DD, PD38026MD
    SERIES_CODES = [
        "PD04638PD",  # TC PEN/USD          → vals[0]
        "PD04659MD",  # spot net purchases   → vals[1]
        "PD04660MD",  # swaps net            → vals[2]
        "PD12301MD",  # Tasa referencia      → vals[3]
        "PD31893DD",  # Bono 10a S/          → vals[4]
        "PD31894DD",  # Bono 10a USD         → vals[5]
        "PD38026MD",  # BVL                  → vals[6]
    ]
    SERIES_KEYS = [
        "fx",
        "spot_net_purchases",
        "swaps_net",
        "reference_rate",
        "bond_sol_10y",
        "bond_usd_10y",
        "bvl",
    ]

    BCRP_START = "2020-01-01"
    BCRP_END = datetime.now().strftime("%Y-%m-%d")

    _MONTH_MAP = {
        "Ene": "01", "Feb": "02", "Mar": "03", "Abr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Ago": "08",
        "Set": "09", "Sep": "09", "Oct": "10", "Nov": "11", "Dic": "12",
    }

    def _parse_date(name: str):
        m = _re.match(r"(\d{2})\.([A-Za-z]{3})\.(\d{2})$", name)
        if not m:
            return None
        day, mon, yr = m.group(1), m.group(2).capitalize(), m.group(3)
        mon_num = _MONTH_MAP.get(mon)
        if not mon_num:
            return None
        return f"20{yr}-{mon_num}-{day}"

    series_str = "-".join(SERIES_CODES)
    url = (
        f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
        f"{series_str}/json/{BCRP_START}/{BCRP_END}/esp"
    )

    try:
        resp = httpx.get(url, timeout=40, follow_redirects=True)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch BCRP financial markets data: {e}")
        return

    records = []
    for period in raw.get("periods", []):
        iso = _parse_date(period["name"])
        if not iso:
            continue
        vals = period.get("values", [])
        row = {"date": iso}
        for i, key in enumerate(SERIES_KEYS):
            v = vals[i] if i < len(vals) else ""
            row[key] = float(v) if v not in ("", "n.d.", None) else None
        records.append(row)

    if not records:
        logger.warning("No BCRP financial market data fetched")
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill reference rate (changes infrequently, BCRP only posts on change dates)
    df["reference_rate"] = df["reference_rate"].ffill()

    # Total daily intervention = spot + swaps (null → 0 for summation)
    df["total_intervention"] = df["spot_net_purchases"].fillna(0) + df["swaps_net"].fillna(0)

    # Monthly aggregation (full history since 2020)
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")
        .agg(
            spot_monthly=("spot_net_purchases", "sum"),
            swaps_monthly=("swaps_net", "sum"),
            fx_avg=("fx", "mean"),
            bond_sol_avg=("bond_sol_10y", "mean"),
            bond_usd_avg=("bond_usd_10y", "mean"),
            bvl_end=("bvl", "last"),
            reference_rate_end=("reference_rate", "last"),
            n_days=("date", "count"),
        )
        .reset_index()
    )
    monthly["total_monthly"] = monthly["spot_monthly"] + monthly["swaps_monthly"]
    monthly["month_str"] = monthly["month"].dt.strftime("%Y-%m")

    # Latest available values
    latest_row = df.dropna(subset=["fx"]).iloc[-1]
    latest = {
        "date": latest_row["date"].strftime("%Y-%m-%d"),
        "fx": round(float(latest_row["fx"]), 4) if pd.notna(latest_row["fx"]) else None,
        "spot_net_purchases": round(float(latest_row["spot_net_purchases"]), 1) if pd.notna(latest_row["spot_net_purchases"]) else None,
        "swaps_net": round(float(latest_row["swaps_net"]), 1) if pd.notna(latest_row["swaps_net"]) else None,
        "reference_rate": round(float(latest_row["reference_rate"]), 2) if pd.notna(latest_row["reference_rate"]) else None,
        "bond_sol_10y": round(float(latest_row["bond_sol_10y"]), 3) if pd.notna(latest_row["bond_sol_10y"]) else None,
        "bond_usd_10y": round(float(latest_row["bond_usd_10y"]), 3) if pd.notna(latest_row["bond_usd_10y"]) else None,
        "bvl": round(float(latest_row["bvl"]), 0) if pd.notna(latest_row["bvl"]) else None,
    }

    # Daily series (last 2 years to keep JSON manageable)
    TWO_YEARS_AGO = pd.Timestamp.now() - pd.DateOffset(years=2)
    df_recent = df[df["date"] >= TWO_YEARS_AGO].copy()

    # Forward-fill market prices to handle BCRP publication lag (TC/bonds/BVL
    # are published 1-2 days late, leaving trailing nulls that break charts)
    for col in ["fx", "reference_rate", "bond_sol_10y", "bond_usd_10y", "bvl"]:
        if col in df_recent.columns:
            df_recent[col] = df_recent[col].ffill()

    daily_series = []
    for _, row in df_recent.iterrows():
        daily_series.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "spot_net_purchases": round(float(row["spot_net_purchases"]), 1) if pd.notna(row["spot_net_purchases"]) else None,
            "swaps_net": round(float(row["swaps_net"]), 1) if pd.notna(row["swaps_net"]) else None,
            "total_intervention": round(float(row["total_intervention"]), 1),
            "fx": round(float(row["fx"]), 4) if pd.notna(row["fx"]) else None,
            "reference_rate": round(float(row["reference_rate"]), 2) if pd.notna(row["reference_rate"]) else None,
            "bond_sol_10y": round(float(row["bond_sol_10y"]), 3) if pd.notna(row["bond_sol_10y"]) else None,
            "bond_usd_10y": round(float(row["bond_usd_10y"]), 3) if pd.notna(row["bond_usd_10y"]) else None,
            "bvl": round(float(row["bvl"]), 0) if pd.notna(row["bvl"]) else None,
        })

    monthly_series = []
    for _, row in monthly.iterrows():
        monthly_series.append({
            "month": row["month_str"],
            "spot_net_purchases": round(float(row["spot_monthly"]), 1) if pd.notna(row["spot_monthly"]) else 0.0,
            "swaps_net": round(float(row["swaps_monthly"]), 1) if pd.notna(row["swaps_monthly"]) else 0.0,
            "total_intervention": round(float(row["total_monthly"]), 1),
            "fx_avg": round(float(row["fx_avg"]), 4) if pd.notna(row["fx_avg"]) else None,
            "bond_sol_avg": round(float(row["bond_sol_avg"]), 3) if pd.notna(row["bond_sol_avg"]) else None,
            "bond_usd_avg": round(float(row["bond_usd_avg"]), 3) if pd.notna(row["bond_usd_avg"]) else None,
            "bvl_end": round(float(row["bvl_end"]), 0) if pd.notna(row["bvl_end"]) else None,
            "reference_rate": round(float(row["reference_rate_end"]), 2) if pd.notna(row["reference_rate_end"]) else None,
            "n_days": int(row["n_days"]),
        })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "sources": {
                "spot": "PD04659MD — Compras netas spot BCRP (Mill. USD)",
                "swaps": "PD04660MD — Swaps cambiarios netos BCRP (Mill. USD)",
                "fx": "PD04638PD — TC Interbancario Venta (PEN/USD)",
                "bond_sol": "PD31893DD — Bono Soberano 10 años en Soles (rendimiento %)",
                "bond_usd": "PD31894DD — Bono Soberano 10 años en USD (rendimiento %)",
                "bvl": "PD38026MD — BVL Índice General",
                "reference_rate": "PD12301MD — Tasa de Referencia BCRP (%)",
            },
            "methodology": (
                "Flotación sucia (managed float): el BCRP interviene en el mercado spot "
                "y via swaps cambiarios para suavizar la volatilidad del tipo de cambio, "
                "sin defender un nivel específico de TC."
            ),
            "units": {
                "spot_net_purchases": "Mill. USD (positivo = compra neta de USD por el BCRP)",
                "swaps_net": "Mill. USD netos (swaps compra - swaps venta)",
                "fx": "Soles por dólar (venta interbancaria)",
                "bond_sol_10y": "Rendimiento anual (%)",
                "bond_usd_10y": "Rendimiento anual (%)",
                "bvl": "Puntos del índice",
                "reference_rate": "Tasa de política monetaria anual (%)",
            },
            "coverage": f"{BCRP_START} a {BCRP_END}",
            "n_days_daily": len(daily_series),
            "n_months_monthly": len(monthly_series),
        },
        "latest": latest,
        "daily_series": daily_series,
        "monthly_series": monthly_series,
    }

    output_path = DATA_DIR / "fx_interventions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Exported BCRP financial markets: {len(daily_series)} days, {len(monthly_series)} months. "
        f"Latest TC={latest.get('fx')} | Spot={latest.get('spot_net_purchases')} Mill.USD"
    )


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


def export_supermarket_daily_index():
    """Build and export the supermarket daily price index JSON."""
    import subprocess

    logger.info("Building supermarket daily price index...")
    script_path = PROJECT_ROOT / "scripts" / "build_price_index.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("Supermarket daily price index built and exported")
    else:
        logger.error(f"Failed to build supermarket price index: {result.stderr[-500:]}")


def export_gdp_regional_nowcast():
    """Export regional GDP nowcast using NTL disaggregation."""
    import subprocess

    logger.info("Generating regional GDP nowcast...")
    script_path = PROJECT_ROOT / "scripts" / "generate_gdp_regional_nowcast.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("Regional GDP nowcast generated successfully")
    else:
        logger.error(f"Failed to generate regional GDP nowcast: {result.stderr}")


def export_inflation_regional_nowcast():
    """Export regional inflation nowcast using food/core basket decomposition."""
    import subprocess

    logger.info("Generating regional inflation nowcast...")
    script_path = PROJECT_ROOT / "scripts" / "generate_inflation_regional_nowcast.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("Regional inflation nowcast generated successfully")
    else:
        logger.error(f"Failed to generate regional inflation nowcast: {result.stderr}")


def export_gdp_contributions():
    """Export sectoral GDP contributions to YoY growth as JSON for the web chart.

    Reads bcrp_quarterly_gdp_contributions.parquet (generated by download_gdp_levels.py).
    Outputs gdp_contributions.json with last 12 quarters.
    """
    contrib_path = RAW_BCRP_DIR / "bcrp_quarterly_gdp_contributions.parquet"
    levels_path  = RAW_BCRP_DIR / "bcrp_quarterly_gdp_levels.parquet"

    if not contrib_path.exists():
        logger.info("GDP contributions not found — skipping (run download_gdp_levels.py first)")
        return

    contrib = pd.read_parquet(contrib_path)
    contrib["date"] = pd.to_datetime(contrib["date"])
    contrib = contrib.sort_values("date")

    # Load total YoY from levels file for reference
    total_yoy = None
    if levels_path.exists():
        levels = pd.read_parquet(levels_path)
        levels["date"] = pd.to_datetime(levels["date"])
        total_wide = levels[levels["series_code"] == "PN37692AQ"].set_index("date")["value"]
        total_yoy = ((total_wide - total_wide.shift(4)) / total_wide.shift(4) * 100).rename("total_yoy")

    SECTORS = [
        "Agropecuario", "Pesca", "Minería e Hidrocarburos", "Manufactura",
        "Electricidad y Agua", "Construcción", "Comercio", "Servicios",
    ]
    # Short labels used as JSON keys (ASCII-safe, unique)
    SECTOR_KEY = {
        "Agropecuario":           "Agro",
        "Pesca":                  "Pesca",
        "Minería e Hidrocarburos":"Mineria",
        "Manufactura":            "Manufactura",
        "Electricidad y Agua":    "Electr",
        "Construcción":           "Construc",
        "Comercio":               "Comercio",
        "Servicios":              "Servicios",
    }
    # Display labels for charts (full Spanish, with accents)
    SECTOR_LABEL = {
        "Agro":         "Agropecuario",
        "Pesca":        "Pesca",
        "Mineria":      "Minería",
        "Manufactura":  "Manufactura",
        "Electr":       "Electricidad",
        "Construc":     "Construcción",
        "Comercio":     "Comercio",
        "Servicios":    "Servicios",
    }

    # ── Chart 2 & 3: Contributions (last 16 quarters) ────────────────────────
    quarters = []
    for _, row in contrib.tail(16).iterrows():
        d = row["date"]
        q_label = f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        entry = {"quarter": q_label, "date": d.strftime("%Y-%m-%d")}

        # Add total YoY
        if total_yoy is not None and d in total_yoy.index:
            entry["total_yoy"] = round(float(total_yoy[d]), 2)

        # Add sector contributions
        for sec in SECTORS:
            if sec in row and pd.notna(row[sec]):
                entry[SECTOR_KEY[sec]] = round(float(row[sec]), 2)

        quarters.append(entry)

    # ── Chart 1: GDP shares over time (last 80 quarters = 20 years) ──────────
    shares = []
    if levels_path.exists():
        levels = pd.read_parquet(levels_path)
        levels["date"] = pd.to_datetime(levels["date"])
        wide = levels.pivot_table(index="date", columns="series_code", values="value").sort_index()
        total_col = "PN37692AQ"
        sector_codes_ordered = [
            "PN37684AQ", "PN37685AQ", "PN37686AQ", "PN37687AQ",
            "PN37688AQ", "PN37689AQ", "PN37690AQ", "PN37691AQ",
        ]
        for d, row in wide.tail(80).iterrows():
            total = row.get(total_col)
            if not total or pd.isna(total) or total == 0:
                continue
            entry = {
                "quarter": f"{d.year}-Q{(d.month-1)//3+1}",
                "date": d.strftime("%Y-%m-%d"),
            }
            sector_names_full = [
                "Agropecuario", "Pesca", "Minería e Hidrocarburos", "Manufactura",
                "Electricidad y Agua", "Construcción", "Comercio", "Servicios",
            ]
            for code, name in zip(sector_codes_ordered, sector_names_full):
                val = row.get(code)
                if pd.notna(val):
                    entry[SECTOR_KEY[name]] = round(float(val) / float(total) * 100, 2)
            shares.append(entry)

    # Latest quarter summary
    latest_q = quarters[-1] if quarters else {}
    out = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "source": "BCRP — PBI por sectores productivos, soles constantes 2007",
            "note": "contribution_i = (GDP_i,t - GDP_i,t-4) / GDP_total,t-4 × 100",
            "latest_quarter": latest_q.get("quarter", ""),
            "total_yoy_latest": latest_q.get("total_yoy"),
        },
        "sector_keys": list(SECTOR_KEY.values()),
        "sector_labels": SECTOR_LABEL,
        "contributions": quarters,
        "shares": shares,
    }

    out_path = DATA_DIR / "gdp_contributions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Exported GDP contributions: %d quarters → %s", len(quarters), out_path)


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


def generate_nota_diaria() -> None:
    """Generate bilingual daily brief (nota_diaria.json) from already-exported JSONs."""
    import json as _json
    from datetime import date as _date

    today = _date.today().isoformat()

    def _load(name):
        p = DATA_DIR / name
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return _json.load(f)

    gdp   = _load("gdp_nowcast.json")
    infl  = _load("inflation_nowcast.json")
    prices= _load("daily_price_index.json")
    pol   = _load("political_index_daily.json")

    highlights = []

    # ── Prices ────────────────────────────────────────────────────────────────
    if prices and prices.get("latest"):
        lat = prices["latest"]
        cum = lat.get("cum_pct", 0.0)
        sign_es = "subió" if cum > 0 else "bajó"
        sign_en = "rose" if cum > 0 else "fell"
        highlights.append({
            "icon": "🛒",
            "text_es": f"Precios de supermercado {sign_es} {abs(cum):.2f}% en lo que va del mes.",
            "text_en": f"Supermarket prices {sign_en} {abs(cum):.2f}% month-to-date.",
        })
        movers = lat.get("top_movers", [])
        for mv in movers[:2]:
            v = mv.get("var", 0.0)
            icon = "📈" if v > 0 else "📉"
            highlights.append({
                "icon": icon,
                "text_es": f"{mv.get('label_es', mv.get('category', ''))}: {v:+.1f}% en el mes.",
                "text_en": f"{mv.get('label_en', mv.get('category', ''))}: {v:+.1f}% this month.",
            })

    # ── GDP ───────────────────────────────────────────────────────────────────
    if gdp and gdp.get("nowcast"):
        nw = gdp["nowcast"]
        val = nw.get("value", 0.0)
        period = nw.get("target_period", "")
        highlights.append({
            "icon": "📊",
            "text_es": f"Nowcast PBI {period}: {val:+.2f}% (variación anual estimada).",
            "text_en": f"GDP nowcast {period}: {val:+.2f}% (estimated year-on-year).",
        })

    # ── Inflation ─────────────────────────────────────────────────────────────
    if infl and infl.get("nowcast"):
        nw = infl["nowcast"]
        val = nw.get("value", 0.0)
        highlights.append({
            "icon": "💹",
            "text_es": f"Nowcast inflación mensual: {val:+.2f}% (IPC mensual estimado).",
            "text_en": f"Monthly inflation nowcast: {val:+.2f}% (estimated monthly CPI).",
        })

    # ── Political risk ────────────────────────────────────────────────────────
    if pol and pol.get("current"):
        cur = pol["current"]
        score = cur.get("score", 0.0)
        level = cur.get("level", "MODERADO")
        level_map_es = {
            "MINIMO": "mínimo", "BAJO": "bajo", "MODERADO": "moderado",
            "ELEVADO": "elevado", "ALTO": "alto", "CRITICO": "crítico",
        }
        level_map_en = {
            "MINIMO": "minimal", "BAJO": "low", "MODERADO": "moderate",
            "ELEVADO": "elevated", "ALTO": "high", "CRITICO": "critical",
        }
        highlights.append({
            "icon": "🏛️",
            "text_es": f"Riesgo político: PRR {round(score)} — nivel {level_map_es.get(level, level.lower())}.",
            "text_en": f"Political risk: PRR {round(score)} — {level_map_en.get(level, level.lower())} level.",
        })

    # ── Headline (from top highlight) ─────────────────────────────────────────
    if prices and prices.get("latest"):
        cum = prices["latest"].get("cum_pct", 0.0)
        headline_es = f"Precios supermercado {'+' if cum >= 0 else ''}{cum:.2f}% MTD · Nowcast PBI {gdp['nowcast']['value']:+.2f}%" if gdp else f"Precios supermercado {'+' if cum >= 0 else ''}{cum:.2f}% acumulado del mes"
        headline_en = f"Supermarket prices {'+' if cum >= 0 else ''}{cum:.2f}% MTD · GDP nowcast {gdp['nowcast']['value']:+.2f}%" if gdp else f"Supermarket prices {'+' if cum >= 0 else ''}{cum:.2f}% month-to-date"
    elif gdp:
        val = gdp["nowcast"]["value"]
        period = gdp["nowcast"].get("target_period", "")
        headline_es = f"Nowcast PBI {period}: {val:+.2f}% variación anual estimada"
        headline_en = f"GDP nowcast {period}: {val:+.2f}% estimated year-on-year"
    else:
        headline_es = "Actualización diaria — Qhawarina"
        headline_en = "Daily update — Qhawarina"

    body_es = "Datos de alta frecuencia actualizados. Revisa las páginas de estadísticas para el detalle completo."
    body_en = "High-frequency data updated. Visit the statistics pages for full detail."

    output = {
        "date": today,
        "headline_es": headline_es,
        "headline_en": headline_en,
        "body_es": body_es,
        "body_en": body_en,
        "highlights": highlights,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = DATA_DIR / "nota_diaria.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"[nota_diaria] Written → {out_path.name} ({len(highlights)} highlights)")


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
    export_inflation_regional_nowcast()  # Add regional inflation disaggregation
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

    # Build + export supermarket daily price index
    logger.info("\nBuilding supermarket price index...")
    export_supermarket_daily_index()

    # Export BCRP financial market data (interventions, FX, bonds, BVL)
    logger.info("\nExporting BCRP financial markets data...")
    export_bcrp_financial_markets()

    # Export GDP sectoral contributions
    logger.info("\nExporting GDP sectoral contributions...")
    export_gdp_contributions()

    # Export backtest results
    logger.info("\nExporting backtest results...")
    export_backtest_results()

    # Generate daily brief (nota_diaria.json) from exported JSONs
    logger.info("\nGenerating nota diaria...")
    generate_nota_diaria()

    logger.info("\n" + "=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info(f"Files generated: {len(list(DATA_DIR.glob('*')))}")
    logger.info("=" * 60)

    # ── Auto-sync to web repo (local runs) ───────────────────────────────────
    import shutil
    WEB_DATA_DIR = Path("D:/qhawarina/public/assets/data")
    if WEB_DATA_DIR.exists():
        copied = []
        for src in DATA_DIR.glob("*.json"):
            dst = WEB_DATA_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)
        logger.info(f"[sync] Copied {len(copied)} JSON files → {WEB_DATA_DIR}")
        csv_src = DATA_DIR / "csv"
        csv_dst = WEB_DATA_DIR / "csv"
        if csv_src.exists():
            csv_dst.mkdir(exist_ok=True)
            csv_files = list(csv_src.glob("*.csv"))
            for f in csv_files:
                shutil.copy2(f, csv_dst / f.name)
            logger.info(f"[sync] Copied {len(csv_files)} CSV files → {csv_dst}")
    else:
        logger.info(f"[sync] Web repo not found at {WEB_DATA_DIR} — skipping auto-copy")


def main_daily():
    """Lightweight daily-only export: political index + FX interventions + price index.
    Skips DFM nowcast regeneration (GDP/inflation/poverty models).
    Runs in ~1-2 minutes vs ~10-15 for full main().
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", action="store_true", help="Run daily-only fast export")
    args, _ = parser.parse_known_args()

    if args.daily:
        logger.info("=" * 60)
        logger.info("QHAWARINA DAILY EXPORT (fast mode)")
        logger.info("=" * 60)

        # Load only political index — GDP/inflation/poverty need gitignored panel data
        political_path = PROCESSED_DAILY_DIR / "daily_index.parquet"
        political_df = pd.read_parquet(political_path).sort_values("date")
        latest_political = political_df.iloc[-1]

        logger.info("Exporting political index...")
        export_political_index(political_df, latest_political)

        logger.info("Exporting BCRP financial markets (FX + bonds + BVL)...")
        export_bcrp_financial_markets()

        logger.info("Building supermarket price index...")
        export_supermarket_daily_index()

        logger.info("Generating nota diaria...")
        generate_nota_diaria()

        logger.info("=" * 60)
        logger.info("DAILY EXPORT COMPLETE")
        logger.info("=" * 60)

        # Auto-sync to web repo
        import shutil
        WEB_DATA_DIR = Path("D:/qhawarina/public/assets/data")
        if WEB_DATA_DIR.exists():
            copied = []
            for src in DATA_DIR.glob("*.json"):
                dst = WEB_DATA_DIR / src.name
                shutil.copy2(src, dst)
                copied.append(src.name)
            logger.info(f"[sync] Copied {len(copied)} JSON files → {WEB_DATA_DIR}")
            csv_src = DATA_DIR / "csv"
            csv_dst = WEB_DATA_DIR / "csv"
            if csv_src.exists():
                csv_dst.mkdir(exist_ok=True)
                csv_files = list(csv_src.glob("*.csv"))
                for f in csv_files:
                    shutil.copy2(f, csv_dst / f.name)
                logger.info(f"[sync] Copied {len(csv_files)} CSV files → {csv_dst}")
    else:
        main()


if __name__ == "__main__":
    main_daily()
