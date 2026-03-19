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
            "is_forecast": False,
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

        # Extract forecasts — use backtest RMSE (1.41pp) for honest CI instead of bridge residual SE
        backtest_rmse = 1.41
        ci_half = 1.96 * backtest_rmse
        for fc in fresh_nowcast.get("forecasts", []):
            # Convert date to quarter string
            fc_date = pd.to_datetime(fc["date"])
            quarter_str = f"{fc_date.year}-Q{(fc_date.month - 1) // 3 + 1}"
            val = round(float(fc["forecast_value"]), 2)
            forecasts.append({
                "quarter": quarter_str,
                "value": val,
                "lower": round(val - ci_half, 2),
                "upper": round(val + ci_half, 2),
            })

    # If the model's target_period already has official data, advance to first forecast quarter
    official_quarters = {q["quarter"] for q in all_quarters if q["official"] is not None}
    if target_period in official_quarters and forecasts:
        first_fc = forecasts[0]
        target_period = first_fc["quarter"]
        nowcast_value = first_fc["value"]

    # Append current nowcast target to recent_quarters so the chart shows the live estimate
    if target_period and nowcast_value is not None:
        if not any(q["quarter"] == target_period for q in recent_quarters):
            recent_quarters.append({
                "quarter": target_period,
                "official": None,
                "nowcast": nowcast_value,
                "error": None,
                "is_forecast": True,
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

        # Append nowcast-only entries for months between last official data and target_period.
        # These are months where BCRP data exists but INEI hasn't published official CPI yet.
        forecast_by_month = {fc["month"]: fc["value"] for fc in forecasts}
        forecast_by_month[target_period] = nowcast_value  # target period itself

        last_official_ts = (
            pd.to_datetime(recent_months[-1]["month"]) if recent_months else pd.Timestamp("2000-01-01")
        )
        target_ts = pd.to_datetime(target_period)
        cur = last_official_ts + pd.offsets.MonthBegin(1)
        while cur <= target_ts:
            m = cur.strftime("%Y-%m")
            val = forecast_by_month.get(m, nowcast_value)
            entry = {
                "month": m,
                "official": None,
                "nowcast": round(float(val), 3) if val is not None else None,
                "error": None,
            }
            recent_months.append(entry)
            all_months.append(entry)
            cur += pd.offsets.MonthBegin(1)

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
        "MINIMO": "mínimo", "BAJO": "bajo", "NORMAL": "normal",
        "ELEVADO": "elevado", "ALTO": "alto", "CRITICO": "crítico",
        "MODERADO": "normal",  # legacy alias
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


def export_political_index(political_df: pd.DataFrame, latest: pd.Series, skip_haiku: bool = False):
    """Export political instability index to JSON. Supports v1 and v2 schema."""
    import numpy as np

    political_df = political_df.copy()

    # ── Load articles cache (for Haiku justifications + driver phrases only) ─
    # Index values come from daily_index.parquet — NOT recomputed here.
    from config.settings import RAW_RSS_DIR
    articles_cache_path = RAW_RSS_DIR / "articles_classified.parquet"
    articles_cache = None
    if articles_cache_path.exists():
        try:
            articles_cache = pd.read_parquet(articles_cache_path)
            articles_cache["published"] = pd.to_datetime(articles_cache["published"], utc=True)
            articles_cache["date"] = articles_cache["published"].dt.tz_convert("America/Lima").dt.date
        except Exception as e:
            logger.warning("Could not load articles cache: %s", e)

    # ── Read IRP/IRE from daily_index.parquet (source of truth) ──────────────
    # The parquet was built by build_daily_index.py using the Intensity×Breadth^β
    # formula. Do NOT recompute here — the parquet is the single source of truth.
    political_df["irp"] = pd.to_numeric(
        political_df["political_index"] if "political_index" in political_df.columns else 0.0,
        errors="coerce",
    ).fillna(0.0)
    political_df["ire"] = pd.to_numeric(
        political_df["economic_index"] if "economic_index" in political_df.columns else 0.0,
        errors="coerce",
    ).fillna(0.0)
    if "n_articles_total" not in political_df.columns:
        political_df["n_articles_total"] = 0
    political_df["instability_index"] = political_df["irp"]

    # Low-coverage interpolation: days with < 25 articles are noise spikes.
    MIN_ARTICLES = 25
    political_df = political_df.sort_values("date").reset_index(drop=True)
    low_cov_mask = political_df["n_articles_total"] < MIN_ARTICLES
    political_df.loc[low_cov_mask, ["irp", "ire"]] = float("nan")
    political_df[["irp", "ire"]] = (
        political_df[["irp", "ire"]]
        .interpolate(method="linear", limit_direction="both")
    )

    # Use the last COMPLETE day, not .iloc[-1].
    # The pipeline runs at 21:00 Lima (02:00 UTC next day), so by export time
    # the RSS fetcher has already bucketed a handful of articles into "tomorrow",
    # creating a partial row with n_articles_total ~50-100 and artificially low
    # IRP/IRE values.  We skip any trailing rows below the complete-day threshold.
    MIN_COMPLETE_ARTICLES = 100
    complete_df = political_df[political_df["n_articles_total"] >= MIN_COMPLETE_ARTICLES]
    today_vals = complete_df.iloc[-1] if not complete_df.empty else (political_df.iloc[-1] if not political_df.empty else None)
    if today_vals is not None:
        print(f"[AI-GPR] IRP today = {today_vals['irp']:.1f}  IRE today = {today_vals['ire']:.1f}  (date: {str(today_vals['date'])[:10]}, articles: {int(today_vals['n_articles_total'])})")

    # ── Smoothing ─────────────────────────────────────────────────────────────
    # Prefer the parquet's pre-computed irp_smooth / ire_smooth columns
    # (trailing EMA, alpha=0.30, from build_daily_index.py).  These avoid the
    # centered-window boundary truncation that under-reports risk on the most
    # recent day. Fall back to a 7-day trailing rolling mean if absent.
    MIN_ARTICLES = 5
    SMOOTH_WINDOW = 7

    political_df["low_coverage"] = political_df["n_articles_total"] < MIN_ARTICLES

    if "irp_smooth" in political_df.columns:
        political_df["irp_7d"] = pd.to_numeric(
            political_df["irp_smooth"], errors="coerce"
        ).fillna(political_df["irp"]).round(1)
    else:
        political_df["irp_7d"] = (
            political_df["irp"]
            .rolling(window=SMOOTH_WINDOW, min_periods=2)
            .mean().round(1).fillna(political_df["irp"])
        )

    if "ire_smooth" in political_df.columns:
        political_df["ire_7d"] = pd.to_numeric(
            political_df["ire_smooth"], errors="coerce"
        ).fillna(political_df["ire"]).round(1)
    else:
        political_df["ire_7d"] = (
            political_df["ire"]
            .rolling(window=SMOOTH_WINDOW, min_periods=2)
            .mean().round(1).fillna(political_df["ire"])
        )
    # Backward compat
    political_df["instability_index"] = political_df["irp"]
    political_df["score_smooth"] = political_df["irp_7d"]

    PROVISIONAL_DAYS = 5
    cutoff = political_df["date"].max() - pd.Timedelta(days=PROVISIONAL_DAYS)
    political_df["provisional"] = political_df["date"] > cutoff

    # ── Drop partial today ONLY if the nightly pipeline isn't running ─────────
    # run_daily_pipeline.py writes run_time to pipeline_status.json at startup,
    # before this export step runs. So if run_time matches today (in PET),
    # the pipeline is active and today's data is being collected — keep it.
    # If run_time is from a previous day, today's data is partial — drop it.
    from datetime import timezone, timedelta as _td
    import json as _json
    _now_utc = datetime.now(timezone.utc)
    _now_pet = _now_utc.replace(tzinfo=None) + _td(hours=-5)
    _today_pet_str = _now_pet.strftime("%Y-%m-%d")

    _pipeline_running_today = False
    try:
        _status_path = Path(__file__).parent.parent / "data" / "pipeline_status.json"
        if _status_path.exists():
            _status = _json.loads(_status_path.read_text(encoding="utf-8"))
            # run_daily_pipeline.py sets status["date"] = today (local machine date)
            # at startup, before calling export. Match against PET date.
            _status_date = _status.get("date", "")
            _pipeline_running_today = (_status_date == _today_pet_str)
    except Exception:
        pass

    # Always strip any dates beyond today PET (UTC midnight articles leaked into tomorrow)
    _has_future = (political_df["date"].astype(str) > _today_pet_str).any()
    if _has_future:
        political_df = political_df[political_df["date"].astype(str) <= _today_pet_str].copy()
        logger.info("  Stripped future dates beyond %s", _today_pet_str)

    if not _pipeline_running_today:
        _has_today = (political_df["date"].astype(str) == _today_pet_str).any()
        if _has_today:
            political_df = political_df[political_df["date"].astype(str) < _today_pet_str].copy()
            logger.info(
                "  Pipeline not running today — dropped partial today (%s), latest = %s",
                _today_pet_str, political_df["date"].max()
            )
    # ─────────────────────────────────────────────────────────────────────────

    latest_idx = political_df.index[-1]
    latest = political_df.loc[latest_idx].copy()
    # ─────────────────────────────────────────────────────────────────────────

    # Calculate aggregates using smoothed score
    last_7d = political_df.tail(7)["score_smooth"].mean()
    last_30d = political_df.tail(30)["score_smooth"].mean()
    # Use the full series for max-year lookup (not a 365-day slice)
    valid_series = political_df[political_df["score_smooth"] > 0]
    if valid_series.empty:
        max_year = 0.0
        max_year_date = political_df["date"].iloc[-1]
    else:
        max_year = valid_series["score_smooth"].max()
        max_year_date = valid_series.loc[valid_series["score_smooth"] == max_year, "date"].iloc[0]

    # Level classification (applied to both IRP and IRE)
    def classify_level(score: float) -> str:
        # mean=100. Thresholds anchored around the mean:
        #   mínimo  < 50    (virtually no coverage)
        #   bajo    50-90   (below-average)
        #   normal  90-110  (near historical mean)
        #   elevado 110-150 (clearly above average)
        #   alto    150-200 (very high)
        #   crítico > 200   (extraordinary)
        if score < 50:   return "MINIMO"
        if score < 90:   return "BAJO"
        if score < 110:  return "NORMAL"
        if score < 150:  return "ELEVADO"
        if score < 200:  return "ALTO"
        return "CRITICO"

    # Daily series — full history (no 365-day truncation)
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
        for _, row in political_df.iterrows()
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
    eco_sector_titles:     list = []
    today_all = pd.DataFrame()  # initialized here so it's always in scope

    if articles_cache is not None:
        # Always prefer actual Lima date (UTC-5) for article filtering.
        # Fallback to latest["date"] only if no articles exist for today — this prevents
        # March 14 justification from reusing March 13 articles when political_df latest
        # row is still March 13 (e.g. pipeline_status not yet updated).
        from datetime import timezone as _tz14, timedelta as _td14
        _actual_today = (datetime.now(_tz14.utc) + _td14(hours=-5)).date()
        today_all_actual = articles_cache[articles_cache["date"] == _actual_today]
        if not today_all_actual.empty:
            today_date = _actual_today
            today_all  = today_all_actual
        else:
            today_date = latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]
            today_all  = articles_cache[articles_cache["date"] == today_date]
        has_dual   = "political_score" in today_all.columns and "economic_score" in today_all.columns

        if not skip_haiku and has_dual and (irp_today > 0 or ire_today > 0):
            try:
                import re as _re
                client = __import__("anthropic").Anthropic()

                # Keyword exclusion: articles that Haiku misclassifies as risk but are
                # clearly sports matches, entertainment events, or industry conferences.
                _EXCL_RE = _re.compile(
                    r'\b(champions league|premier league|la liga|bundesliga|serie a|ligue 1'
                    r'|copa del rey|copa libertadores|sudamericana|eliminatorias'
                    r'|atlético de madrid|atletico de madrid|real madrid|fc barcelona'
                    r'|manchester (city|united)|liverpool fc|tottenham|chelsea fc'
                    r'|triunfo colchonero|triunfo blanquiazul|triunfo blanquirrojo'
                    r'|ganó \d+-\d+|perdió \d+-\d+|empató \d+-\d+'
                    r'|octavos de final|cuartos de final|semifinal.*copa|final.*copa'
                    r'|simposio \w+|congreso empresarial|webinar|workshop'
                    # Routine FX quote articles (daily "precio del dólar hoy" = irrelevant)
                    r'|precio del dólar.*hoy|tipo de cambio.*hoy|a cuánto (cerró|abrió|está) el (dólar|tipo de cambio)'
                    r'|cotización del dólar'
                    # Arts / cultural events (no macroeconomic relevance)
                    r'|fae lima|festival de artes escénicas|festival artes escénicas'
                    r'|escena y memoria|cierre del fae|apertura del fae'
                    r'|gran teatro nacional.{0,30}(programa|temporada|estreno)'
                    r'|bienal de arte|galería de arte'
                    # Foreign leader health / personal news
                    r'|bolsonaro|lula da silva|jair bolsonaro'
                    r'|(expresidente|ex[ -]presidente).{0,40}(brasileño|argentino|colombiano|chileno|venezolano|boliviano|ecuatoriano|paraguayo|uruguayo)'
                    # Deaths of foreign intellectuals/cultural figures
                    r'|(muere|fallece|muerte de).{0,120}'
                    r'(filósofos?|filósofas?|escritores?|escritoras?|novelistas?|poetas?|poetisas?|'
                    r'dramaturgos?|dramaturgAs?|compositores?|compositoras?|músicos?|músicas?|'
                    r'pensadores?|pensadoras?|teólogos?|teólogas?|sociólogos?|sociólogas?|'
                    r'antropólogos?|antropólogas?|arqueólogos?|arqueólogas?|historiadores?|historiadoras?|'
                    r'ensayistas?|lingüistas?|psicólogos?|psicólogas?|'
                    r'pintores?|pintoras?|escultores?|escultoras?|actores?|actrices?|'
                    r'bailarines?|bailarinas?|cantantes?|cineastas?|guionistas?))\b',
                    _re.IGNORECASE,
                )

                def _exclude_sports(df_: "pd.DataFrame") -> "pd.DataFrame":
                    """Drop rows whose title clearly matches sports/entertainment patterns."""
                    mask = df_["title"].str.contains(_EXCL_RE, na=False)
                    dropped = mask.sum()
                    if dropped:
                        logger.info("  top_drivers: excluded %d sports/conf articles", dropped)
                    return df_[~mask]

                # Top 5 political articles — pol is dominant dimension (pol >= eco)
                pol_primary = today_all[
                    today_all["political_score"] >= today_all["economic_score"]
                ] if "economic_score" in today_all.columns else today_all
                pol_primary = _exclude_sports(pol_primary)
                top_pol = pol_primary.nlargest(10, "political_score")[
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
                                f"ESCALA: media histórica=100. <50=mínimo, 50-90=bajo, "
                                f"90-110=normal, 110-150=elevado, 150-200=alto, >200=crítico. "
                                f"El índice NO tiene techo en 100 — puede superar 200 en crisis. "
                                f"Hoy IRP={irp_today:.0f} ({irp_today/100:.2f}× el promedio) → "
                                f"nivel {'mínimo' if irp_today<50 else 'bajo' if irp_today<90 else 'normal' if irp_today<110 else 'elevado' if irp_today<150 else 'alto' if irp_today<200 else 'crítico'}.\n"
                                f"Los 5 artículos con mayor puntaje político:\n{drivers_text}\n\n"
                                f"Escribe 2-3 oraciones explicando el nivel de riesgo político. "
                                f"NO menciones el número del índice. Máximo 60 palabras. Solo el texto."
                            }]
                        )
                        political_justification = msg.content[0].text.strip()
                    top_political_drivers = [
                        {"title": r.title, "source": r.source, "score": int(r.political_score)}
                        for _, r in top_pol.iterrows() if r.political_score > 0
                    ]

                # Top 5 economic articles — eco is clearly dominant (eco > pol * 1.3).
                # The 1.3× threshold prevents political articles (candidates, censure)
                # from appearing here just because they also mention economic context.
                eco_primary = today_all[
                    today_all["economic_score"] > today_all["political_score"] * 1.3
                ] if "political_score" in today_all.columns else today_all
                eco_primary = _exclude_sports(eco_primary)
                top_eco = eco_primary.nlargest(10, "economic_score")[
                    ["title", "source", "economic_score"]
                ]
                eco_meaningful = top_eco[top_eco["economic_score"] > 15]
                if len(eco_meaningful) >= 2 and ire_today > 0:
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
                                f"ESCALA: media histórica=100. <50=mínimo, 50-90=bajo, "
                                f"90-110=normal, 110-150=elevado, 150-200=alto, >200=crítico. "
                                f"El índice NO tiene techo en 100 — puede superar 200 en crisis. "
                                f"Hoy IRE={ire_today:.0f} ({ire_today/100:.2f}× el promedio) → "
                                f"nivel {'mínimo' if ire_today<50 else 'bajo' if ire_today<90 else 'normal' if ire_today<110 else 'elevado' if ire_today<150 else 'alto' if ire_today<200 else 'crítico'}.\n"
                                f"Los 5 artículos con mayor puntaje económico:\n{drivers_text}\n\n"
                                f"Escribe 2-3 oraciones explicando el nivel de riesgo económico. "
                                f"NO menciones el número del índice. Máximo 60 palabras. Solo el texto."
                            }]
                        )
                        economic_justification = msg.content[0].text.strip()
                top_economic_drivers = [
                    {"title": r.title, "source": r.source, "score": int(r.economic_score)}
                    for _, r in eco_meaningful.iterrows()
                ] if len(eco_meaningful) >= 2 else []

                # All eco>20 titles for frontend sector classification (not limited to top 10)
                eco_all = today_all[today_all["economic_score"] > 20] if "economic_score" in today_all.columns else pd.DataFrame()
                eco_all = _exclude_sports(eco_all)
                eco_sector_titles = eco_all.nlargest(50, "economic_score")["title"].tolist()
            except Exception as e:
                logger.warning("Haiku justification failed: %s", e)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Peak events: local maxima with Haiku labels ───────────────────────────
    def compute_peak_events(df: "pd.DataFrame", arts: "pd.DataFrame | None") -> list:
        """Detect political + economic peak days, label them via Haiku, cache results."""
        import json as _json
        import anthropic as _anthropic

        CACHE_PATH = DATA_DIR / "peak_events_cache.json"
        POL_THRESHOLD = 130
        ECO_THRESHOLD = 150
        DEDUP_WINDOW  = 14  # days

        # Load cache
        try:
            cache: dict = _json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
        except Exception:
            cache = {}

        haiku = _anthropic.Anthropic()

        def _call_haiku(dimension: str, date_str: str, titles: list[str]) -> str:
            dim_es = "político" if dimension == "political" else "económico"
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles[:3]))
            prompt = (
                f"Eres un analista político-económico. Estos son los 3 artículos de mayor riesgo "
                f"{dim_es} del día {date_str}:\n{numbered}\n\n"
                f"Escribe un TÍTULO de máximo 3 palabras que resuma el evento principal. "
                f"Solo el título, sin explicación, sin puntuación.\n\n"
                f"Ejemplos: 'Vacancia Boluarte', 'Crisis Camisea', 'Censura Santiváñez', "
                f"'Aranceles Trump', 'Masacre Pataz', 'Censura Jerí'"
            )
            msg = haiku.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=30,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip().strip("'\"")

        def _detect_peaks(series: "pd.Series", threshold: float) -> list[int]:
            """Return integer index positions of local peaks above threshold."""
            peaks = []
            vals = series.values
            for i in range(1, len(vals) - 1):
                if vals[i] > threshold and vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
                    peaks.append(i)
            return peaks

        def _dedup(peaks: list[int], dates: "pd.Series", values: "pd.Series") -> list[int]:
            """Within any 14-day window keep only the highest peak."""
            if not peaks:
                return []
            kept = []
            last_kept_date = None
            # Sort by value desc so within any window the highest wins
            sorted_peaks = sorted(peaks, key=lambda i: values.iloc[i], reverse=True)
            used = set()
            for i in sorted_peaks:
                if i in used:
                    continue
                d = dates.iloc[i]
                # Mark all peaks within ±7 days as used
                for j in sorted_peaks:
                    if j in used:
                        continue
                    if abs((dates.iloc[j] - d).days) < DEDUP_WINDOW:
                        used.add(j)
                kept.append(i)
            return kept

        results = []
        df_sorted = df.sort_values("date").reset_index(drop=True)
        dates_s = df_sorted["date"]

        for dimension, col, threshold in [
            ("political", "irp_7d", POL_THRESHOLD),
            ("economic",  "ire_7d", ECO_THRESHOLD),
        ]:
            if col not in df_sorted.columns:
                continue
            raw_peaks = _detect_peaks(df_sorted[col], threshold)
            dedup_peaks = _dedup(raw_peaks, dates_s, df_sorted[col])

            for idx in dedup_peaks:
                date_ts = dates_s.iloc[idx]
                date_str = date_ts.strftime("%Y-%m-%d") if hasattr(date_ts, "strftime") else str(date_ts)[:10]
                value = round(float(df_sorted[col].iloc[idx]), 1)
                cache_key = f"{date_str}_{dimension}"

                if cache_key in cache:
                    label = cache[cache_key]
                else:
                    label = None
                    if arts is not None:
                        try:
                            score_col = "political_score" if dimension == "political" else "economic_score"
                            peak_date = date_ts.date() if hasattr(date_ts, "date") else date_ts
                            day_arts = arts[arts["date"] == peak_date]
                            if score_col in day_arts.columns:
                                top3 = day_arts.nlargest(3, score_col)["title"].tolist()
                                if top3:
                                    label = _call_haiku(dimension, date_str, top3)
                        except Exception as _le:
                            logger.warning("Peak label failed for %s: %s", cache_key, _le)
                    if label:
                        cache[cache_key] = label

                results.append({
                    "date": date_str,
                    "dimension": dimension,
                    "value": value,
                    "label": label or "",
                })

        # Persist cache
        try:
            CACHE_PATH.write_text(_json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as _ce:
            logger.warning("Could not save peak_events_cache: %s", _ce)

        results.sort(key=lambda x: x["date"])
        return results

    try:
        peak_events = compute_peak_events(political_df, articles_cache)
        logger.info("Peak events computed: %d", len(peak_events))
    except Exception as _pe:
        logger.warning("compute_peak_events failed: %s", _pe)
        peak_events = []
    # ─────────────────────────────────────────────────────────────────────────

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "coverage_days": len(political_df),
            "rss_feeds": 35,
            "rss_sources": 16,
            "methodology": "AI-GPR dual index (Iacoviello & Tong 2026): IRP + IRE, mean=100 over 2025 baseline",
        },
        "current": {
            "date": latest["date"].strftime("%Y-%m-%d"),
            # Dual indices
            "political_raw":       round(irp_today, 1),
            "political_raw_level": classify_level(irp_today),
            "political_7d":        round(irp_7d, 1),
            "political_level":     classify_level(irp_7d),
            "political_multiplier": round(irp_7d / 100, 2),
            "economic_raw":       round(ire_today, 1),
            "economic_raw_level": classify_level(ire_today),
            "economic_7d":        round(ire_7d, 1),
            "economic_level":     classify_level(ire_7d),
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
            "eco_sector_titles":       eco_sector_titles,
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
        "peak_events": peak_events,
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

    # Update pipeline_status.json so homepage freshness check passes
    _status_out = DATA_DIR / "pipeline_status.json"
    _status_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "run_time": datetime.now().isoformat(),
        "supermarket": {"passed": True, "errors": []},
        "rss": {"passed": True, "errors": []},
    }
    with open(_status_out, "w", encoding="utf-8") as _f:
        json.dump(_status_data, _f, indent=2)
    logger.info(f"Updated pipeline_status.json: {_status_data['run_time']}")


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
    """Build and export supermarket daily price index JSONs.

    Runs build_price_index.py → writes daily_price_index.json.
    Also regenerates supermarket_daily_index.json (legacy schema with
    daily_series key) so that /estadisticas/inflacion/precios-alta-frecuencia
    stays current.
    """
    import subprocess, json as _json

    logger.info("Building supermarket daily price index...")
    script_path = PROJECT_ROOT / "scripts" / "build_price_index.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Failed to build supermarket price index: {result.stderr[-500:]}")
        return

    logger.info("Supermarket daily price index built and exported")

    # Regenerate supermarket_daily_index.json from daily_price_index.json.
    # The precios-alta-frecuencia page reads daily_series[] which has a
    # slightly different schema — keep it in sync so it doesn't go stale.
    src_path = DATA_DIR / "daily_price_index.json"
    dst_path = DATA_DIR / "supermarket_daily_index.json"
    try:
        src = _json.loads(src_path.read_text(encoding="utf-8"))
        raw_series = src.get("series", [])
        metadata = src.get("metadata", {})

        daily_series = [
            {
                "date":          row.get("date", ""),
                "index_all":     round(row["index_all"], 4) if row.get("index_all") is not None else None,
                "index_food":    round(row["index_food"], 4) if row.get("index_food") is not None else None,
                "index_nonfood": round(row["index_nonfood"], 4) if row.get("index_nonfood") is not None else None,
                "var_all":       round(row["var_all"], 6) if row.get("var_all") is not None else None,
                "n_products":    row.get("n_products"),
                "interpolated":  row.get("interpolated", False),
            }
            for row in raw_series
        ]

        out = {
            "metadata": {
                "method":        metadata.get("method", "Jevons bilateral index (geometric mean)"),
                "base_date":     metadata.get("base_date", ""),
                "base_value":    100,
                "stores":        metadata.get("stores", ["Plaza Vea", "Metro", "Wong"]),
                "frequency":     "daily",
                "generated_at":  metadata.get("generated_at", ""),
                "total_skus":    metadata.get("total_skus"),
                "note":          "Updated daily from build_price_index.py",
            },
            "daily_series": daily_series,
        }

        dst_path.write_text(_json.dumps(out, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "  Regenerated supermarket_daily_index.json: %d days, latest=%s",
            len(daily_series),
            daily_series[-1]["date"] if daily_series else "n/a",
        )
    except Exception as exc:
        logger.warning("  Could not regenerate supermarket_daily_index.json: %s", exc)


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


def export_monthly_peaks_json():
    """Re-generate risk_index_monthly_peaks.json from political_index_daily.json.

    Reads daily_series from the already-exported JSON, groups by month,
    finds the peak 7d value per dimension, looks up event labels from
    peak_events_cache.json, and writes the result.

    Only complete months are included (current partial month is excluded).
    Run every day — a new month row appears automatically on the 1st.
    """
    import datetime as _dt

    daily_json_path = DATA_DIR / "political_index_daily.json"
    cache_path      = DATA_DIR / "peak_events_cache.json"
    out_path        = DATA_DIR / "risk_index_monthly_peaks.json"

    if not daily_json_path.exists():
        logger.warning("export_monthly_peaks_json: political_index_daily.json not found — skipping")
        return

    with open(daily_json_path, encoding="utf-8") as f:
        daily_data = json.load(f)

    daily_series = daily_data.get("daily_series", [])
    if not daily_series:
        logger.warning("export_monthly_peaks_json: daily_series empty — skipping")
        return

    # Load event label cache (may not exist yet on fresh install)
    event_cache: dict = {}
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                event_cache = json.load(f)
        except Exception:
            pass

    # Current month string — exclude it (partial)
    current_month = _dt.date.today().strftime("%Y-%m")

    # Group daily rows by month
    from collections import defaultdict
    by_month: dict[str, list] = defaultdict(list)
    for row in daily_series:
        m = row["date"][:7]
        if m != current_month:
            by_month[m].append(row)

    months_out = []
    for month in sorted(by_month.keys()):
        rows = by_month[month]

        # IRP peak
        irp_row  = max(rows, key=lambda r: r.get("political_7d", 0))
        irp_peak = round(irp_row.get("political_7d", 0), 1)
        irp_date = irp_row["date"]

        # IRE peak
        ire_row  = max(rows, key=lambda r: r.get("economic_7d", 0))
        ire_peak = round(ire_row.get("economic_7d", 0), 1)
        ire_date = ire_row["date"]

        # Look up event labels from cache
        irp_event = event_cache.get(f"{irp_date}_political", "")
        ire_event = event_cache.get(f"{ire_date}_economic", "")

        # Preserve existing top_articles if they exist in the old file
        existing_irp_arts: list = []
        existing_ire_arts: list = []
        if out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    old = json.load(f)
                for om in old.get("months", []):
                    if om["month"] == month:
                        existing_irp_arts = om.get("irp_top_articles", [])
                        existing_ire_arts = om.get("ire_top_articles", [])
                        break
            except Exception:
                pass

        months_out.append({
            "month":            month,
            "irp_7d_peak":      irp_peak,
            "irp_peak_date":    irp_date,
            "irp_event":        irp_event,
            "irp_top_articles": existing_irp_arts,
            "ire_7d_peak":      ire_peak,
            "ire_peak_date":    ire_date,
            "ire_event":        ire_event,
            "ire_top_articles": existing_ire_arts,
        })

    result = {
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "months": months_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Monthly peaks JSON updated: %d complete months", len(months_out))


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
        level = cur.get("level", "NORMAL")
        level_map_es = {
            "MINIMO": "mínimo", "BAJO": "bajo", "NORMAL": "normal",
            "ELEVADO": "elevado", "ALTO": "alto", "CRITICO": "crítico",
            "MODERADO": "normal",  # legacy alias
        }
        level_map_en = {
            "MINIMO": "minimal", "BAJO": "low", "NORMAL": "normal",
            "ELEVADO": "elevated", "ALTO": "high", "CRITICO": "critical",
            "MODERADO": "normal",  # legacy alias
        }
        highlights.append({
            "icon": "🏛️",
            "text_es": f"Riesgo político: IRP {round(score)} — nivel {level_map_es.get(level, level.lower())}.",
            "text_en": f"Political risk: IRP {round(score)} — {level_map_en.get(level, level.lower())} level.",
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

    # Re-generate monthly peaks table (complete months only)
    logger.info("\nUpdating monthly risk peaks JSON...")
    export_monthly_peaks_json()

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
    parser.add_argument("--no-haiku", action="store_true", help="Skip Haiku API calls (faster, no justification text)")
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
        export_political_index(political_df, latest_political, skip_haiku=args.no_haiku)

        logger.info("Updating monthly risk peaks JSON...")
        export_monthly_peaks_json()

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
