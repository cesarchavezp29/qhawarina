"""Generate real-time nowcasts for GDP, inflation, and poverty.

Uses VintageManager with value_col="auto" for correct panel construction:
  - GDP series: picks value_yoy for level/index series, value_raw for rate/variation
  - Inflation series: correctly uses value_raw for CPI variation series
  - Ragged-edge truncation: drops months where <50% of model series have data

Usage:
    python scripts/generate_nowcast.py
    python scripts/generate_nowcast.py --gdp-only
    python scripts/generate_nowcast.py --inflation-only
    python scripts/generate_nowcast.py --poverty-only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CONFIG_DIR,
    PROCESSED_DEPARTMENTAL_DIR,
    PROCESSED_NATIONAL_DIR,
    TARGETS_DIR,
)
from src.backtesting.vintage import VintageManager
from src.models.dfm import (
    AR1Benchmark,
    GDP_SERIES,
    INFLATION_SERIES,
    MLNowcaster,
    NowcastDFM,
    PhillipsCurveNowcaster,
    RandomWalkBenchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.nowcast")


def _truncate_ragged_edge(
    panel_wide: pd.DataFrame,
    model_series: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Truncate panel to the last month with sufficient series coverage.

    For GDP nowcasting, the ragged edge (different publication lags)
    means the latest months may only have a few fast-release series.
    Using these sparse months leads to extreme factor values and
    unreliable nowcasts.

    Parameters
    ----------
    panel_wide : pd.DataFrame
        Wide-format panel (date index x series columns).
    model_series : list[str]
        Series used by the model (used to count coverage).
    threshold : float
        Minimum fraction of model series that must have data for a
        month to be kept (default: 0.5 = 50%).

    Returns
    -------
    pd.DataFrame
        Truncated panel ending at the last month with >= threshold coverage.
    """
    available = [s for s in model_series if s in panel_wide.columns]
    n_model = len(available)
    if n_model == 0:
        return panel_wide

    # Count non-NaN model series per month
    coverage = panel_wide[available].notna().sum(axis=1)
    min_series = max(int(threshold * n_model), 3)

    # Find last month with sufficient coverage
    good_months = coverage[coverage >= min_series]
    if good_months.empty:
        return panel_wide

    last_good = good_months.index.max()
    truncated = panel_wide.loc[panel_wide.index <= last_good]

    n_dropped = len(panel_wide) - len(truncated)
    if n_dropped > 0:
        logger.info(
            "Ragged edge: dropped %d months after %s (coverage < %d/%d series)",
            n_dropped, last_good.strftime("%Y-%m"), min_series, n_model,
        )

    return truncated


def nowcast_gdp(vm: VintageManager, as_of: pd.Timestamp) -> dict:
    """Generate GDP nowcast with ragged-edge truncation.

    Parameters
    ----------
    vm : VintageManager
        Initialized vintage manager.
    as_of : pd.Timestamp
        Evaluation date (today or simulated).

    Returns
    -------
    dict
        nowcast_value, bridge_r2, target_period, n_series, n_months
    """
    panel_wide = vm.get_vintage_wide(as_of, value_col="auto")

    # Truncate ragged edge
    panel_wide = _truncate_ragged_edge(panel_wide, GDP_SERIES, threshold=0.5)

    # Load target
    gdp_df = pd.read_parquet(TARGETS_DIR / "gdp_quarterly.parquet")
    gdp_df["date"] = pd.to_datetime(gdp_df["date"])

    # Filter target to prevent look-ahead (everything before latest quarter)
    latest_date = panel_wide.index.max()
    # Target period: the most recent complete quarter
    quarter_start = latest_date - pd.offsets.QuarterBegin(startingMonth=1)
    target_period = pd.Timestamp(quarter_start.year, quarter_start.month, 1)
    target_available = gdp_df[gdp_df["date"] < target_period].copy()

    # Fit and nowcast — Ridge bridge + rolling window + COVID exclusion
    dfm = NowcastDFM(
        k_factors=3, factor_order=1, target="gdp",
        bridge_method="ridge", bridge_alpha=1.0,
        rolling_window_years=7,
        exclude_covid=True,
    )
    dfm.fit(panel_wide)
    nc = dfm.nowcast(panel_wide, target_available)

    # Generate forecasts (6 quarters ahead)
    try:
        forecasts = dfm.forecast(panel_wide, target_available, horizons=6)
        forecast_list = forecasts.to_dict('records') if not forecasts.empty else []
    except Exception as e:
        logger.warning("GDP forecast failed: %s", e)
        forecast_list = []

    # Also get benchmark
    ar1 = AR1Benchmark(target="gdp")
    ar1_nc = ar1.nowcast(panel_wide, target_available)
    rw = RandomWalkBenchmark(target="gdp")
    rw_nc = rw.nowcast(panel_wide, target_available)

    # ML nowcasts (BCRP DT 003-2024 methods)
    ml_results = {}
    for method in ("lasso", "elastic_net", "gbm"):
        try:
            ml = MLNowcaster(
                method=method, target="gdp",
                rolling_window_years=7,
                include_target_ar=True,
            )
            ml.fit(panel_wide)
            ml_nc = ml.nowcast(panel_wide, target_available)
            ml_results[method] = ml_nc
        except Exception as e:
            logger.warning("ML %s GDP failed: %s", method, e)
            ml_results[method] = {"nowcast_value": np.nan}

    n_series = len([s for s in GDP_SERIES if s in panel_wide.columns])

    result = {
        "target": "GDP YoY (%)",
        "target_period": f"{target_period.year}-Q{(target_period.month - 1) // 3 + 1}",
        "nowcast_value": nc.get("nowcast_value", np.nan),
        "bridge_r2": nc.get("bridge_r2", np.nan),
        "ar1_nowcast": ar1_nc.get("nowcast_value", np.nan),
        "rw_nowcast": rw_nc.get("nowcast_value", np.nan),
        "forecasts": forecast_list,
        "n_series": n_series,
        "n_months": len(panel_wide),
        "panel_end": panel_wide.index.max().strftime("%Y-%m"),
    }
    for method, ml_nc in ml_results.items():
        result[f"ml_{method}"] = ml_nc.get("nowcast_value", np.nan)
        result[f"ml_{method}_cv_rmse"] = ml_nc.get("cv_rmse", np.nan)

    return result


def nowcast_inflation(vm: VintageManager, as_of: pd.Timestamp) -> dict:
    """Generate inflation nowcast using VintageManager auto panel.

    Parameters
    ----------
    vm : VintageManager
        Initialized vintage manager.
    as_of : pd.Timestamp
        Evaluation date.

    Returns
    -------
    dict
        nowcast_value, bridge_r2, target_period, n_series, n_months
    """
    panel_wide = vm.get_vintage_wide(as_of, value_col="auto")

    # Load target
    inf_df = pd.read_parquet(TARGETS_DIR / "inflation_monthly.parquet")
    inf_df["date"] = pd.to_datetime(inf_df["date"])

    # Target period: previous month
    target_period = as_of - pd.offsets.MonthBegin(1)
    target_available = inf_df[inf_df["date"] < target_period].copy()

    # Fit and nowcast — DFM with lagged factors + AR(1)
    # Using ipc_3m_ma (3-month moving average) for smoother underlying inflation
    dfm = NowcastDFM(
        k_factors=2, factor_order=1, target="inflation",
        inflation_col="ipc_3m_ma",
        include_factor_lags=1,
        include_target_ar=True,
    )
    dfm.fit(panel_wide)
    nc = dfm.nowcast(panel_wide, target_available)

    # Generate forecasts (6 months ahead)
    try:
        forecasts = dfm.forecast(panel_wide, target_available, horizons=6)
        forecast_list = forecasts.to_dict('records') if not forecasts.empty else []
    except Exception as e:
        logger.warning("Inflation forecast failed: %s", e)
        forecast_list = []

    # Benchmarks
    ar1 = AR1Benchmark(target="inflation", inflation_col="ipc_monthly_var")
    ar1_nc = ar1.nowcast(panel_wide, target_available)
    rw = RandomWalkBenchmark(target="inflation", inflation_col="ipc_monthly_var")
    rw_nc = rw.nowcast(panel_wide, target_available)
    phillips = PhillipsCurveNowcaster(inflation_col="ipc_monthly_var")
    phillips_nc = phillips.nowcast(panel_wide, target_available)

    # ML nowcasts (BCRP DT 003-2024 methods)
    ml_results = {}
    for method in ("lasso", "elastic_net", "gbm"):
        try:
            ml = MLNowcaster(
                method=method, target="inflation",
                inflation_col="ipc_monthly_var",
                include_target_ar=True,
            )
            ml.fit(panel_wide)
            ml_nc = ml.nowcast(panel_wide, target_available)
            ml_results[method] = ml_nc
        except Exception as e:
            logger.warning("ML %s inflation failed: %s", method, e)
            ml_results[method] = {"nowcast_value": np.nan}

    n_series = len([s for s in INFLATION_SERIES if s in panel_wide.columns])

    result = {
        "target": "CPI 3M-MA Var (%)",
        "target_period": target_period.strftime("%Y-%m"),
        "nowcast_value": nc.get("nowcast_value", np.nan),
        "bridge_r2": nc.get("bridge_r2", np.nan),
        "ar1_nowcast": ar1_nc.get("nowcast_value", np.nan),
        "rw_nowcast": rw_nc.get("nowcast_value", np.nan),
        "phillips_nowcast": phillips_nc.get("nowcast_value", np.nan),
        "forecasts": forecast_list,
        "n_series": n_series,
        "n_months": len(panel_wide),
        "panel_end": panel_wide.index.max().strftime("%Y-%m"),
    }
    for method, ml_nc in ml_results.items():
        result[f"ml_{method}"] = ml_nc.get("nowcast_value", np.nan)
        result[f"ml_{method}_cv_rmse"] = ml_nc.get("cv_rmse", np.nan)

    return result


def nowcast_poverty() -> dict:
    """Generate poverty nowcast from departmental panel.

    Returns
    -------
    dict
        nowcast_value (national avg), dept breakdown, benchmarks
    """
    from src.models.poverty import (
        PanelPovertyNowcaster,
        PovertyAR1Benchmark,
        PovertyRandomWalkBenchmark,
        _aggregate_dept_panel_annual,
    )

    panel_path = PROCESSED_DEPARTMENTAL_DIR / "panel_departmental_monthly.parquet"
    poverty_path = TARGETS_DIR / "poverty_departmental.parquet"

    if not panel_path.exists():
        return {"target": "Poverty Rate (%)", "error": "Departmental panel not found"}
    if not poverty_path.exists():
        return {"target": "Poverty Rate (%)", "error": "Poverty targets not found"}

    dept_panel = pd.read_parquet(panel_path)
    poverty_df = pd.read_parquet(poverty_path)

    # Aggregate to annual features
    dept_features = _aggregate_dept_panel_annual(dept_panel, value_col="value_yoy")

    # Models — GradientBoosting with AR(1) lag as feature
    target_cols = ["poverty_rate"]
    model = PanelPovertyNowcaster(
        target_cols=target_cols, include_ar=True, model_type="gbr",
    )
    model.fit(dept_features, poverty_df)
    nc = model.nowcast(dept_features, poverty_df)

    ar1 = PovertyAR1Benchmark(target_cols=target_cols)
    ar1_nc = ar1.nowcast(dept_features, poverty_df)

    rw = PovertyRandomWalkBenchmark(target_cols=target_cols)
    rw_nc = rw.nowcast(dept_features, poverty_df)

    # Nowcast year
    feature_years = set(dept_features["year"].unique())
    poverty_years = set(poverty_df["year"].unique())
    nowcast_years = feature_years - poverty_years
    nowcast_year = max(nowcast_years) if nowcast_years else max(feature_years)

    # Department details
    dept_nowcasts = nc.get("dept_nowcasts", {})
    n_depts = len(dept_nowcasts)

    return {
        "target": "Poverty Rate (%)",
        "target_period": str(nowcast_year),
        "nowcast_value": nc.get("nowcast_value", np.nan),
        "ar1_nowcast": ar1_nc.get("nowcast_value", np.nan),
        "rw_nowcast": rw_nc.get("nowcast_value", np.nan),
        "n_depts": n_depts,
        "dept_nowcasts": dept_nowcasts,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate real-time nowcasts")
    parser.add_argument("--gdp-only", action="store_true")
    parser.add_argument("--inflation-only", action="store_true")
    parser.add_argument("--poverty-only", action="store_true")
    parser.add_argument(
        "--as-of", default=None,
        help="Evaluation date (default: today). Format: YYYY-MM-DD",
    )
    args = parser.parse_args()

    run_all = not (args.gdp_only or args.inflation_only or args.poverty_only)

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()

    print("=" * 65)
    print(f"  NEXUS NOWCAST — as of {as_of.strftime('%Y-%m-%d')}")
    print("=" * 65)

    # Initialize VintageManager (shared for GDP + inflation)
    vm = None
    panel_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"
    lags_path = CONFIG_DIR / "publication_lags.yaml"
    if (run_all or args.gdp_only or args.inflation_only) and panel_path.exists():
        vm = VintageManager(panel_path, lags_path)
        logger.info(
            "VintageManager: %d rows, %d series",
            len(vm.panel), vm.panel["series_id"].nunique(),
        )

    results = {}

    # ── GDP ────────────────────────────────────────────────────────────────
    if run_all or args.gdp_only:
        print(f"\n{'─'*65}")
        print("  GDP Nowcast")
        print(f"{'─'*65}")
        try:
            gdp = nowcast_gdp(vm, as_of)
            results["gdp"] = gdp
            print(f"  Target period : {gdp['target_period']}")
            print(f"  Panel end     : {gdp['panel_end']} ({gdp['n_series']} series, {gdp['n_months']} months)")
            print(f"  Bridge R2     : {gdp['bridge_r2']:.3f}")
            print(f"  ┌──────────────────────────────────────")
            print(f"  │ DFM Nowcast  : {gdp['nowcast_value']:+.2f}%")
            for m in ("lasso", "elastic_net", "gbm"):
                key = f"ml_{m}"
                if key in gdp and not np.isnan(gdp[key]):
                    cv_key = f"ml_{m}_cv_rmse"
                    cv_s = f"  (CV-RMSE {gdp[cv_key]:.2f})" if cv_key in gdp and not np.isnan(gdp.get(cv_key, np.nan)) else ""
                    label = m.upper().replace("_", " ")
                    print(f"  │ ML {label:<10s}: {gdp[key]:+.2f}%{cv_s}")
            print(f"  │ AR(1)        : {gdp['ar1_nowcast']:+.2f}%")
            print(f"  │ Random Walk  : {gdp['rw_nowcast']:+.2f}%")
            print(f"  └──────────────────────────────────────")
        except Exception as e:
            logger.error("GDP nowcast failed: %s", e)
            results["gdp"] = {"error": str(e)}

    # ── Inflation ─────────────────────────────────────────────────────────
    if run_all or args.inflation_only:
        print(f"\n{'─'*65}")
        print("  Inflation Nowcast")
        print(f"{'─'*65}")
        try:
            inf = nowcast_inflation(vm, as_of)
            results["inflation"] = inf
            print(f"  Target period : {inf['target_period']}")
            print(f"  Panel end     : {inf['panel_end']} ({inf['n_series']} series, {inf['n_months']} months)")
            print(f"  Direct Reg R2 : {inf['bridge_r2']:.3f}")
            print(f"  ┌──────────────────────────────────────")
            print(f"  │ DFM Nowcast  : {inf['nowcast_value']:+.3f}%")
            for m in ("lasso", "elastic_net", "gbm"):
                key = f"ml_{m}"
                if key in inf and not np.isnan(inf[key]):
                    cv_key = f"ml_{m}_cv_rmse"
                    cv_s = f"  (CV-RMSE {inf[cv_key]:.3f})" if cv_key in inf and not np.isnan(inf.get(cv_key, np.nan)) else ""
                    label = m.upper().replace("_", " ")
                    print(f"  │ ML {label:<10s}: {inf[key]:+.3f}%{cv_s}")
            print(f"  │ AR(1)        : {inf['ar1_nowcast']:+.3f}%")
            print(f"  │ Random Walk  : {inf['rw_nowcast']:+.3f}%")
            print(f"  │ Phillips     : {inf.get('phillips_nowcast', np.nan):+.3f}%")
            print(f"  └──────────────────────────────────────")
        except Exception as e:
            logger.error("Inflation nowcast failed: %s", e)
            results["inflation"] = {"error": str(e)}

    # ── Poverty ───────────────────────────────────────────────────────────
    if run_all or args.poverty_only:
        print(f"\n{'─'*65}")
        print("  Poverty Nowcast")
        print(f"{'─'*65}")
        try:
            pov = nowcast_poverty()
            results["poverty"] = pov
            if "error" in pov:
                print(f"  ERROR: {pov['error']}")
            else:
                nowcast_pct = pov['nowcast_value'] * 100
                ar1_pct = pov['ar1_nowcast'] * 100
                rw_pct = pov['rw_nowcast'] * 100
                print(f"  Target period : {pov['target_period']}")
                print(f"  Departments   : {pov.get('n_depts', 0)}")
                print(f"  ┌──────────────────────────────────────")
                print(f"  │ Panel Nowcast: {nowcast_pct:.1f}%")
                print(f"  │ AR(1)        : {ar1_pct:.1f}%")
                print(f"  │ Random Walk  : {rw_pct:.1f}%")
                print(f"  └──────────────────────────────────────")

                # Top 5 highest/lowest poverty departments
                dept_nc = pov.get("dept_nowcasts", {})
                if dept_nc:
                    from config.settings import DEPARTMENTS
                    dept_rates = [
                        (code, vals.get("poverty_rate", np.nan) * 100)
                        for code, vals in dept_nc.items()
                        if "poverty_rate" in vals
                    ]
                    dept_rates.sort(key=lambda x: x[1], reverse=True)
                    print(f"\n  Top 5 highest poverty:")
                    for code, rate in dept_rates[:5]:
                        name = DEPARTMENTS.get(code, code)
                        print(f"    {name:<20s} {rate:.1f}%")
                    print(f"  Top 5 lowest poverty:")
                    for code, rate in dept_rates[-5:]:
                        name = DEPARTMENTS.get(code, code)
                        print(f"    {name:<20s} {rate:.1f}%")

        except Exception as e:
            logger.error("Poverty nowcast failed: %s", e)
            results["poverty"] = {"error": str(e)}

    # ── Summary table ─────────────────────────────────────────────────────
    ml_methods = ("lasso", "elastic_net", "gbm")
    ml_headers = ("LASSO", "ENET", "GBM")

    print(f"\n{'='*100}")
    print("  SUMMARY")
    print(f"{'='*100}")
    header = f"  {'Target':<20s} {'Period':<10s} {'DFM':>8s}"
    for h in ml_headers:
        header += f" {h:>8s}"
    header += f" {'AR(1)':>8s} {'RW':>8s}"
    print(header)
    print(f"  {'─'*96}")

    for key in ("gdp", "inflation", "poverty"):
        r = results.get(key, {})
        if "error" in r:
            print(f"  {r.get('target', key):<20s} {'ERROR':<10s}")
            continue
        if not r:
            continue

        target = r.get("target", key)
        period = r.get("target_period", "?")
        nc = r.get("nowcast_value", np.nan)
        ar1 = r.get("ar1_nowcast", np.nan)
        rw = r.get("rw_nowcast", np.nan)

        # Format function based on target type
        def _fmt(val, key=key):
            if isinstance(val, float) and np.isnan(val):
                return "N/A"
            if key == "poverty":
                return f"{val*100:.1f}%"
            elif key == "inflation":
                return f"{val:+.3f}%"
            else:
                return f"{val:+.2f}%"

        row = f"  {target:<20s} {period:<10s} {_fmt(nc):>8s}"
        for m in ml_methods:
            ml_val = r.get(f"ml_{m}", np.nan)
            row += f" {_fmt(ml_val):>8s}"
        row += f" {_fmt(ar1):>8s} {_fmt(rw):>8s}"
        print(row)

    print(f"  {'─'*96}")
    print()


if __name__ == "__main__":
    main()
