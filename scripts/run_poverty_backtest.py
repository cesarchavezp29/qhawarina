"""Run poverty nowcasting backtest: Panel Regression vs DFM vs AR(1) vs RW.

Usage:
    python scripts/run_poverty_backtest.py
    python scripts/run_poverty_backtest.py --target-col extreme_poverty_rate
    python scripts/run_poverty_backtest.py --disaggregate
    python scripts/run_poverty_backtest.py --monthly
    python scripts/run_poverty_backtest.py --monthly --eval-months 3 6 9 12
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DEPARTMENTAL_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGETS_DIR,
)
from src.backtesting.backtester import MonthlyPovertyBacktester, PovertyBacktester
from src.models.poverty import (
    DFMPovertyNowcaster,
    PanelPovertyNowcaster,
    PovertyAR1Benchmark,
    PovertyRandomWalkBenchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.poverty_backtest")


def main():
    parser = argparse.ArgumentParser(description="Poverty nowcasting backtest")
    parser.add_argument(
        "--target-col",
        default="poverty_rate",
        choices=["poverty_rate", "extreme_poverty_rate", "mean_consumption"],
        help="Target variable to evaluate (default: poverty_rate)",
    )
    parser.add_argument(
        "--eval-start", type=int, default=2012,
        help="First evaluation year (default: 2012)",
    )
    parser.add_argument(
        "--eval-end", type=int, default=2024,
        help="Last evaluation year (default: 2024)",
    )
    parser.add_argument(
        "--disaggregate", action="store_true",
        help="Run district disaggregation on final nowcast",
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Run monthly-resolution backtest with rolling windows",
    )
    parser.add_argument(
        "--eval-months", nargs="+", type=int, default=[3, 6, 9, 12],
        help="Months to evaluate within each year (default: 3 6 9 12)",
    )
    parser.add_argument(
        "--window-months", type=int, default=12,
        help="Rolling window size in months (default: 12)",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading departmental panel...")
    panel_path = PROCESSED_DEPARTMENTAL_DIR / "panel_departmental_monthly.parquet"
    if not panel_path.exists():
        logger.error("Departmental panel not found: %s", panel_path)
        sys.exit(1)
    dept_panel = pd.read_parquet(panel_path)

    logger.info("Loading poverty targets...")
    poverty_path = TARGETS_DIR / "poverty_departmental.parquet"
    if not poverty_path.exists():
        logger.error("Poverty targets not found: %s", poverty_path)
        sys.exit(1)
    poverty_df = pd.read_parquet(poverty_path)

    logger.info(
        "Data loaded: %d panel rows, %d poverty obs (%d depts × %d years)",
        len(dept_panel), len(poverty_df),
        poverty_df["department_code"].nunique(),
        poverty_df["year"].nunique(),
    )

    # ── Create models ────────────────────────────────────────────────────────
    target_cols = [args.target_col]

    panel_model = PanelPovertyNowcaster(
        target_cols=target_cols, alpha=1.0, include_ar=True,
        model_type="gbr", exclude_covid=True,
    )
    dfm_model = DFMPovertyNowcaster(
        k_factors=3, target_cols=target_cols, alpha=1.0,
    )

    benchmarks = {
        "ar1": PovertyAR1Benchmark(target_cols=target_cols),
        "rw": PovertyRandomWalkBenchmark(target_cols=target_cols),
    }

    # ── Run Panel backtest ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Panel Regression backtest...")
    panel_bt = PovertyBacktester(
        dept_panel=dept_panel,
        model=panel_model,
        benchmarks=benchmarks,
        poverty_df=poverty_df,
        target_col=args.target_col,
        value_col="value_yoy",
    )
    panel_results = panel_bt.run(
        eval_start_year=args.eval_start,
        eval_end_year=args.eval_end,
        min_train_years=8,
    )

    # ── Run DFM backtest ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running DFM Hybrid backtest...")
    dfm_bt = PovertyBacktester(
        dept_panel=dept_panel,
        model=dfm_model,
        benchmarks={},  # don't re-run benchmarks
        poverty_df=poverty_df,
        target_col=args.target_col,
        value_col="value_yoy",
    )
    dfm_results = dfm_bt.run(
        eval_start_year=args.eval_start,
        eval_end_year=args.eval_end,
        min_train_years=8,
    )

    # ── Merge results ────────────────────────────────────────────────────────
    if not panel_results.empty and not dfm_results.empty:
        combined = panel_results.copy()
        dfm_cols = dfm_results[["year", "department_code", "panel_nowcast", "panel_error"]]
        dfm_cols = dfm_cols.rename(columns={
            "panel_nowcast": "dfm_nowcast",
            "panel_error": "dfm_error",
        })
        combined = combined.merge(
            dfm_cols, on=["year", "department_code"], how="left",
        )
    else:
        combined = panel_results if not panel_results.empty else dfm_results

    # ── Save results ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "backtest_poverty.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Results saved: %s (%d rows)", output_path, len(combined))

    # ── Print summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("POVERTY BACKTEST RESULTS (%s)", args.target_col)
    logger.info("=" * 60)

    panel_summary = panel_bt.summary()
    _print_summary(panel_summary, "Panel Regression")

    dfm_summary = dfm_bt.summary()
    _print_summary(dfm_summary, "DFM Hybrid")

    # ── Monthly backtest ────────────────────────────────────────────────────
    if args.monthly:
        _run_monthly_backtest(dept_panel, poverty_df, target_cols, benchmarks, args)

    # ── District disaggregation ──────────────────────────────────────────────
    if args.disaggregate:
        _run_disaggregation(dept_panel, poverty_df, target_cols, args)


def _print_summary(summary: dict, model_name: str):
    """Print backtest metrics."""
    if not summary:
        logger.info("  %s: no results", model_name)
        return

    logger.info("  %s:", model_name)
    for name, metrics in summary.items():
        rmse = metrics.get("rmse", np.nan)
        mae = metrics.get("mae", np.nan)
        logger.info("    %-10s RMSE=%.3f  MAE=%.3f", name, rmse, mae)
        for key, val in metrics.items():
            if key.startswith("rel_rmse"):
                logger.info("    %-10s %s=%.3f", "", key, val)


def _run_monthly_backtest(dept_panel, poverty_df, target_cols, benchmarks, args):
    """Run monthly-resolution poverty backtest with rolling windows."""
    logger.info("=" * 60)
    logger.info("Running MONTHLY poverty backtest (rolling %d-month window)...",
                args.window_months)
    logger.info("Eval months: %s", args.eval_months)

    monthly_model = PanelPovertyNowcaster(
        target_cols=target_cols, alpha=1.0, include_ar=True,
        model_type="gbr", exclude_covid=True,
    )

    monthly_bt = MonthlyPovertyBacktester(
        dept_panel=dept_panel,
        model=monthly_model,
        benchmarks=benchmarks,
        poverty_df=poverty_df,
        target_col=args.target_col,
        value_col="value_yoy",
        eval_months=args.eval_months,
        window_months=args.window_months,
    )
    monthly_results = monthly_bt.run(
        eval_start_year=args.eval_start,
        eval_end_year=args.eval_end,
        min_train_years=8,
    )

    if not monthly_results.empty:
        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "backtest_poverty_monthly.parquet"
        monthly_results.to_parquet(out_path, index=False)
        logger.info("Monthly results saved: %s (%d rows)", out_path, len(monthly_results))

        # Print convergence analysis
        convergence = monthly_bt.convergence_analysis()
        if not convergence.empty:
            logger.info("=" * 60)
            logger.info("CONVERGENCE ANALYSIS")
            logger.info("=" * 60)
            logger.info("%-8s %-8s %-8s %-12s %-12s",
                        "Month", "RMSE", "MAE", "W/Y Std", "Revision")
            for _, row in convergence.iterrows():
                logger.info(
                    "%-8d %-8.3f %-8.3f %-12s %-12s",
                    int(row["month"]),
                    row.get("rmse", np.nan),
                    row.get("mae", np.nan),
                    f"{row['within_year_std']:.3f}" if "within_year_std" in row and not pd.isna(row.get("within_year_std")) else "N/A",
                    f"{row['mean_revision']:.3f}" if "mean_revision" in row and not pd.isna(row.get("mean_revision")) else "N/A",
                )

        # Print overall summary
        monthly_summary = monthly_bt.summary()
        _print_summary(monthly_summary, "Monthly GBR Panel")
    else:
        logger.warning("No monthly backtest results produced")


def _run_disaggregation(dept_panel, poverty_df, target_cols, args):
    """Run district disaggregation on the most recent nowcast."""
    from src.models.poverty import PanelPovertyNowcaster, _aggregate_dept_panel_annual
    from src.processing.spatial_disagg import nowcast_districts

    ntl_path = PROCESSED_DIR / "ntl_monthly_district.parquet"
    if not ntl_path.exists():
        logger.warning("District NTL data not found: %s", ntl_path)
        return

    logger.info("Running district disaggregation...")
    model = PanelPovertyNowcaster(target_cols=target_cols, alpha=1.0, include_ar=True, model_type="gbr", exclude_covid=True)
    features = _aggregate_dept_panel_annual(dept_panel, value_col="value_yoy")
    model.fit(features, poverty_df)
    nc = model.nowcast(features, poverty_df)

    dept_nowcasts = nc.get("dept_nowcasts", {})
    if not dept_nowcasts:
        logger.warning("No department nowcasts to disaggregate")
        return

    nowcast_year = max(features["year"].unique())
    district_df = nowcast_districts(
        dept_nowcasts, ntl_path, year=nowcast_year, target_cols=target_cols,
    )

    if not district_df.empty:
        out_path = RESULTS_DIR / "district_poverty_nowcast.parquet"
        district_df.to_parquet(out_path, index=False)
        logger.info(
            "District nowcasts saved: %s (%d districts)", out_path, len(district_df),
        )

        # Bounds check
        for tc in target_cols:
            col = f"{tc}_nowcast"
            if col in district_df.columns:
                vals = district_df[col].dropna()
                logger.info(
                    "  %s: min=%.2f, max=%.2f, mean=%.2f",
                    col, vals.min(), vals.max(), vals.mean(),
                )


if __name__ == "__main__":
    main()
