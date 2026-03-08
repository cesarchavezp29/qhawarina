"""Run inflation nowcasting backtest: DFM vs AR(1) vs Random Walk vs Phillips Curve.

Evaluates monthly CPI variation nowcasts from 2010-01 to 2024-12
using expanding-window backtesting with publication lag awareness.

Usage:
    python scripts/run_inflation_backtest.py
    python scripts/run_inflation_backtest.py --target-col ipc_12m_var
    python scripts/run_inflation_backtest.py --show-12m
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BACKTEST_EVAL_END,
    BACKTEST_EVAL_START,
    CONFIG_DIR,
    PROCESSED_NATIONAL_DIR,
    RESULTS_DIR,
    TARGETS_DIR,
)
from src.backtesting.backtester import RollingBacktester
from src.backtesting.vintage import VintageManager
from src.models.dfm import (
    AR1Benchmark,
    CombinationNowcaster,
    NowcastDFM,
    PhillipsCurveNowcaster,
    RandomWalkBenchmark,
    reconstruct_12m_from_monthly,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.inflation_backtest")


def _print_12m_reconstruction(results: pd.DataFrame, inf_df: pd.DataFrame):
    """Show reconstructed 12-month metrics from monthly predictions."""
    from src.backtesting.metrics import compute_all_metrics

    # Need both actual monthly and predicted monthly to reconstruct 12m
    model_cols = [c for c in results.columns if c.startswith("pred_")]

    print(f"\n{'='*60}")
    print("RECONSTRUCTED 12-MONTH INFLATION METRICS")
    print(f"{'='*60}")

    # Get actual 12m series from target
    actual_12m = inf_df.set_index("date")["ipc_12m_var"].sort_index()

    for col in model_cols:
        model_name = col.replace("pred_", "")
        # Build a monthly series: actual where available, predicted for last period
        monthly_actual = inf_df.set_index("date")["ipc_monthly_var"].sort_index()

        # Reconstruct 12m from actual monthly (as baseline)
        recon_actual = reconstruct_12m_from_monthly(monthly_actual)

        # For predicted: replace last month with prediction, reconstruct
        pred_series = results.set_index("target_date")[col].dropna()
        if pred_series.empty:
            continue

        # Align predictions with actual 12m
        common = pred_series.index.intersection(actual_12m.index)
        if len(common) < 2:
            continue

        # Simple approach: use actual 12m as baseline comparison
        pred_vals = pred_series.loc[common].values
        actual_vals = actual_12m.loc[common].values

        m = compute_all_metrics(actual_vals, pred_vals)
        print(f"  {model_name:<12} 12m-proxy RMSE: {m['rmse']:.3f}  (n={m['n_obs']})")

    print("  Note: these are approximate — true 12m reconstruction requires")
    print("  cumulating 12 consecutive monthly predictions.")


def main():
    parser = argparse.ArgumentParser(description="Inflation nowcasting backtest")
    parser.add_argument(
        "--target-col", default="ipc_3m_ma",
        choices=["ipc_monthly_var", "ipc_12m_var", "ipc_3m_ma"],
        help="Inflation target column (default: ipc_3m_ma = 3-month MA underlying inflation)",
    )
    parser.add_argument(
        "--show-12m", action="store_true",
        help="Show reconstructed 12-month inflation metrics",
    )
    args = parser.parse_args()

    target_col = args.target_col

    panel_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"
    lags_path = CONFIG_DIR / "publication_lags.yaml"
    inflation_path = TARGETS_DIR / "inflation_monthly.parquet"
    output_path = RESULTS_DIR / "backtest_inflation.parquet"

    # Validate inputs
    if not panel_path.exists():
        logger.error("Panel not found: %s", panel_path)
        sys.exit(1)
    if not inflation_path.exists():
        logger.error("Inflation target not found: %s", inflation_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("INFLATION NOWCASTING BACKTEST (target: %s)", target_col)
    logger.info("=" * 60)

    # Load target
    inf_df = pd.read_parquet(inflation_path)
    inf_df["date"] = pd.to_datetime(inf_df["date"])
    logger.info("Inflation target: %d months, %s to %s",
                len(inf_df), inf_df["date"].min().strftime("%Y-%m"),
                inf_df["date"].max().strftime("%Y-%m"))

    # Create vintage manager
    logger.info("Loading vintage manager...")
    vm = VintageManager(panel_path, lags_path)
    logger.info("Panel: %d rows, %d series",
                len(vm.panel), vm.panel["series_id"].nunique())

    # Create models — lagged factors + AR(1) target
    # include_factor_lags=1 captures delayed transmission effects
    # include_target_ar=True captures CPI persistence (R2: 0.139 -> 0.199)
    dfm = NowcastDFM(
        k_factors=2, factor_order=1, target="inflation",
        inflation_col=target_col,
        include_factor_lags=1,
        include_target_ar=True,
    )
    benchmarks = {
        "ar1": AR1Benchmark(target="inflation", inflation_col=target_col),
        "rw": RandomWalkBenchmark(target="inflation", inflation_col=target_col),
        "phillips": PhillipsCurveNowcaster(inflation_col=target_col),
        "combo": CombinationNowcaster(inflation_col=target_col),
    }

    # Run backtest
    backtester = RollingBacktester(
        vintage_manager=vm,
        model=dfm,
        benchmarks=benchmarks,
        target_df=inf_df,
        target_col=target_col,
        frequency="M",
        value_col="auto",
    )

    logger.info("Running backtest: %s to %s", BACKTEST_EVAL_START, BACKTEST_EVAL_END)
    results = backtester.run(
        eval_start=BACKTEST_EVAL_START,
        eval_end=BACKTEST_EVAL_END,
        min_train_periods=36,
    )

    if results.empty:
        logger.error("No backtest results generated!")
        sys.exit(1)

    # Save results
    backtester.save(output_path)

    # Print summary
    summary = backtester.summary()
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    print(f"\n{'Model':<12} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>8} {'N':>5}")
    print("-" * 45)
    for model_name, metrics in summary.items():
        print(
            f"{model_name:<12} {metrics['rmse']:8.3f} {metrics['mae']:8.3f} "
            f"{metrics['directional_accuracy']:8.1%} {metrics['n_obs']:5d}"
        )

    # Print relative RMSE
    if "dfm" in summary:
        print(f"\nRelative RMSE (DFM vs benchmarks):")
        for key, val in summary["dfm"].items():
            if key.startswith("rel_rmse_vs_"):
                bname = key.replace("rel_rmse_vs_", "")
                print(f"  vs {bname}: {val:.3f} {'(DFM wins)' if val < 1 else '(benchmark wins)'}")

    # Optional 12m reconstruction
    if args.show_12m and target_col == "ipc_monthly_var":
        _print_12m_reconstruction(results, inf_df)

    logger.info("Results saved to %s", output_path)
    return results


if __name__ == "__main__":
    main()
