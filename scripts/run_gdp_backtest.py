"""Run GDP nowcasting backtest: DFM vs AR(1) vs Random Walk.

Evaluates quarterly GDP yoy nowcasts from 2010-Q1 to 2024-Q4
using expanding-window backtesting with publication lag awareness.

Usage:
    python scripts/run_gdp_backtest.py
"""

import logging
import sys
from pathlib import Path

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
from src.models.dfm import AR1Benchmark, NowcastDFM, RandomWalkBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.gdp_backtest")


def main():
    panel_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"
    lags_path = CONFIG_DIR / "publication_lags.yaml"
    gdp_path = TARGETS_DIR / "gdp_quarterly.parquet"
    output_path = RESULTS_DIR / "backtest_gdp.parquet"

    # Validate inputs
    if not panel_path.exists():
        logger.error("Panel not found: %s", panel_path)
        sys.exit(1)
    if not gdp_path.exists():
        logger.error("GDP target not found: %s", gdp_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("GDP NOWCASTING BACKTEST")
    logger.info("=" * 60)

    # Load target
    gdp_df = pd.read_parquet(gdp_path)
    gdp_df["date"] = pd.to_datetime(gdp_df["date"])
    logger.info("GDP target: %d quarters, %s to %s",
                len(gdp_df), gdp_df["date"].min().strftime("%Y-%m"),
                gdp_df["date"].max().strftime("%Y-%m"))

    # Create vintage manager
    logger.info("Loading vintage manager...")
    vm = VintageManager(panel_path, lags_path)
    logger.info("Panel: %d rows, %d series",
                len(vm.panel), vm.panel["series_id"].nunique())

    # Create models — Ridge bridge + rolling 7yr window
    # Ridge(alpha=1.0) prevents OLS coefficient explosion (-81 on factor_1)
    # Rolling 7yr window avoids COVID structural break contamination
    dfm = NowcastDFM(
        k_factors=3, factor_order=1, target="gdp",
        bridge_method="ridge", bridge_alpha=1.0,
        rolling_window_years=7,
        exclude_covid=True,  # Exclude 2020-2021 in post-COVID evaluations
    )
    benchmarks = {
        "ar1": AR1Benchmark(target="gdp"),
        "rw": RandomWalkBenchmark(target="gdp"),
    }

    # Run backtest
    backtester = RollingBacktester(
        vintage_manager=vm,
        model=dfm,
        benchmarks=benchmarks,
        target_df=gdp_df,
        target_col="gdp_yoy",
        frequency="Q",
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

    logger.info("Results saved to %s", output_path)
    return results


if __name__ == "__main__":
    main()
