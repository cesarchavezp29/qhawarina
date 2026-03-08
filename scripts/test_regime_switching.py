"""Test regime-switching model with GDP and inflation data."""

import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.regime_switching import (
    build_regime_detector_gdp,
    build_regime_detector_inflation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_regime_switching")


def main():
    # Load targets
    gdp = pd.read_parquet(project_root / "data/targets/gdp_quarterly.parquet")
    inflation = pd.read_parquet(project_root / "data/targets/inflation_monthly.parquet")
    panel = pd.read_parquet(
        project_root / "data/processed/national/panel_national_monthly.parquet"
    )

    # Extract financial stress index
    fsi = panel[panel.series_id == "FINANCIAL_STRESS_INDEX"].copy()
    fsi = fsi.set_index("date")["value_raw"].sort_index()

    logger.info("Loaded %d GDP quarters, %d inflation months", len(gdp), len(inflation))
    logger.info("Financial stress: %d months", len(fsi))

    # ── GDP regime detection ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GDP REGIME DETECTION")
    logger.info("=" * 60)

    gdp_yoy = gdp.set_index("date")["gdp_yoy"].sort_index()

    # Resample financial stress to quarterly (mean)
    fsi_q = fsi.resample("QS").mean()

    # Align
    df_gdp = pd.concat([gdp_yoy, fsi_q], axis=1).dropna()
    df_gdp.columns = ["gdp_yoy", "financial_stress"]

    detector_gdp, probs_gdp = build_regime_detector_gdp(
        df_gdp["gdp_yoy"], df_gdp[["financial_stress"]]
    )

    # Show transition matrix
    logger.info("\nGDP Transition Matrix:")
    logger.info("\n%s", detector_gdp.get_transition_matrix())

    # Show crisis periods
    crisis_periods = probs_gdp[probs_gdp["prob_crisis"] > 0.5]
    logger.info("\nGDP Crisis periods (prob > 0.5):")
    for _, row in crisis_periods.iterrows():
        date = row["date"]
        quarter = (date.month - 1) // 3 + 1
        logger.info("  %d-Q%d: %.1f%%", date.year, quarter, row["prob_crisis"] * 100)

    # ── Inflation regime detection ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("INFLATION REGIME DETECTION")
    logger.info("=" * 60)

    inf_monthly = inflation.set_index("date")["ipc_monthly_var"].sort_index()

    # Align
    df_inf = pd.concat([inf_monthly, fsi], axis=1).dropna()
    df_inf.columns = ["inflation", "financial_stress"]

    detector_inf, probs_inf = build_regime_detector_inflation(
        df_inf["inflation"], df_inf[["financial_stress"]]
    )

    # Show transition matrix
    logger.info("\nInflation Transition Matrix:")
    logger.info("\n%s", detector_inf.get_transition_matrix())

    # Show crisis periods
    crisis_periods_inf = probs_inf[probs_inf["prob_crisis"] > 0.5]
    logger.info("\nInflation Crisis periods (prob > 0.5):")
    logger.info("  Total: %d/%d months", len(crisis_periods_inf), len(probs_inf))
    if len(crisis_periods_inf) > 0:
        logger.info("  First: %s", crisis_periods_inf["date"].iloc[0].strftime("%Y-%m"))
        logger.info("  Last: %s", crisis_periods_inf["date"].iloc[-1].strftime("%Y-%m"))

    # ── Save results ───────────────────────────────────────────────────────────
    output_dir = project_root / "data/results"
    output_dir.mkdir(exist_ok=True)

    probs_gdp.to_parquet(output_dir / "regime_probabilities_gdp.parquet", index=False)
    probs_inf.to_parquet(
        output_dir / "regime_probabilities_inflation.parquet", index=False
    )

    logger.info("\n✓ Regime probabilities saved to data/results/")
    logger.info("  - regime_probabilities_gdp.parquet")
    logger.info("  - regime_probabilities_inflation.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
