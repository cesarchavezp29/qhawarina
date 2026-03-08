"""
Run counterfactual scenario analysis and export results.

This script:
1. Loads trained nowcasting models
2. Runs pre-built scenarios from the library
3. Calculates baseline vs counterfactual
4. Propagates cross-model impacts
5. Exports results for web visualization

Usage:
    python scripts/run_scenario_analysis.py --scenario mild_recession
    python scripts/run_scenario_analysis.py --all  # Run all scenarios
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scenarios import ScenarioEngine, ShockPropagator, SCENARIO_LIBRARY, get_scenario
from src.models.dfm import NowcastDFM
from src.backtesting.vintage import VintageManager


def load_models():
    """Load trained nowcasting models."""
    print("Loading models...")

    # GDP model
    gdp_model = NowcastDFM(
        target="gdp",
        k_factors=2,
        bridge_method="ridge",
        bridge_alpha=1.0,
        rolling_window_years=7,
        exclude_covid=True,
        include_target_ar=False,
    )

    # Inflation model
    inflation_model = NowcastDFM(
        target="inflation",
        inflation_col="ipc_3m_ma",
        k_factors=2,
        bridge_method="ridge",
        bridge_alpha=1.0,
        include_factor_lags=1,
        include_target_ar=True,
        exclude_covid=False,
    )

    return {
        "gdp": gdp_model,
        "inflation": inflation_model,
    }


def run_scenario_analysis(scenario_name: str, output_dir: Path):
    """
    Run a single scenario analysis.

    Args:
        scenario_name: Name of scenario from library
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"{'='*60}")

    # Get scenario
    scenario = get_scenario(scenario_name)
    print(f"Description: {scenario.description}")
    print(f"Tags: {', '.join(scenario.tags)}")

    # Load data
    print("\nLoading national panel...")
    panel_path = PROJECT_ROOT / "data/processed/national/panel_national_monthly.parquet"
    panel_df = pd.read_parquet(panel_path)

    # Load targets
    gdp_targets = pd.read_parquet(PROJECT_ROOT / "data/targets/gdp_quarterly.parquet")
    inflation_targets = pd.read_parquet(
        PROJECT_ROOT / "data/targets/inflation_monthly.parquet"
    )

    # Determine target periods
    latest_gdp_date = gdp_targets["date"].max()
    latest_inflation_date = inflation_targets["date"].max()

    # Next quarter/month for nowcasting
    gdp_target_period = (latest_gdp_date + pd.DateOffset(months=3)).strftime("%Y-%m")
    inflation_target_period = (
        latest_inflation_date + pd.DateOffset(months=1)
    ).strftime("%Y-%m")

    print(f"\nTarget periods:")
    print(f"  GDP: {gdp_target_period}")
    print(f"  Inflation: {inflation_target_period}")

    # Load models
    models = load_models()

    # Train models on latest data
    print("\nTraining models...")

    # Prepare wide panel using VintageManager
    vintage_mgr = VintageManager(panel_path=panel_path)
    panel_wide = vintage_mgr._pivot_auto(panel_df)

    models["gdp"].fit(panel_wide)
    models["inflation"].fit(panel_wide)

    # Run scenario
    print("\n" + "="*60)
    print("RUNNING SCENARIO ENGINE")
    print("="*60)

    engine = ScenarioEngine()
    results = engine.run_scenario(
        scenario=scenario,
        panel_df=panel_df,
        gdp_model=models["gdp"],
        inflation_model=models["inflation"],
        gdp_targets=gdp_targets,
        inflation_targets=inflation_targets,
        target_period=gdp_target_period,  # Use for both (will handle internally)
    )

    # Display results
    print("\n" + "="*60)
    print("BASELINE vs COUNTERFACTUAL")
    print("="*60)

    if "gdp" in results["baseline"]:
        gdp_base = results["baseline"]["gdp"]["gdp_yoy"]
        gdp_counter = results["counterfactual"]["gdp"]["gdp_yoy"]
        gdp_impact = results["impacts"]["gdp"]

        print(f"\nGDP (YoY %):")
        print(f"  Baseline:        {gdp_base:>6.2f}%")
        print(f"  Counterfactual:  {gdp_counter:>6.2f}%")
        print(f"  Impact:          {gdp_impact:>+6.2f}pp")

    if "inflation" in results["baseline"]:
        inf_base = results["baseline"]["inflation"]["ipc_monthly_var"]
        inf_counter = results["counterfactual"]["inflation"]["ipc_monthly_var"]
        inf_impact = results["impacts"]["inflation"]

        print(f"\nInflation (Monthly %):")
        print(f"  Baseline:        {inf_base:>6.2f}%")
        print(f"  Counterfactual:  {inf_counter:>6.2f}%")
        print(f"  Impact:          {inf_impact:>+6.2f}pp")

    # Propagate shocks across models
    print("\n" + "="*60)
    print("CROSS-MODEL IMPACT PROPAGATION")
    print("="*60)

    propagator = ShockPropagator()
    propagation = propagator.propagate_full_scenario(
        gdp_impact=results["impacts"].get("gdp"),
        inflation_impact=results["impacts"].get("inflation"),
    )

    print("\nAggregate Impacts:")
    for key, value in propagation["aggregate_impacts"].items():
        if abs(value) > 0.01:
            print(f"  {key}: {value:+.2f}")

    print(f"\nInterpretation:")
    print(f"  {propagation['interpretation']}")

    # Prepare export
    export_data = {
        "metadata": {
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "tags": scenario.tags,
            "generated_at": datetime.now().isoformat(),
            "target_period_gdp": gdp_target_period,
            "target_period_inflation": inflation_target_period,
        },
        "baseline": results["baseline"],
        "counterfactual": results["counterfactual"],
        "direct_impacts": results["impacts"],
        "propagated_impacts": {
            "individual": propagation["individual_impacts"],
            "aggregate": propagation["aggregate_impacts"],
            "interpretation": propagation["interpretation"],
        },
        "shocks": {
            "exogenous": [
                {
                    "series_id": s.series_id,
                    "shock_type": s.shock_type,
                    "shock_value": s.shock_value,
                    "description": s.description,
                }
                for s in scenario.exogenous_shocks
            ],
            "endogenous": [
                {
                    "target": s.target,
                    "forced_value": s.forced_value,
                    "description": s.description,
                    "period": s.period,
                }
                for s in scenario.endogenous_shocks
            ],
        },
    }

    # Save results
    output_path = output_dir / f"scenario_{scenario_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nExported to {output_path}")

    return export_data


def run_all_scenarios(output_dir: Path):
    """Run all scenarios in the library."""
    print("="*60)
    print(f"Running ALL scenarios ({len(SCENARIO_LIBRARY)} total)")
    print("="*60)

    all_results = []

    for scenario_name in SCENARIO_LIBRARY.keys():
        try:
            result = run_scenario_analysis(scenario_name, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in scenario '{scenario_name}': {e}")
            import traceback

            traceback.print_exc()
            continue

    # Create comparison table
    comparison_rows = []

    for result in all_results:
        row = {
            "scenario": result["metadata"]["scenario_name"],
            "description": result["metadata"]["scenario_description"],
        }

        # Extract key metrics
        if "gdp" in result["baseline"]:
            row["gdp_baseline"] = result["baseline"]["gdp"]["gdp_yoy"]
            row["gdp_counterfactual"] = result["counterfactual"]["gdp"]["gdp_yoy"]
            row["gdp_impact"] = result["direct_impacts"]["gdp"]

        if "inflation" in result["baseline"]:
            row["inflation_baseline"] = result["baseline"]["inflation"]["ipc_monthly_var"]
            row["inflation_counterfactual"] = result["counterfactual"]["inflation"][
                "ipc_monthly_var"
            ]
            row["inflation_impact"] = result["direct_impacts"]["inflation"]

        # Propagated impacts
        agg = result["propagated_impacts"]["aggregate"]
        row["poverty_impact_pp"] = agg.get("poverty_total_pp", 0.0)

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Save comparison
    comparison_path = output_dir / "scenario_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\n{'='*60}")
    print("SCENARIO COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print(f"\n✓ Comparison table saved to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Run counterfactual scenario analysis")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name from library (e.g., mild_recession, inflation_spike)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios in the library",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/scenarios",
        help="Output directory for results (default: exports/scenarios)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        run_all_scenarios(output_dir)
    elif args.scenario:
        run_scenario_analysis(args.scenario, output_dir)
    else:
        print("Error: Must specify --scenario NAME or --all")
        print("\nAvailable scenarios:")
        for name, scenario in SCENARIO_LIBRARY.items():
            print(f"  {name:20s} - {scenario.description}")
        sys.exit(1)


if __name__ == "__main__":
    main()
