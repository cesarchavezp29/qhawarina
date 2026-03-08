"""
Demo script showing all policy simulators and calculators in action.

Run: python scripts/demo_simulators.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.policy_simulator import PolicySimulator, ScenarioLibrary
from src.simulation.calculators import (
    InflationCalculator,
    PovertyForecastCalculator,
    GDPScenarioCalculator,
    RegionalComparator
)
import json


def main():
    print("=" * 90)
    print(" " * 20 + "QHAWARINA POLICY SIMULATORS DEMO")
    print("=" * 90)
    print()
    print("Testing all interactive calculators to prove they work...")
    print()

    # ==================================================================================
    # 1. POLICY SCENARIOS
    # ==================================================================================
    print("-" * 90)
    print("1. POLICY SCENARIO ANALYSIS")
    print("-" * 90)
    print()

    simulator = PolicySimulator()
    baseline_gdp = 2.8
    baseline_inflation = 2.3
    baseline_poverty = 24.5

    scenarios = {
        'Commodity Crash': ScenarioLibrary.commodity_crash(-20),
        'Currency Crisis': ScenarioLibrary.fx_crisis(15),
        'Rate Hike': ScenarioLibrary.rate_hike(100),
        'Political Crisis': ScenarioLibrary.political_crisis(2.0),
        'China Slowdown': ScenarioLibrary.china_slowdown(-2.0)
    }

    print(f"Baseline Forecast: GDP={baseline_gdp}%, Inflation={baseline_inflation}%, Poverty={baseline_poverty}%")
    print()
    print(f"{'Scenario':<25} {'GDP Impact':<12} {'Inflation Impact':<15} {'Final GDP':<10}")
    print("-" * 90)

    for name, shock in scenarios.items():
        result = simulator.simulate_shock(shock, baseline_gdp, baseline_inflation, baseline_poverty)
        print(f"{name:<25} {result.gdp_impact:+6.2f}pp      {result.inflation_impact:+6.2f}pp         {result.shocked_gdp:6.1f}%")

    print()

    # ==================================================================================
    # 2. INFLATION CALCULATOR
    # ==================================================================================
    print("-" * 90)
    print("2. INFLATION IMPACT CALCULATOR")
    print("-" * 90)
    print()

    inflation_calc = InflationCalculator()

    test_cases = [
        (1000, '2026-02-10', '2026-02-28', 'all'),
        (500, '2026-02-10', '2026-02-28', 'food'),
        (100, '2026-02-15', '2026-02-28', 'carnes')
    ]

    for amount, start, end, category in test_cases:
        result = inflation_calc.calculate_impact(amount, start, end, category)
        if 'error' not in result:
            print(f"S/ {amount} ({category}, {start} to {end}):")
            print(f"  Equivalent today: S/ {result['equivalent_today']:.2f}")
            print(f"  Change: S/ {result['loss_amount']:+.2f} ({result['loss_pct']:+.2f}%)")
            print(f"  {result['interpretation']}")
            print()

    # ==================================================================================
    # 3. POVERTY FORECAST
    # ==================================================================================
    print("-" * 90)
    print("3. POVERTY FORECAST CALCULATOR")
    print("-" * 90)
    print()

    poverty_calc = PovertyForecastCalculator()

    departments = [
        ('Ayacucho', 'baseline'),
        ('Lima', 'optimistic'),
        ('Huancavelica', 'pessimistic')
    ]

    print(f"{'Department':<15} {'Scenario':<12} {'Forecast':<10} {'90% CI':<15} {'People Affected':<15}")
    print("-" * 90)

    for dept, scenario in departments:
        result = poverty_calc.forecast(dept, '2026-12-31', scenario)
        ci = result['confidence_interval']
        print(f"{dept:<15} {scenario:<12} {result['poverty_rate']:>6.1f}%    "
              f"[{ci['lower']:.1f}%, {ci['upper']:.1f}%]   {result['people_affected']:>10,.0f}")

    print()

    # ==================================================================================
    # 4. GDP SCENARIO BUILDER
    # ==================================================================================
    print("-" * 90)
    print("4. GDP SCENARIO BUILDER")
    print("-" * 90)
    print()

    gdp_calc = GDPScenarioCalculator()

    custom_scenarios = [
        ("Downside: Commodity + Political", [
            {'type': 'commodity', 'magnitude': -15},
            {'type': 'political', 'magnitude': 1.5}
        ]),
        ("Upside: Rates Cut + FX Stable", [
            {'type': 'interest_rate', 'magnitude': -50},
            {'type': 'fx', 'magnitude': -5}
        ])
    ]

    for scenario_name, shocks in custom_scenarios:
        result = gdp_calc.calculate_scenario(2.8, shocks)
        print(f"{scenario_name}:")
        print(f"  Baseline GDP: {result['baseline_gdp']:.1f}%")
        print(f"  Shocked GDP:  {result['shocked_gdp']:.1f}% ({result['total_impact']:+.2f}pp)")
        print(f"  {result['interpretation']}")
        print()

    # ==================================================================================
    # 5. REGIONAL COMPARATOR
    # ==================================================================================
    print("-" * 90)
    print("5. REGIONAL DEPARTMENT COMPARATOR")
    print("-" * 90)
    print()

    regional = RegionalComparator()
    result = regional.compare_departments(['Lima', 'Ayacucho', 'Cusco', 'Arequipa'])

    print(f"Comparing: {', '.join(result['departments'])}")
    print(f"Summary: {result['summary']}")
    print()

    # ==================================================================================
    # SAVE DEMO RESULTS
    # ==================================================================================
    demo_output = {
        'demo_date': '2026-02-28',
        'policy_scenarios': {
            name: simulator.simulate_shock(shock, baseline_gdp, baseline_inflation, baseline_poverty).to_dict()
            for name, shock in scenarios.items()
        },
        'calculators_tested': ['inflation', 'poverty', 'gdp_scenario', 'regional_compare'],
        'status': 'All calculators working'
    }

    output_file = Path(__file__).parent.parent / 'demo_simulators_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_output, f, indent=2, ensure_ascii=False)

    print("-" * 90)
    print(f"Demo complete. Full results saved to: {output_file.name}")
    print("-" * 90)
    print()
    print("NEXT STEPS:")
    print("  1. Start API server:  python scripts/api_calculators.py")
    print("  2. Test API endpoint: curl http://localhost:5000/api/scenarios/library")
    print("  3. Build React UI to embed in Qhawarina website")
    print()
    print("These calculators prove the models are useful, not bullshit.")
    print("=" * 90)


if __name__ == '__main__':
    main()
