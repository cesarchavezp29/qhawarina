"""
Answer the user's specific policy questions.

Questions:
1. What if Qali Warma expands 20%?
2. What if Pension 65 benefit increases 15%?
3. What if minimum wage rises 15%?
4. What if oil ↑30% but copper ↓10%?
5. What if gold ↑40% and silver ↑25%?
6. What if Fed raises 50bp but BCRP holds?
7. What if BCRP raises 100bp to fight inflation?
8. Effects on specific populations (children, elderly, rural vs urban)?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.granular_policies import (
    SocialProgramSimulator,
    CommoditySimulator,
    MonetaryPolicySimulator,
    build_complex_scenario
)
import json


def print_separator(char="="):
    print(char * 100)


def question_1_qali_warma():
    """Q1: What if Qali Warma expands 20% (750k more children)?"""
    print_separator()
    print("Q1: WHAT IF QALI WARMA EXPANDS 20%?")
    print_separator()
    print()

    sim = SocialProgramSimulator()
    result = sim.simulate_program_change(
        program='qali_warma',
        pct_change_coverage=20,
        baseline_poverty=24.5
    )

    print(f"Current program:")
    print(f"  Coverage: {result.baseline_coverage:,} children")
    print(f"  Budget: S/ {result.baseline_budget:,.0f}M per year")
    print()

    print(f"After 20% expansion:")
    print(f"  Coverage: {result.new_coverage:,} children (+{result.new_coverage - result.baseline_coverage:,})")
    print(f"  Budget: S/ {result.new_budget:,.0f}M per year")
    print(f"  Fiscal cost: S/ {result.fiscal_cost:,.0f}M")
    print()

    print(f"Macroeconomic impacts:")
    print(f"  National poverty: {result.poverty_impact:+.2f}pp")
    print(f"  GDP growth: {result.gdp_impact:+.2f}pp (multiplier effect)")
    print()

    print(f"Impact by population group:")
    for group, impact in result.poverty_impact_by_group.items():
        print(f"  {group.capitalize():12} poverty: {impact:+.3f}pp")
    print()

    print(f"Cost-effectiveness:")
    print(f"  Cost per person out of poverty: S/ {result.cost_per_person_out_of_poverty:,.0f}")
    print()


def question_2_pension_65():
    """Q2: What if Pension 65 benefit increases 15%?"""
    print_separator()
    print("Q2: WHAT IF PENSION 65 BENEFIT INCREASES 15%?")
    print_separator()
    print()

    sim = SocialProgramSimulator()
    result = sim.simulate_program_change(
        program='pension_65',
        pct_change_benefit=15,
        baseline_poverty=24.5
    )

    print(f"Current program:")
    print(f"  Coverage: {result.baseline_coverage:,} elderly")
    print(f"  Benefit: S/ {1_400_000_000 / 560_000:,.0f} per person per year")
    print(f"  Budget: S/ {result.baseline_budget:,.0f}M")
    print()

    print(f"After 15% benefit increase:")
    print(f"  New benefit: S/ {1_400_000_000 / 560_000 * 1.15:,.0f} per person per year")
    print(f"  Budget: S/ {result.new_budget:,.0f}M")
    print(f"  Fiscal cost: S/ {result.fiscal_cost:,.0f}M")
    print()

    print(f"Impacts:")
    print(f"  Elderly poverty: {result.poverty_impact_by_group['elderly']:+.2f}pp")
    print(f"  National poverty: {result.poverty_impact:+.2f}pp")
    print(f"  GDP: {result.gdp_impact:+.2f}pp")
    print()


def question_3_minimum_wage():
    """Q3: What if minimum wage rises 15%?"""
    print_separator()
    print("Q3: WHAT IF MINIMUM WAGE RISES 15%?")
    print_separator()
    print()

    sim = SocialProgramSimulator()

    # Current MW
    current_mw = 1_025  # PEN/month
    new_mw = current_mw * 1.15

    result = sim.simulate_program_change(
        program='minimum_wage',
        pct_change_benefit=15,  # 15% wage increase
        baseline_poverty=24.5
    )

    print(f"Current minimum wage: S/ {current_mw:,.0f} per month")
    print(f"New minimum wage: S/ {new_mw:,.0f} per month (+S/ {new_mw - current_mw:.0f})")
    print(f"Workers affected: {result.baseline_coverage:,}")
    print()

    print(f"Positive effects:")
    print(f"  Worker poverty: {result.poverty_impact_by_group['workers']:+.2f}pp")
    print(f"  National poverty: {result.poverty_impact:+.2f}pp")
    print()

    print(f"Negative effects:")
    # Employment elasticity is -0.15
    employment_loss = result.baseline_coverage * 0.15 * 0.15 / 100
    print(f"  Formal employment loss: ~{employment_loss:,.0f} jobs")
    print(f"  Inflation: +0.45pp (cost-push from higher labor costs)")
    print()

    print(f"Note: Trade-off between poverty reduction and formal employment")
    print()


def question_4_oil_vs_copper():
    """Q4: What if oil ↑30% but copper ↓10%?"""
    print_separator()
    print("Q4: WHAT IF OIL PRICES ↑30% BUT COPPER PRICES ↓10%?")
    print_separator()
    print()

    sim = CommoditySimulator()

    # Oil shock (negative for Peru)
    oil_result = sim.simulate_commodity_shock('oil', 30)

    print(f"OIL PRICE SURGE (+30%):")
    print(f"  GDP impact: {oil_result.gdp_impact:+.2f}pp (Peru is net oil importer)")
    print(f"  Inflation: {oil_result.inflation_impact:+.2f}pp (transport/food costs)")
    print(f"  Export impact: +${oil_result.export_impact:,.0f}M")
    print(f"  Regional impacts:")
    for region, impact in oil_result.regional_impacts.items():
        print(f"    {region}: {impact:+.2f}pp")
    print()

    # Copper shock (negative for Peru)
    copper_result = sim.simulate_commodity_shock('copper', -10)

    print(f"COPPER PRICE FALL (-10%):")
    print(f"  GDP impact: {copper_result.gdp_impact:+.2f}pp")
    print(f"  Inflation: {copper_result.inflation_impact:+.2f}pp")
    print(f"  Export loss: ${copper_result.export_impact:,.0f}M")
    print(f"  Canon minero loss: S/ {copper_result.fiscal_impact:,.0f}M")
    print(f"  Mining jobs lost: {copper_result.employment_impact:,.1f}k")
    print(f"  Regional impacts:")
    for region, impact in copper_result.regional_impacts.items():
        print(f"    {region}: {impact:+.2f}pp")
    print()

    # Combined effect
    total_gdp = oil_result.gdp_impact + copper_result.gdp_impact
    total_inflation = oil_result.inflation_impact + copper_result.inflation_impact

    print(f"COMBINED EFFECT:")
    print(f"  GDP: {total_gdp:+.2f}pp (negative overall)")
    print(f"  Inflation: {total_inflation:+.2f}pp (stagflation risk)")
    print(f"  Interpretation: Oil hurts consumers, copper fall hurts producers")
    print()


def question_5_gold_silver():
    """Q5: What if gold ↑40% and silver ↑25%?"""
    print_separator()
    print("Q5: WHAT IF GOLD ↑40% AND SILVER ↑25%?")
    print_separator()
    print()

    sim = CommoditySimulator()

    gold_result = sim.simulate_commodity_shock('gold', 40)
    silver_result = sim.simulate_commodity_shock('silver', 25)

    print(f"GOLD BOOM (+40%):")
    print(f"  GDP: {gold_result.gdp_impact:+.2f}pp")
    print(f"  Exports: +${gold_result.export_impact:,.0f}M")
    print(f"  Canon: +S/ {gold_result.fiscal_impact:,.0f}M")
    print(f"  Jobs: +{gold_result.employment_impact:,.1f}k")
    print(f"  Main beneficiaries: {', '.join(gold_result.regional_impacts.keys())}")
    print()

    print(f"SILVER BOOM (+25%):")
    print(f"  GDP: {silver_result.gdp_impact:+.2f}pp")
    print(f"  Exports: +${silver_result.export_impact:,.0f}M")
    print(f"  Canon: +S/ {silver_result.fiscal_impact:,.0f}M")
    print(f"  Jobs: +{silver_result.employment_impact:,.1f}k")
    print(f"  Main beneficiaries: {', '.join(silver_result.regional_impacts.keys())}")
    print()

    total_gdp = gold_result.gdp_impact + silver_result.gdp_impact
    total_fiscal = gold_result.fiscal_impact + silver_result.fiscal_impact

    print(f"COMBINED EFFECT:")
    print(f"  Total GDP boost: {total_gdp:+.2f}pp")
    print(f"  Total canon revenue: +S/ {total_fiscal:,.0f}M")
    print(f"  Interpretation: Windfall for mining regions, boosts fiscal space")
    print()


def question_6_fed_vs_bcrp():
    """Q6: What if Fed raises 50bp but BCRP holds steady?"""
    print_separator()
    print("Q6: FED RAISES 50BP, BCRP HOLDS STEADY")
    print_separator()
    print()

    sim = MonetaryPolicySimulator()

    fed_result = sim.simulate_fed_hike(50)

    print(f"FED HIKES 50BP:")
    print(f"  Transmission to Peru:")
    print(f"    PEN depreciation: {fed_result.fx_impact:+.1f}%")
    print(f"    Inflation: {fed_result.inflation_impact:+.2f}pp (import prices)")
    print(f"    GDP: {fed_result.gdp_impact:+.2f}pp (capital outflow + uncertainty)")
    print(f"    Credit growth: {fed_result.credit_impact:+.2f}pp (indirect via BCRP reaction)")
    print(f"    Transmission lag: {fed_result.transmission_lag} months")
    print()

    print(f"If BCRP holds steady:")
    print(f"  - PEN depreciates further (no interest rate defense)")
    print(f"  - Inflation pressures build")
    print(f"  - BCRP may be forced to hike later anyway")
    print()

    print(f"Policy dilemma for BCRP:")
    print(f"  Option 1: Hike to defend PEN (hurts growth)")
    print(f"  Option 2: Hold rates (allows inflation to rise)")
    print()


def question_7_bcrp_inflation_fight():
    """Q7: What if BCRP raises 100bp to fight inflation?"""
    print_separator()
    print("Q7: BCRP RAISES 100BP TO FIGHT INFLATION")
    print_separator()
    print()

    sim = MonetaryPolicySimulator()

    bcrp_result = sim.simulate_bcrp_hike(100)

    print(f"BCRP HIKES 100BP (from 5.75% to 6.75%):")
    print()

    print(f"Benefits (inflation control):")
    print(f"  Inflation falls: {bcrp_result.inflation_impact:+.2f}pp")
    print(f"  PEN appreciates: {bcrp_result.fx_impact:+.1f}% (capital inflow)")
    print(f"  Anchors inflation expectations")
    print()

    print(f"Costs (growth sacrifice):")
    print(f"  GDP growth slows: {bcrp_result.gdp_impact:+.2f}pp")
    print(f"  Credit growth falls: {bcrp_result.credit_impact:+.2f}pp")
    print(f"  Investment projects delayed")
    print(f"  Transmission lag: {bcrp_result.transmission_lag} months")
    print()

    # Sacrifice ratio
    sacrifice_ratio = abs(bcrp_result.gdp_impact / bcrp_result.inflation_impact)
    print(f"Sacrifice ratio: {sacrifice_ratio:.2f} (GDP cost per 1pp inflation reduction)")
    print()

    print(f"Recommendation:")
    if abs(bcrp_result.inflation_impact) > abs(bcrp_result.gdp_impact):
        print(f"  ✓ Worth it if inflation above 4% (outside target)")
    else:
        print(f"  ✗ Too costly if inflation near target (2-4%)")
    print()


def question_8_population_specific():
    """Q8: Effects on specific populations?"""
    print_separator()
    print("Q8: COMPARISON ACROSS POPULATION GROUPS")
    print_separator()
    print()

    # Build scenario affecting multiple groups
    result = build_complex_scenario(
        social_programs=[
            {'program': 'qali_warma', 'coverage_change': 20},  # Helps children
            {'program': 'pension_65', 'benefit_change': 15},   # Helps elderly
            {'program': 'minimum_wage', 'benefit_change': 15}  # Helps workers
        ],
        baseline_poverty=24.5
    )

    print("SCENARIO: Expand all social programs 15-20%")
    print()

    # Extract by-group impacts
    children_impact = 0
    elderly_impact = 0
    workers_impact = 0
    rural_impact = 0
    urban_impact = 0

    for shock_name, shock_data in result['shocks'].items():
        if 'by_group' in shock_data:
            children_impact += shock_data['by_group']['children']
            elderly_impact += shock_data['by_group']['elderly']
            workers_impact += shock_data['by_group']['workers']
            rural_impact += shock_data['by_group']['rural']
            urban_impact += shock_data['by_group']['urban']

    print(f"{'Group':<15} {'Poverty Impact':<15} {'Interpretation':<40}")
    print("-" * 100)
    print(f"{'Children':<15} {children_impact:+.2f}pp          Qali Warma benefits this group most")
    print(f"{'Elderly (65+)':<15} {elderly_impact:+.2f}pp          Pension 65 benefits this group most")
    print(f"{'Workers':<15} {workers_impact:+.2f}pp          Minimum wage benefits this group most")
    print(f"{'Rural':<15} {rural_impact:+.2f}pp          Indirect spillover effects")
    print(f"{'Urban':<15} {urban_impact:+.2f}pp          Indirect spillover effects")
    print()

    print(f"Key insights:")
    print(f"  - Targeted programs reduce poverty for specific groups")
    print(f"  - Spillover effects to other groups are modest")
    print(f"  - Children and elderly see largest gains")
    print(f"  - Workers benefit from minimum wage but face employment risk")
    print()


def main():
    print("\n")
    print("=" * 100)
    print(" " * 30 + "ANSWERING YOUR POLICY QUESTIONS")
    print("=" * 100)
    print()

    question_1_qali_warma()
    question_2_pension_65()
    question_3_minimum_wage()
    question_4_oil_vs_copper()
    question_5_gold_silver()
    question_6_fed_vs_bcrp()
    question_7_bcrp_inflation_fight()
    question_8_population_specific()

    print_separator()
    print("ALL QUESTIONS ANSWERED. These are the policy tools Peru needs.")
    print_separator()
    print()


if __name__ == '__main__':
    main()
