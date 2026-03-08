"""
Granular Policy Simulators - Answer specific policy questions.

Questions this module answers:
1. "What if we expand Qali Warma to cover X more children?"
2. "What if Pension 65 benefit increases 20%?"
3. "What if minimum wage rises 15%?"
4. "What if oil prices surge 30% but copper falls 10%?"
5. "What if Fed raises rates 50bp but BCRP holds steady?"
6. "What happens to rural vs urban poverty?"
7. "What's the effect on children vs elderly?"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SOCIAL PROGRAM SIMULATORS
# ============================================================================

@dataclass
class SocialProgramImpact:
    """Results from social program expansion/contraction."""
    program_name: str
    baseline_coverage: int  # Number of beneficiaries
    new_coverage: int
    baseline_budget: float  # Million PEN
    new_budget: float
    poverty_impact: float  # pp change in poverty rate
    poverty_impact_by_group: Dict[str, float]  # children, elderly, rural, urban
    gdp_impact: float  # pp (multiplier effect)
    fiscal_cost: float  # Million PEN
    cost_per_person_out_of_poverty: float  # PEN


class SocialProgramSimulator:
    """
    Simulate expansions/contractions of Peru's social programs.

    Key programs:
    - Qali Warma: School feeding program (3.7M children)
    - Juntos: Conditional cash transfer (750k households)
    - Pension 65: Elderly cash transfer (560k beneficiaries)
    - Minimum wage: Affects 2.5M formal workers
    """

    # Program parameters (2024 data)
    PROGRAMS = {
        'qali_warma': {
            'baseline_coverage': 3_700_000,  # Children
            'baseline_budget': 2_100,  # Million PEN/year
            'benefit_per_person': 2_100_000_000 / 3_700_000,  # ~567 PEN/child/year
            'poverty_elasticity': -0.08,  # 10% coverage ↑ → -0.8pp child poverty
            'target_group': 'children',
            'gdp_multiplier': 0.4  # Local demand effect
        },
        'juntos': {
            'baseline_coverage': 750_000,  # Households
            'baseline_budget': 1_800,  # Million PEN/year
            'benefit_per_person': 1_800_000_000 / 750_000,  # 2,400 PEN/household/year
            'poverty_elasticity': -0.15,  # 10% coverage ↑ → -1.5pp poverty
            'target_group': 'rural',
            'gdp_multiplier': 0.6
        },
        'pension_65': {
            'baseline_coverage': 560_000,  # Elderly
            'baseline_budget': 1_400,  # Million PEN/year
            'benefit_per_person': 1_400_000_000 / 560_000,  # 2,500 PEN/elderly/year
            'poverty_elasticity': -0.12,  # 10% coverage ↑ → -1.2pp elderly poverty
            'target_group': 'elderly',
            'gdp_multiplier': 0.3  # Low multiplier (mostly consumption)
        },
        'minimum_wage': {
            'baseline_coverage': 2_500_000,  # Formal workers affected
            'baseline_budget': 0,  # Not a fiscal expense
            'benefit_per_person': 1_025 * 12,  # Current MW: 1,025 PEN/month
            'poverty_elasticity': -0.25,  # 10% MW ↑ → -2.5pp poverty (working poor)
            'target_group': 'workers',
            'gdp_multiplier': 0.2,  # Some disemployment effect
            'employment_elasticity': -0.15  # 10% MW ↑ → -1.5% employment
        }
    }

    # Population shares by group (2024 INEI)
    POPULATION_SHARES = {
        'children': 0.28,  # Under 18
        'elderly': 0.12,   # 65+
        'rural': 0.23,
        'urban': 0.77,
        'workers': 0.45    # Economically active
    }

    def simulate_program_change(
        self,
        program: str,
        pct_change_coverage: float = 0,
        pct_change_benefit: float = 0,
        baseline_poverty: float = 24.5
    ) -> SocialProgramImpact:
        """
        Simulate change in social program.

        Parameters
        ----------
        program : str
            'qali_warma', 'juntos', 'pension_65', 'minimum_wage'
        pct_change_coverage : float
            % change in number of beneficiaries
        pct_change_benefit : float
            % change in benefit amount per person
        baseline_poverty : float
            Current national poverty rate (%)

        Returns
        -------
        SocialProgramImpact
        """
        if program not in self.PROGRAMS:
            raise ValueError(f"Unknown program: {program}. Choose from {list(self.PROGRAMS.keys())}")

        params = self.PROGRAMS[program]

        # Calculate new coverage and budget
        baseline_cov = params['baseline_coverage']
        baseline_budget = params['baseline_budget']
        benefit_pp = params['benefit_per_person']

        new_cov = baseline_cov * (1 + pct_change_coverage / 100)
        new_benefit_pp = benefit_pp * (1 + pct_change_benefit / 100)
        new_budget = (new_cov * new_benefit_pp) / 1_000_000  # Million PEN

        # Calculate poverty impact
        total_pct_change = pct_change_coverage + pct_change_benefit
        poverty_impact = params['poverty_elasticity'] * (total_pct_change / 10)

        # Disaggregate by population group
        target_group = params['target_group']
        poverty_by_group = self._distribute_poverty_impact(
            total_impact=poverty_impact,
            target_group=target_group,
            baseline_poverty=baseline_poverty
        )

        # GDP multiplier effect
        budget_change = new_budget - baseline_budget  # Million PEN
        gdp_impact = (budget_change / 1000) * params['gdp_multiplier']  # pp of GDP

        # Fiscal cost
        fiscal_cost = budget_change

        # Cost-effectiveness
        people_out_of_poverty = abs(poverty_impact) * 0.34_000_000 / 100  # 34M population
        cost_per_person = fiscal_cost * 1_000_000 / people_out_of_poverty if people_out_of_poverty > 0 else 0

        return SocialProgramImpact(
            program_name=program,
            baseline_coverage=int(baseline_cov),
            new_coverage=int(new_cov),
            baseline_budget=baseline_budget,
            new_budget=new_budget,
            poverty_impact=poverty_impact,
            poverty_impact_by_group=poverty_by_group,
            gdp_impact=gdp_impact,
            fiscal_cost=fiscal_cost,
            cost_per_person_out_of_poverty=cost_per_person
        )

    def _distribute_poverty_impact(
        self,
        total_impact: float,
        target_group: str,
        baseline_poverty: float
    ) -> Dict[str, float]:
        """
        Distribute poverty impact across demographic groups.

        Target group gets 80% of impact, others get remainder.
        """
        result = {}

        for group in ['children', 'elderly', 'rural', 'urban', 'workers']:
            if group == target_group:
                result[group] = total_impact * 0.8
            else:
                # Spillover effect
                result[group] = total_impact * 0.2 / 4

        return result


# ============================================================================
# COMMODITY-SPECIFIC SHOCKS
# ============================================================================

@dataclass
class CommodityShockResult:
    """Results from commodity-specific shock."""
    commodity: str
    price_change_pct: float
    gdp_impact: float
    inflation_impact: float
    export_impact: float  # Million USD
    fiscal_impact: float  # Million PEN (canon minero)
    employment_impact: float  # Thousand jobs
    regional_impacts: Dict[str, float]  # By department


class CommoditySimulator:
    """
    Simulate commodity-specific price shocks.

    Peru's export structure (2023):
    - Copper: 28% of exports (Ancash, Moquegua, Arequipa, Tacna)
    - Gold: 15% of exports (Cajamarca, La Libertad, Arequipa)
    - Oil & gas: 8% of exports (Loreto, Ucayali, Piura)
    - Silver: 4% of exports (Junín, Pasco, Lima)
    - Zinc: 7% of exports (Pasco, Junín, Lima)
    - Fishmeal: 5% of exports (Coastal: Piura, Ancash, Ica)
    """

    # Export shares and elasticities
    COMMODITIES = {
        'copper': {
            'export_share': 0.28,
            'gdp_elasticity': 0.20,  # 10% price ↑ → +2.0pp GDP
            'inflation_elasticity': 0.05,
            'canon_rate': 0.50,  # 50% of profits go to canon
            'employment': 120_000,
            'main_regions': ['Ancash', 'Moquegua', 'Arequipa', 'Tacna', 'Cusco']
        },
        'gold': {
            'export_share': 0.15,
            'gdp_elasticity': 0.12,
            'inflation_elasticity': 0.03,
            'canon_rate': 0.50,
            'employment': 80_000,
            'main_regions': ['Cajamarca', 'La Libertad', 'Arequipa', 'Madre de Dios']
        },
        'oil': {
            'export_share': 0.08,
            'gdp_elasticity': -0.08,  # Peru is small net importer (hurts when ↑)
            'inflation_elasticity': 0.35,  # High pass-through to transport/food
            'canon_rate': 0.50,
            'employment': 25_000,
            'main_regions': ['Loreto', 'Ucayali', 'Piura', 'Tumbes']
        },
        'silver': {
            'export_share': 0.04,
            'gdp_elasticity': 0.04,
            'inflation_elasticity': 0.01,
            'canon_rate': 0.50,
            'employment': 30_000,
            'main_regions': ['Junín', 'Pasco', 'Lima', 'Ayacucho']
        },
        'zinc': {
            'export_share': 0.07,
            'gdp_elasticity': 0.08,
            'inflation_elasticity': 0.02,
            'canon_rate': 0.50,
            'employment': 40_000,
            'main_regions': ['Pasco', 'Junín', 'Lima', 'Ancash']
        }
    }

    def simulate_commodity_shock(
        self,
        commodity: str,
        price_change_pct: float,
        baseline_gdp: float = 2.8,
        baseline_inflation: float = 2.3
    ) -> CommodityShockResult:
        """
        Simulate commodity-specific price shock.

        Parameters
        ----------
        commodity : str
            'copper', 'gold', 'oil', 'silver', 'zinc'
        price_change_pct : float
            % change in commodity price
        baseline_gdp : float
            Baseline GDP growth (%)
        baseline_inflation : float
            Baseline inflation (%)

        Returns
        -------
        CommodityShockResult
        """
        if commodity not in self.COMMODITIES:
            raise ValueError(f"Unknown commodity: {commodity}. Choose from {list(self.COMMODITIES.keys())}")

        params = self.COMMODITIES[commodity]

        # GDP and inflation impacts
        gdp_impact = params['gdp_elasticity'] * (price_change_pct / 10)
        inflation_impact = params['inflation_elasticity'] * (price_change_pct / 10)

        # Export impact (assume exports = $60B, commodity share * price change)
        total_exports_usd = 60_000  # Million USD
        commodity_exports = total_exports_usd * params['export_share']
        export_impact = commodity_exports * (price_change_pct / 100)

        # Fiscal impact (canon minero)
        # Assume profit margin = 40% of export value, canon = 50% of profits
        canon_impact = export_impact * 0.40 * params['canon_rate'] * 3.35  # Convert to PEN

        # Employment impact (assume elasticity of employment to price = 0.3)
        employment_impact = params['employment'] * (price_change_pct / 100) * 0.3

        # Regional impacts (distribute GDP impact to main regions)
        regional_impacts = {}
        for region in params['main_regions']:
            # Main region gets proportional share
            regional_impacts[region] = gdp_impact / len(params['main_regions'])

        return CommodityShockResult(
            commodity=commodity,
            price_change_pct=price_change_pct,
            gdp_impact=gdp_impact,
            inflation_impact=inflation_impact,
            export_impact=export_impact,
            fiscal_impact=canon_impact,
            employment_impact=employment_impact / 1000,  # Thousand jobs
            regional_impacts=regional_impacts
        )


# ============================================================================
# MONETARY POLICY: FED vs BCRP
# ============================================================================

@dataclass
class MonetaryPolicyResult:
    """Results from monetary policy shock."""
    policy_authority: str  # 'Fed' or 'BCRP'
    rate_change_bp: float
    gdp_impact: float
    inflation_impact: float
    fx_impact: float  # % change in PEN/USD
    credit_impact: float  # % change in credit growth
    transmission_lag: int  # Months until full effect


class MonetaryPolicySimulator:
    """
    Simulate Fed vs BCRP rate hikes.

    Key difference: Fed hikes affect Peru via capital flows + FX,
    BCRP hikes affect via domestic credit channel.
    """

    def simulate_fed_hike(
        self,
        rate_change_bp: float,
        baseline_gdp: float = 2.8,
        baseline_inflation: float = 2.3
    ) -> MonetaryPolicyResult:
        """
        Simulate Fed rate hike impact on Peru.

        Transmission:
        1. Capital outflow from Peru (carry trade unwind)
        2. PEN depreciation
        3. Import inflation
        4. BCRP may need to raise rates to defend currency

        Parameters
        ----------
        rate_change_bp : float
            Fed rate hike in basis points

        Returns
        -------
        MonetaryPolicyResult
        """
        rate_change_pp = rate_change_bp / 100

        # Capital flow effect → FX depreciation
        # Empirical: 100bp Fed hike → ~3% PEN depreciation
        fx_impact = 0.03 * (rate_change_bp / 100)

        # FX depreciation → inflation (pass-through = 0.25)
        inflation_impact = fx_impact * 100 * 0.25

        # FX depreciation → GDP (net effect small, exports ↑ but imports ↓)
        gdp_impact = -0.10 * rate_change_pp  # Small negative (uncertainty effect dominates)

        # Credit tightening (indirect via BCRP reaction)
        credit_impact = -0.5 * rate_change_pp  # pp change in credit growth

        return MonetaryPolicyResult(
            policy_authority='Fed',
            rate_change_bp=rate_change_bp,
            gdp_impact=gdp_impact,
            inflation_impact=inflation_impact,
            fx_impact=fx_impact * 100,  # Convert to %
            credit_impact=credit_impact,
            transmission_lag=3  # Months
        )

    def simulate_bcrp_hike(
        self,
        rate_change_bp: float,
        baseline_gdp: float = 2.8,
        baseline_inflation: float = 2.3
    ) -> MonetaryPolicyResult:
        """
        Simulate BCRP rate hike impact.

        Transmission:
        1. Domestic credit tightening
        2. Lower consumption/investment
        3. PEN appreciation (capital inflow)
        4. Lower inflation

        Parameters
        ----------
        rate_change_bp : float
            BCRP rate hike in basis points

        Returns
        -------
        MonetaryPolicyResult
        """
        rate_change_pp = rate_change_bp / 100

        # Credit channel (BCRP has strong control over domestic rates)
        credit_impact = -1.2 * rate_change_pp  # pp change in credit growth

        # GDP impact (via investment + consumption)
        gdp_impact = -0.20 * rate_change_pp

        # Inflation impact (demand channel + expectations)
        inflation_impact = -0.15 * rate_change_pp

        # FX impact (PEN appreciates slightly)
        fx_impact = -0.02 * (rate_change_bp / 100)  # Appreciation (negative)

        return MonetaryPolicyResult(
            policy_authority='BCRP',
            rate_change_bp=rate_change_bp,
            gdp_impact=gdp_impact,
            inflation_impact=inflation_impact,
            fx_impact=fx_impact * 100,
            credit_impact=credit_impact,
            transmission_lag=6  # Months (longer than Fed due to domestic frictions)
        )


# ============================================================================
# COMBINED SCENARIO BUILDER
# ============================================================================

def build_complex_scenario(
    social_programs: List[Dict] = None,
    commodity_shocks: List[Dict] = None,
    monetary_policy: List[Dict] = None,
    baseline_gdp: float = 2.8,
    baseline_inflation: float = 2.3,
    baseline_poverty: float = 24.5
) -> Dict:
    """
    Build complex multi-shock scenario.

    Example:
    --------
    scenario = build_complex_scenario(
        social_programs=[
            {'program': 'qali_warma', 'coverage_change': 20},
            {'program': 'pension_65', 'benefit_change': 15}
        ],
        commodity_shocks=[
            {'commodity': 'copper', 'price_change': -15},
            {'commodity': 'oil', 'price_change': 30}
        ],
        monetary_policy=[
            {'authority': 'Fed', 'rate_change': 50},
            {'authority': 'BCRP', 'rate_change': 25}
        ]
    )
    """
    results = {
        'baseline': {
            'gdp': baseline_gdp,
            'inflation': baseline_inflation,
            'poverty': baseline_poverty
        },
        'shocks': {},
        'final': {}
    }

    total_gdp_impact = 0
    total_inflation_impact = 0
    total_poverty_impact = 0
    total_fiscal_cost = 0

    # Process social programs
    if social_programs:
        social_sim = SocialProgramSimulator()
        for shock in social_programs:
            result = social_sim.simulate_program_change(
                program=shock['program'],
                pct_change_coverage=shock.get('coverage_change', 0),
                pct_change_benefit=shock.get('benefit_change', 0),
                baseline_poverty=baseline_poverty
            )
            results['shocks'][f"social_{shock['program']}"] = {
                'gdp_impact': result.gdp_impact,
                'poverty_impact': result.poverty_impact,
                'fiscal_cost': result.fiscal_cost,
                'coverage': result.new_coverage,
                'by_group': result.poverty_impact_by_group
            }
            total_gdp_impact += result.gdp_impact
            total_poverty_impact += result.poverty_impact
            total_fiscal_cost += result.fiscal_cost

    # Process commodity shocks
    if commodity_shocks:
        commodity_sim = CommoditySimulator()
        for shock in commodity_shocks:
            result = commodity_sim.simulate_commodity_shock(
                commodity=shock['commodity'],
                price_change_pct=shock['price_change']
            )
            results['shocks'][f"commodity_{shock['commodity']}"] = {
                'gdp_impact': result.gdp_impact,
                'inflation_impact': result.inflation_impact,
                'export_impact_usd': result.export_impact,
                'fiscal_impact': result.fiscal_impact,
                'employment_impact_k': result.employment_impact,
                'regional_impacts': result.regional_impacts
            }
            total_gdp_impact += result.gdp_impact
            total_inflation_impact += result.inflation_impact
            total_fiscal_cost -= result.fiscal_impact  # Negative = revenue gain

    # Process monetary policy
    if monetary_policy:
        mp_sim = MonetaryPolicySimulator()
        for shock in monetary_policy:
            if shock['authority'].lower() == 'fed':
                result = mp_sim.simulate_fed_hike(shock['rate_change'])
            else:
                result = mp_sim.simulate_bcrp_hike(shock['rate_change'])

            results['shocks'][f"monetary_{shock['authority']}"] = {
                'gdp_impact': result.gdp_impact,
                'inflation_impact': result.inflation_impact,
                'fx_impact': result.fx_impact,
                'credit_impact': result.credit_impact
            }
            total_gdp_impact += result.gdp_impact
            total_inflation_impact += result.inflation_impact

    # Final outcomes
    results['final'] = {
        'gdp': baseline_gdp + total_gdp_impact,
        'gdp_impact': total_gdp_impact,
        'inflation': baseline_inflation + total_inflation_impact,
        'inflation_impact': total_inflation_impact,
        'poverty': baseline_poverty + total_poverty_impact,
        'poverty_impact': total_poverty_impact,
        'fiscal_cost': total_fiscal_cost,
        'interpretation': _interpret_scenario(
            total_gdp_impact,
            total_inflation_impact,
            total_poverty_impact
        )
    }

    return results


def _interpret_scenario(gdp_impact, inflation_impact, poverty_impact):
    """Generate narrative interpretation."""
    parts = []

    if abs(gdp_impact) > 2:
        gdp_msg = "severe contraction" if gdp_impact < 0 else "strong boom"
    elif abs(gdp_impact) > 1:
        gdp_msg = "significant slowdown" if gdp_impact < 0 else "robust growth"
    else:
        gdp_msg = "modest impact on growth"
    parts.append(f"GDP: {gdp_msg} ({gdp_impact:+.1f}pp)")

    if abs(inflation_impact) > 1:
        inf_msg = "above BCRP target" if inflation_impact > 0 else "below target"
        parts.append(f"Inflation: {inf_msg} ({inflation_impact:+.1f}pp)")

    if poverty_impact != 0:
        pov_msg = "worsens" if poverty_impact > 0 else "improves"
        parts.append(f"Poverty: {pov_msg} ({poverty_impact:+.1f}pp)")

    return " | ".join(parts)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 90)
    print("GRANULAR POLICY SIMULATORS - DEMO")
    print("=" * 90)
    print()

    # Test scenario
    result = build_complex_scenario(
        social_programs=[
            {'program': 'qali_warma', 'coverage_change': 20},
            {'program': 'pension_65', 'benefit_change': 15}
        ],
        commodity_shocks=[
            {'commodity': 'copper', 'price_change': -15},
            {'commodity': 'oil', 'price_change': 30}
        ],
        monetary_policy=[
            {'authority': 'Fed', 'rate_change': 50},
            {'authority': 'BCRP', 'rate_change': 25}
        ],
        baseline_gdp=2.8,
        baseline_inflation=2.3,
        baseline_poverty=24.5
    )

    print("Baseline:")
    print(f"  GDP: {result['baseline']['gdp']:.1f}%")
    print(f"  Inflation: {result['baseline']['inflation']:.1f}%")
    print(f"  Poverty: {result['baseline']['poverty']:.1f}%")
    print()

    print("Final Outcome:")
    print(f"  GDP: {result['final']['gdp']:.1f}% ({result['final']['gdp_impact']:+.2f}pp)")
    print(f"  Inflation: {result['final']['inflation']:.1f}% ({result['final']['inflation_impact']:+.2f}pp)")
    print(f"  Poverty: {result['final']['poverty']:.1f}% ({result['final']['poverty_impact']:+.2f}pp)")
    print(f"  Fiscal cost: S/ {result['final']['fiscal_cost']:,.0f}M")
    print()
    print(f"Interpretation: {result['final']['interpretation']}")
    print()
    print("=" * 90)
