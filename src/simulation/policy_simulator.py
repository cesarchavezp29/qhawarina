"""
Policy Simulator - Answer "what if" questions using nowcasting models.

Examples:
- "What if commodity prices fall 10%?"
- "What if BCRP raises rates 50bp?"
- "What if political instability doubles?"
- "What if minimum wage increases 15%?"

Uses DFM models to translate shocks into GDP/inflation/poverty impacts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyShock:
    """Represents a policy intervention or external shock."""
    name: str
    variable: str  # Series ID from panel
    shock_type: str  # 'level', 'growth', 'std_dev'
    magnitude: float  # Size of shock
    persistence: float = 1.0  # How many periods it lasts (1.0 = permanent)
    description: str = ""


@dataclass
class SimulationResult:
    """Results from policy simulation."""
    scenario_name: str
    baseline_gdp: float
    shocked_gdp: float
    gdp_impact: float  # pp change
    baseline_inflation: float
    shocked_inflation: float
    inflation_impact: float  # pp change
    baseline_poverty: Optional[float] = None
    shocked_poverty: Optional[float] = None
    poverty_impact: Optional[float] = None
    transmission_channels: Dict[str, float] = None
    confidence_interval: Tuple[float, float] = None

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            'scenario': self.scenario_name,
            'gdp': {
                'baseline': round(self.baseline_gdp, 2),
                'shocked': round(self.shocked_gdp, 2),
                'impact': round(self.gdp_impact, 2),
                'impact_pct': round(self.gdp_impact, 2)
            },
            'inflation': {
                'baseline': round(self.baseline_inflation, 2),
                'shocked': round(self.shocked_inflation, 2),
                'impact': round(self.inflation_impact, 2)
            },
            'poverty': {
                'baseline': round(self.baseline_poverty, 2) if self.baseline_poverty else None,
                'shocked': round(self.shocked_poverty, 2) if self.shocked_poverty else None,
                'impact': round(self.poverty_impact, 2) if self.poverty_impact else None
            } if self.baseline_poverty else None,
            'transmission': self.transmission_channels or {},
            'confidence': {
                'lower': round(self.confidence_interval[0], 2),
                'upper': round(self.confidence_interval[1], 2)
            } if self.confidence_interval else None
        }


class PolicySimulator:
    """
    Simulate policy scenarios using fitted DFM models.

    Approach:
    1. Load latest panel data and fitted models
    2. Apply shock to specific series
    3. Re-run DFM factor extraction with shocked data
    4. Generate new nowcast
    5. Compare baseline vs shocked outcomes
    """

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parents[2]
        self.data_dir = self.project_root / "data"

        # Calibrated semi-elasticities (from VAR/SVAR literature on Peru)
        # These map shocks to outcomes when DFM not available
        self.calibrated_effects = {
            'commodity_prices': {
                'gdp_elasticity': 0.15,  # 10% commodity price ↑ → +1.5pp GDP
                'inflation_elasticity': 0.08,  # 10% commodity ↑ → +0.8pp inflation
            },
            'fx_rate': {
                'gdp_elasticity': -0.12,  # 10% depreciation → -1.2pp GDP
                'inflation_elasticity': 0.25,  # 10% depreciation → +2.5pp inflation (pass-through)
            },
            'interest_rate': {
                'gdp_elasticity': -0.20,  # 100bp hike → -0.20pp GDP
                'inflation_elasticity': -0.15,  # 100bp hike → -0.15pp inflation
            },
            'political_instability': {
                'gdp_elasticity': -0.10,  # +1 std dev instability → -0.10pp GDP
                'inflation_elasticity': 0.05,  # +1 std dev → +0.05pp inflation (via risk premium)
            },
            'minimum_wage': {
                'poverty_elasticity': -0.30,  # 10% MW increase → -3pp poverty
                'inflation_elasticity': 0.03,  # 10% MW → +0.3pp inflation
            }
        }

    def simulate_shock(
        self,
        shock: PolicyShock,
        baseline_gdp: float,
        baseline_inflation: float,
        baseline_poverty: float = None,
        use_dfm: bool = False
    ) -> SimulationResult:
        """
        Simulate a policy shock and compute impacts.

        Parameters
        ----------
        shock : PolicyShock
            The policy intervention to simulate
        baseline_gdp : float
            Baseline GDP growth forecast (%)
        baseline_inflation : float
            Baseline inflation forecast (%)
        baseline_poverty : float, optional
            Baseline poverty rate (%)
        use_dfm : bool
            If True, re-run DFM with shocked data (slower but more accurate)
            If False, use calibrated semi-elasticities (faster)

        Returns
        -------
        SimulationResult
        """
        logger.info(f"Simulating shock: {shock.name}")

        if use_dfm:
            return self._simulate_with_dfm(shock, baseline_gdp, baseline_inflation, baseline_poverty)
        else:
            return self._simulate_with_elasticities(shock, baseline_gdp, baseline_inflation, baseline_poverty)

    def _simulate_with_elasticities(
        self,
        shock: PolicyShock,
        baseline_gdp: float,
        baseline_inflation: float,
        baseline_poverty: Optional[float]
    ) -> SimulationResult:
        """Fast simulation using calibrated elasticities."""

        # Map shock to calibration category
        shock_category = self._categorize_shock(shock)

        if shock_category not in self.calibrated_effects:
            logger.warning(f"No calibration for {shock_category}, using generic elasticities")
            shock_category = 'commodity_prices'  # Default

        effects = self.calibrated_effects[shock_category]

        # Compute impacts (magnitude is in % terms)
        shock_magnitude = shock.magnitude * shock.persistence

        gdp_impact = effects.get('gdp_elasticity', 0) * shock_magnitude
        inflation_impact = effects.get('inflation_elasticity', 0) * shock_magnitude
        poverty_impact = effects.get('poverty_elasticity', 0) * shock_magnitude if baseline_poverty else None

        # Add uncertainty (assume ±50% standard error)
        se_gdp = abs(gdp_impact) * 0.5
        ci = (gdp_impact - 1.96*se_gdp, gdp_impact + 1.96*se_gdp)

        return SimulationResult(
            scenario_name=shock.name,
            baseline_gdp=baseline_gdp,
            shocked_gdp=baseline_gdp + gdp_impact,
            gdp_impact=gdp_impact,
            baseline_inflation=baseline_inflation,
            shocked_inflation=baseline_inflation + inflation_impact,
            inflation_impact=inflation_impact,
            baseline_poverty=baseline_poverty,
            shocked_poverty=baseline_poverty + poverty_impact if baseline_poverty else None,
            poverty_impact=poverty_impact,
            transmission_channels=self._get_transmission_channels(shock_category, shock_magnitude),
            confidence_interval=ci
        )

    def _categorize_shock(self, shock: PolicyShock) -> str:
        """Map shock variable to calibration category."""
        var_lower = shock.variable.lower()

        if 'copper' in var_lower or 'commodity' in var_lower or 'export' in var_lower:
            return 'commodity_prices'
        elif 'fx' in var_lower or 'tc' in var_lower or 'exchange' in var_lower:
            return 'fx_rate'
        elif 'rate' in var_lower or 'tasa' in var_lower:
            return 'interest_rate'
        elif 'political' in var_lower or 'instability' in var_lower:
            return 'political_instability'
        elif 'wage' in var_lower or 'salario' in var_lower:
            return 'minimum_wage'
        else:
            return 'commodity_prices'  # Default

    def _get_transmission_channels(self, category: str, magnitude: float) -> Dict[str, float]:
        """Explain how shock transmits to outcomes."""
        channels = {
            'commodity_prices': {
                'exports': 0.4 * magnitude,
                'investment': 0.3 * magnitude,
                'fiscal_revenue': 0.2 * magnitude,
                'confidence': 0.1 * magnitude
            },
            'fx_rate': {
                'import_prices': 0.6 * magnitude,
                'export_competitiveness': 0.3 * magnitude,
                'debt_service': 0.1 * magnitude
            },
            'interest_rate': {
                'investment': 0.5 * magnitude,
                'consumption': 0.3 * magnitude,
                'credit': 0.2 * magnitude
            },
            'political_instability': {
                'investment': 0.5 * magnitude,
                'consumer_confidence': 0.3 * magnitude,
                'risk_premium': 0.2 * magnitude
            },
            'minimum_wage': {
                'labor_income': 0.6 * magnitude,
                'employment': -0.2 * magnitude,
                'prices': 0.2 * magnitude
            }
        }
        return channels.get(category, {})

    def _simulate_with_dfm(
        self,
        shock: PolicyShock,
        baseline_gdp: float,
        baseline_inflation: float,
        baseline_poverty: Optional[float]
    ) -> SimulationResult:
        """
        More accurate simulation by re-running DFM with shocked data.

        Steps:
        1. Load latest panel
        2. Apply shock to specified series
        3. Re-extract factors
        4. Re-run bridge equation
        5. Compare to baseline

        NOT IMPLEMENTED YET - requires loading fitted DFM models
        """
        raise NotImplementedError(
            "DFM-based simulation not yet implemented. "
            "Use use_dfm=False for elasticity-based simulation."
        )


class ScenarioLibrary:
    """Pre-configured policy scenarios."""

    @staticmethod
    def commodity_crash(magnitude: float = -20) -> PolicyShock:
        """Global commodity price crash (e.g., copper -20%)."""
        return PolicyShock(
            name=f"Commodity Crash ({magnitude:+.0f}%)",
            variable="copper_price",
            shock_type="growth",
            magnitude=magnitude,
            persistence=1.0,
            description=f"Global commodity prices fall {abs(magnitude):.0f}% due to China slowdown"
        )

    @staticmethod
    def fx_crisis(magnitude: float = 15) -> PolicyShock:
        """Currency crisis (e.g., PEN depreciates 15%)."""
        return PolicyShock(
            name=f"Currency Crisis ({magnitude:+.0f}%)",
            variable="fx_rate",
            shock_type="growth",
            magnitude=magnitude,
            persistence=0.8,  # Partially reversed by BCRP intervention
            description=f"PEN depreciates {magnitude:.0f}% due to capital flight"
        )

    @staticmethod
    def rate_hike(magnitude: float = 100) -> PolicyShock:
        """BCRP rate hike (e.g., +100bp)."""
        return PolicyShock(
            name=f"Rate Hike ({magnitude:+.0f}bp)",
            variable="reference_rate",
            shock_type="level",
            magnitude=magnitude / 100,  # Convert bp to pp
            persistence=1.0,
            description=f"BCRP raises reference rate by {magnitude:.0f}bp to fight inflation"
        )

    @staticmethod
    def political_crisis(magnitude: float = 2.0) -> PolicyShock:
        """Political instability surge (e.g., +2 std dev)."""
        return PolicyShock(
            name=f"Political Crisis ({magnitude:+.1f}σ)",
            variable="political_index",
            shock_type="std_dev",
            magnitude=magnitude,
            persistence=0.5,  # Crisis fades over time
            description=f"Political instability index rises {magnitude:.1f} standard deviations (protests, cabinet changes)"
        )

    @staticmethod
    def minimum_wage_hike(magnitude: float = 15) -> PolicyShock:
        """Minimum wage increase (e.g., +15%)."""
        return PolicyShock(
            name=f"Minimum Wage Hike ({magnitude:+.0f}%)",
            variable="minimum_wage",
            shock_type="growth",
            magnitude=magnitude,
            persistence=1.0,
            description=f"Government raises minimum wage by {magnitude:.0f}%"
        )

    @staticmethod
    def china_slowdown(magnitude: float = -2.0) -> PolicyShock:
        """China GDP slowdown (indirect via commodity/export demand)."""
        return PolicyShock(
            name=f"China Slowdown ({magnitude:+.1f}pp)",
            variable="export_demand",
            shock_type="growth",
            magnitude=magnitude * 5,  # Amplified via commodity channel
            persistence=1.0,
            description=f"China GDP growth slows by {abs(magnitude):.1f}pp, reducing demand for Peruvian exports"
        )

    @staticmethod
    def el_nino(magnitude: float = -3.0) -> PolicyShock:
        """El Niño climate shock (agricultural production fall)."""
        return PolicyShock(
            name="El Niño (Strong)",
            variable="agriculture_production",
            shock_type="growth",
            magnitude=magnitude,
            persistence=0.3,  # Temporary shock (one season)
            description="Strong El Niño reduces agricultural output and increases food prices"
        )


def compare_scenarios(
    scenarios: List[PolicyShock],
    baseline_gdp: float,
    baseline_inflation: float,
    baseline_poverty: float = None
) -> pd.DataFrame:
    """
    Compare multiple policy scenarios side-by-side.

    Returns DataFrame with impacts for easy comparison.
    """
    simulator = PolicySimulator()

    results = []
    for scenario in scenarios:
        result = simulator.simulate_shock(
            scenario,
            baseline_gdp,
            baseline_inflation,
            baseline_poverty
        )
        results.append({
            'Scenario': result.scenario_name,
            'GDP Impact (pp)': result.gdp_impact,
            'Inflation Impact (pp)': result.inflation_impact,
            'Poverty Impact (pp)': result.poverty_impact,
            'GDP Final (%)': result.shocked_gdp,
            'Inflation Final (%)': result.shocked_inflation,
            'Poverty Final (%)': result.shocked_poverty
        })

    df = pd.DataFrame(results)
    df = df.round(2)
    return df


if __name__ == '__main__':
    # Example: Compare multiple downside scenarios
    logging.basicConfig(level=logging.INFO)

    # Current nowcasts (example values)
    baseline_gdp = 2.8  # Q1 2026 GDP growth forecast
    baseline_inflation = 2.3  # March 2026 inflation forecast
    baseline_poverty = 24.5  # 2026 poverty rate forecast

    scenarios = [
        ScenarioLibrary.commodity_crash(-20),
        ScenarioLibrary.fx_crisis(15),
        ScenarioLibrary.political_crisis(2.0),
        ScenarioLibrary.rate_hike(100),
        ScenarioLibrary.el_nino(-3.0),
        ScenarioLibrary.china_slowdown(-2.0)
    ]

    print("=" * 80)
    print("POLICY SCENARIO COMPARISON")
    print("=" * 80)
    print(f"\nBaseline Forecast:")
    print(f"  GDP Growth:      {baseline_gdp:.1f}%")
    print(f"  Inflation:       {baseline_inflation:.1f}%")
    print(f"  Poverty Rate:    {baseline_poverty:.1f}%")
    print("\n" + "=" * 80)

    results_df = compare_scenarios(scenarios, baseline_gdp, baseline_inflation, baseline_poverty)
    print("\nScenario Impacts:")
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)
