"""
Cross-model shock propagation.

Propagates shocks across models using empirical elasticities:
- GDP → Poverty (Okun-style relationship)
- Inflation → Poverty (real income effect)
- Political Risk → GDP (uncertainty channel)
- FX → Inflation (pass-through)
"""

from typing import Dict, Optional
import numpy as np


class ShockPropagator:
    """
    Propagates shocks across economic indicators using empirical elasticities.

    Elasticities are estimated from historical Peruvian data or literature.
    """

    # Empirical elasticities (approximate, based on Peru data)
    ELASTICITIES = {
        # GDP → Poverty: 1pp GDP growth → -0.5pp poverty (semi-elasticity)
        "gdp_to_poverty": -0.5,
        # Inflation → Poverty: 1pp inflation → +0.3pp poverty
        "inflation_to_poverty": 0.3,
        # Political Risk → GDP: 1 sigma political shock → -0.8pp GDP
        "political_to_gdp": -0.8,
        # FX depreciation → Inflation: 1% depreciation → 0.15% inflation (pass-through)
        "fx_to_inflation": 0.15,
        # GDP → Employment: 1pp GDP → 0.6pp employment (Okun coefficient)
        "gdp_to_employment": 0.6,
        # Confidence → GDP: 1 sigma confidence drop → -0.5pp GDP
        "confidence_to_gdp": 0.5,
    }

    def __init__(self, custom_elasticities: Optional[Dict[str, float]] = None):
        """
        Initialize propagator with elasticities.

        Args:
            custom_elasticities: Override default elasticities
        """
        self.elasticities = self.ELASTICITIES.copy()
        if custom_elasticities:
            self.elasticities.update(custom_elasticities)

    def propagate_gdp_shock(self, gdp_impact: float) -> Dict[str, float]:
        """
        Propagate GDP shock to other indicators.

        Args:
            gdp_impact: GDP growth impact in pp (e.g., -2.0 for -2pp)

        Returns:
            Dictionary of impacts on other indicators
        """
        return {
            "poverty_impact_pp": gdp_impact * self.elasticities["gdp_to_poverty"],
            "employment_impact_pp": gdp_impact * self.elasticities["gdp_to_employment"],
            "mechanism": "GDP shock → Poverty (Okun-style), Employment",
        }

    def propagate_inflation_shock(self, inflation_impact: float) -> Dict[str, float]:
        """
        Propagate inflation shock to other indicators.

        Args:
            inflation_impact: Inflation impact in pp monthly (e.g., 1.0 for +1pp)

        Returns:
            Dictionary of impacts
        """
        # Annualize for poverty impact (poverty responds to annual inflation)
        annual_inflation_impact = inflation_impact * 12

        return {
            "poverty_impact_pp": annual_inflation_impact
            * self.elasticities["inflation_to_poverty"],
            "real_income_loss_pct": annual_inflation_impact,  # Direct purchasing power
            "mechanism": "Inflation → Real income erosion → Poverty",
        }

    def propagate_political_shock(self, political_sigma: float) -> Dict[str, float]:
        """
        Propagate political risk shock to economic indicators.

        Args:
            political_sigma: Political instability in sigma units (e.g., +2.0)

        Returns:
            Dictionary of impacts
        """
        gdp_impact = political_sigma * self.elasticities["political_to_gdp"]

        # Political shock → GDP → Poverty (second-round effect)
        poverty_impact = gdp_impact * self.elasticities["gdp_to_poverty"]

        return {
            "gdp_impact_pp": gdp_impact,
            "poverty_impact_pp": poverty_impact,
            "investment_impact": "Negative (uncertainty channel)",
            "mechanism": "Political instability → Uncertainty → Lower investment/GDP → Poverty",
        }

    def propagate_fx_shock(self, fx_depreciation_pct: float) -> Dict[str, float]:
        """
        Propagate FX shock to inflation.

        Args:
            fx_depreciation_pct: FX depreciation in % (e.g., 10.0 for +10%)

        Returns:
            Dictionary of impacts
        """
        inflation_impact = fx_depreciation_pct * self.elasticities["fx_to_inflation"]

        # FX → Inflation → Poverty (second-round)
        poverty_impact = (inflation_impact * 12) * self.elasticities["inflation_to_poverty"]

        return {
            "inflation_impact_monthly_pp": inflation_impact,
            "inflation_impact_annual_pp": inflation_impact * 12,
            "poverty_impact_pp": poverty_impact,
            "mechanism": "FX depreciation → Import prices → Inflation → Real income → Poverty",
        }

    def propagate_full_scenario(
        self,
        gdp_impact: Optional[float] = None,
        inflation_impact: Optional[float] = None,
        political_sigma: Optional[float] = None,
        fx_depreciation_pct: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """
        Propagate multiple shocks simultaneously and aggregate impacts.

        Args:
            gdp_impact: GDP growth impact in pp
            inflation_impact: Inflation impact in monthly pp
            political_sigma: Political instability in sigma
            fx_depreciation_pct: FX depreciation in %

        Returns:
            Dictionary with:
            - individual_impacts: Each shock's propagation
            - aggregate_impacts: Sum of all impacts on each indicator
        """
        individual = {}
        aggregate = {
            "gdp_total_pp": 0.0,
            "inflation_total_monthly_pp": 0.0,
            "poverty_total_pp": 0.0,
        }

        if gdp_impact is not None:
            individual["gdp_shock"] = self.propagate_gdp_shock(gdp_impact)
            aggregate["gdp_total_pp"] += gdp_impact
            aggregate["poverty_total_pp"] += individual["gdp_shock"]["poverty_impact_pp"]

        if inflation_impact is not None:
            individual["inflation_shock"] = self.propagate_inflation_shock(inflation_impact)
            aggregate["inflation_total_monthly_pp"] += inflation_impact
            aggregate["poverty_total_pp"] += individual["inflation_shock"][
                "poverty_impact_pp"
            ]

        if political_sigma is not None:
            individual["political_shock"] = self.propagate_political_shock(political_sigma)
            aggregate["gdp_total_pp"] += individual["political_shock"]["gdp_impact_pp"]
            aggregate["poverty_total_pp"] += individual["political_shock"]["poverty_impact_pp"]

        if fx_depreciation_pct is not None:
            individual["fx_shock"] = self.propagate_fx_shock(fx_depreciation_pct)
            aggregate["inflation_total_monthly_pp"] += individual["fx_shock"][
                "inflation_impact_monthly_pp"
            ]
            aggregate["poverty_total_pp"] += individual["fx_shock"]["poverty_impact_pp"]

        return {
            "individual_impacts": individual,
            "aggregate_impacts": aggregate,
            "interpretation": self._generate_interpretation(aggregate),
        }

    def _generate_interpretation(self, aggregate: Dict[str, float]) -> str:
        """Generate human-readable interpretation of aggregate impacts."""
        parts = []

        if abs(aggregate["gdp_total_pp"]) > 0.1:
            direction = "increase" if aggregate["gdp_total_pp"] > 0 else "decrease"
            parts.append(
                f"GDP would {direction} by {abs(aggregate['gdp_total_pp']):.1f}pp"
            )

        if abs(aggregate["inflation_total_monthly_pp"]) > 0.05:
            direction = "increase" if aggregate["inflation_total_monthly_pp"] > 0 else "decrease"
            annual = aggregate["inflation_total_monthly_pp"] * 12
            parts.append(
                f"Inflation would {direction} by {abs(annual):.1f}pp annually"
            )

        if abs(aggregate["poverty_total_pp"]) > 0.1:
            direction = "increase" if aggregate["poverty_total_pp"] > 0 else "decrease"
            parts.append(
                f"Poverty would {direction} by {abs(aggregate['poverty_total_pp']):.1f}pp"
            )

        if parts:
            return ". ".join(parts) + "."
        else:
            return "Minimal aggregate impact detected."
