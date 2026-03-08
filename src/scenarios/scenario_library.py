"""
Library of pre-built scenarios for common counterfactual analyses.

These scenarios are ready-to-use for firms doing strategic planning.
"""

from .scenario_engine import Scenario, ExogenousShock, EndogenousShock

# =============================================================================
# RECESSION SCENARIOS
# =============================================================================

MILD_RECESSION = Scenario(
    name="Mild Recession",
    description="GDP growth slows to 0%, mild political instability increase",
    endogenous_shocks=[
        EndogenousShock(
            target="gdp",
            forced_value=0.0,
            description="GDP growth constrained to 0%",
            period="2026-Q2",
        )
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=1.0,
            description="Financial stress rises to +1 sigma",
        )
    ],
    tags=["recession", "mild", "domestic"],
)

SEVERE_RECESSION = Scenario(
    name="Severe Recession",
    description="GDP contracts -2%, high political instability, FX depreciation",
    endogenous_shocks=[
        EndogenousShock(
            target="gdp",
            forced_value=-2.0,
            description="GDP contracts -2%",
            period="2026-Q2",
        )
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=2.0,
            description="Political crisis (+2 sigma)",
        ),
        ExogenousShock(
            series_id="PN01246PM",  # FX rate
            shock_type="percentage",
            shock_value=0.15,
            description="Sol depreciates 15%",
        ),
    ],
    tags=["recession", "severe", "crisis"],
)

# =============================================================================
# INFLATION SCENARIOS
# =============================================================================

INFLATION_SPIKE = Scenario(
    name="Inflation Spike",
    description="Inflation jumps to 0.5% monthly (6% annual), driven by food and FX",
    endogenous_shocks=[
        EndogenousShock(
            target="inflation",
            forced_value=0.5,
            description="Inflation = 0.5% monthly",
            period="2026-03",
        )
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="MIDAGRI_CHICKEN_VAR",
            shock_type="percentage",
            shock_value=0.30,
            description="Chicken prices +30%",
        ),
        ExogenousShock(
            series_id="PN01246PM",  # FX rate
            shock_type="percentage",
            shock_value=0.10,
            description="Sol depreciates 10%",
        ),
    ],
    tags=["inflation", "food", "external"],
)

DEFLATION = Scenario(
    name="Deflation",
    description="Prices fall -0.2% monthly due to demand collapse",
    endogenous_shocks=[
        EndogenousShock(
            target="inflation",
            forced_value=-0.2,
            description="Deflation -0.2% monthly",
            period="2026-03",
        )
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="SUPERMARKET_FOOD_VAR",
            shock_type="absolute",
            shock_value=-0.5,
            description="Food prices fall sharply",
        ),
    ],
    tags=["deflation", "demand_shock"],
)

# =============================================================================
# POLITICAL CRISIS SCENARIOS
# =============================================================================

POLITICAL_CRISIS = Scenario(
    name="Political Crisis",
    description="Major political instability (+3 sigma), business confidence collapses",
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=3.0,
            description="Extreme political instability (protests, resignations)",
        ),
        ExogenousShock(
            series_id="PD37981AM",  # Business confidence
            shock_type="percentage",
            shock_value=-0.30,
            description="Business confidence drops 30%",
        ),
    ],
    tags=["political", "crisis", "uncertainty"],
)

INSTITUTIONAL_REFORM = Scenario(
    name="Institutional Reform",
    description="Political stability improves significantly (-1.5 sigma)",
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=-1.5,
            description="Political stability improves (reforms, elections)",
        ),
        ExogenousShock(
            series_id="PD37981AM",  # Business confidence
            shock_type="percentage",
            shock_value=0.20,
            description="Business confidence rises 20%",
        ),
    ],
    tags=["political", "reform", "positive"],
)

# =============================================================================
# EXTERNAL SHOCK SCENARIOS
# =============================================================================

COMMODITY_BOOM = Scenario(
    name="Commodity Boom",
    description="Copper/gold prices surge, driving GDP and FX appreciation",
    exogenous_shocks=[
        ExogenousShock(
            series_id="PN01450PM",  # Exports (if available)
            shock_type="percentage",
            shock_value=0.25,
            description="Export prices +25%",
        ),
        ExogenousShock(
            series_id="PN01246PM",  # FX rate
            shock_type="percentage",
            shock_value=-0.10,
            description="Sol appreciates 10%",
        ),
    ],
    tags=["external", "commodity", "positive"],
)

GLOBAL_RECESSION = Scenario(
    name="Global Recession",
    description="China slowdown, US recession, commodity prices crash",
    exogenous_shocks=[
        ExogenousShock(
            series_id="PN01450PM",  # Exports
            shock_type="percentage",
            shock_value=-0.30,
            description="Export demand collapses -30%",
        ),
        ExogenousShock(
            series_id="PN01246PM",  # FX rate
            shock_type="percentage",
            shock_value=0.20,
            description="Sol depreciates 20%",
        ),
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=1.5,
            description="Political stress from economic headwinds",
        ),
    ],
    tags=["external", "global", "recession"],
)

# =============================================================================
# COMBINED/STRESS TEST SCENARIOS
# =============================================================================

PERFECT_STORM = Scenario(
    name="Perfect Storm",
    description="Simultaneous political crisis, external shock, and inflation surge",
    endogenous_shocks=[
        EndogenousShock(
            target="gdp",
            forced_value=-3.0,
            description="Severe GDP contraction -3%",
            period="2026-Q2",
        ),
        EndogenousShock(
            target="inflation",
            forced_value=0.8,
            description="High inflation 0.8% monthly",
            period="2026-03",
        ),
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=2.5,
            description="Major political crisis",
        ),
        ExogenousShock(
            series_id="PN01246PM",  # FX rate
            shock_type="percentage",
            shock_value=0.25,
            description="Currency crisis: Sol depreciates 25%",
        ),
    ],
    tags=["stress_test", "extreme", "crisis"],
)

GOLDILOCKS = Scenario(
    name="Goldilocks Scenario",
    description="Ideal conditions: Moderate growth, low inflation, political stability",
    endogenous_shocks=[
        EndogenousShock(
            target="gdp",
            forced_value=3.5,
            description="Healthy GDP growth 3.5%",
            period="2026-Q2",
        ),
        EndogenousShock(
            target="inflation",
            forced_value=0.17,
            description="Target inflation ~2% annual",
            period="2026-03",
        ),
    ],
    exogenous_shocks=[
        ExogenousShock(
            series_id="FINANCIAL_STRESS_INDEX",
            shock_type="sigma",
            shock_value=-1.0,
            description="Political stability (-1 sigma)",
        ),
    ],
    tags=["positive", "ideal", "planning"],
)

# =============================================================================
# SCENARIO LIBRARY
# =============================================================================

SCENARIO_LIBRARY = {
    # Recession
    "mild_recession": MILD_RECESSION,
    "severe_recession": SEVERE_RECESSION,
    # Inflation
    "inflation_spike": INFLATION_SPIKE,
    "deflation": DEFLATION,
    # Political
    "political_crisis": POLITICAL_CRISIS,
    "institutional_reform": INSTITUTIONAL_REFORM,
    # External
    "commodity_boom": COMMODITY_BOOM,
    "global_recession": GLOBAL_RECESSION,
    # Combined
    "perfect_storm": PERFECT_STORM,
    "goldilocks": GOLDILOCKS,
}


def get_scenario(name: str) -> Scenario:
    """
    Get a pre-built scenario by name.

    Args:
        name: Scenario name (e.g., "mild_recession", "inflation_spike")

    Returns:
        Scenario object

    Raises:
        KeyError if scenario not found
    """
    if name not in SCENARIO_LIBRARY:
        available = ", ".join(SCENARIO_LIBRARY.keys())
        raise KeyError(
            f"Scenario '{name}' not found. Available: {available}"
        )

    return SCENARIO_LIBRARY[name]


def list_scenarios() -> dict:
    """
    List all available scenarios with descriptions.

    Returns:
        Dictionary mapping scenario names to descriptions
    """
    return {
        name: scenario.description
        for name, scenario in SCENARIO_LIBRARY.items()
    }
