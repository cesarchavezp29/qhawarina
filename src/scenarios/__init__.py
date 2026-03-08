"""
Scenario and counterfactual analysis module.

Enables "what-if" analysis by injecting shocks into nowcasting models
and propagating impacts across indicators.
"""

from .scenario_engine import (
    Scenario,
    ExogenousShock,
    EndogenousShock,
    ScenarioEngine,
)
from .shock_propagator import ShockPropagator
from .scenario_library import SCENARIO_LIBRARY, get_scenario

__all__ = [
    "Scenario",
    "ExogenousShock",
    "EndogenousShock",
    "ScenarioEngine",
    "ShockPropagator",
    "SCENARIO_LIBRARY",
    "get_scenario",
]
