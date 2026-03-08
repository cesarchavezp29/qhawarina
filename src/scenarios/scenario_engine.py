"""
Core scenario engine for counterfactual analysis.

Supports two types of shocks:
1. ExogenousShock: Override input series (e.g., "FX rate +10%")
2. EndogenousShock: Force model output (e.g., "GDP growth = 0%")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class ExogenousShock:
    """
    Shock to an input series/feature.

    Examples:
    - "FX volatility increases by 50%"
    - "Political events spike to +2 sigma"
    - "Credit spreads widen by 200bp"
    """

    series_id: str  # e.g., "PN01246PM" (FX rate)
    shock_type: Literal["absolute", "percentage", "sigma"]
    shock_value: float  # e.g., 0.10 for +10%, 2.0 for +2 sigma
    description: str
    start_date: Optional[str] = None  # ISO format YYYY-MM-DD
    end_date: Optional[str] = None  # If None, shock applies to forecast period only

    def apply(self, series: pd.Series) -> pd.Series:
        """Apply shock to a pandas Series."""
        result = series.copy()

        # Determine date range
        if self.start_date:
            mask = result.index >= pd.to_datetime(self.start_date)
            if self.end_date:
                mask &= result.index <= pd.to_datetime(self.end_date)
        else:
            # Apply to all dates (useful for historical counterfactuals)
            mask = pd.Series(True, index=result.index)

        # Apply shock
        if self.shock_type == "absolute":
            result.loc[mask] += self.shock_value
        elif self.shock_type == "percentage":
            result.loc[mask] *= (1 + self.shock_value)
        elif self.shock_type == "sigma":
            std = series.std()
            result.loc[mask] += self.shock_value * std
        else:
            raise ValueError(f"Unknown shock_type: {self.shock_type}")

        return result


@dataclass
class EndogenousShock:
    """
    Force a model's output to a specific value.

    Examples:
    - "GDP growth = 0%"
    - "Inflation = 5%"
    - "Poverty rate = 30%"

    Used to explore downstream impacts when one indicator is constrained.
    """

    target: Literal["gdp", "inflation", "poverty", "political_risk"]
    forced_value: float
    description: str
    period: str  # e.g., "2026-Q1" or "2026-01"


@dataclass
class Scenario:
    """
    A complete scenario with multiple shocks.

    Scenarios can combine exogenous and endogenous shocks to model
    complex situations like:
    - Recession: GDP forced to -2%, political instability +1 sigma
    - Inflation spike: Commodity prices +30%, FX depreciation +15%
    - Political crisis: Political events +3 sigma, business confidence -20%
    """

    name: str
    description: str
    exogenous_shocks: List[ExogenousShock] = field(default_factory=list)
    endogenous_shocks: List[EndogenousShock] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # e.g., ["recession", "external"]

    def apply_to_panel(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all exogenous shocks to a panel DataFrame.

        Args:
            panel_df: Long-format panel with columns [date, series_id, value_*, ...]

        Returns:
            Modified panel with shocks applied
        """
        result = panel_df.copy()

        for shock in self.exogenous_shocks:
            # Filter to series
            mask = result["series_id"] == shock.series_id
            if not mask.any():
                print(f"Warning: series_id '{shock.series_id}' not found in panel")
                continue

            # Get the series
            series_data = result.loc[mask].set_index("date")

            # Apply shock to relevant value columns
            for col in ["value_raw", "value_sa", "value_yoy", "value_dlog"]:
                if col in series_data.columns:
                    shocked = shock.apply(series_data[col])
                    result.loc[mask, col] = shocked.values

        return result


class ScenarioEngine:
    """
    Engine for running counterfactual scenarios through nowcasting models.

    Usage:
        engine = ScenarioEngine()

        # Create scenario
        scenario = Scenario(
            name="Mild Recession",
            description="GDP growth slows to 0%, political instability rises",
            endogenous_shocks=[
                EndogenousShock("gdp", 0.0, "GDP growth = 0%", "2026-Q2")
            ],
            exogenous_shocks=[
                ExogenousShock("POLITICAL_INDEX", "sigma", 1.5,
                              "Political instability +1.5 sigma")
            ]
        )

        # Run scenario
        results = engine.run_scenario(scenario, panel_df, models)
    """

    def __init__(self):
        self.baseline_results = {}
        self.scenario_results = {}

    def run_scenario(
        self,
        scenario: Scenario,
        panel_df: pd.DataFrame,
        gdp_model=None,
        inflation_model=None,
        poverty_model=None,
        political_model=None,
        gdp_targets=None,
        inflation_targets=None,
        target_period: Optional[str] = None,
    ) -> Dict:
        """
        Run a scenario through all nowcasting models.

        Args:
            scenario: The scenario to run
            panel_df: National panel DataFrame
            gdp_model: Trained GDP nowcaster
            inflation_model: Trained inflation nowcaster
            poverty_model: Trained poverty nowcaster (needs dept panel)
            political_model: Political index builder
            gdp_targets: GDP quarterly targets DataFrame
            inflation_targets: Inflation monthly targets DataFrame
            target_period: Period to nowcast (e.g., "2026-Q1" or "2026-02")

        Returns:
            Dictionary with baseline and counterfactual results
        """
        results = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "tags": scenario.tags,
            },
            "baseline": {},
            "counterfactual": {},
            "impacts": {},
        }

        # Apply exogenous shocks to panel (long format)
        shocked_panel_long = scenario.apply_to_panel(panel_df)

        # Convert both baseline and shocked panels to wide format
        # We need to pivot from long (date, series_id, value_*) to wide (date x series_id)
        from src.backtesting.vintage import VintageManager

        # Create a temporary VM instance just for pivoting
        # (we don't actually need the full vintage functionality here)
        def pivot_panel(long_panel):
            """Helper to pivot long panel to wide."""
            result = long_panel.pivot(
                index="date", columns="series_id", values="value_raw"
            )
            result.index = pd.to_datetime(result.index)
            return result.sort_index()

        panel_wide = pivot_panel(panel_df)
        shocked_panel_wide = pivot_panel(shocked_panel_long)

        # Run each model with baseline and shocked data
        if gdp_model and gdp_targets is not None:
            results["baseline"]["gdp"] = self._run_gdp_baseline(
                gdp_model, panel_wide, gdp_targets
            )
            results["counterfactual"]["gdp"] = self._run_gdp_counterfactual(
                gdp_model, shocked_panel_wide, gdp_targets, scenario
            )
            results["impacts"]["gdp"] = (
                results["counterfactual"]["gdp"]["gdp_yoy"]
                - results["baseline"]["gdp"]["gdp_yoy"]
            )

        if inflation_model and inflation_targets is not None:
            results["baseline"]["inflation"] = self._run_inflation_baseline(
                inflation_model, panel_wide, inflation_targets
            )
            results["counterfactual"]["inflation"] = self._run_inflation_counterfactual(
                inflation_model, shocked_panel_wide, inflation_targets, scenario
            )
            results["impacts"]["inflation"] = (
                results["counterfactual"]["inflation"]["ipc_monthly_var"]
                - results["baseline"]["inflation"]["ipc_monthly_var"]
            )

        # Poverty would need departmental panel
        # Political risk needs event classification + financial data

        return results

    def _run_gdp_baseline(self, model, panel_wide, target_df):
        """Run GDP model with baseline data (expects wide panel)."""
        try:
            result = model.nowcast(panel_wide, target_df)
            return {
                "gdp_yoy": result.get("nowcast_value", np.nan),
                "confidence_lower": result.get("confidence_lower"),
                "confidence_upper": result.get("confidence_upper"),
            }
        except Exception as e:
            print(f"GDP baseline failed: {e}")
            import traceback

            traceback.print_exc()
            return {"gdp_yoy": np.nan}

    def _run_gdp_counterfactual(self, model, shocked_panel, target_df, scenario):
        """Run GDP model with shocked data."""
        # Check for endogenous GDP shocks
        gdp_shocks = [s for s in scenario.endogenous_shocks if s.target == "gdp"]

        if gdp_shocks:
            # Force GDP to specific value
            shock = gdp_shocks[0]
            return {
                "gdp_yoy": shock.forced_value,
                "forced": True,
                "description": shock.description,
            }
        else:
            # Run model with shocked panel
            return self._run_gdp_baseline(model, shocked_panel, target_df)

    def _run_inflation_baseline(self, model, panel_wide, target_df):
        """Run inflation model with baseline data (expects wide panel)."""
        try:
            result = model.nowcast(panel_wide, target_df)
            # The DFM returns nowcast_value for inflation
            return {
                "ipc_monthly_var": result.get("nowcast_value", np.nan),
                "ipc_3m_ma": result.get("ipc_3m_ma"),
            }
        except Exception as e:
            print(f"Inflation baseline failed: {e}")
            import traceback

            traceback.print_exc()
            return {"ipc_monthly_var": np.nan}

    def _run_inflation_counterfactual(self, model, shocked_panel, target_df, scenario):
        """Run inflation model with shocked data."""
        inflation_shocks = [s for s in scenario.endogenous_shocks if s.target == "inflation"]

        if inflation_shocks:
            shock = inflation_shocks[0]
            return {
                "ipc_monthly_var": shock.forced_value,
                "forced": True,
                "description": shock.description,
            }
        else:
            return self._run_inflation_baseline(model, shocked_panel, target_df)

    def compare_scenarios(self, scenarios: List[Scenario], **kwargs) -> pd.DataFrame:
        """
        Run multiple scenarios and return comparison table.

        Returns DataFrame with columns:
        - scenario_name
        - gdp_baseline, gdp_counterfactual, gdp_impact
        - inflation_baseline, inflation_counterfactual, inflation_impact
        - etc.
        """
        comparison = []

        for scenario in scenarios:
            result = self.run_scenario(scenario, **kwargs)

            row = {
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
            }

            for indicator in ["gdp", "inflation", "poverty", "political_risk"]:
                if indicator in result["baseline"]:
                    baseline_key = list(result["baseline"][indicator].keys())[0]
                    row[f"{indicator}_baseline"] = result["baseline"][indicator][baseline_key]
                    row[f"{indicator}_counterfactual"] = result["counterfactual"][indicator][baseline_key]
                    row[f"{indicator}_impact"] = result["impacts"][indicator]

            comparison.append(row)

        return pd.DataFrame(comparison)
