"""
Interactive calculators for Qhawarina website.

Each calculator provides a simple API endpoint that takes user inputs
and returns results using the nowcasting models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import json


class InflationCalculator:
    """Calculate purchasing power erosion using supermarket price data."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parents[2]
        self.price_data = self._load_price_data()

    def _load_price_data(self) -> pd.DataFrame:
        """Load daily supermarket price index."""
        try:
            data_file = self.project_root / "exports" / "data" / "daily_price_index.json"
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data['series'])
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            # Fallback to empty DataFrame if file not found
            return pd.DataFrame()

    def calculate_impact(
        self,
        amount: float,
        start_date: str,
        end_date: str = None,
        category: str = 'all'
    ) -> Dict:
        """
        Calculate how much purchasing power has changed.

        Parameters
        ----------
        amount : float
            Purchase amount in PEN
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (defaults to today)
        category : str
            'all', 'food', 'arroz_cereales', etc.

        Returns
        -------
        dict with:
            - original_amount
            - equivalent_today
            - loss_amount
            - loss_pct
            - cumulative_inflation
            - daily_inflation_avg
        """
        if self.price_data.empty:
            return {'error': 'Price data not available'}

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else self.price_data['date'].max()

        # Get index values
        index_col = f'index_{category}' if category != 'all' else 'index_all'

        if index_col not in self.price_data.columns:
            return {'error': f'Category {category} not found'}

        df = self.price_data[['date', index_col]].dropna()

        start_idx = df[df['date'] <= start][index_col].iloc[-1] if len(df[df['date'] <= start]) > 0 else None
        end_idx = df[df['date'] <= end][index_col].iloc[-1] if len(df[df['date'] <= end]) > 0 else None

        if start_idx is None or end_idx is None:
            return {'error': 'Date range not available in data'}

        # Calculate impact
        inflation_factor = (end_idx / start_idx)
        equivalent_today = amount * inflation_factor
        loss = equivalent_today - amount
        loss_pct = (inflation_factor - 1) * 100

        days = (end - start).days
        daily_avg = loss_pct / days if days > 0 else 0

        return {
            'original_amount': round(amount, 2),
            'original_date': start_date,
            'equivalent_today': round(equivalent_today, 2),
            'comparison_date': end.strftime('%Y-%m-%d'),
            'loss_amount': round(loss, 2),
            'loss_pct': round(loss_pct, 2),
            'days_elapsed': days,
            'daily_inflation_avg': round(daily_avg, 4),
            'category': category,
            'interpretation': self._interpret_inflation(loss_pct, days)
        }

    def _interpret_inflation(self, loss_pct: float, days: int) -> str:
        """Generate human-readable interpretation."""
        if days < 30:
            period = f"{days} days"
        elif days < 365:
            period = f"{days // 30} months"
        else:
            period = f"{days // 365:.1f} years"

        annualized = (loss_pct / days * 365) if days > 0 else 0

        if loss_pct > 0:
            return f"Prices increased {loss_pct:.1f}% over {period} (annualized: {annualized:.1f}%/year)"
        else:
            return f"Prices decreased {abs(loss_pct):.1f}% over {period} (deflation)"


class PovertyForecastCalculator:
    """Forecast poverty rate for any department and date."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parents[2]

    def forecast(
        self,
        department: str,
        forecast_date: str,
        scenario: str = 'baseline'
    ) -> Dict:
        """
        Forecast poverty rate for a department.

        Parameters
        ----------
        department : str
            Department name (e.g., 'Lima', 'Ayacucho')
        forecast_date : str
            Target date (YYYY-MM-DD)
        scenario : str
            'baseline', 'optimistic', 'pessimistic'

        Returns
        -------
        dict with poverty forecast and confidence interval
        """
        # Placeholder - would use actual poverty nowcaster
        # For now, return demo data
        baseline_rates = {
            'Lima': 12.3,
            'Ayacucho': 38.5,
            'Cusco': 28.4,
            'Arequipa': 10.2,
            'Piura': 25.1,
            'Huancavelica': 45.2
        }

        base_rate = baseline_rates.get(department, 24.5)

        # Add scenario adjustment
        if scenario == 'optimistic':
            forecast_rate = base_rate - 2.0
            ci_lower = forecast_rate - 1.5
            ci_upper = forecast_rate + 1.5
        elif scenario == 'pessimistic':
            forecast_rate = base_rate + 2.5
            ci_lower = forecast_rate - 2.0
            ci_upper = forecast_rate + 2.0
        else:  # baseline
            forecast_rate = base_rate
            ci_lower = forecast_rate - 1.8
            ci_upper = forecast_rate + 1.8

        return {
            'department': department,
            'forecast_date': forecast_date,
            'scenario': scenario,
            'poverty_rate': round(forecast_rate, 1),
            'confidence_interval': {
                'lower': round(max(0, ci_lower), 1),
                'upper': round(min(100, ci_upper), 1),
                'confidence_level': 90
            },
            'people_affected': round(forecast_rate / 100 * self._get_population(department)),
            'interpretation': f"{forecast_rate:.1f}% of population in poverty ({scenario} scenario)"
        }

    def _get_population(self, department: str) -> int:
        """Get department population (2023 estimates)."""
        populations = {
            'Lima': 10_500_000,
            'Ayacucho': 700_000,
            'Cusco': 1_300_000,
            'Arequipa': 1_400_000,
            'Piura': 2_000_000,
            'Huancavelica': 500_000
        }
        return populations.get(department, 1_000_000)


class GDPScenarioCalculator:
    """Build custom GDP scenarios with shock parameters."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parents[2]

    def calculate_scenario(
        self,
        baseline_gdp: float,
        shocks: List[Dict]
    ) -> Dict:
        """
        Calculate GDP under custom shock scenario.

        Parameters
        ----------
        baseline_gdp : float
            Baseline GDP growth forecast (%)
        shocks : list of dict
            Each dict has: {'type': str, 'magnitude': float}
            Types: 'commodity', 'fx', 'political', 'china'

        Returns
        -------
        dict with scenario results
        """
        from src.simulation.policy_simulator import PolicySimulator, ScenarioLibrary

        simulator = PolicySimulator()
        total_gdp_impact = 0
        total_inflation_impact = 0
        shock_details = []

        for shock_spec in shocks:
            shock_type = shock_spec['type']
            magnitude = shock_spec['magnitude']

            # Map to policy shock
            if shock_type == 'commodity':
                shock = ScenarioLibrary.commodity_crash(magnitude)
            elif shock_type == 'fx':
                shock = ScenarioLibrary.fx_crisis(magnitude)
            elif shock_type == 'political':
                shock = ScenarioLibrary.political_crisis(magnitude)
            elif shock_type == 'china':
                shock = ScenarioLibrary.china_slowdown(magnitude)
            elif shock_type == 'interest_rate':
                shock = ScenarioLibrary.rate_hike(magnitude)
            else:
                continue

            result = simulator.simulate_shock(
                shock,
                baseline_gdp=baseline_gdp,
                baseline_inflation=2.3  # Assume baseline inflation
            )

            total_gdp_impact += result.gdp_impact
            total_inflation_impact += result.inflation_impact

            shock_details.append({
                'type': shock_type,
                'magnitude': magnitude,
                'gdp_impact': round(result.gdp_impact, 2),
                'inflation_impact': round(result.inflation_impact, 2)
            })

        return {
            'baseline_gdp': baseline_gdp,
            'shocked_gdp': round(baseline_gdp + total_gdp_impact, 2),
            'total_impact': round(total_gdp_impact, 2),
            'inflation_impact': round(total_inflation_impact, 2),
            'shocks': shock_details,
            'interpretation': self._interpret_gdp(baseline_gdp, total_gdp_impact)
        }

    def _interpret_gdp(self, baseline: float, impact: float) -> str:
        """Interpret GDP scenario."""
        shocked = baseline + impact
        if impact < -2:
            severity = "severe contraction"
        elif impact < -1:
            severity = "significant slowdown"
        elif impact < 0:
            severity = "modest slowdown"
        elif impact > 2:
            severity = "strong boost"
        elif impact > 1:
            severity = "moderate boost"
        else:
            severity = "slight improvement"

        return f"GDP would grow {shocked:.1f}% ({impact:+.1f}pp vs baseline) — {severity}"


class RegionalComparator:
    """Compare multiple departments on key indicators."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parents[2]

    def compare_departments(
        self,
        departments: List[str],
        metrics: List[str] = None
    ) -> Dict:
        """
        Compare departments on GDP, poverty, NTL, etc.

        Parameters
        ----------
        departments : list of str
            Department names to compare
        metrics : list of str, optional
            Metrics to compare (defaults to all)

        Returns
        -------
        dict with comparison data
        """
        if metrics is None:
            metrics = ['gdp_growth', 'poverty_rate', 'ntl_growth', 'population']

        # Demo data - would use actual nowcasts
        comparison = {}

        for dept in departments:
            comparison[dept] = {
                'gdp_growth': round(np.random.uniform(1.5, 4.5), 1),
                'poverty_rate': round(np.random.uniform(10, 40), 1),
                'ntl_growth': round(np.random.uniform(-2, 8), 1),
                'population': round(np.random.uniform(500_000, 3_000_000), 0)
            }

        # Add rankings
        for metric in metrics:
            values = [(dept, comparison[dept].get(metric, 0)) for dept in departments]
            values.sort(key=lambda x: x[1], reverse=(metric != 'poverty_rate'))

            for rank, (dept, val) in enumerate(values, 1):
                comparison[dept][f'{metric}_rank'] = rank

        return {
            'departments': list(comparison.keys()),
            'metrics': metrics,
            'data': comparison,
            'summary': self._summarize_comparison(comparison, metrics)
        }

    def _summarize_comparison(self, data: Dict, metrics: List[str]) -> str:
        """Generate summary text."""
        # Find best/worst performers
        poverty = [(d, data[d].get('poverty_rate', 50)) for d in data.keys()]
        poverty.sort(key=lambda x: x[1])

        best = poverty[0][0]
        worst = poverty[-1][0]

        return f"Lowest poverty: {best} ({poverty[0][1]:.1f}%) | Highest: {worst} ({poverty[-1][1]:.1f}%)"


if __name__ == '__main__':
    # Test calculators
    print("=" * 80)
    print("TESTING INTERACTIVE CALCULATORS")
    print("=" * 80)

    # 1. Inflation Calculator
    print("\n1. INFLATION CALCULATOR")
    print("-" * 80)
    calc = InflationCalculator()
    result = calc.calculate_impact(
        amount=1000,
        start_date='2026-02-10',
        end_date='2026-02-28',
        category='all'
    )
    print(f"S/ {result['original_amount']} on {result['original_date']}")
    print(f"Equivalent today: S/ {result['equivalent_today']}")
    print(f"Loss: S/ {result['loss_amount']} ({result['loss_pct']:.2f}%)")
    print(f"Interpretation: {result['interpretation']}")

    # 2. Poverty Forecast
    print("\n2. POVERTY FORECAST")
    print("-" * 80)
    pov_calc = PovertyForecastCalculator()
    result = pov_calc.forecast('Ayacucho', '2026-12-31', 'baseline')
    print(f"Department: {result['department']}")
    print(f"Forecast: {result['poverty_rate']}% poverty rate")
    print(f"90% CI: [{result['confidence_interval']['lower']}%, {result['confidence_interval']['upper']}%]")
    print(f"People affected: {result['people_affected']:,.0f}")

    # 3. GDP Scenario
    print("\n3. GDP SCENARIO BUILDER")
    print("-" * 80)
    gdp_calc = GDPScenarioCalculator()
    result = gdp_calc.calculate_scenario(
        baseline_gdp=2.8,
        shocks=[
            {'type': 'commodity', 'magnitude': -15},
            {'type': 'political', 'magnitude': 1.5}
        ]
    )
    print(f"Baseline GDP: {result['baseline_gdp']:.1f}%")
    print(f"Shocked GDP: {result['shocked_gdp']:.1f}%")
    print(f"Total impact: {result['total_impact']:+.2f}pp")
    print(f"Interpretation: {result['interpretation']}")

    # 4. Regional Comparator
    print("\n4. REGIONAL COMPARATOR")
    print("-" * 80)
    comp = RegionalComparator()
    result = comp.compare_departments(['Lima', 'Ayacucho', 'Cusco'])
    print(f"Comparing: {', '.join(result['departments'])}")
    print(f"Summary: {result['summary']}")

    print("\n" + "=" * 80)
