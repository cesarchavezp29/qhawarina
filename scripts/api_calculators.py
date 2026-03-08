"""
Simple JSON API for Qhawarina interactive calculators.

Usage:
    python scripts/api_calculators.py

This starts a Flask server on http://localhost:5000 with endpoints:
    - POST /api/inflation - Calculate inflation impact
    - POST /api/poverty - Forecast poverty
    - POST /api/gdp_scenario - Build GDP scenario
    - POST /api/regional_compare - Compare departments
    - POST /api/policy_simulate - Simulate policy shock

Example:
    curl -X POST http://localhost:5000/api/inflation \
         -H "Content-Type: application/json" \
         -d '{"amount": 1000, "start_date": "2026-01-01", "category": "food"}'
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.simulation.calculators import (
    InflationCalculator,
    PovertyForecastCalculator,
    GDPScenarioCalculator,
    RegionalComparator
)
from src.simulation.policy_simulator import PolicySimulator, ScenarioLibrary

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Qhawarina website

# Initialize calculators
inflation_calc = InflationCalculator(project_root)
poverty_calc = PovertyForecastCalculator(project_root)
gdp_calc = GDPScenarioCalculator(project_root)
regional_comp = RegionalComparator(project_root)
policy_sim = PolicySimulator(project_root)


@app.route('/api/inflation', methods=['POST'])
def calculate_inflation():
    """
    Calculate purchasing power erosion.

    Body:
        {
            "amount": 1000,
            "start_date": "2026-01-01",
            "end_date": "2026-02-28",  // optional
            "category": "all"  // or "food", "arroz_cereales", etc.
        }
    """
    data = request.json
    result = inflation_calc.calculate_impact(
        amount=float(data['amount']),
        start_date=data['start_date'],
        end_date=data.get('end_date'),
        category=data.get('category', 'all')
    )
    return jsonify(result)


@app.route('/api/poverty', methods=['POST'])
def forecast_poverty():
    """
    Forecast poverty rate for a department.

    Body:
        {
            "department": "Ayacucho",
            "forecast_date": "2026-12-31",
            "scenario": "baseline"  // or "optimistic", "pessimistic"
        }
    """
    data = request.json
    result = poverty_calc.forecast(
        department=data['department'],
        forecast_date=data['forecast_date'],
        scenario=data.get('scenario', 'baseline')
    )
    return jsonify(result)


@app.route('/api/gdp_scenario', methods=['POST'])
def build_gdp_scenario():
    """
    Build custom GDP scenario with shocks.

    Body:
        {
            "baseline_gdp": 2.8,
            "shocks": [
                {"type": "commodity", "magnitude": -15},
                {"type": "political", "magnitude": 1.5}
            ]
        }

    Shock types: commodity, fx, political, china, interest_rate
    """
    data = request.json
    result = gdp_calc.calculate_scenario(
        baseline_gdp=float(data['baseline_gdp']),
        shocks=data['shocks']
    )
    return jsonify(result)


@app.route('/api/regional_compare', methods=['POST'])
def compare_departments():
    """
    Compare departments on key indicators.

    Body:
        {
            "departments": ["Lima", "Ayacucho", "Cusco"],
            "metrics": ["gdp_growth", "poverty_rate"]  // optional
        }
    """
    data = request.json
    result = regional_comp.compare_departments(
        departments=data['departments'],
        metrics=data.get('metrics')
    )
    return jsonify(result)


@app.route('/api/policy_simulate', methods=['POST'])
def simulate_policy():
    """
    Simulate a policy shock.

    Body:
        {
            "shock_type": "commodity_crash",  // or fx_crisis, rate_hike, etc.
            "magnitude": -20,
            "baseline_gdp": 2.8,
            "baseline_inflation": 2.3,
            "baseline_poverty": 24.5  // optional
        }

    Shock types:
        - commodity_crash (magnitude = % change in prices)
        - fx_crisis (magnitude = % depreciation)
        - rate_hike (magnitude = basis points)
        - political_crisis (magnitude = std deviations)
        - china_slowdown (magnitude = pp change in China GDP)
        - el_nino (magnitude = % fall in agriculture)
    """
    data = request.json

    # Map shock type to scenario
    shock_map = {
        'commodity_crash': ScenarioLibrary.commodity_crash,
        'fx_crisis': ScenarioLibrary.fx_crisis,
        'rate_hike': ScenarioLibrary.rate_hike,
        'political_crisis': ScenarioLibrary.political_crisis,
        'china_slowdown': ScenarioLibrary.china_slowdown,
        'el_nino': ScenarioLibrary.el_nino,
    }

    shock_fn = shock_map.get(data['shock_type'])
    if not shock_fn:
        return jsonify({'error': f"Unknown shock type: {data['shock_type']}"}), 400

    shock = shock_fn(magnitude=float(data['magnitude']))

    result = policy_sim.simulate_shock(
        shock=shock,
        baseline_gdp=float(data['baseline_gdp']),
        baseline_inflation=float(data['baseline_inflation']),
        baseline_poverty=float(data.get('baseline_poverty')) if data.get('baseline_poverty') else None
    )

    return jsonify(result.to_dict())


@app.route('/api/scenarios/library', methods=['GET'])
def get_scenario_library():
    """Get list of pre-configured policy scenarios."""
    scenarios = [
        {
            'id': 'commodity_crash',
            'name': 'Commodity Price Crash',
            'description': 'Global commodity prices fall (copper, gold, zinc)',
            'default_magnitude': -20,
            'unit': '% change'
        },
        {
            'id': 'fx_crisis',
            'name': 'Currency Crisis',
            'description': 'PEN depreciation due to capital flight',
            'default_magnitude': 15,
            'unit': '% depreciation'
        },
        {
            'id': 'rate_hike',
            'name': 'Interest Rate Hike',
            'description': 'BCRP raises reference rate to fight inflation',
            'default_magnitude': 100,
            'unit': 'basis points'
        },
        {
            'id': 'political_crisis',
            'name': 'Political Instability Surge',
            'description': 'Protests, cabinet changes, institutional crisis',
            'default_magnitude': 2.0,
            'unit': 'std deviations'
        },
        {
            'id': 'china_slowdown',
            'name': 'China Economic Slowdown',
            'description': 'Reduced demand for Peruvian exports',
            'default_magnitude': -2.0,
            'unit': 'pp change in China GDP'
        },
        {
            'id': 'el_nino',
            'name': 'El Niño Climate Shock',
            'description': 'Agricultural production fall, food price increase',
            'default_magnitude': -3.0,
            'unit': '% fall in agriculture'
        }
    ]
    return jsonify({'scenarios': scenarios})


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'Qhawarina Calculators API'})


if __name__ == '__main__':
    print("=" * 80)
    print("QHAWARINA INTERACTIVE CALCULATORS API")
    print("=" * 80)
    print("\nEndpoints available:")
    print("  POST /api/inflation          - Calculate inflation impact")
    print("  POST /api/poverty            - Forecast poverty rate")
    print("  POST /api/gdp_scenario       - Build GDP scenario")
    print("  POST /api/regional_compare   - Compare departments")
    print("  POST /api/policy_simulate    - Simulate policy shock")
    print("  GET  /api/scenarios/library  - List pre-configured scenarios")
    print("  GET  /health                 - Health check")
    print("\nStarting server on http://localhost:5000")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
