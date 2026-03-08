# Counterfactual Analysis Feature

## Overview

The counterfactual analysis system allows users to simulate economic scenarios and evaluate their impact before they occur. This is a **premium B2B feature** designed for strategic planning, risk analysis, and decision-making.

## Architecture

### Backend (NEXUS)

**Location**: `src/scenarios/`

#### 1. Scenario Engine (`scenario_engine.py`)
- **ExogenousShock**: Override input series (e.g., "FX rate +10%")
  - `shock_type`: absolute, percentage, sigma
  - Applied to specific series_id in the panel
- **EndogenousShock**: Force model output (e.g., "GDP = 0%")
  - Constrains target variable to specific value
- **Scenario**: Combination of shocks with metadata
- **ScenarioEngine**: Runs scenarios through all models
  - Compares baseline vs counterfactual
  - Returns direct impacts

#### 2. Shock Propagator (`shock_propagator.py`)
- Cross-model impact propagation using empirical elasticities
- **Elasticities** (estimated from Peruvian data):
  - GDP → Poverty: -0.5pp per 1pp GDP (Okun-style)
  - Inflation → Poverty: +0.3pp per 1pp inflation
  - Political Risk → GDP: -0.8pp per 1 sigma shock
  - FX → Inflation: 15% pass-through
  - GDP → Employment: 0.6pp per 1pp GDP
  - Confidence → GDP: 0.5pp per 1 sigma
- Methods:
  - `propagate_gdp_shock()`: GDP → Poverty, Employment
  - `propagate_inflation_shock()`: Inflation → Poverty (via real income)
  - `propagate_political_shock()`: Political → GDP → Poverty
  - `propagate_fx_shock()`: FX → Inflation → Poverty
  - `propagate_full_scenario()`: Aggregates all impacts

#### 3. Scenario Library (`scenario_library.py`)
Pre-built scenarios ready for use:

**Recession:**
- `mild_recession`: GDP=0%, financial stress +1σ
- `severe_recession`: GDP=-2%, political crisis +2σ, FX +15%

**Inflation:**
- `inflation_spike`: Inflation=0.5% monthly, chicken +30%, FX +10%
- `deflation`: Inflation=-0.2%, food prices fall

**Political:**
- `political_crisis`: Financial stress +3σ, confidence -30%
- `institutional_reform`: Stability improves -1.5σ, confidence +20%

**External:**
- `commodity_boom`: Exports +25%, Sol appreciates -10%
- `global_recession`: Exports -30%, FX +20%, political stress +1.5σ

**Stress Tests:**
- `perfect_storm`: GDP=-3%, Inflation=0.8%, political +2.5σ, FX +25%
- `goldilocks`: GDP=+3.5%, Inflation=0.17%, stability -1σ

### Frontend (Qhawarina)

**Location**: `app/escenarios/page.tsx`

#### Features:
1. **Scenario Selector**: Dropdown with 10 pre-built scenarios
2. **Scenario Description**:
   - Name, description, tags
   - List of exogenous and endogenous shocks
3. **Comparison Table**:
   - Baseline vs Counterfactual
   - Direct impacts with color coding (red/green)
4. **Propagated Impacts**:
   - Aggregate impacts (GDP, Inflation, Poverty)
   - Mechanism explanations
   - Employment effects
5. **Premium Badge**: Highlights feature as PRO/subscription
6. **CTA**: Call-to-action for premium subscriptions

#### UI Components:
- Gradient blue/indigo header with PREMIUM badge
- Interactive comparison table with arrow visualization
- Impact cards with trend icons (TrendingUp/TrendingDown)
- Color-coded impacts (green=positive, red=negative)
- Example scenario box on homepage

## Usage

### Backend

```bash
# Run single scenario
python scripts/run_scenario_analysis.py --scenario mild_recession

# Run all scenarios
python scripts/run_scenario_analysis.py --all

# Output directory
--output-dir exports/scenarios
```

### Example Output

```json
{
  "metadata": {
    "scenario_name": "Mild Recession",
    "target_period_gdp": "2026-Q2",
    "target_period_inflation": "2026-03"
  },
  "baseline": {
    "gdp": {"gdp_yoy": 2.5},
    "inflation": {"ipc_monthly_var": 0.25}
  },
  "counterfactual": {
    "gdp": {"gdp_yoy": 0.0, "forced": true},
    "inflation": {"ipc_monthly_var": 0.25}
  },
  "direct_impacts": {
    "gdp": -2.5,
    "inflation": 0.0
  },
  "propagated_impacts": {
    "aggregate": {
      "gdp_total_pp": -2.5,
      "poverty_total_pp": 1.25
    },
    "interpretation": "GDP would decrease by 2.5pp. Poverty would increase by 1.2pp."
  }
}
```

## Business Model

### Free Tier:
- View pre-built scenarios (read-only)
- See baseline vs counterfactual comparison
- Basic impact visualization

### Premium Tier (B2B):
- **Custom scenarios**: Create your own shocks
- **Parameter tuning**: Adjust shock magnitudes with sliders
- **Sectoral analysis**: Construction, Manufacturing, Services
- **API access**: Programmatic scenario execution
- **Historical counterfactuals**: "What if 2008 crisis hit today?"
- **Sensitivity analysis**: Monte Carlo simulations
- **White-label reports**: Branded PDFs for clients
- **Priority support**: Dedicated account manager

## Technical Details

### Model Integration
- GDP: DFM with Ridge bridge
- Inflation: DFM with AR(1) and lagged factors
- Poverty: GBR on departmental panel (future)
- Political Risk: NLP + financial stress (future)

### Shock Application
1. **Long panel**: Shocks applied to `panel_df` (long format)
2. **Pivot**: Convert to wide format for DFM
3. **Nowcast**: Run models on shocked panel
4. **Compare**: Baseline vs counterfactual
5. **Propagate**: Cross-model impacts via elasticities

### Performance
- Scenario execution: ~10-30 seconds
- Pre-built scenarios cached
- JSON export for web visualization
- Future: Real-time execution with WebSockets

## Future Enhancements

1. **Interactive Scenario Builder**:
   - Sliders for shock magnitudes
   - Multi-shock composer
   - Real-time preview

2. **Monte Carlo Simulations**:
   - Stochastic shocks
   - Confidence intervals
   - Risk distributions

3. **Time-Series Scenarios**:
   - Multi-period shocks
   - Dynamic paths
   - Impulse response functions

4. **Sectoral Disaggregation**:
   - Industry-specific impacts
   - Regional heterogeneity
   - Firm-level analysis

5. **Policy Evaluation**:
   - Fiscal stimulus scenarios
   - Monetary policy shocks
   - Structural reforms

## References

- Giannone, D., Reichlin, L., & Small, D. (2008). "Nowcasting: The real-time informational content of macroeconomic data."
- Stock, J.H., & Watson, M.W. (2002). "Forecasting using principal components from a large number of predictors."
- Timmermann, A. (2006). "Forecast combinations."

## Contact

For premium subscriptions or custom scenarios:
- Email: info@qhawarina.pe
- Website: /escenarios

---

**Built with ❤️ for Peru • Open source, siempre gratis (core), premium para empresas**
