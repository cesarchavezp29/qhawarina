# Qhawarina.pe Website Design Document

**Version**: 1.0
**Date**: 2026-02-14
**Status**: Pre-production design (domain not yet purchased)

---

## Executive Summary

Qhawarina (Quechua: "tomorrow's view") is Peru's first real-time economic nowcasting platform. The website provides daily-updated predictions for GDP growth, inflation, poverty rates, and political instability using machine learning models trained on 490+ economic indicators.

**Target Audience**: Economists, policy analysts, investors, journalists, researchers

**Core Value Proposition**: Real-time economic intelligence for Peru, updated daily, methodologically transparent, and free to access.

---

## Site Architecture

```
qhawarina.pe/
├── index.html                    # Homepage - Dashboard of latest nowcasts
├── gdp/
│   ├── index.html               # GDP nowcast detail page
│   └── methodology.html         # GDP methodology
├── inflation/
│   ├── index.html               # Inflation nowcast detail page
│   └── methodology.html         # Inflation methodology
├── poverty/
│   ├── index.html               # Poverty nowcast detail page
│   ├── map.html                 # Interactive district map
│   └── methodology.html         # Poverty methodology
├── political/
│   ├── index.html               # Political instability index
│   └── daily-reports/           # Archive of daily reports
├── supermarket/
│   ├── index.html               # Supermarket price index
│   └── categories.html          # Category breakdowns
├── data/
│   ├── index.html               # Data downloads page
│   └── api.html                 # API documentation
├── about/
│   ├── index.html               # About the project
│   ├── team.html                # Team page
│   └── methodology.html         # General methodology
└── assets/
    ├── data/                    # JSON exports (daily updated)
    ├── charts/                  # PNG/SVG chart exports
    └── maps/                    # GeoJSON + rendered maps
```

---

## Page Designs

### 1. Homepage (`/index.html`)

**Layout**: Dashboard-style with 4 main cards

```
┌────────────────────────────────────────────────────────────┐
│  QHAWARINA                                [ES/EN]  [DATA]  │
│  La economía del Perú en tiempo real                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐│
│  │ GDP GROWTH     │  │ INFLATION      │  │ POVERTY RATE ││
│  │                │  │                │  │              ││
│  │   +2.14%       │  │   +0.169%      │  │   24.9%      ││
│  │   YoY Q4 2025  │  │   MoM Feb 2026 │  │   2025 est.  ││
│  │                │  │                │  │              ││
│  │ [chart]        │  │ [chart]        │  │ [chart]      ││
│  │                │  │                │  │              ││
│  │ Ver más →      │  │ Ver más →      │  │ Ver más →    ││
│  └────────────────┘  └────────────────┘  └──────────────┘│
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ POLITICAL INSTABILITY INDEX          Latest: Feb 14  │ │
│  │                                                       │ │
│  │  0.340 / 1.00  (BAJO)                                │ │
│  │                                                       │ │
│  │  [30-day sparkline chart]                            │ │
│  │                                                       │ │
│  │  Peak recent: Jan 21 (0.991) - José Jerí scandal    │ │
│  │                                                       │ │
│  │  Ver más →                                           │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  Last updated: 2026-02-14 08:00 PET                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ABOUT QHAWARINA                                      │ │
│  │                                                       │ │
│  │ Real-time nowcasting for Peru using Dynamic Factor   │ │
│  │ Models trained on 490+ indicators. Updated daily.    │ │
│  │                                                       │ │
│  │ [Methodology] [Download Data] [API Access]           │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Large, prominent nowcast values with directional indicators (↑↓)
- Inline mini-charts (last 12 months trend)
- Color coding: Green (positive), Red (negative), Yellow (warning)
- Timestamp showing last data update
- Clean, minimalist design (inspired by FiveThirtyEight, FRED)

---

### 2. GDP Nowcast Detail Page (`/gdp/index.html`)

**Layout**: Full-width chart + metrics table

```
┌────────────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                       │
├────────────────────────────────────────────────────────────┤
│  GDP GROWTH NOWCAST                                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  +2.14% YoY (Q4 2025)                                │ │
│  │                                                       │ │
│  │  Model: Dynamic Factor Model with Ridge Bridge       │ │
│  │  Data through: November 2025 (83 series)             │ │
│  │  Bridge R²: 0.934                                    │ │
│  │  Out-of-sample RMSE: 1.41pp (vs AR1: 0.76pp)        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  [CHART: GDP Growth Quarterly - Line chart]          │ │
│  │  2010-2025, showing actual (solid) and nowcast (dot) │ │
│  │  With confidence intervals                           │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  QUARTERLY BREAKDOWN                                       │
│  ┌────────┬──────────┬──────────┬──────────┬────────────┐│
│  │Quarter │ Official │ Nowcast  │ Error    │ Coverage   ││
│  ├────────┼──────────┼──────────┼──────────┼────────────┤│
│  │2025-Q4 │   --     │  +2.14%  │   --     │ 83/84 (99%)││
│  │2025-Q3 │ +2.68%   │  +2.85%  │ +0.17pp  │100%        ││
│  │2025-Q2 │ +2.01%   │  +1.89%  │ -0.12pp  │100%        ││
│  │2025-Q1 │ +1.48%   │  +1.62%  │ +0.14pp  │100%        ││
│  └────────┴──────────┴──────────┴──────────┴────────────┘│
│                                                             │
│  TOP CONTRIBUTING SERIES (Factor loadings)                 │
│  1. Manufacturing production index    (loading: +0.84)     │
│  2. Non-traditional exports          (loading: +0.71)     │
│  3. Tax revenue (IGV)                (loading: +0.68)     │
│  4. Formal employment (urban)        (loading: +0.65)     │
│  5. Credit to private sector         (loading: +0.59)     │
│                                                             │
│  [Download CSV] [View Methodology] [API Endpoint]          │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Prominent nowcast value with model details
- Interactive quarterly chart (Plotly or Chart.js)
- Comparison table: Official vs Nowcast vs Error
- Series contribution breakdown (transparency)
- Export options (CSV, JSON)

---

### 3. Inflation Nowcast Detail Page (`/inflation/index.html`)

**Layout**: Similar to GDP, adapted for monthly frequency

```
┌────────────────────────────────────────────────────────────┐
│  INFLATION NOWCAST                                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  +0.169% MoM (February 2026)                         │ │
│  │                                                       │ │
│  │  Model: DFM with lagged factors + AR(1)              │ │
│  │  Data through: January 2026 (22 series)              │ │
│  │  Factor R²: 0.199                                    │ │
│  │  Out-of-sample RMSE: 0.319pp (vs AR1: 0.322pp)      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  [CHART: Monthly Inflation - Dual Y-axis]            │ │
│  │  Left: MoM variation (bars)                          │ │
│  │  Right: 12m cumulative (line)                        │ │
│  │  2020-2026                                           │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  RECENT MONTHS                                             │
│  ┌────────┬──────────┬──────────┬──────────┬────────────┐│
│  │Month   │ Official │ Nowcast  │ Error    │ 12m Accum  ││
│  ├────────┼──────────┼──────────┼──────────┼────────────┤│
│  │2026-02 │   --     │ +0.169%  │   --     │    --      ││
│  │2026-01 │ +0.17%   │ +0.166%  │ -0.004pp │  +2.17%    ││
│  │2025-12 │ +0.04%   │ +0.11%   │ +0.07pp  │  +2.00%    ││
│  │2025-11 │ +0.07%   │ +0.05%   │ -0.02pp  │  +1.96%    ││
│  └────────┴──────────┴──────────┴──────────┴────────────┘│
│                                                             │
│  CATEGORY CONTRIBUTIONS (Supermarket Price Index - NEW!)   │
│  ┌────────────────────────┬─────────┬──────────────────┐  │
│  │Category                │MoM Var  │ CPI Weight       │  │
│  ├────────────────────────┼─────────┼──────────────────┤  │
│  │Food (supermarket-based)│  --     │ 22.22% (tracked) │  │
│  │  - Meat               │  --     │  5.23%           │  │
│  │  - Dairy              │  --     │  2.85%           │  │
│  │  - Bread/Cereals      │  --     │  4.23%           │  │
│  │  - Fruits/Vegetables  │  --     │  5.37%           │  │
│  │Non-food               │  --     │  6.00% (tracked) │  │
│  └────────────────────────┴─────────┴──────────────────┘  │
│                                                             │
│  Note: Supermarket index active from Feb 2026 (2 days of   │
│  data). Will contribute to DFM from Feb 2027 (12m history).│
│                                                             │
│  [Download CSV] [View Methodology] [Supermarket Details]   │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Monthly frequency (vs quarterly for GDP)
- Dual visualization: MoM (bars) + 12m cumulative (line)
- Category breakdown table
- Supermarket price index integration (NEW - highlighted)
- Forward-looking note about data availability

---

### 4. Poverty Nowcast Detail Page (`/poverty/index.html`)

**Layout**: Map-first design with district drill-down

```
┌────────────────────────────────────────────────────────────┐
│  POVERTY RATE NOWCAST                                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  24.9% National Average (2025)                       │ │
│  │                                                       │ │
│  │  Model: Gradient Boosting Regressor (GBR)            │ │
│  │  Department estimates: 26 regions                    │ │
│  │  District estimates: 1,891 districts                 │ │
│  │  Out-of-sample RMSE: 2.54pp (vs AR1: 2.65pp)        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  [MAP: Peru choropleth - Department level]           │ │
│  │  Color gradient: Green (low) → Red (high)            │ │
│  │  Interactive: Hover shows dept name + rate           │ │
│  │  Click → drill down to district map                  │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  DEPARTMENT RANKING (2025 Nowcast)                         │
│  ┌──────────────┬──────────┬──────────┬─────────────────┐ │
│  │Department    │ 2024 Obs │ 2025 Now │ Change (pp)     │ │
│  ├──────────────┼──────────┼──────────┼─────────────────┤ │
│  │Puno          │  52.3%   │  49.8%   │  -2.5  ↓        │ │
│  │Cajamarca     │  42.6%   │  40.1%   │  -2.5  ↓        │ │
│  │Ayacucho      │  39.2%   │  37.4%   │  -1.8  ↓        │ │
│  │...           │   ...    │   ...    │   ...           │ │
│  │Lima          │  14.2%   │  13.8%   │  -0.4  ↓        │ │
│  │Madre de Dios │   6.1%   │   5.7%   │  -0.4  ↓        │ │
│  └──────────────┴──────────┴──────────┴─────────────────┘ │
│                                                             │
│  [Interactive District Map] [Download Full Data]           │
│  [Methodology] [Nighttime Lights Methodology]              │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Interactive Peru map** (using Leaflet.js or Mapbox GL)
- Choropleth coloring by poverty rate
- Department ranking table with year-over-year changes
- Link to district-level disaggregation page
- GeoJSON export for researchers

---

### 5. Political Instability Index (`/political/index.html`)

**Layout**: Timeline-first with daily drill-down

```
┌────────────────────────────────────────────────────────────┐
│  POLITICAL INSTABILITY INDEX                               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Current: 0.340 / 1.00  (BAJO)                       │ │
│  │  Date: February 14, 2026                             │ │
│  │                                                       │ │
│  │  7-day average: 0.385  (BAJO)                        │ │
│  │  30-day average: 0.447  (MEDIO)                      │ │
│  │  Peak (last 90d): 0.991 on Jan 21, 2026              │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  [CHART: Daily instability - Line + area fill]       │ │
│  │  Last 180 days                                       │ │
│  │  Color zones: 0-0.33 (green), 0.33-0.66 (yellow),   │ │
│  │               0.66-1.0 (red)                         │ │
│  │  Annotations on major events (hover to see details)  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  MAJOR EVENTS (Score > 0.75)                               │
│  ┌────────────┬───────┬────────────────────────────────┐  │
│  │Date        │Score  │ Event Summary                  │  │
│  ├────────────┼───────┼────────────────────────────────┤  │
│  │2026-01-21  │ 0.991 │ José Jerí scandal - 6 censure  │  │
│  │            │       │ motions, cabinet crisis        │  │
│  │            │       │ [View full report →]           │  │
│  │2025-12-03  │ 0.812 │ Mining conflict in Espinar     │  │
│  │            │       │ [View full report →]           │  │
│  │2025-10-15  │ 0.776 │ Congress-Executive standoff    │  │
│  │            │       │ [View full report →]           │  │
│  └────────────┴───────┴────────────────────────────────┘  │
│                                                             │
│  METHODOLOGY                                               │
│  - 81 RSS feeds monitored daily (government, media, think  │
│    tanks, economic outlets)                                │
│  - GPT-4o classification: Political vs Economic relevance  │
│  - Composite score: Weighted average of political (60%) +  │
│    economic (40%) intensity                                │
│  - Updated daily at 08:00 PET                              │
│                                                             │
│  [Download Daily Data] [View Methodology] [RSS Sources]    │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Real-time daily index (most unique feature)
- Color-coded risk zones
- Major event annotations on chart
- Clickable events → full daily reports
- RSS feed transparency

---

### 6. Supermarket Price Index (`/supermarket/index.html`)

**Layout**: BPP-style index display

```
┌────────────────────────────────────────────────────────────┐
│  SUPERMARKET PRICE INDEX                                   │
│  Peru's Billion Prices Project                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Status: BETA (2 days of data)                       │ │
│  │                                                       │ │
│  │  Latest: February 11, 2026                           │ │
│  │  Products tracked: 41,970 (3 stores)                 │ │
│  │  Categories: 13 food + 2 non-food                    │ │
│  │  CPI basket coverage: 28.2%                          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  [CHART: Daily index - Coming soon]                  │ │
│  │  Requires 12+ months of data for reliable trends     │ │
│  │                                                       │ │
│  │  Expected launch: February 2027                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  CURRENT SNAPSHOT (Feb 11, 2026)                           │
│  ┌──────────────────────┬─────────┬──────────┬──────────┐ │
│  │Category              │Products │Price Range│CPI Weight│ │
│  ├──────────────────────┼─────────┼──────────┼──────────┤ │
│  │Beverages             │  7,294  │S/0.40-228│  1.92%   │ │
│  │Personal care         │  6,198  │S/0.70-180│  4.79%   │ │
│  │Rice/Cereals          │  6,194  │S/0.60-350│  4.23%   │ │
│  │Cleaning              │  4,377  │S/1.20-120│  1.32%   │ │
│  │Sugar/Sweets          │  4,358  │S/0.90-180│  0.89%   │ │
│  │Meat                  │  4,163  │S/3.50-800│  5.23%   │ │
│  │Dairy                 │  2,880  │S/1.20-120│  2.85%   │ │
│  │Fruits                │  2,614  │S/0.80-90 │  2.50%   │ │
│  │Vegetables            │  1,971  │S/0.60-70 │  2.87%   │ │
│  │Oils/Fats             │  1,331  │S/2.50-140│  0.54%   │ │
│  │Bread/Flour           │    958  │S/1.20-80 │  4.23%   │ │
│  │Fish/Seafood          │    304  │S/8.00-180│  1.09%   │ │
│  │Eggs                  │     68  │S/4.00-35 │  0.64%   │ │
│  └──────────────────────┴─────────┴──────────┴──────────┘ │
│                                                             │
│  STORES MONITORED                                          │
│  - Plaza Vea (www.plazavea.com.pe) - 15,186 products       │
│  - Metro (www.metro.pe) - 11,763 products                  │
│  - Wong (www.wong.pe) - 15,021 products                    │
│                                                             │
│  METHODOLOGY                                               │
│  - Jevons bilateral index (geometric mean of price ratios) │
│  - CPI-weighted aggregation (INEI official weights)        │
│  - Chain-linked daily index (base = first day = 100)       │
│  - Following Cavallo & Rigobon (2016) BPP methodology      │
│                                                             │
│  [View Raw Data] [Methodology] [Cavallo Paper]             │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- BETA status badge (data too new for trends)
- Product count statistics
- Category breakdown table
- Store coverage transparency
- Academic citation (Cavallo & Rigobon)

---

### 7. Data Downloads Page (`/data/index.html`)

**Layout**: File browser with metadata

```
┌────────────────────────────────────────────────────────────┐
│  DATA DOWNLOADS                                            │
│                                                             │
│  All datasets are updated daily at 08:00 PET and freely    │
│  available for research and commercial use under CC BY 4.0.│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ NOWCAST OUTPUTS (Latest Predictions)                   ││
│  ├────────────────────────────────────────────────────────┤│
│  │ gdp_nowcast.json                         Updated: 02/14││
│  │   Q4 2025 GDP growth nowcast                     2.1 KB││
│  │   [JSON] [CSV]                                         ││
│  │                                                         ││
│  │ inflation_nowcast.json                   Updated: 02/14││
│  │   Feb 2026 inflation nowcast                     1.8 KB││
│  │   [JSON] [CSV]                                         ││
│  │                                                         ││
│  │ poverty_nowcast.json                     Updated: 02/14││
│  │   2025 poverty rates (26 depts, 1891 districts)  45 KB││
│  │   [JSON] [CSV] [GeoJSON]                               ││
│  │                                                         ││
│  │ political_index_daily.json               Updated: 02/14││
│  │   Daily instability scores (last 90 days)        12 KB││
│  │   [JSON] [CSV]                                         ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ HISTORICAL PANEL DATA                                  ││
│  ├────────────────────────────────────────────────────────┤│
│  │ panel_national_monthly.parquet           Updated: 02/14││
│  │   84 national series (2003-2026)                 850 KB││
│  │   [Parquet] [CSV]                                      ││
│  │                                                         ││
│  │ panel_departmental_monthly.parquet       Updated: 02/14││
│  │   406 regional series (26 depts, 2007-2025)     2.1 MB││
│  │   [Parquet] [CSV]                                      ││
│  │                                                         ││
│  │ supermarket_snapshots_daily.parquet      Updated: 02/14││
│  │   41,970 products x 2 days                       1.3 MB││
│  │   [Parquet] [CSV]                                      ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ BACKTEST RESULTS                                       ││
│  ├────────────────────────────────────────────────────────┤│
│  │ backtest_gdp.parquet                                   ││
│  │   2009-2025 GDP nowcast backtest (RMSE: 1.41pp)  18 KB││
│  │   [Parquet] [CSV]                                      ││
│  │                                                         ││
│  │ backtest_inflation.parquet                             ││
│  │   2012-2026 inflation backtest (RMSE: 0.32pp)    25 KB││
│  │   [Parquet] [CSV]                                      ││
│  │                                                         ││
│  │ backtest_poverty.parquet                               ││
│  │   2012-2024 poverty backtest (RMSE: 2.54pp)      14 KB││
│  │   [Parquet] [CSV]                                      ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  API ACCESS                                                │
│  For programmatic access, see [API Documentation]          │
│                                                             │
│  LICENSE                                                   │
│  All data is released under Creative Commons Attribution   │
│  4.0 International (CC BY 4.0). Please cite:               │
│                                                             │
│    Qhawarina (2026). "Economic Nowcasting for Peru."       │
│    Retrieved from https://qhawarina.pe                     │
└────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Three categories: Nowcasts, Panel Data, Backtest Results
- File size and last-updated metadata
- Multiple format options (JSON, CSV, Parquet, GeoJSON)
- License information (CC BY 4.0)
- API link for programmatic access

---

## Data Export Formats

### 1. GDP Nowcast (`/assets/data/gdp_nowcast.json`)

```json
{
  "metadata": {
    "generated_at": "2026-02-14T08:00:00-05:00",
    "model": "DynamicFactorModel",
    "model_params": {
      "k_factors": 2,
      "factor_order": 2,
      "bridge_method": "ridge",
      "bridge_alpha": 1.0,
      "rolling_window_years": 7
    },
    "data_vintage": "2025-11",
    "series_coverage": "83/84 (98.8%)"
  },
  "nowcast": {
    "target_period": "2025-Q4",
    "value": 2.14,
    "unit": "percent_yoy",
    "bridge_r2": 0.934
  },
  "recent_quarters": [
    {"quarter": "2025-Q4", "official": null, "nowcast": 2.14, "error": null},
    {"quarter": "2025-Q3", "official": 2.68, "nowcast": 2.85, "error": 0.17},
    {"quarter": "2025-Q2", "official": 2.01, "nowcast": 1.89, "error": -0.12}
  ],
  "top_contributors": [
    {"series": "Manufacturing production index", "loading": 0.84},
    {"series": "Non-traditional exports", "loading": 0.71},
    {"series": "Tax revenue (IGV)", "loading": 0.68}
  ],
  "backtest_metrics": {
    "rmse": 1.41,
    "mae": 1.05,
    "r2": 0.89,
    "relative_rmse_vs_ar1": 1.85
  }
}
```

### 2. Inflation Nowcast (`/assets/data/inflation_nowcast.json`)

```json
{
  "metadata": {
    "generated_at": "2026-02-14T08:00:00-05:00",
    "model": "DynamicFactorModel",
    "model_params": {
      "k_factors": 2,
      "include_factor_lags": 1,
      "include_target_ar": true
    },
    "data_vintage": "2026-01",
    "series_coverage": "22/25 (88.0%)"
  },
  "nowcast": {
    "target_period": "2026-02",
    "value": 0.169,
    "unit": "percent_mom",
    "factor_r2": 0.199
  },
  "recent_months": [
    {"month": "2026-02", "official": null, "nowcast": 0.169, "error": null},
    {"month": "2026-01", "official": 0.17, "nowcast": 0.166, "error": -0.004},
    {"month": "2025-12", "official": 0.04, "nowcast": 0.11, "error": 0.07}
  ],
  "category_contributions": [
    {"category": "Food (supermarket)", "weight": 0.2222, "data_available": false},
    {"category": "Non-food", "weight": 0.0600, "data_available": false}
  ],
  "backtest_metrics": {
    "rmse": 0.319,
    "mae": 0.245,
    "r2": 0.199,
    "relative_rmse_vs_ar1": 0.991
  }
}
```

### 3. Poverty Nowcast (`/assets/data/poverty_nowcast.json`)

```json
{
  "metadata": {
    "generated_at": "2026-02-14T08:00:00-05:00",
    "model": "GradientBoostingRegressor",
    "target_year": 2025,
    "departments": 26,
    "districts": 1891
  },
  "national": {
    "poverty_rate": 24.9,
    "unit": "percent"
  },
  "departments": [
    {
      "code": "21",
      "name": "Puno",
      "poverty_rate_2024": 52.3,
      "poverty_rate_2025_nowcast": 49.8,
      "change_pp": -2.5
    },
    {
      "code": "06",
      "name": "Cajamarca",
      "poverty_rate_2024": 42.6,
      "poverty_rate_2025_nowcast": 40.1,
      "change_pp": -2.5
    }
  ],
  "districts": [
    {
      "ubigeo": "210101",
      "department_code": "21",
      "name": "Puno",
      "poverty_rate_nowcast": 38.2,
      "ntl_weight": 0.045
    }
  ],
  "backtest_metrics": {
    "rmse": 2.54,
    "mae": 1.89,
    "r2": 0.76,
    "relative_rmse_vs_ar1": 0.958
  }
}
```

### 4. Political Index (`/assets/data/political_index_daily.json`)

```json
{
  "metadata": {
    "generated_at": "2026-02-14T08:00:00-05:00",
    "coverage_days": 90,
    "rss_feeds": 81
  },
  "current": {
    "date": "2026-02-14",
    "score": 0.340,
    "level": "BAJO",
    "articles_total": 127,
    "articles_political": 42,
    "articles_economic": 85
  },
  "aggregates": {
    "7d_avg": 0.385,
    "30d_avg": 0.447,
    "90d_max": 0.991,
    "90d_max_date": "2026-01-21"
  },
  "major_events": [
    {
      "date": "2026-01-21",
      "score": 0.991,
      "level": "MUY ALTO",
      "summary": "José Jerí scandal - 6 censure motions",
      "report_url": "/political/daily-reports/2026-01-21.html"
    }
  ],
  "daily_series": [
    {"date": "2026-02-14", "score": 0.340},
    {"date": "2026-02-13", "score": 0.412},
    {"date": "2026-02-12", "score": 0.378}
  ]
}
```

---

## Visual Design System

### Color Palette

**Primary Colors**:
- Qhawarina Blue: `#1E40AF` (trust, stability)
- Qhawarina Green: `#059669` (positive growth)
- Qhawarina Red: `#DC2626` (negative/alert)
- Qhawarina Yellow: `#F59E0B` (warning/caution)

**Neutral Colors**:
- Dark Gray: `#1F2937` (text)
- Medium Gray: `#6B7280` (secondary text)
- Light Gray: `#F3F4F6` (backgrounds)
- White: `#FFFFFF`

### Typography

**Headings**: Inter (Google Fonts) - Clean, modern, excellent readability
- H1: 2.5rem, Bold (600)
- H2: 2rem, Bold (600)
- H3: 1.5rem, Semibold (500)

**Body Text**: Inter Regular (400)
- Base: 1rem (16px)
- Small: 0.875rem (14px)

**Monospace** (for data tables): JetBrains Mono
- Numbers and metrics

### Interactive Charts (CRITICAL: All charts must be dynamic)

**Primary Library**: **Plotly.js** (preferred) or Chart.js
- **Why Plotly**: Publication-quality, highly interactive, zoom/pan, hover tooltips, responsive
- **Fallback**: Chart.js for simpler charts (faster loading)

**Interactivity Features** (MUST HAVE):
1. **Hover tooltips**: Show exact values on mouseover
2. **Zoom/Pan**: Double-click to reset, scroll to zoom, drag to pan
3. **Toggle series**: Click legend to show/hide series (e.g., hide confidence intervals)
4. **Export**: Download chart as PNG/SVG (built-in Plotly toolbar)
5. **Responsive**: Auto-resize on mobile/tablet
6. **Animations**: Smooth transitions when data updates

**Chart Colors**:
- Line charts: Qhawarina Blue (#1E40AF)
- Bar charts: Gradient from Blue → Green (positive), Blue → Red (negative)
- Confidence intervals: Light blue with 30% opacity, toggleable
- Gridlines: Light gray (#E5E7EB), dashed

**Annotations** (Interactive):
- Nowcast values: Dashed line, hover shows "Nowcast: +2.14%"
- Official values: Solid line, hover shows "Official: +2.68%"
- Target bands: Horizontal shaded region (e.g., BCRP inflation target 1-3%)
- Event markers: Vertical lines for major events (e.g., COVID-19 start, elections)

**Example Chart Configurations**:

```javascript
// GDP Growth Chart (Plotly.js)
const gdpChart = {
  data: [
    {
      x: dates,
      y: officialGDP,
      name: 'Official GDP',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#1E40AF', width: 2 },
      marker: { size: 6 }
    },
    {
      x: nowcastDates,
      y: nowcastGDP,
      name: 'Nowcast',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#059669', width: 2, dash: 'dash' },
      marker: { size: 6, symbol: 'diamond' }
    },
    {
      x: dates,
      y: upperBound,
      name: '90% CI Upper',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#1E40AF', width: 0 },
      showlegend: false,
      hoverinfo: 'skip'
    },
    {
      x: dates,
      y: lowerBound,
      name: '90% CI',
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(30, 64, 175, 0.2)',
      line: { color: '#1E40AF', width: 0 }
    }
  ],
  layout: {
    title: 'Peru GDP Growth (YoY %)',
    xaxis: { title: 'Quarter', gridcolor: '#E5E7EB' },
    yaxis: { title: 'Growth Rate (%)', gridcolor: '#E5E7EB', zeroline: true },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#1F2937', font: { color: '#FFF' } },
    legend: { orientation: 'h', y: -0.2 },
    font: { family: 'Inter, sans-serif' }
  },
  config: {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { filename: 'qhawarina_gdp', height: 600, width: 1000 }
  }
};
```

**Advanced Features**:
- **Range slider**: For long time series (Plotly rangeslider)
- **Dropdown filters**: Select time period (1Y, 5Y, All)
- **Play button**: Animate time series (show GDP evolution quarter by quarter)
- **Comparison mode**: Overlay multiple models (DFM vs AR1 vs RW)

### Interactive Maps

**Library**: **Mapbox GL JS** (preferred) or Leaflet.js
- **Why Mapbox**: Smooth zoom, vector tiles, beautiful styling, fast rendering, 50,000 free loads/month
- **Fallback**: Leaflet.js (completely free, no API limits, but slower rendering)

**Map Interactivity Features** (MUST HAVE):
1. **Choropleth coloring**: Dynamic color scale based on poverty rate (green → yellow → red)
2. **Hover tooltips**: Show department name + poverty rate on mouseover
3. **Click drill-down**: Click department → zoom to district-level map
4. **Zoom/Pan**: Smooth navigation, pinch-zoom on mobile, double-click to reset
5. **Legend**: Interactive color scale showing poverty rate ranges
6. **Search box**: Find specific department or district by name
7. **Year toggle**: Switch between 2024 actual vs 2025 nowcast
8. **Mobile-responsive**: Touch-friendly on phones/tablets

**Color Scale** (Continuous gradient):
- Low poverty (5-15%): Green (#059669) → "Baja"
- Medium (15-30%): Yellow (#F59E0B) → "Media"
- High (30-50%): Red (#DC2626) → "Alta"
- Very high (50%+): Dark Red (#7F1D1D) → "Muy Alta"

**Example Map Configuration**:

```javascript
// Mapbox GL poverty choropleth map
mapboxgl.accessToken = 'YOUR_MAPBOX_TOKEN';
const map = new mapboxgl.Map({
  container: 'poverty-map',
  style: 'mapbox://styles/mapbox/light-v11',
  center: [-75.015, -9.19], // Peru geographic center
  zoom: 5,
  minZoom: 4,
  maxZoom: 10
});

map.on('load', () => {
  // Load Peru GeoJSON (departments or districts)
  map.addSource('poverty', {
    type: 'geojson',
    data: '/assets/data/poverty_map.geojson'
  });

  // Choropleth fill layer with color gradient
  map.addLayer({
    id: 'poverty-fill',
    type: 'fill',
    source: 'poverty',
    paint: {
      'fill-color': [
        'interpolate',
        ['linear'],
        ['get', 'poverty_rate_nowcast'],
        5, '#059669',   // Green (low)
        15, '#F59E0B',  // Yellow (medium)
        30, '#DC2626',  // Red (high)
        50, '#7F1D1D'   // Dark red (very high)
      ],
      'fill-opacity': [
        'case',
        ['boolean', ['feature-state', 'hover'], false],
        0.9,  // Darker on hover
        0.7   // Normal opacity
      ]
    }
  });

  // Department borders
  map.addLayer({
    id: 'poverty-outline',
    type: 'line',
    source: 'poverty',
    paint: {
      'line-color': '#1F2937',
      'line-width': 1
    }
  });

  // Hover effect with popup
  let popup = new mapboxgl.Popup({
    closeButton: false,
    closeOnClick: false
  });

  map.on('mousemove', 'poverty-fill', (e) => {
    map.getCanvas().style.cursor = 'pointer';

    if (e.features.length > 0) {
      const feature = e.features[0];
      popup.setLngLat(e.lngLat)
        .setHTML(`
          <div class="p-2">
            <div class="font-bold text-lg">${feature.properties.name}</div>
            <div class="mt-1">
              <span class="text-gray-600">Poverty Rate:</span>
              <span class="font-semibold text-red-600 ml-1">
                ${feature.properties.poverty_rate_nowcast.toFixed(1)}%
              </span>
            </div>
            <div class="text-xs text-gray-500 mt-1">
              Click to view districts →
            </div>
          </div>
        `)
        .addTo(map);
    }
  });

  map.on('mouseleave', 'poverty-fill', () => {
    map.getCanvas().style.cursor = '';
    popup.remove();
  });

  // Click to drill down to district level
  map.on('click', 'poverty-fill', (e) => {
    const deptCode = e.features[0].properties.department_code;
    const deptName = e.features[0].properties.name;

    // Load district-level GeoJSON for this department
    loadDistrictMap(deptCode, deptName);
  });
});

// Download PNG button
document.getElementById('download-map').addEventListener('click', () => {
  const canvas = map.getCanvas();
  const link = document.createElement('a');
  link.download = 'qhawarina_poverty_map.png';
  link.href = canvas.toDataURL();
  link.click();
});
```

**Map Performance Optimization**:
- Use vector tiles (not GeoJSON) for >1000 polygons
- Lazy load: Don't render map until user scrolls to it
- Simplify geometries: Use mapshaper to reduce file size (tolerance=0.001)
- Cache GeoJSON in browser localStorage

---

## Technical Stack (Recommendations)

### Frontend (Modern Stack for Professional Dashboard)
- **Framework**: **Next.js 14+** (React, App Router, static site generation, image optimization)
- **Styling**: **Tailwind CSS v3** (utility-first, responsive, dark mode support)
- **Charts**: **Plotly.js v2** (interactive zoom/pan/hover, export PNG/SVG, mobile-friendly)
  - Alternative: Recharts (React-native, simpler API, but less features)
- **Maps**: **Mapbox GL JS v3** (vector tiles, 3D terrain, smooth zoom, free tier: 50k loads/month)
  - Fallback: Leaflet.js v1.9 (completely free, but slower performance)
- **Icons**: **Lucide Icons** (modern, consistent, tree-shakeable)
- **Animations**: **Framer Motion** (smooth page transitions, chart entrance animations)
- **Data Fetching**: **SWR** or **TanStack Query** (auto-refresh, caching, stale-while-revalidate)
- **TypeScript**: Strongly recommended for type safety (catch errors before deployment)

### Data Pipeline
- **Export Scripts**: Python scripts to generate JSON/CSV/GeoJSON
- **Scheduling**: Windows Task Scheduler (daily at 08:00 PET)
- **Storage**: Static files in `/assets/data/` (no database needed)

### Hosting
- **Static Host**: Vercel, Netlify, or GitHub Pages
- **CDN**: Cloudflare (Peru-optimized edge caching)
- **SSL**: Let's Encrypt (free)

### Analytics
- **Privacy-first**: Plausible or Simple Analytics (GDPR-compliant, no cookies)
- **Metrics**: Page views, most-viewed nowcasts, download counts

---

## Development Phases

### Phase 1: Core Structure (Week 1)
- Set up Next.js project
- Implement homepage dashboard
- Create GDP, Inflation, Poverty detail pages
- Wire up JSON data exports

### Phase 2: Visualizations (Week 2)
- Integrate Plotly.js charts
- Build interactive Peru map (Leaflet + GeoJSON)
- Add political instability timeline
- Responsive design testing

### Phase 3: Data Integration (Week 3)
- Write Python export scripts for all JSON files
- Test daily update workflow
- Implement data versioning (timestamped exports)
- Add download page

### Phase 4: Polish (Week 4)
- Methodology pages (LaTeX equations)
- About/Team pages
- SEO optimization (meta tags, sitemaps)
- Performance optimization (image compression, lazy loading)

### Phase 5: Launch Prep
- Domain purchase: qhawarina.pe
- DNS setup (Cloudflare)
- Deploy to Vercel/Netlify
- Announce on Twitter/LinkedIn

---

## SEO Strategy

### Meta Tags (Example for Homepage)
```html
<title>Qhawarina - Nowcasting económico para Perú</title>
<meta name="description" content="Predicciones diarias de PBI, inflación y pobreza para Perú usando modelos de factores dinámicos. Datos abiertos y metodología transparente.">
<meta name="keywords" content="Peru GDP, nowcasting, inflation, poverty, economic indicators, BCRP, INEI">
<meta property="og:title" content="Qhawarina - Economic Nowcasting for Peru">
<meta property="og:description" content="Daily GDP, inflation, and poverty predictions for Peru">
<meta property="og:image" content="https://qhawarina.pe/assets/og-image.png">
<meta property="og:url" content="https://qhawarina.pe">
<meta name="twitter:card" content="summary_large_image">
```

### Target Keywords
- "Peru GDP forecast"
- "Peruvian inflation nowcast"
- "Poverty rate Peru 2025"
- "Billion Prices Project Peru"
- "Political instability Peru"
- "Economic indicators Peru real-time"

---

## Launch Checklist

- [ ] Purchase domain: qhawarina.pe
- [ ] Set up Cloudflare DNS
- [ ] Deploy Next.js site to Vercel
- [ ] Configure daily data export cron job
- [ ] Test all data exports (JSON, CSV, GeoJSON)
- [ ] Validate all charts render correctly
- [ ] Test responsive design (mobile, tablet, desktop)
- [ ] Add Google Analytics / Plausible
- [ ] Create sitemap.xml
- [ ] Submit to Google Search Console
- [ ] Write launch blog post
- [ ] Share on Twitter, LinkedIn, Reddit r/Peru
- [ ] Email BCRP, INEI, MEF for feedback

---

## Future Enhancements (Post-Launch)

### Short-term (3 months)
- Email alerts for major events (political index > 0.75)
- API endpoint with rate limiting (500 req/day free tier)
- English translation toggle
- Mobile app (React Native)

### Medium-term (6 months)
- User-contributed economic indicators (crowdsourced data)
- Regional GDP disaggregation (department-level)
- Sectoral breakdowns (agriculture, mining, services)
- Export forecast PDF reports (automated)

### Long-term (12 months)
- Subscription tier for high-frequency data (hourly political index)
- White-label API for institutional clients
- Expand to other Latin American countries (Colombia, Chile)
- Academic partnerships (PUCP, GRADE, IEP)

---

## Appendix A: Data Update Workflow

```bash
# Daily update script (runs at 08:00 PET via Task Scheduler)

# 1. Update all data sources (3 hours)
python scripts/update_nexus.py

# 2. Generate nowcasts (2 minutes)
python scripts/generate_nowcast.py

# 3. Export to JSON for website (1 minute)
python scripts/export_web_data.py

# 4. Sync to website assets folder
robocopy D:\Nexus\nexus\exports D:\qhawarina-web\public\assets\data /MIR

# 5. Trigger Vercel deployment
vercel --prod
```

---

## Appendix B: JSON Schema Validation

All JSON exports should validate against schemas:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["metadata", "nowcast"],
  "properties": {
    "metadata": {
      "type": "object",
      "required": ["generated_at", "model"],
      "properties": {
        "generated_at": {"type": "string", "format": "date-time"},
        "model": {"type": "string"}
      }
    },
    "nowcast": {
      "type": "object",
      "required": ["target_period", "value", "unit"],
      "properties": {
        "target_period": {"type": "string"},
        "value": {"type": "number"},
        "unit": {"type": "string"}
      }
    }
  }
}
```

---

## Contact

For questions about the design: [Your contact info]

**Last Updated**: 2026-02-14
**Version**: 1.0
**Status**: Ready for development
