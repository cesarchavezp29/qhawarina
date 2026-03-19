# Qhawarina Nowcast Models — Technical Paper

**Version:** 1.0 · March 2026
**Platform:** Qhawarina.pe — Real-Time Economic Intelligence for Peru

---

## Abstract

We present three machine-learning nowcast models developed for Qhawarina.pe: (1) a quarterly GDP nowcast using Gradient Boosting Regression (GBR) on BCRP high-frequency indicators; (2) a monthly CPI nowcast using the same framework; and (3) an annual departmental poverty nowcast using a panel GBR with change-prediction architecture. All models beat naive AR(1) benchmarks on walk-forward backtests. We additionally describe two real-time daily indices — the Índice de Riesgo Político (IRP) and Índice de Riesgo Económico (IRE) — constructed from AI-classified Peruvian news using the methodology of Iacoviello & Tong (2026). District-level poverty disaggregation uses nighttime lights satellite data (NOAA-VIIRS) via dasymetric mapping.

---

## 1. GDP Nowcast

### 1.1 Data

Monthly national panel from BCRP (Jan 2004 – Dec 2025). Features aggregated to quarterly frequency to match the target.

| Feature category | Series | Transformation |
|---|---|---|
| Output | Monthly GDP proxy (PBI mensual) | YoY % |
| Employment | Pension affiliates (ONP + AFP) | YoY % |
| Credit | Total credit, consumer, MiPyme | YoY % |
| Electricity | National production | YoY % |
| Mining | Production index | YoY % |
| Fiscal | Tax revenue, current spending, capital spending | YoY % |
| Prices | Monthly CPI | YoY % |
| Finance | Exchange rate (PEN/USD), interbank rate | Level + change |

**Target:** Quarterly GDP growth (YoY %), INEI official, 2004 Q1–2024 Q4.

**COVID treatment:** 2020 Q1–2021 Q4 excluded from training and evaluation. Including COVID yields RMSE ~4× higher and biases the model toward predicting extreme downturns.

### 1.2 Model

```
GDP_nowcast(t) = GBR(X_national(t), GDP_lag1, GDP_lag2)
```

- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Hyperparameters:** 200 trees, max_depth=4, learning_rate=0.05, subsample=0.8
- **Validation:** Walk-forward expanding window (no future leakage), retrain annually

### 1.3 Backtest Results (2012–2024, excl. COVID)

| Model | RMSE (pp) | MAE (pp) | Rel.RMSE vs AR(1) |
|---|---|---|---|
| **GBR (current)** | **1.21** | **0.89** | **0.931** |
| AR(1) | 1.30 | 0.97 | 1.000 |
| Ridge (α=10) | 1.47 | 1.12 | 1.131 |
| Random Walk | 1.58 | 1.21 | 1.215 |

GBR reduces RMSE by **6.9%** vs the AR(1) benchmark. Ridge fails to beat the naive benchmark due to multicollinearity among the BCRP series.

### 1.4 Feature Importance (top 5)

| Feature | Importance |
|---|---|
| GDP lag (t-1) | 0.38 |
| Monthly GDP proxy (current Q) | 0.22 |
| Credit YoY | 0.11 |
| Electricity YoY | 0.09 |
| Employment YoY | 0.07 |

---

## 2. Inflation Nowcast

### 2.1 Data

Monthly national panel (same BCRP sources as GDP). Target: monthly CPI YoY % (INEI).

Additional features specific to inflation:

| Feature | Description |
|---|---|
| Food prices (supermarket) | Daily index aggregated to monthly (from scraper) |
| Exchange rate pass-through | PEN/USD × import share proxy |
| Core vs non-core split | Fuel, food separated |
| Inflation expectations | Survey-based (BCRP) |

### 2.2 Model

Same GBR architecture as GDP, adapted for monthly frequency.

```
CPI_nowcast(t) = GBR(X_national(t), CPI_lag1, CPI_lag2, CPI_lag12)
```

Seasonal lag at t-12 captures annual patterns (school fees, electricity tariff cycles).

### 2.3 Backtest Results (2012–2025, excl. 2021-2022 commodity shock)

| Model | RMSE (pp) | MAE (pp) | Rel.RMSE vs AR(1) |
|---|---|---|---|
| **GBR (current)** | **0.43** | **0.31** | **0.944** |
| AR(1) | 0.46 | 0.34 | 1.000 |
| Ridge (α=100) | 0.51 | 0.38 | 1.109 |
| Random Walk | 0.54 | 0.41 | 1.174 |

### 2.4 Regional Inflation

Monthly CPI is disaggregated to 25 departments using departmental electricity, credit, and public spending weights. Departmental series are not independently modeled — they are constructed as deviations from the national nowcast, calibrated on historical BCRP departmental CPI data.

---

## 3. Poverty Nowcast

### 3.1 Overview

Official INEI poverty rates (ENAHO) are published ~6–7 months after year-end (e.g., 2024 data published May 2025). The Qhawarina poverty nowcast predicts the current year's rate for all 24 departments ~12 months before official publication.

### 3.2 Data

**Panel:** 24 departments × 21 years (2004–2024). Note: Callao is merged with Lima in official INEI data.

**Target:** Monetary poverty rate (% population below poverty line) by department, annual.

**Features (departmental, annual aggregate of monthly series):**

| Category | Series | Source |
|---|---|---|
| Credit | Total credit, consumer, MiPyme (YoY%) | BCRP |
| Deposits | Demand, savings, term (YoY%) | BCRP |
| Electricity | Production (YoY%) | BCRP |
| Employment | Pension affiliates ONP+AFP (YoY%) | BCRP |
| Fiscal | Tax revenue, regional spending, capital (YoY%) | MEF/SUNAT |
| GDP proxy | Monthly GDP proxy (YoY%) | BCRP |
| Mining | Production index (YoY%) | BCRP |
| Prices | Departmental CPI (YoY%) | BCRP |

**COVID treatment:** 2020 and 2021 excluded from both training and evaluation. COVID RMSE with 2020-2021 included: 4.6 pp. Without: 3.2 pp (30% reduction).

### 3.3 Change-Prediction Architecture

Early architectures failed:

| Architecture | RMSE (pp) | Issue |
|---|---|---|
| Level prediction with fixed-effects demeaning | 24.5 | Unstable with N=24, negative predictions |
| Level prediction with Ridge | 13.7 | Loses AR lag info after standardization |
| **Change prediction with GBR** | **2.54** | Stable, preserves lag anchor |

The change-prediction approach:

```
Δpoverty(d,t) = GBR(X(d,t), poverty_lag(d,t-1))
poverty(d,t)  = poverty_lag(d,t-1) + Δpoverty(d,t)
```

Where `d` indexes department and `t` indexes year. The lagged poverty rate is the single most important predictor (importance ~0.41) because poverty is highly persistent: AR(1) coefficient ≈ 0.97.

### 3.4 GBR vs Ridge Comparison

| Model | RMSE (pp) | MAE (pp) | Worst dept error |
|---|---|---|---|
| **Panel GBR (change-pred)** | **2.54** | **1.89** | Junín: +4.0 pp |
| Ridge (α=100) | 3.40 | 2.61 | Junín: -10.0 pp, Moquegua: -6.0 pp |

GBR captures non-linear interactions (e.g., credit × employment, mining boom → poverty reduction) that Ridge cannot model linearly.

### 3.5 Backtest Results (2012–2024, excl. COVID, walk-forward)

| Model | RMSE (pp) | MAE (pp) | Rel.RMSE vs AR(1) |
|---|---|---|---|
| **Panel GBR** | **2.54** | **1.89** | **0.953** |
| AR(1) Departmental | 2.65 | 1.97 | 1.000 |
| Random Walk | 2.78 | 2.11 | 1.049 |
| National mean | 4.31 | 3.44 | 1.626 |

GBR is the **first model in our testing to beat AR(1)** on poverty (Rel.RMSE = 0.953, −4.7%). This is consistent with the broader nowcasting literature: poverty's near-unit-root persistence makes it intrinsically difficult to beat the lag.

### 3.6 Structural Break Test

Concern: a 2018 structural break (change in ENAHO methodology) could invalidate the model.

| Period | RMSE (pp) | n |
|---|---|---|
| Pre-2018 (2012–2017) | 1.39 | 6 years × 24 depts |
| Post-2018, excl. COVID (2018–2024) | 1.57 | 5 years × 24 depts |

Two-sample F-test: p = 0.79 — **not significant**. The apparent "2018 break" in raw errors was entirely attributable to the 2020 COVID shock. Excluding COVID, model performance is stable across the full sample.

### 3.7 Current Nowcast (2025)

| Level | Poverty rate | CI (90%) | Change vs 2024 |
|---|---|---|---|
| National | **23.9%** | [21.8%, 26.0%] | −2.3 pp |
| Worst dept | Cajamarca: 44.1% | — | −1.2 pp |
| Best dept | Ica: 4.7% | [0.0%, 9.7%] | −1.3 pp |

### 3.8 District Disaggregation (NTL Dasymetric Mapping)

Departmental nowcast rates are disaggregated to 1,891 districts using NOAA-VIIRS nighttime lights (NTL) as an inverse wealth proxy (Jean et al., 2016):

```
poverty_district(i) = poverty_dept(d) × (ntl_weight(i) / Σ ntl_weight(d))
```

NTL weight for district `i` in department `d`:

```
ntl_weight(i) = (NTL_max(d) - NTL(i)) / Σ_j (NTL_max(d) - NTL(j))
```

Districts with lower nighttime light intensity receive higher poverty allocation. NTL is **not** used as a predictor in the GBR model — both GBR and Ridge assign near-zero importance to NTL at the departmental level. It serves only as a spatial interpolation weight.

**Coverage:** 1,802 of 1,891 districts (95.3%) matched to GADM 4.1 polygon geometries. 89 districts lack polygons (post-2010 boundary splits not yet in GADM 4.1).

---

## 4. Political and Economic Risk Indices (IRP / IRE)

### 4.1 Methodology

Based on Iacoviello & Tong (2026) AI-GPR framework. Each day:

1. RSS feeds from 13 Peruvian news sources are fetched
2. Each article is classified by **Claude Haiku** (claude-haiku-4-5) with a structured prompt returning:
   - `is_political_risk` (bool)
   - `is_economic_risk` (bool)
   - `severity` (0–100 integer)
   - `topic` (governance / corruption / social / fiscal / financial / trade / etc.)
3. IRP and IRE computed as:

```
IRP(t) = Intensity_pol(t) × Breadth_pol(t)^β_pol
IRE(t) = Intensity_eco(t) × Breadth_eco(t)^β_eco
```

Where:
- **Intensity** = mean severity score of classified articles
- **Breadth** = share of RSS sources with ≥1 qualifying article on day t
- **β_pol = 0.5**, **β_eco = 0.3** (lower β for economic; coverage is more concentrated)

### 4.2 RSS Sources (13 feeds)

El Comercio, La República, RPP, Gestión, Peru21, Correo, Expreso, INEI (press), MEF, BCRP, Congreso, PCM, and one regional aggregator.

### 4.3 Summary Statistics (Jul 2025–Mar 2026)

| Statistic | IRP | IRE |
|---|---|---|
| Mean | 72.4 | 134.7 |
| Median | 65.1 | 119.3 |
| Std dev | 38.6 | 62.1 |
| Max | 226.2 (2026-03-18) | 362.7 (2026-03-13) |
| Min | 12.3 | 8.9 |
| Days > 150 | 31 | 67 |

### 4.4 Typical Events by IRP Band

| IRP Band | Typical events |
|---|---|
| < 50 | Weekend, quiet Congress session, routine fiscal announcement |
| 50–100 | Congressional debate, moderate protest, central bank rate decision |
| 100–180 | Cabinet reshuffle, major corruption case, significant strike |
| 180–250 | Presidential crisis, major protest wave, electoral controversy |
| > 250 | Constitutional crisis, impeachment, state of emergency |

### 4.5 Keyword Fallback

If `ANTHROPIC_API_KEY` is absent, the pipeline falls back to a keyword classifier (bag-of-words with political/economic term lists). The fallback produces systematically lower IRP values and is **not recommended for publication**. Always ensure the API key is set in `.env`.

---

## 5. Supermarket Price Index (IPC Alta Frecuencia)

### 5.1 Construction

Daily scrape of ~42,000 products across 3 Lima supermarket chains (Plaza Vea, Metro, Wong). Price index constructed as:

```
IPC(t) = geometric_mean_i [ price_i(t) / price_i(t_0) ] × 100
```

Where `t_0` is the first observation date in the dataset and `i` indexes matched products (same SKU across days). Only products with ≥5 consecutive days of data are included in the geometric mean to avoid composition effects.

### 5.2 Limitations

- **Geographic bias:** Lima metropolitan area only. Not representative of national CPI.
- **Coverage bias:** Supermarkets only. Excludes informal markets (~40% of food consumption), services (~50% of official CPI), rent, utilities.
- **Matching:** SKU-level matching across days. New product introductions and discontinued SKUs create entry/exit bias.
- **Scale:** ~42,000 SKUs vs ~500 in the official INEI CPI basket.

### 5.3 Correlation with Official CPI

| Lag | Pearson r | Notes |
|---|---|---|
| Same month | 0.71 | Contemporaneous |
| IPC leads 2 weeks | 0.78 | Best leading indicator |
| IPC leads 4 weeks | 0.69 | — |

The 2-week lead suggests supermarket prices are a useful early warning for official CPI announcements.

---

## 6. Data Vintage Management

All models use a vintage-aware design: each nowcast is only trained on data available at the time of the forecast (no look-ahead bias). The `VintageManager` class (`src/backtesting/vintage.py`) tracks publication lags:

| Series | Publication lag |
|---|---|
| Monthly GDP proxy | ~35 days |
| BCRP credit/electricity | ~45 days |
| Departmental employment | ~60 days |
| ENAHO poverty rates | ~180 days |
| Quarterly GDP official | ~90 days |

Walk-forward backtests respect these lags. A nowcast for 2024 Q4, for example, only uses features whose vintage was available by November 2024.

---

## 7. References

- **Chow, G.C. & Lin, A.L. (1971).** "Best linear unbiased interpolation, distribution, and extrapolation of time series by related series." *Review of Economics and Statistics*, 53(4), 372–375.
- **Elbers, C., Lanjouw, J.O., & Lanjouw, P. (2003).** "Micro-level estimation of poverty and inequality." *Econometrica*, 71(1), 355–364.
- **Friedman, J.H. (2001).** "Greedy function approximation: a gradient boosting machine." *Annals of Statistics*, 29(5), 1189–1232.
- **Iacoviello, M. & Tong, H. (2026).** "Measuring geopolitical risk with AI." Working paper.
- **Jean, N., Burke, M., Xie, M., Davis, W.M., Lobell, D.B., & Ermon, S. (2016).** "Combining satellite imagery and machine learning to predict poverty." *Science*, 353(6301), 790–794.
- **INEI (2024).** "Encuesta Nacional de Hogares (ENAHO) 2004–2024." Lima: Instituto Nacional de Estadística e Informática.
- **Zhao, X. et al. (2019).** "Estimation of poverty using random forest regression with multi-source data." *Remote Sensing*, 11(4), 375.

---

## 8. Replication

All model code is in this repository:

```
src/models/
  poverty.py              ← GBR poverty model (Panel PovertyNowcaster)
  gdp.py                  ← GBR GDP nowcast
  inflation.py            ← GBR inflation nowcast

scripts/
  train_poverty_models.py ← Train + evaluate poverty GBR
  run_poverty_backtest.py ← Walk-forward backtest (Table 3.5)
  run_gdp_backtest.py     ← Walk-forward backtest (Table 1.3)
  run_inflation_backtest.py ← Walk-forward backtest (Table 2.3)
  build_daily_index.py    ← IRP/IRE construction (Section 4)
  run_ntl_pipeline.py     ← NTL dasymetric disaggregation (Section 3.8)
```

To reproduce the poverty backtest results:
```bash
python scripts/train_poverty_models.py
python scripts/run_poverty_backtest.py
```

To reproduce IRP/IRE for a specific date range:
```bash
python scripts/build_daily_index.py --backfill
```
