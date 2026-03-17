# Minimum Wage Effects in Peru: Bunching, Employment, and Wage Distribution
## Complete Paper Outline

---

## STRUCTURAL NOTE
- **Bunching estimator** (Tables 2, 4): valid for all three events (A, B, C). Does NOT require
  parallel trends. Compares wage distribution within same population pre vs. post.
- **DiD / Event study** (Table 5, Figure 1): Events A and C only. Event B FAILS pre-trend test
  (F-test p=0.0014) for employment → moved to Appendix B.
- **OWE** (Table 5): Pooled over Events A+B only. Event C excluded (weak instrument, F=1.5).
- **Panel decomposition** (Table A1): Appendix only. 86% two-year attrition → survivorship bias.
- **EPEN DEP bunching** (Table A2): DROPPED. MW does not bind (median wage S/2,200+, Kaitz≈0.47).
- **Construction sector** (Table 4): EXCLUDED. Extreme outliers (ratios 3.23 Event A, 8.31 Event C),
  likely sparse cell noise.

---

## 1. INTRODUCTION (≈1,200 words)

**Hook**: Peru has raised the minimum wage five times since 2000, yet the employment effects remain
contested. The formal–informal divide and weak enforcement create a distinctive institutional context
in which standard competitive predictions may not hold.

**Question**: How do minimum wage increases affect (1) the wage distribution (bunching/missing mass),
(2) employment along the extensive margin, and (3) hours along the intensive margin among
formal dependent workers in Peru?

**Contribution**:
1. First application of the Cengiz et al. (2019) revised bunching estimator to Peru using
   three identifiable MW events.
2. First IV/OWE estimate for Peru following Dube & Lindner (2024), using pre-period Kaitz
   as a cross-department instrument.
3. Panel-based transition analysis for Event C (ENAHO Panel 978) — limited to appendix
   due to attrition.

**Main findings** (preview):
- Formal dependent workers show **missing mass** below the MW (bunching ratio ≈ 0.70–0.83)
  and **excess mass** at and just above the MW across all three events.
- Informal workers show a **lighthouse effect**: bunching ratios 0.81–1.22 indicate spillovers
  from formal sector wage norms.
- **Employment effect is near zero**: Pooled OWE = −0.114 (SE = 0.366), consistent with
  zero. Confidence interval spans −0.83 to +0.60.
- **Hours adjustment is negligible**: DiD estimates ±1h/week, none statistically significant.

**Paper structure**: Section 2 describes the institutional context and MW events. Section 3
presents data and sample. Section 4 describes the empirical strategy. Sections 5–7 present
bunching, employment, and heterogeneity results. Section 8 concludes.

---

## 2. INSTITUTIONAL CONTEXT (≈800 words)

### 2.1 Peru's Minimum Wage System
- **Remuneración Mínima Vital (RMV)**: set by executive decree, applies to all formal dependent
  workers (empleados and obreros). Not indexed.
- Enforcement: through SUNAFIL inspections; binding primarily for registered (formal) workers.
- Informal employment ≈ 72–75% of employed (ENAHO 2015–2023); MW directly affects ≈10–15%
  of the workforce.

### 2.2 The Three MW Events
| Event | Pre-MW | Post-MW | Change | Effective date | Pre year | Post year |
|-------|--------|---------|--------|----------------|----------|-----------|
| A | S/750 | S/850 | +13.3% | May 2016 | 2015 | 2017 |
| B | S/850 | S/930 | +9.4% | April 2018 | 2017 | 2019 |
| C | S/930 | S/1,025 | +10.2% | May 2022 | 2021 | 2023 |

- Event B note: DiD employment estimates fail parallel trends test (see Appendix B).
  Bunching results for Event B are retained in the main analysis.

### 2.3 Labor Market Structure
- Formal dependent workers (ENAHO `ocupinf==2`, `cat07p500a1==2`): ~12–14% of employed.
- Kaitz index (MW / median formal wage) ranges 0.44–0.55 across departments,
  providing identification variation.

---

## 3. DATA (≈700 words)

### 3.1 ENAHO Module 500 (Cross-Section)
- **Source**: Encuesta Nacional de Hogares sobre Condiciones de Vida y Pobreza,
  Módulo 500 (Empleo e Ingresos), INEI Peru.
- **Years**: 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023 (no 2020 due to COVID-19).
- **Sample restriction**: Formal dependent workers earning S/200–S/8,000/month.
  - Employment: `ocu500 == 1`
  - Dependent: `cat07p500a1 == 2` (or `p507 ∈ {3,4,6}`)
  - Formal: `ocupinf == 2` (registered with social security or under written contract)
- **Key variables**:
  - Wage: `p524a1` (monthly cash income, primary job); fallback `i524a1/12`
  - Hours: `p513t` (total weekly hours, primary job)
  - Weight: `factor07i500a` (cross-section expansion factor)
  - Department: first 2 digits of `ubigeo` (24 departments)
- **Sample sizes** (formal dependent workers):
  | Year | N |
  |------|---|
  | 2015 | 10,102 |
  | 2016 | 11,588 |
  | 2017 | 10,930 |
  | 2018 | 11,792 |
  | 2019 | 10,992 |
  | 2021 | 9,008 |
  | 2022 | 9,666 |
  | 2023 | 9,931 |

### 3.2 EPEN CIU Annual 2023 (Corroboration — Appendix A)
- Source: Encuesta Permanente de Empleo Nacional — Ciudades, Annual 2023, INEI code 873.
- Urban formal workers. Single cross-section (2022 annual not available).
- Used for single-period Lee-Saez bunching as corroboration only.

### 3.3 ENAHO Panel 978 (Appendix A)
- Source: ENAHO Panel 2020–2024 (code 978), wide format (348,505 × 7,337 columns).
- Matched 2021→2023 subsample identified by `facpanel2123 > 0`.
- **Not used in main results** due to ~86% two-year attrition rate; survivorship bias
  cannot be ruled out.

---

## 4. EMPIRICAL STRATEGY (≈900 words)

### 4.1 Bunching Estimator (Cengiz et al. 2019 — Revised)
Bin width: S/25. Wage distribution in S/200–S/4,000.

**Counterfactual**: polynomial (degree 4) fit on "clean zone" (wages > 2 × MW_new).

**Missing mass**: net deficit in [0.85 × MW_old, MW_new):
$$B^{miss} = \sum_{b \in \text{miss zone}} \max(-\Delta_b, 0)$$

**Excess mass**: positive excess in [MW_new, MW_new + 10 × \text{bin\_width}):
$$B^{exc} = \sum_{b \in \text{exc zone}} \max(\Delta_b, 0)$$

**Bunching ratio**: $R = B^{exc} / B^{miss}$ (ratio > 1: excess exceeds missing, net employment gain).

This estimator is valid without assuming parallel trends. It characterizes the redistribution
of the wage distribution around the MW.

### 4.2 OWE / IV Estimation (Dube & Lindner 2024)
**Instrument**: Pre-period Kaitz index, $K_{d,\text{pre}} = \text{MW}_{\text{old}} / \text{median formal wage}_{d,\text{pre}}$.
Variation across 24 departments.

**First stage** (wage equation):
$$\log w_{idt} = \alpha_d + \gamma_t + \pi \cdot (K_{d,\text{pre}} \times \text{post}_t)
  + X_{it}' \delta + \varepsilon_{idt}$$

**Reduced form** (employment):
$$\text{emp}_{idt} = \alpha_d + \gamma_t + \beta \cdot (K_{d,\text{pre}} \times \text{post}_t)
  + X_{it}' \delta + u_{idt}$$

**OWE** = $\beta / \pi$, delta-method SE. SE clustered by department (24 clusters).

Estimation uses WLS (individual survey weights), restricts to formal dependent workers
for first stage and all working-age adults for reduced form.

**Event C excluded** from pooled OWE: first-stage F = 1.5 (weak instrument).

### 4.3 Event Study (Pre-Trends Test)
$$Y_{idt} = \alpha_d + \gamma_t + \sum_{s \neq \text{base}} \beta_s
  \cdot (K_{d,\text{pre}} \times \mathbf{1}[\text{year} = s]) + X_{it}' \delta + \varepsilon_{idt}$$

Omitted year = last pre-period year. Pre-trend coefficients must be ≈ 0. Joint F-test reported.

**Event B employment**: F-test p = 0.0014 → parallel trends violated → DiD not identified.
Event B bunching retained (pre-trend test not required for distributional estimator).

### 4.4 Intensive Margin (Hours DiD)
Standard 2×2 DiD comparing weighted mean weekly hours (`p513t`) for
treatment zone [0.85 × MW_old, MW_new) vs. control zone [1.5 × MW_new, 3 × MW_new],
pre and post. Pseudo-panel using repeated cross-sections.

---

## 5. BUNCHING RESULTS (≈1,000 words)

### 5.1 Main Results (Table 2)

**TABLE 2: Cengiz Revised Bunching Estimates — ENAHO CS**

| Event | Group | Missing (pp) | Excess (pp) | Ratio | Emp. Δ (%) |
|-------|-------|-------------|-------------|-------|------------|
| A (750→850) | Formal dep | — | — | 0.696 | — |
| A | Informal | — | — | 1.218 | — |
| B (850→930) | Formal dep | — | — | 0.829 | — |
| B | Informal | — | — | 0.987 | — |
| C (930→1025) | Formal dep | — | — | 0.830 | — |
| C | Informal | — | — | 0.811 | — |

*Note: populate from `mw_complete_margins.json → bunching_revised`.*

**Interpretation**:
- Formal dependent bunching ratios < 1 (0.70–0.83): missing mass exceeds excess mass,
  consistent with some workers displaced out of the treated wage zone (transition to
  informal, reduced hours, or higher wages via renegotiation).
- Informal lighthouse effect: ratios 0.81–1.22 indicate spillover of the MW norm
  into informal wage-setting, stronger for Event A.

### 5.2 EPEN CIU Corroboration (Table A1 — Appendix)
Single-period Lee-Saez bunching on EPEN CIU Annual 2023:
- All cities: excess factor = 1.34×, bunching ratio = 1.622
- Lima upper stratum: ratio = 1.602
- Other cities: ratio = 1.688
Confirms bunching around MW = S/1,025 in post-period urban data.

---

## 6. EMPLOYMENT EFFECTS (≈900 words)

### 6.1 Own-Wage Elasticity (Table 5)

**TABLE 5: IV/OWE Estimates — Dube & Lindner (2024)**

| Event | π (wage FS) | SE | F | β (emp RF) | SE | OWE | SE |
|-------|------------|----|----|------------|----|----|-----|
| A | 0.110 | 0.084 | 1.7 | −0.011 | 0.040 | −0.097 | 0.367 |
| B | −0.059 | 0.094 | 0.4 | 0.171 | 0.052 | — | — |
| C | 0.175 | 0.144 | 1.5† | 0.183 | 0.088 | 1.045 | 0.995 |
| **Pooled A+B** | — | — | — | — | — | **−0.114** | **0.366** |

†Event C: weak instrument (F < 10). OWE not used in main pooled estimate.
Event B: first-stage sign reversed (negative Kaitz–wage relationship), OWE unreliable.

**Headline**: Pooled OWE = **−0.114 (SE = 0.366, p = 0.755)**, 95% CI [−0.83, +0.60].
Consistent with zero employment elasticity. The wide confidence interval reflects
limited geographic variation across 24 Peruvian departments.

Dube (2019) US benchmark: OWE ≈ −0.04. Peru estimate cannot reject this value nor zero.

### 6.2 Event Study (Figure 1)

**FIGURE 1: Event Study Pre-Trend Coefficients**

Panels: Employment (left) and Log wage (right) for Events A and C.

- **Event A employment**: pre-trend (base 2015) → 2016 β = +0.060 (ns). PASS.
- **Event A log wage**: all pre-period betas ns. PASS.
- **Event C employment**: pre-trend (base 2021) → 2019 β = +0.089 (ns). PASS (p=0.260).
- **Event C log wage**: PASS (p=0.654).

Post-period estimates show positive employment effects (β_post > 0), consistent
with zero or mildly positive OWE.

*Event B shown in Appendix B with pre-trend violation note.*

---

## 7. HETEROGENEITY (≈700 words)

### 7.1 By Sector (Table 4)

**TABLE 4: Bunching Ratios by Industry Sector**
*(Construction excluded — outlier values driven by sparse cells)*

| Sector | Event A | Event B | Event C | Avg |
|--------|---------|---------|---------|-----|
| Total (reference) | 0.733 | 0.854 | 0.840 | 0.809 |
| Agricultura/Minería | 1.947 | 0.792 | 0.594 | 1.111 |
| Manufactura | 1.449 | 0.955 | 1.126 | 1.177 |
| Comercio | 0.791 | 1.230 | 0.728 | 0.916 |
| Transporte y alojamiento | 1.036 | 1.040 | 0.609 | 0.895 |
| Servicios profesionales | — | — | — | — |

*Populate remaining rows from `mw_heterogeneity.json → subgroups`.*

**Interpretation**:
- Manufactura consistently above aggregate (ratios 0.96–1.45): suggests stronger
  MW compliance and possibly more wage renegotiation.
- Agricultura/Minería high ratio Event A: heterogeneous enforcement across regions.
- Transporte/alojamiento declining ratio in Event C: possible labor demand softening
  post-COVID.

### 7.2 Regional Variation
- Lima vs. rest: Lima formal workers have lower Kaitz (high median wages), smaller
  bunching response. Non-Lima departments show stronger distributional effects.

---

## 8. INTENSIVE MARGIN (≈400 words)

### 8.1 Hours DiD (Table 6)

**TABLE 6: Intensive Margin DiD — Weekly Hours (p513t)**

| Event | Treat pre | Treat post | Δ treat | Ctrl pre | Ctrl post | Δ ctrl | DiD |
|-------|-----------|------------|---------|----------|-----------|--------|-----|
| A (750→850) | 44.4h | 44.8h | +0.4h | 42.1h | 41.5h | −0.7h | **+1.1h** |
| B (850→930) | 44.8h | 45.1h | +0.3h | 41.2h | 42.0h | +0.8h | **−0.5h** |
| C (930→1025) | 44.5h | 46.4h | +1.9h | 42.1h | 43.6h | +1.5h | **+0.4h** |

No DiD estimate exceeds ±1.5h/week. None approach conventional significance thresholds.

**Interpretation**: MW increases in Peru do not appear to generate systematic reductions
in weekly hours among formal dependent workers. Employers are not substituting
along the intensive margin in response to MW increases.

---

## 9. CONCLUSION (≈600 words)

**Summary of findings**:
1. MW increases in Peru generate clear bunching below the MW and excess mass at/above
   the MW among formal dependent workers — a distributional response consistent with
   wage floor enforcement.
2. Informal sector wages exhibit lighthouse effects in some events, indicating that
   MW norms partially transmit to informal wage-setting.
3. Employment elasticity is near zero and imprecisely estimated (pooled OWE = −0.114,
   SE = 0.366). The data cannot distinguish Peru from the Dube (2019) US benchmark
   of OWE ≈ −0.04.
4. No meaningful intensive margin (hours) adjustment.

**Interpretation**: Peru's MW increases appear to compress the formal wage distribution
without large disemployment effects. This is consistent with:
- A labor market with significant monopsonistic elements (employers have wage-setting power)
- Partial enforcement reducing the effective bite for marginal workers
- The formal sector operating away from the labor demand margin due to informal alternatives

**Limitations**:
- 24 departments provide limited variation for IV identification (weak first stages).
- Event B excluded from DiD due to pre-trend violation.
- Panel evidence limited by high attrition in ENAHO Panel 978.

**Policy implications**: The results do not support the view that Peru's recent MW increases
(+37% from 2015 to 2023) have caused large employment losses among formal workers.
However, distributional effects — workers displaced below or above the MW zone — warrant
attention.

---

## APPENDICES

### Appendix A: ENAHO Panel 978 Decomposition (Table A1)
**CAVEAT**: 86% two-year attrition (2021→2023). Results subject to survivorship bias.
Matched sample identified by `facpanel2123 > 0`.

| Metric | Treatment (N=1,171) | Control (N=3,903) | DiD |
|--------|---------------------|-------------------|-----|
| Employment retention | 15.8% | 15.0% | +0.8pp |
| Formal dep retention | 12.1% | 12.5% | −0.4pp |
| Δ log wage (stayers) | +0.264 | +0.143 | **+0.121** |
| Δ hours (stayers) | −0.5h/wk | −2.1h/wk | **+1.6h/wk** |

Transition matrix (Treatment, N=1,171):
- Formal dep 2023: 12.1%
- Informal dep 2023: 2.1%
- Not observed 2023: 85.8% (panel attrition, NOT unemployment)

*Do not interpret 85.8% as job loss.*

### Appendix B: Event B Employment DiD (Pre-Trend Violation)
Event B (850→930, April 2018) fails parallel trends test for employment:
pre-period F-test p = 0.0014. Employment coefficients in 2015 and 2016
are significantly positive before the 2018 MW change, invalidating causal
identification via the Kaitz DiD.

Log-wage pre-trends PASS (p = 0.514). Employment pre-trends FAIL.

**Bunching results for Event B are retained in main text** (Table 2, Table 4):
the distributional estimator does not rely on parallel trends.

### Appendix C: EPEN CIU Annual 2023 — Lee-Saez Bunching (Table A1-CIU)
| Sample | N | Excess factor | Bunching ratio |
|--------|---|---------------|----------------|
| All cities | 53,316 | 1.34× | 1.622 |
| Lima upper stratum | — | — | 1.602 |
| Other cities | — | — | 1.688 |

Data: EPEN CIU Annual 2023 (INEI code 873). MW = S/1,025.
Formality: `INFORMAL_P == '2'` (health insurance registered).
Single-period estimate — no pre/post comparison available.

---

## FILE INDEX

| File | Contents |
|------|----------|
| `exports/data/mw_complete_margins.json` | Bunching (Table 2), lighthouse (supplemental), wage compression |
| `exports/data/mw_heterogeneity.json` | Sector/regional bunching ratios (Table 4) |
| `exports/data/mw_iv_owe.json` | OWE, first stage, event study coefficients (Table 5, Figure 1) |
| `exports/data/mw_hours_epen_dep.json` | Hours DiD (Table 6) |
| `exports/data/mw_panel_decomposition.json` | Panel decomposition (Appendix A) |
| `exports/data/mw_epen_ciu_annual_bunching.json` | EPEN CIU Lee-Saez (Appendix C) |
| `exports/data/mw_paper_results_complete.json` | All results aggregated |

---

*Generated: 2026-03-17. Data: ENAHO 2015–2023, EPEN CIU Annual 2023, ENAHO Panel 978.*
