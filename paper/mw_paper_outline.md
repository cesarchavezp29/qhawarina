# Missing Mass and Minimum Wages: Distributional Effects of Three MW Increases in Peru
### [Alt. title: "Where Do the Workers Go? Bunching Evidence from Peru's Minimum Wage Increases"]
## Complete Paper Outline — Revised v4

---

## IDENTIFICATION SCORECARD

| Method | Event A | Event B | Event C | Verdict |
|--------|---------|---------|---------|---------|
| Cengiz bunching | ✓ valid | ✓ valid | ✓ valid | **Main result** |
| Event study / DiD | ⚠ no pre-period† | ✗ fail (p=0.010) | ✗ fail (p=0.034) | Appendix B — not identified |
| IV / OWE | ✗ F=2.9 (ns) | ✗ F=0.6, reversed sign | ✗ F=2.4 (ns) | Appendix C — not identified |
| Panel decomp | — | — | ✗ 86% attrition | Appendix A — unreliable |
| EPEN DEP bunching | — | — | ✗ MW doesn't bind (Kaitz=0.47) | Dropped |

†Event A: 2015 is the base year; no prior ENAHO years available for pre-trend test.
Event B and C: joint employment pre-trend F-tests fail (p=0.010 and p=0.034). DiD reported descriptively.

---

## 1. INTRODUCTION (≈1,000 words)

**Hook**: Peru raised the minimum wage from S/750 to S/1,025 between 2016 and 2022 — a 37%
increase over six years — while formal employment remained stable. This paper uses three
identifiable MW events to study how the wage distribution responded.

**Question**: Do Peru's MW increases produce missing mass below the threshold and bunching at
it? Do they affect employment and hours? We use the Cengiz et al. (2019) revised bunching
estimator as our primary tool, with event-study DiD as supplementary evidence where
pre-trends hold.

**Contribution**:
1. First application of the Cengiz et al. (2019) bunching estimator to Peru, using three
   MW events across 2015–2023.
2. Direct test for MW spillovers to informal wages using the same estimator on formal and
   informal workers simultaneously.
3. We attempted the Dube & Lindner (2024) IV/OWE approach; first stages are uniformly
   weak (F ≤ 3, reversed sign in Event B), and employment pre-trends fail in all testable
   events. We document why these approaches fail in Peru and report results in Appendices B–C.

**Main findings**:
- Formal dependent workers show clear missing mass and excess mass at the MW in all three
  events (bunching ratios 0.70–0.83), with increasing missing mass across events as the MW
  rises relative to the wage distribution.
- Wage compression: workers just above the new MW see 3–5pp faster log-wage growth than
  high-wage workers, consistent with the distributional shift.
- Informal workers: partial response in Event A (ratio 1.22) but not Events B and C
  (0.99 and 0.81). No consistent spillover pattern.
- Event study DiD: pre-trends fail for all testable events (B: p=0.010, C: p=0.034).
  Results are descriptive only; no identified employment effect is reported.
- Hours adjustment is negligible across all events (DiD ≤ ±1.1h/week).
- The IV/OWE approach is not identified; no credible employment elasticity is reported.

**Paper structure**: Section 2 describes the institutional context. Section 3 presents data
and summary statistics (Table 1). Section 4 describes the empirical strategies. Sections 5–7
present bunching, heterogeneity, and supplementary DiD results. Section 8 concludes.

---

## 2. INSTITUTIONAL CONTEXT (≈700 words)

### 2.1 Peru's Minimum Wage System
- **Remuneración Mínima Vital (RMV)**: set by executive decree, applies to registered
  formal dependent workers (empleados and obreros). Not indexed to inflation or productivity.
- Enforcement through SUNAFIL labor inspections, concentrated in Lima and large firms.
  High informality (≈72–75% of employed) and weak enforcement limit direct reach.
- The MW directly affects ≈10–15% of the workforce classified as formal dependent workers
  under INEI's comprehensive formality definition (`ocupinf == 2`).

### 2.2 The Three MW Events
| Event | Pre-MW | Post-MW | Change | Effective date | Analysis years |
|-------|--------|---------|--------|----------------|----------------|
| A | S/750 | S/850 | +13.3% | May 2016 | Pre: 2015, Post: 2017 |
| B | S/850 | S/930 | +9.4% | April 2018 | Pre: 2017, Post: 2019 |
| C | S/930 | S/1,025 | +10.2% | May 2022 | Pre: 2021, Post: 2023 |

Event B: employment pre-trends fail (F-test p=0.010). Event C: employment pre-trends also
fail (p=0.034). All regression-based event study results are in Appendix B. Bunching results
are retained in the main analysis (Cengiz estimator does not require parallel trends).

### 2.3 Department-Level Kaitz Variation
The Kaitz index (MW / weighted-median formal wage by department) ranges from 0.44 to 1.09
across departments and years, with Huancavelica consistently the most exposed (Kaitz 0.72–1.09
depending on year). For Event C specifically (2021 pre-period), the range is 0.45–0.72, median
0.60. This variation does not generate sufficient first-stage power for IV identification (F≤3
in all events — see Appendix C), and employment pre-trends fail for Events B and C (Appendix B).
The Kaitz is used here as a descriptive measure of MW bindingness, not as an IV.

---

## 3. DATA (≈700 words)

### 3.1 ENAHO Module 500 (Cross-Section)
- **Source**: ENAHO Module 500 (Empleo e Ingresos), INEI Peru.
- **Years**: 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023. No 2020 (COVID-19).
- **Sample**: Formal dependent workers, S/0–S/6,000/month.
  - Employed: `ocu500 == 1`
  - Dependent: `p507 ∈ {3, 4, 6}` (empleado, obrero, trabajador del hogar)
  - Formal: `ocupinf == 2` (social security registration or written contract)
- **Key variables**:
  - Wage: `p524a1` (monthly cash income); fallback `i524a1 / 12`
  - Hours: `p513t` (total weekly hours, primary job)
  - Weight: `fac500a`
  - Department: first 2 digits of `ubigeo` (24 departments)

### 3.2 Summary Statistics (Table 1)

**TABLE 1: Summary Statistics — Formal Dependent Workers, ENAHO 2015–2023**

| Year | MW | N | Med. wage (S/.) | % at MW | % below MW | Median Kaitz | Mean hrs/wk | Formality rate |
|------|----|---|:---------------:|:-------:|:----------:|:------------:|:-----------:|:--------------:|
| 2015 | 750 | 10,134 | 1,374 | 5.1% | 11.3% | 0.546 | 46.4 | 26.7% |
| 2016 | 850 | 11,667 | 1,500 | 5.3% | 13.9% | 0.567 | 46.2 | 27.9% |
| 2017 | 850 | 11,026 | 1,500 | 6.9% | 10.1% | 0.567 | 46.4 | 27.4% |
| 2018 | 930 | 11,804 | 1,580 | 6.2% | 14.6% | 0.589 | 46.3 | 27.6% |
| 2019 | 930 | 10,991 | 1,609 | 7.6% | 11.4% | 0.578 | 46.2 | 27.3% |
| 2021 | 930 | 9,123 | 1,500 | 10.3% | 11.2% | 0.620 | 46.8 | 23.2% |
| 2022 | 1,025 | 9,720 | 1,700 | 6.1% | 16.9% | 0.603 | 46.6 | 24.3% |
| 2023 | 1,025 | 9,996 | 1,800 | 10.1% | 12.3% | 0.569 | 46.5 | 26.1% |

*N = unweighted formal dependent workers with valid wage. % at MW = share earning MW±12.5 (within one half-bin).
% below MW = share earning below MW. Median Kaitz = MW / median wage, national. Formality rate = formal dep / all employed.
Hours from p513t (formal dep with positive hours). COVID-19 year 2021 shows depressed formality and elevated Kaitz.*

**Key patterns**:
- Median wage rises from S/1,374 (2015) to S/1,800 (2023), a 31% nominal increase, lagging
  the 37% MW increase — Kaitz rises nationally from 0.546 to 0.620 by 2021 before recovering.
- Share at MW nearly doubles: 5.1% (2015) → 10.1% (2023), indicating increasing MW bindingness.
- Formality rate dips sharply in 2021 (23.2%) due to COVID-19 — context for Event C identification.
- Hours stable at 46–47h/week throughout; no aggregate hours response to MW increases.

### 3.3 EPEN CIU Annual 2023 (Appendix D)
INEI code 873. Urban formal workers, single cross-section. No 2022 annual available.

---

## 4. EMPIRICAL STRATEGY (≈800 words)

### 4.1 Cengiz Bunching Estimator (Primary)
Bin width: S/25. Range: S/0–S/6,000 (survey-weighted by `fac500a`).

**Counterfactual**: weighted-average background shift from clean zone (wages > 2 × MW_new),
applied uniformly as a shift correction to the full pre-post delta.

**Missing mass** (net deficit below MW):
$$B^{miss} = \sum_{b \in [0.85 \times \text{MW}_{old},\; \text{MW}_{new})} \max(-\Delta_b^{adj},\; 0)$$

**Excess mass** (net surplus at MW):
$$B^{exc} = \sum_{b \in [\text{MW}_{new},\; \text{MW}_{new} + 250)} \max(\Delta_b^{adj},\; 0)$$

where $\Delta_b^{adj} = \Delta_b - \bar{\Delta}_{clean}$ subtracts the background trend.

**Bunching ratio**: $R = B^{exc} / B^{miss}$.

This estimator is valid for all three events. It does not require parallel trends.

### 4.2 Event Study DiD (Appendix B — Not Identified)
Employment pre-trends fail for Events B (p=0.010) and C (p=0.034). Event A has no
testable pre-period (2015 = base). All event study results reported in Appendix B as
descriptive only — not causally identified.

$$Y_{idt} = \alpha_d + \gamma_t
  + \sum_{s \neq \text{base}} \beta_s \cdot (K_{d,\text{pre}} \times \mathbf{1}[\text{year}=s])
  + \mathbf{X}_{it}'\delta + \varepsilon_{idt}$$

$K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.
SE clustered by department (24 clusters). WLS with survey weights.

### 4.3 Intensive Margin: Hours DiD (Supplementary)
2×2 DiD, weighted mean weekly hours (`p513t`):
- Treatment: [0.85 × MW_old, MW_new)
- Control: [1.5 × MW_new, 3.0 × MW_new]

### 4.4 IV / OWE Approach (Appendix C — Not Identified)
Following Dube & Lindner (2024): OWE = $\hat{\beta}_\text{emp} / \hat{\pi}_\text{wage}$.
F-statistics: A=2.9, B=0.6 (reversed sign), C=2.4. All below Stock-Yogo threshold of F>10.
See Appendix C.

---

## 5. BUNCHING RESULTS (≈900 words)

### 5.1 Main Bunching Estimates (Table 2)

**TABLE 2: Cengiz Revised Bunching — ENAHO CS**

| Event | Group | Missing (pp) | Excess (pp) | Ratio |
|-------|-------|:-----------:|:-----------:|:-----:|
| **A** (750→850) | Formal dep | 6.78 | 4.72 | 0.696 |
| | Informal dep | 4.45 | 5.42 | 1.218 |
| | All workers | 5.41 | 4.72 | 0.872 |
| **B** (850→930) | Formal dep | 8.03 | 6.66 | 0.829 |
| | Informal dep | 5.28 | 5.21 | 0.987 |
| | All workers | 6.44 | 5.81 | 0.901 |
| **C** (930→1025) | Formal dep | 13.02 | 10.80 | 0.830 |
| | Informal dep | 5.50 | 4.46 | 0.811 |
| | All workers | 8.29 | 6.84 | 0.824 |

*Missing (pp) = net share lost from [0.85×MW_old, MW_new). Excess (pp) = net share gained
at [MW_new, MW_new+250). Ratio = Excess/Missing.*

**Interpretation**:
- Formal dep ratios < 1 across all events (0.70–0.83): more workers leave the affected zone
  than reappear at the MW, indicating transitions out of the treated zone (informal employment,
  non-employment, or wages above the S/1,275 excess window).
- Missing mass increases across events: 6.78pp (A) → 8.03pp (B) → 13.02pp (C), reflecting
  higher MW bite as the RMV rises relative to the wage distribution.
- Informal workers: ratio > 1 in Event A (1.22), approximately 1 in Event B (0.99), below 1
  in Event C (0.81). No consistent spillover pattern.

**Placebo test**: Applying the same estimator to artificial thresholds (S/1,100→1,200 and
S/1,400→1,500) on the Event B population produces ratios of 0.114 and 0.013, compared to
0.829 at the actual S/930 threshold. Bunching is specific to the true MW.

### 5.2 Figure 2: Wage Distributions Pre vs. Post
Three panels (one per event), formal-dep workers, S/25 bins. Pre-period (blue bars),
post-period (orange bars), polynomial counterfactual (dashed gray). MW marked by vertical
line. Shaded region = affected zone [0.85×MW_old, MW_new).

*Generated by `scripts/mw_paper_figures.py`. File: `exports/figures/fig2_bunching_distributions.png`.*

### 5.3 Wage Compression (Table 3)
Workers in the compression zone [MW_new, 1.3×MW_new) saw faster wage growth than workers
in the high-wage zone [1.5×MW_new, 3×MW_new) in all three events.

**TABLE 3: Wage Compression DiD — ENAHO CS**

| Event | Compression DiD (log pts) | % compression |
|-------|:-------------------------:|:-------------:|
| A (750→850) | −0.036 | −3.1% |
| B (850→930) | −0.050 | −4.4% |
| C (930→1025) | −0.069 | −5.4% |

Mean DiD = −0.030 log points, t-test p < 0.001. Effect grows across events, consistent with
increasing MW bite (Table 1: Kaitz rises from 0.546 in 2015 to 0.620 in 2021).
The compression is not driven by Kaitz-intensity (Kaitz regression: β=+0.045, p=0.786),
indicating a uniform distributional shift rather than an enforcement-intensity channel.
This finding is descriptive, not causal.

**DRAFT NOTE — mechanism framing (critical for referee)**: The compression zone workers
[MW_new, 1.3×MW_new) were NOT directly hit by the MW increase — their wages were already
above the new floor. The DiD being negative means their wages grew *slower* than high
earners, not that the MW pushed their wages down. The mechanism is: floor workers at
[0.85×MW_old, MW_new) received mandatory raises to the new MW, compressing the gap between
them and just-above-floor workers who did not receive proportional raises. The compression
is driven by the bottom of the distribution being lifted, not by any suppression at the
top. The draft must state this explicitly: "Wages in the compression zone grew more slowly
*relative to high earners*, reflecting the mechanical narrowing of the gap between the
new floor and workers just above it — not a negative effect of the MW on above-floor wages."

---

## 6. HETEROGENEITY (≈700 words)

### 6.1 By Industry Sector (Table 4 — Panel A)
*Construction excluded: outlier ratios (A=3.23, C=8.31) driven by sparse cells (<400 obs).*
*Table 4 uses the same wage variable (`p524a1`) and dependent definition (`p507 ∈ {3,4,6}`)
as Table 2. "Total" row reproduces the Table 2 formal_dep ratios.*

**TABLE 4, Panel A: Bunching Ratios by Industry**

| Sector | Event A | Event B | Event C | Average |
|--------|:-------:|:-------:|:-------:|:-------:|
| **Total (= Table 2 formal dep)** | **0.696** | **0.829** | **0.830** | **0.785** |
| Agricultura / Minería† | 1.886 | 0.801 | 0.570 | 1.086 |
| Manufactura | 1.449 | 0.947 | 1.133 | 1.176 |
| Comercio | 0.793 | 1.242 | 0.723 | 0.919 |
| Transporte y alojamiento | 1.060 | 1.034 | 0.592 | 0.895 |
| Finanzas / Servicios prof. | 0.808 | 1.089 | 0.806 | 0.901 |
| Administración pública | 1.139 | 0.504 | 0.809 | 0.817 |
| Educación y salud | 0.644 | 0.724 | 0.997 | 0.788 |

†Agriculture/Mining Event A ratio (1.886) sensitive to outlier bins; excluding Event A,
average is 0.69 — consistent with other sectors.

### 6.2 By Firm Size (Table 4 — Panel B)

**TABLE 4, Panel B: Bunching Ratios by Firm Size**

| Firm size | Event A | Event B | Event C | Average |
|-----------|:-------:|:-------:|:-------:|:-------:|
| Micro (≤10 workers) | 0.902 | 0.925 | 0.707 | 0.845 |
| Pequeña (11–50) | 1.110 | 0.949 | 0.598 | 0.886 |
| Mediana/Grande (51+) | 0.700 | 0.814 | 0.919 | 0.811 |

### 6.3 By Age and Sex (Table 4 — Panel C)

**TABLE 4, Panel C: Bunching Ratios by Age and Sex**

| Group | Event A | Event B | Event C | Average |
|-------|:-------:|:-------:|:-------:|:-------:|
| Age 18–24 | 0.729 | 1.074 | 0.848 | 0.884 |
| Age 25–44 | 0.774 | 0.813 | 0.821 | 0.803 |
| Age 45–64 | 0.713 | 0.939 | 0.853 | 0.835 |
| Men | 0.759 | 0.801 | 0.797 | 0.786 |
| Women | 0.714 | 0.947 | 0.882 | 0.848 |

**Interpretation**:
- Ratios cluster near 0.80–0.85 for most cells. No single subgroup shows dramatically
  different behavior.
- Manufactura consistently above aggregate (0.95–1.45): stronger bunching response,
  likely reflecting higher SUNAFIL enforcement density in manufacturing.
- Mediana/Grande firms: ratio rises from 0.70 (A) to 0.92 (C) — large-firm compliance
  increases as MW bite grows.
- Young workers (18–24): ratio 1.074 in Event B, suggesting MW serves as an entry wage
  anchor for new hires in that cycle.
- Women have slightly higher ratios than men in Events B and C.

---

## 7. SUPPLEMENTARY: EVENT STUDY AND HOURS (≈600 words)

### 7.1 Event Study DiD — Employment (Table 5, Figure 1, Appendix B)
Pre-trend tests fail for all identified events (B: p=0.010; C: p=0.034; A: no pre-period).
Results reported as descriptive only.

**TABLE 5: Event Study Post-Period Employment Coefficients (Descriptive)**

| Event | Post β | SE | p | Pre-trend p | Note |
|-------|-------:|---:|---|:-----------:|------|
| A (2017) | +0.007 | 0.034 | 0.84 | — | No pre-period; untestable |
| B (2019) | +0.120 | 0.044 | 0.01 | 0.010 | Pre-trend FAIL → Appendix B |
| C (2023) | +0.224 | 0.078 | 0.004 | 0.034 | Pre-trend FAIL → Appendix B |

*β = coefficient on Kaitz_pre × post_t. SE clustered by department (24 clusters).*

**Event A**: no prior ENAHO waves in our dataset; only 2015 (base year) available.
Post β=+0.007 (p=0.84) — no employment effect, but identification relies on parallel trends
that cannot be verified.

**Events B and C**: pre-trend F-tests fail. The positive post coefficients likely reflect
pre-existing employment growth in higher-Kaitz departments rather than a MW effect. For
Event C specifically, higher-Kaitz departments (more low-wage workers) may have recovered
faster from the COVID-19 shock by 2023 regardless of the MW change.

**FIGURE 1: Event Study Coefficients — Employment (All Events)**
Three panels, year × Kaitz coefficients with 95% CI, base year = last pre-period.
*Caption will note pre-trend failures for Events B and C.*

### 7.2 Intensive Margin: Hours DiD (Table 6)

**TABLE 6: Weekly Hours DiD — ENAHO CS, p513t**

| Event | Treat pre | Treat post | Δ treat | Ctrl pre | Ctrl post | Δ ctrl | DiD |
|-------|:---------:|:----------:|:-------:|:--------:|:---------:|:------:|:---:|
| A (750→850) | 44.4h | 44.8h | +0.4h | 42.1h | 41.5h | −0.7h | +1.1h |
| B (850→930) | 44.8h | 45.1h | +0.3h | 41.2h | 42.0h | +0.8h | −0.5h |
| C (930→1025) | 44.5h | 46.4h | +1.9h | 42.1h | 43.6h | +1.5h | +0.4h |

*Treatment: [0.85×MW_old, MW_new). Control: [1.5×MW_new, 3.0×MW_new]. Weighted means.*

No DiD exceeds ±1.1h/week. No evidence of intensive-margin hours adjustment.

---

## 8. CONCLUSION (≈500 words)

1. MW increases in Peru produce a clear distributional response: formal dep bunching ratios
   of 0.70–0.83 across three events. Missing mass grows from 6.78pp (A) to 13.02pp (C) as
   the MW rises relative to wages. The placebo test (artificial thresholds produce ratios of
   0.01–0.11) confirms the result is MW-specific.

2. Wage compression of 3–5pp is present in all events (mean DiD = −3.0%, p < 0.001),
   consistent with the bunching pattern.

3. No consistent informal spillover: the informal ratio exceeds 1 only in Event A (1.22).

4. Employment effects are not identified by any available method. Event study pre-trends
   fail for all testable events (B: p=0.010; C: p=0.034). The IV/OWE approach fails due
   to weak first stages (F≤3) even with corrected weighted-median Kaitz. The data are
   consistent with small or zero employment effects, but this cannot be established causally.

5. No meaningful hours adjustment.

**What can be concluded**: Peru's MW increases produce a clear distributional shift without
detectably large employment losses. High informality, uneven enforcement, and a Kaitz index
below 1 for most departments likely limit employment effects. The data cannot distinguish
low labor demand elasticity from partial non-compliance.

---

## APPENDICES

### Appendix A: ENAHO Panel 978 — Event C (Table A1)
**Caveat**: ~86% attrition over 2021→2023. Results subject to survivorship bias.
Do not use for causal inference. Matched subsample: `facpanel2123 > 0`.

**TABLE A1: Panel Transition Results**

| Metric | Treatment (N=1,171) | Control (N=3,903) | DiD |
|--------|:-------------------:|:-----------------:|:---:|
| Employment retention | 15.8% | 15.0% | +0.8pp |
| Formal dep retention | 12.1% | 12.5% | −0.4pp |
| Δ log wage (stayers) | +0.264 | +0.143 | +0.121 |
| Δ hours (stayers) | −0.5h/wk | −2.1h/wk | +1.6h/wk |

Transition (treatment): 12.1% → formal dep, 2.1% → informal dep, 85.8% not re-interviewed
(**panel attrition — not unemployment**).

Treatment: formal dep 2021, wage ∈ [S/791, S/1,025). Control: wage ∈ [S/1,230, S/2,563].

### Appendix B: Event Study — Pre-Trend Violations (Table B1)
All three events have identification problems for the event study DiD.

**Event A**: No pre-period data (2015 = base year, no 2014 or earlier in our sample).
Parallel trends assumption untestable. Post β=+0.007 (p=0.84), no employment effect,
but cannot be claimed as causal.

**Event B**: Joint pre-trend F-test p=0.010 (employment). Pre-period coefficients (base 2017):
2015: β=−0.019 (ns); 2016: β=+0.051 (ns, p=0.052). Higher-Kaitz departments had
systematically higher employment before the 2018 MW change. DiD not identified.
IV first stage: π=−0.065 (SE=0.085, F=0.6). Reversed sign. IV ratio uninterpretable.

**Event C**: Joint pre-trend F-test p=0.034 (employment). Pre-period 2019: β=+0.168 (p=0.024).
Higher-Kaitz departments had higher employment in 2019 (pre-COVID), which by 2021 (our
base year) collapsed disproportionately. The 2019→2021 pre-trend renders the 2021→2023
comparison non-parallel.

All bunching results remain in main text — Cengiz estimator does not require parallel trends.

### Appendix C: IV / OWE — Not Identified (Table C1)
Instrument: $K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.

**TABLE C1: First Stage, Reduced Form, and OWE**

| Event | π (FS) | SE | F | β (RF emp) | SE | OWE | SE |
|-------|-------:|---:|:-:|----------:|---:|----:|---:|
| A | +0.124 | 0.073 | 2.9 | +0.007 | 0.034 | +0.054 | 0.276 |
| B | −0.065 | 0.085 | 0.6 | +0.120 | 0.044 | — | — |
| C | +0.142 | 0.092 | 2.4 | +0.224 | 0.078 | +1.576 | 1.155 |
| Pooled A+B+C | — | — | — | — | — | +0.114 | 0.267 |
| Pooled A+B | — | — | — | — | — | +0.031 | 0.275 |

Event B OWE not reported (reversed FS sign). F < 10 in all events (Stock-Yogo threshold).
Peru has a single national MW across 24 departments; Kaitz variation does not generate
sufficient first-stage power even with corrected weighted-median Kaitz construction.
Pooled A+B+C OWE = +0.114 (SE=0.267, p=0.67) — not significant, consistent with zero.

### Appendix D: EPEN CIU Annual 2023 — Lee-Saez Bunching (Table D1)
Single-period Lee-Saez estimate (INEI code 873, N=53,316 formal urban workers).
**Note**: Lee-Saez single-period ratios are not directly comparable to the Cengiz pre-post
ratios in Table 2. The Lee-Saez method measures the excess of the current distribution over
a smooth counterfactual; the Cengiz method measures the pre-to-post change in distribution
shares. The high Lee-Saez ratios (1.6×) simply confirm excess mass at the MW post-period,
consistent with Table 2.

**TABLE D1: EPEN CIU Bunching (Post 2023, Lee-Saez)**

| Sample | N | Excess factor | Ratio |
|--------|--:|:-------------:|:-----:|
| All cities | 53,316 | 1.34× | 1.622 |
| Lima upper stratum | — | — | 1.602 |
| Other cities | — | — | 1.688 |

---

## FILE INDEX

| File | Contents |
|------|----------|
| `exports/data/mw_complete_margins.json` | Table 2, Table 3, lighthouse |
| `exports/data/mw_heterogeneity.json` | Table 4 all panels |
| `exports/data/mw_iv_owe.json` | Table C1, Figure 1, event study |
| `exports/data/mw_hours_epen_dep.json` | Table 6 |
| `exports/data/mw_panel_decomposition.json` | Table A1 |
| `exports/data/mw_epen_ciu_annual_bunching.json` | Table D1 |
| `exports/data/mw_data_audit_complete.json` | Table 1 (sample_counts, wage_dist, hours) |
| `exports/data/mw_sanity_checks.json` | Placebo test (Section 5.1) |
| `exports/figures/fig2_bunching_distributions.png` | Figure 2 |
| `scripts/mw_paper_figures.py` | Figure 2 generation |
| `scripts/mw_heterogeneity_bunching.py` | Table 4 (p524a1, p507 {3,4,6}, fac500a) |

---

*Revised v5: 2026-03-17. Kaitz fix (weighted median); Event C pre-trend updated to FAIL (p=0.034); all F-stats and OWE values updated. Data: ENAHO 2015–2023 (Módulo 500), EPEN CIU Annual 2023, ENAHO Panel 978.*
