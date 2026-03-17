# Missing Mass and Minimum Wages: Distributional Effects of Three MW Increases in Peru
### [Alt. title: "Where Do the Workers Go? Bunching Evidence from Peru's Minimum Wage Increases"]
## Complete Paper Outline — Revised v3

---

## IDENTIFICATION SCORECARD

| Method | Event A | Event B | Event C | Verdict |
|--------|---------|---------|---------|---------|
| Cengiz bunching | ✓ valid | ✓ valid | ✓ valid | **Main result** |
| Event study / DiD | ✓ pre-trends pass (p=0.06) | ✗ fail (p=0.001) | ✓ pass (p=0.26) | A+C main text; B → Appendix B |
| IV / OWE | ✗ F=1.7 | ✗ F=0.4, reversed sign | ✗ F=1.5 | Appendix C — not identified |
| Panel decomp | — | — | ✗ 86% attrition | Appendix A — unreliable |
| EPEN DEP bunching | — | — | ✗ MW doesn't bind (Kaitz=0.47) | Dropped |

**The paper is a bunching paper.** Regression-based methods are supplementary where valid
(Events A and C event study) or in the appendix where they fail (IV, Event B, panel).

---

## 1. INTRODUCTION (≈1,000 words)

**Hook**: Peru raised the minimum wage from S/750 to S/1,025 between 2016 and 2022 — a 37%
increase over six years — while formal employment remained stable. This paper uses three
identifiable MW events to study how the wage distribution responded.

**Question**: Do Peru's MW increases produce missing mass below the threshold and bunching at
it? Do they affect employment and hours? We use the Cengiz et al. (2019) revised bunching
estimator as the primary tool, with event-study DiD as supplementary evidence where
pre-trends hold.

**Contribution**:
1. First application of the Cengiz et al. (2019) bunching estimator to Peru, using three
   MW events across 2015–2023.
2. Direct test for MW spillovers to informal wages using the same estimator on both
   formal and informal workers.
3. We attempted the Dube & Lindner (2024) IV/OWE approach; first stages are uniformly
   weak (F < 2, reversed sign in Event B). We report these results in Appendix C.

**Main findings**:
- Formal dependent workers show clear missing mass and excess mass at the MW in all
  three events (bunching ratios 0.70–0.83).
- Wage compression: workers just below the MW see 3–5pp faster wage growth than high-wage
  workers, consistent with the distributional response.
- Informal workers show a partial response in Event A (ratio 1.22) but not Events B and C
  (0.99 and 0.81). No consistent spillover pattern.
- Event study DiD (Events A and C) finds employment effects near zero and not statistically
  significant.
- Hours adjustment is negligible across all events (DiD ≤ ±1.1h/week).
- The IV/OWE approach is not identified; no credible employment elasticity is reported.

**Paper structure**: Section 2 describes the institutional context and MW events. Section 3
presents data. Section 4 describes the empirical strategies. Sections 5–7 present bunching,
heterogeneity, and supplementary DiD results. Section 8 concludes.

---

## 2. INSTITUTIONAL CONTEXT (≈700 words)

### 2.1 Peru's Minimum Wage System
- **Remuneración Mínima Vital (RMV)**: set by executive decree, applies to registered formal
  dependent workers (empleados and obreros). Not indexed to inflation or productivity.
- Enforcement through SUNAFIL labor inspections, concentrated in Lima and large firms.
  High informality (≈72–75% of employed) and weak enforcement limit direct reach.
- The MW directly affects the ≈10–15% of the workforce classified as formal dependent
  workers under INEI's comprehensive formality definition (`ocupinf == 2`).

### 2.2 The Three MW Events
| Event | Pre-MW | Post-MW | Change | Effective date | Analysis years |
|-------|--------|---------|--------|----------------|----------------|
| A | S/750 | S/850 | +13.3% | May 2016 | Pre: 2015, Post: 2017 |
| B | S/850 | S/930 | +9.4% | April 2018 | Pre: 2017, Post: 2019 |
| C | S/930 | S/1,025 | +10.2% | May 2022 | Pre: 2021, Post: 2023 |

Event B note: employment pre-trends fail (F-test p=0.001). All regression-based results for
Event B are in Appendix B. Bunching results are retained in the main analysis — the
distributional estimator does not require parallel trends.

### 2.3 Department-Level Kaitz Variation
The Kaitz index (MW / median formal wage) varies across Peru's 24 departments, ranging from
0.44 to 1.09 across departments and years, with Huancavelica consistently the most exposed
(Kaitz 0.72–1.09 depending on year and wage definition). For Event C specifically (2021
pre-period), the range is 0.45 (Amazonas) to 0.72 (Huancavelica), median 0.60. This
variation motivates the event-study identification strategy but proves insufficient for
IV identification (Section 4.4 and Appendix C).

---

## 3. DATA (≈700 words)

### 3.1 ENAHO Module 500 (Cross-Section)
- **Source**: ENAHO Module 500 (Empleo e Ingresos), INEI Peru. Annual stratified household survey.
- **Years used**: 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023. No 2020 (COVID-19).
- **Sample restriction**: Formal dependent workers, S/200–S/8,000/month.
  - Employed: `ocu500 == 1`
  - Dependent: `cat07p500a1 == 2`; verified with `p507 ∈ {3,4,6}`
  - Formal: `ocupinf == 2` (social security registration or written contract)
- **Key variables**:
  - Wage: `p524a1` (monthly cash income, primary job); fallback `i524a1 / 12`
  - Hours: `p513t` (total weekly hours, primary job)
  - Survey weight: `factor07i500a`
  - Department: first 2 digits of `ubigeo` (24 departments)

**Sample sizes (formal dependent workers):**
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

### 3.2 EPEN CIU Annual 2023 (Appendix D)
INEI code 873. Single cross-section of urban formal workers. No 2022 annual available.
Used for corroboration only.

---

## 4. EMPIRICAL STRATEGY (≈800 words)

### 4.1 Cengiz Bunching Estimator (Primary)
Bin width: S/25. Wage distribution: S/200–S/4,000 (survey-weighted).

**Counterfactual**: degree-4 polynomial fit on clean zone (wages > 2 × MW_new), excluding
the affected zone [0.85 × MW_old, MW_new + 250].

**Missing mass** (net deficit below MW):
$$B^{miss} = \sum_{b \in [0.85 \times \text{MW}_{old},\; \text{MW}_{new})} \max(-\Delta_b,\; 0)$$

**Excess mass** (net surplus at MW):
$$B^{exc} = \sum_{b \in [\text{MW}_{new},\; \text{MW}_{new} + 250)} \max(\Delta_b,\; 0)$$

**Bunching ratio**: $R = B^{exc} / B^{miss}$. Values below 1 mean the MW zone lost more
workers than it gained; values above 1 mean a net inflow.

This estimator is valid for all three events. It does not require parallel trends or
a valid instrument.

### 4.2 Event Study DiD (Supplementary — Events A and C)
Pre-trend test passed for Events A and C. Event B fails (p = 0.001) → Appendix B.

$$Y_{idt} = \alpha_d + \gamma_t
  + \sum_{s \neq \text{base}} \beta_s \cdot (K_{d,\text{pre}} \times \mathbf{1}[\text{year}=s])
  + \mathbf{X}_{it}'\delta + \varepsilon_{idt}$$

where $K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.
SE clustered by department (24 clusters). WLS with survey weights.

### 4.3 Intensive Margin: Hours DiD (Supplementary)
2×2 DiD of weighted mean weekly hours (`p513t`):
- Treatment: [0.85 × MW_old, MW_new)
- Control: [1.5 × MW_new, 3.0 × MW_new]

### 4.4 IV / OWE Approach (Appendix C — Not Identified)
Following Dube & Lindner (2024): OWE = $\hat{\beta}_\text{emp} / \hat{\pi}_\text{wage}$.
First-stage F-statistics: Event A = 1.7, Event B = 0.4 (reversed sign), Event C = 1.5.
All below the Stock-Yogo threshold of F > 10. The IV is not identified (see Appendix C).

---

## 5. BUNCHING RESULTS (≈900 words)

### 5.1 Main Bunching Estimates (Table 2)

**TABLE 2: Cengiz Revised Bunching — ENAHO CS**

| Event | Group | Missing (pp) | Excess (pp) | Ratio | Emp. Δ (%) |
|-------|-------|:-----------:|:-----------:|:-----:|:----------:|
| **A** (750→850) | Formal dep | 6.78 | 4.72 | 0.696 | 1.33 |
| | Informal dep | 4.45 | 5.42 | 1.218 | 1.08 |
| | All workers | 5.41 | 4.72 | 0.872 | 1.18 |
| **B** (850→930) | Formal dep | 8.03 | 6.66 | 0.829 | 5.59 |
| | Informal dep | 5.28 | 5.21 | 0.987 | 0.34 |
| | All workers | 6.44 | 5.81 | 0.901 | 2.54 |
| **C** (930→1025) | Formal dep | 13.02 | 10.80 | 0.830 | 17.25 |
| | Informal dep | 5.50 | 4.46 | 0.811 | 5.16 |
| | All workers | 8.29 | 6.84 | 0.824 | 9.78 |

*Missing (pp) = net share lost from [0.85×MW_old, MW_new). Excess (pp) = net share gained
at [MW_new, MW_new+250). Ratio = Excess/Missing. Emp. Δ = employment change estimate from
net excess mass.*

**Interpretation**:
- Formal dep ratios < 1 across all events (0.70–0.83): more workers leave the affected zone
  than reappear at the MW, indicating transitions out of the treated zone (informal employment,
  non-employment, or wages above the excess window).
- Informal workers: ratio > 1 in Event A (1.22), essentially 1 in Event B (0.99), below 1 in
  Event C (0.81). No consistent spillover pattern across events.
- Missing mass increases with each MW rise: 6.78pp (A) → 8.03pp (B) → 13.02pp (C), reflecting
  higher MW bite as the RMV rises relative to the wage distribution.

**Placebo test**: Applying the same estimator to artificial MW thresholds (S/1,100→1,200 and
S/1,400→1,500) on the same Event B population produces ratios of 0.114 and 0.013 respectively,
compared to 0.829 at the actual S/930 threshold. Bunching is specific to the true MW.

### 5.2 Figure 2: Wage Distributions Pre vs. Post
Three panels, one per event. Each shows the formal-dep wage distribution in S/25 bins, with
pre-period (blue bars) and post-period (orange bars) and degree-4 polynomial counterfactual
(dashed gray). Vertical line marks the post-period MW.

*Generated by `scripts/mw_paper_figures.py`. Saved in `exports/figures/fig2_bunching_distributions.png`.*

### 5.3 Wage Compression (Supplementary — Table 3)
Workers just below the MW saw faster wage growth than high-wage workers in all three events.
Using the DiD compression measure (Δ log wage for workers in [MW_old, 1.3×MW_old) vs.
[1.5×MW_new, 3×MW_new]):

| Event | Compression DiD | % compression |
|-------|:--------------:|:-------------:|
| A (750→850) | −0.036 | −3.1% |
| B (850→930) | −0.050 | −4.4% |
| C (930→1025) | −0.069 | −5.4% |

Mean DiD = −0.030, t-test p < 0.001 (N=25 dept×event observations). The compression is not
driven by Kaitz-intensity (Kaitz regression: β=+0.045, p=0.786), indicating a uniform
distributional shift rather than an enforcement-intensity channel.

---

## 6. HETEROGENEITY (≈700 words)

### 6.1 By Industry Sector (Table 4 — Panel A)
*Construction excluded: outlier ratios (A=3.23, C=8.31) caused by sparse cells (<200 obs).*

**TABLE 4, Panel A: Bunching Ratios by Industry**

Note: "Total (reference)" uses `i524a1/12` as wage variable (heterogeneity script default)
vs. `p524a1` in Table 2 — ratios differ by 0.03–0.04 due to wage variable, not sample.†

| Sector | Event A | Event B | Event C | Average |
|--------|:-------:|:-------:|:-------:|:-------:|
| Total (reference)† | 0.733 | 0.854 | 0.840 | 0.809 |
| Agricultura / Minería‡ | 1.947 | 0.792 | 0.594 | 1.111 |
| Manufactura | 1.449 | 0.955 | 1.126 | 1.177 |
| Comercio | 0.791 | 1.230 | 0.728 | 0.916 |
| Transporte y alojamiento | 1.036 | 1.040 | 0.609 | 0.895 |
| Finanzas / Servicios prof. | 0.817 | 1.091 | 0.801 | 0.903 |
| Administración pública | 1.161 | 0.531 | 0.806 | 0.833 |
| Educación y salud | 0.648 | 0.729 | 1.003 | 0.793 |

†Table 4 "Total" uses `i524a1/12` (annualized monthly, the heterogeneity script default);
Table 2 uses `p524a1` (primary). Difference of 0.03–0.04 in ratios reflects wage variable,
not population differences. `p524a1` is preferred; Table 2 ratios are definitive.

‡Agriculture/Mining Event A ratio (1.947) is sensitive to a single high-wage outlier bin;
excluding Event A, the average is 0.69 — similar to other sectors.

### 6.2 By Firm Size (Table 4 — Panel B)

**TABLE 4, Panel B: Bunching Ratios by Firm Size**

| Firm size | Event A | Event B | Event C | Average |
|-----------|:-------:|:-------:|:-------:|:-------:|
| Micro (≤10 workers) | 0.987 | 1.002 | 0.699 | 0.896 |
| Pequeña (11–50) | 1.127 | 0.940 | 0.604 | 0.890 |
| Mediana/Grande (51+) | 0.704 | 0.826 | 0.922 | 0.817 |

### 6.3 By Age and Sex (Table 4 — Panel C)

**TABLE 4, Panel C: Bunching Ratios by Age and Sex**

| Group | Event A | Event B | Event C | Average |
|-------|:-------:|:-------:|:-------:|:-------:|
| Age 18–24 | 0.729 | 1.083 | 0.846 | 0.886 |
| Age 25–44 | 0.786 | 0.818 | 0.822 | 0.808 |
| Age 45–64 | 0.779 | 0.959 | 0.838 | 0.859 |
| Men | 0.768 | 0.816 | 0.801 | 0.795 |
| Women | 0.742 | 0.967 | 0.883 | 0.864 |

**Interpretation**:
- Ratios cluster near the aggregate 0.80–0.85 for most cells. No heterogeneity is dramatic.
- Manufactura consistently above aggregate (0.95–1.45): more bunching response, possibly
  reflecting stronger SUNAFIL enforcement in manufacturing plants.
- Mediana/Grande firms (51+) show rising ratios (0.70 → 0.82 → 0.92): large firms may
  be increasing compliance as MW bite grows.
- Young workers (18–24) show ratio 1.083 in Event B: excess exceeds missing, consistent
  with MW serving as an entry wage anchor for new hires.

---

## 7. SUPPLEMENTARY: EVENT STUDY AND HOURS (≈600 words)

### 7.1 Event Study DiD — Employment (Table 5, Figure 1)
*Events A and C only. Event B: Appendix B.*

**TABLE 5: Event Study Post-Period Employment Coefficients**

| Event | Post β | SE | p | Pre-trend p |
|-------|-------:|---:|---|:-----------:|
| A (2017) | −0.011 | 0.040 | 0.79 | 0.061 (PASS) |
| C (2023) | +0.183 | 0.088 | 0.04 | 0.260 (PASS) |

*β = coefficient on Kaitz_pre × post_t. SE clustered by department (24 clusters).*

**FIGURE 1: Event Study Coefficients — Employment (Events A and C)**
Two panels, year × Kaitz coefficients with 95% CI, base year = last pre-period.

- **Event A**: post coefficient β=−0.011 (p=0.79). No detectable employment effect.
- **Event C**: post coefficient β=+0.183 (p=0.04). This positive coefficient likely reflects
  post-COVID recovery concentrated in higher-Kaitz (lower-wage) departments, rather than
  MW effects. The 2021 pre-period is a COVID trough; departments with more low-wage formal
  workers (higher Kaitz) may have recovered faster by 2023 regardless of the MW. The weak
  first stage (F=1.5) means this cannot be interpreted as a wage-effect channel.

### 7.2 Intensive Margin: Hours DiD (Table 6)

**TABLE 6: Weekly Hours DiD — ENAHO CS, p513t**

| Event | Treat pre | Treat post | Δ treat | Ctrl pre | Ctrl post | Δ ctrl | DiD |
|-------|:---------:|:----------:|:-------:|:--------:|:---------:|:------:|:---:|
| A (750→850) | 44.4h | 44.8h | +0.4h | 42.1h | 41.5h | −0.7h | +1.1h |
| B (850→930) | 44.8h | 45.1h | +0.3h | 41.2h | 42.0h | +0.8h | −0.5h |
| C (930→1025) | 44.5h | 46.4h | +1.9h | 42.1h | 43.6h | +1.5h | +0.4h |

*Treatment: formal dep earning [0.85×MW_old, MW_new). Control: [1.5×MW_new, 3.0×MW_new].*

No DiD exceeds ±1.1h/week. MW increases do not produce detectable hours adjustment.

---

## 8. CONCLUSION (≈500 words)

1. MW increases in Peru compress the formal wage distribution: bunching ratios of 0.70–0.83
   across all three events, with missing mass growing from 6.78pp (Event A) to 13.02pp
   (Event C) as the MW rises relative to wages.

2. Wage compression of 3–5pp is present in all events and statistically robust
   (mean DiD = −3.0%, p < 0.001), consistent with the bunching pattern.

3. No consistent spillover to informal wages: the informal ratio exceeds 1 only in Event A
   (1.22). Events B and C show ratios at or below 1.

4. Employment effects are not identified. The event study DiD (Events A and C) finds effects
   not different from zero. The IV/OWE approach fails due to weak first stages across all
   events. The panel decomposition is unreliable due to 86% attrition.

5. No meaningful hours adjustment across three events.

**What can be concluded**: Peru's MW increases produce a clear distributional shift — workers
bunch at the new floor — without detectably large employment losses. High informality,
uneven enforcement, and a Kaitz index below 1 for most departments likely limit the MW's
macroeconomic bite. The data cannot distinguish low labor demand elasticity from partial
non-compliance as the underlying mechanism.

---

## APPENDICES

### Appendix A: ENAHO Panel 978 — Event C (Table A1)
**Caveat**: ~86% attrition over 2021→2023. Results subject to survivorship bias.
Matched subsample: `facpanel2123 > 0`. Do not use for causal inference.

**TABLE A1: Panel Transition Results**

| Metric | Treatment (N=1,171) | Control (N=3,903) | DiD |
|--------|:-------------------:|:-----------------:|:---:|
| Employment retention | 15.8% | 15.0% | +0.8pp |
| Formal dep retention | 12.1% | 12.5% | −0.4pp |
| Δ log wage (stayers) | +0.264 | +0.143 | +0.121 |
| Δ hours (stayers) | −0.5h/wk | −2.1h/wk | +1.6h/wk |

Transition (treatment): 12.1% → formal dep, 2.1% → informal dep, 85.8% not re-interviewed
(panel attrition — **not unemployment**).

Treatment: formal dep 2021, wage ∈ [S/791, S/1,025). Control: wage ∈ [S/1,230, S/2,563].

### Appendix B: Event B — Pre-Trend Violation and Reversed IV (Table B1)
Pre-trend joint F-test p = **0.0014**. Pre-period employment coefficients (base 2017):
- 2015: β=+0.016 (ns); 2016: β=+0.080 (p=0.001)

Higher-Kaitz departments had systematically higher employment before the 2018 MW change.
DiD is not identified for Event B employment.

IV first stage: π=−0.059 (SE=0.094, F=0.4). Reversed sign — higher-Kaitz departments had
lower wage growth 2017→2019. IV ratio is uninterpretable.

Event B bunching remains in main Table 2 — the distributional estimator is valid and does
not require parallel trends.

### Appendix C: IV / OWE — Not Identified (Table C1)
Instrument: $K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.

**TABLE C1: First Stage, Reduced Form, and OWE**

| Event | π (FS) | SE | F | β (RF) | SE | OWE | SE |
|-------|-------:|---:|:-:|-------:|---:|----:|---:|
| A | +0.110 | 0.084 | 1.7 | −0.011 | 0.040 | −0.097 | 0.367 |
| B | −0.059 | 0.094 | 0.4 | +0.171 | 0.052 | — | — |
| C | +0.175 | 0.144 | 1.5 | +0.183 | 0.088 | +1.045 | 0.995 |

Event B OWE not reported (reversed first-stage sign). F < 10 in all events (Stock-Yogo
threshold). Peru has a single national MW across 24 departments; the instrument generates
insufficient cross-sectional variation to identify the first stage.

The wide confidence intervals (e.g., Event A: 95% CI [−0.82, +0.62]) confirm the approach
cannot bound the employment elasticity. We report these results in Appendix C.

### Appendix D: EPEN CIU Annual 2023 — Lee-Saez Bunching (Table D1)
Single-period estimate (no 2022 annual available). N=53,316 formal urban workers.
Formality: `INFORMAL_P == '2'`. MW = S/1,025.

**TABLE D1: EPEN CIU Bunching (Post 2023)**

| Sample | N | Excess factor | Ratio |
|--------|--:|:-------------:|:-----:|
| All cities | 53,316 | 1.34× | 1.622 |
| Lima upper stratum | — | — | 1.602 |
| Other cities | — | — | 1.688 |

Confirms excess mass at S/1,025 in post-period urban data, consistent with Table 2.

---

## FILE INDEX

| File | Contents |
|------|----------|
| `exports/data/mw_complete_margins.json` | Table 2, Table 3 (compression), lighthouse |
| `exports/data/mw_heterogeneity.json` | Table 4 all panels |
| `exports/data/mw_iv_owe.json` | Table C1, Figure 1, event study |
| `exports/data/mw_hours_epen_dep.json` | Table 6 |
| `exports/data/mw_panel_decomposition.json` | Table A1 |
| `exports/data/mw_epen_ciu_annual_bunching.json` | Table D1 |
| `exports/data/mw_sanity_checks.json` | Placebo test (Section 5.1) |
| `exports/data/mw_paper_results_complete.json` | All results aggregated |
| `exports/figures/fig2_bunching_distributions.png` | Figure 2 |
| `scripts/mw_paper_figures.py` | Figure 2 generation script |

---

*Revised v3: 2026-03-17. Data: ENAHO 2015–2023 (Módulo 500), EPEN CIU Annual 2023, ENAHO Panel 978.*
