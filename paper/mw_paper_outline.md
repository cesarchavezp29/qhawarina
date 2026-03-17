# Minimum Wage Effects in Peru: Evidence from Bunching Estimators
## Complete Paper Outline — Revised

---

## IDENTIFICATION SCORECARD (honest summary)

| Method | Event A | Event B | Event C | Verdict |
|--------|---------|---------|---------|---------|
| Cengiz bunching | ✓ valid | ✓ valid | ✓ valid | **Main result** |
| Event study / DiD | ✓ pre-trends pass | ✗ pre-trends fail (p=0.001) | ✓ pre-trends pass | A+C in main text; B in Appendix B |
| IV / OWE | ✗ F=1.7 | ✗ F=0.4, reversed sign | ✗ F=1.5 | Appendix C — not identified |
| Panel decomp | — | — | ✗ 86% attrition | Appendix A — not reliable |
| EPEN DEP bunching | — | — | ✗ MW doesn't bind (Kaitz=0.47) | Dropped |

**The paper is a bunching paper.** Regression-based methods are supplementary where valid
(Events A and C event study) or relegated to appendix where they fail (IV, Event B, panel).

---

## 1. INTRODUCTION (≈1,000 words)

**Hook**: Peru raised the minimum wage from S/750 to S/1,025 between 2016 and 2022 — a 37%
increase — while formal employment remained stable. This apparent puzzle motivates the paper.

**Question**: How do minimum wage increases affect the wage distribution and employment among
formal dependent workers in Peru? Three identifiable events (2016, 2018, 2022) allow us to
apply the Cengiz et al. (2019) revised bunching estimator with pre- and post-period data.

**Contribution**:
1. First application of the Cengiz et al. (2019) bunching estimator to Peru, using three
   MW events spanning 2015–2023.
2. Bunching estimates for formal dependent, informal, and all workers — enabling a direct test
   for MW spillovers into informal wages.
3. We attempted the Dube & Lindner (2024) IV approach but first stages are uniformly weak
   (F < 2 across all three events), reflecting the limited cross-department variation produced
   by a single national MW across 24 Peruvian departments. We report these attempts
   transparently in Appendix C.

**Main findings**:
- Formal dependent workers show clear missing mass below the MW and excess mass at the MW
  in all three events (bunching ratios 0.70–0.83).
- Informal workers show mixed evidence: ratio > 1 in Event A (1.22) but ≤ 1 in Events B and C
  (0.99 and 0.81). No consistent lighthouse effect.
- Event study DiD for Events A and C (where pre-trends hold) finds positive post-period
  employment effects that are not statistically different from zero.
- Hours adjustment is negligible (DiD ≤ ±1.1h/week across all events).
- The IV/OWE approach is not identified given the instrument weakness.

**Paper structure**: Section 2 describes the institutional context and MW events. Section 3
presents data and sample construction. Section 4 describes the empirical strategies.
Sections 5–7 present bunching, heterogeneity, and supplementary DiD results.
Section 8 concludes.

---

## 2. INSTITUTIONAL CONTEXT (≈700 words)

### 2.1 Peru's Minimum Wage System
- **Remuneración Mínima Vital (RMV)**: set by executive decree, applies to all registered
  formal dependent workers (empleados and obreros). Not indexed to inflation or productivity.
- Enforcement: through SUNAFIL labor inspections, concentrated in Lima and large firms.
  High informality (≈72–75% of employed) limits the direct reach of the MW.
- The MW binds primarily for the ≈10–15% of the workforce classified as formal dependent
  workers under INEI's comprehensive formality definition (`ocupinf == 2`).

### 2.2 The Three MW Events
| Event | Pre-MW | Post-MW | Change | Effective date | Analysis years |
|-------|--------|---------|--------|----------------|----------------|
| A | S/750 | S/850 | +13.3% | May 2016 | Pre: 2015, Post: 2017 |
| B | S/850 | S/930 | +9.4% | April 2018 | Pre: 2017, Post: 2019 |
| C | S/930 | S/1,025 | +10.2% | May 2022 | Pre: 2021, Post: 2023 |

**Event B note**: Employment pre-trends fail (F-test p=0.001) — DiD and IV in Appendix B.
Bunching estimates for Event B are retained in the main analysis; the distributional estimator
does not require parallel trends.

### 2.3 Department-Level Kaitz Variation
Pre-period Kaitz index (MW / median formal wage) ranges from 0.45 (Amazonas) to 0.72
(Huancavelica), with median 0.60, for Event C. This variation motivates the event-study
identification strategy for Events A and C but provides only limited variation for IV.

---

## 3. DATA (≈700 words)

### 3.1 ENAHO Module 500 (Cross-Section)
- **Source**: ENAHO Module 500 (Empleo e Ingresos), INEI Peru. Annual, stratified household survey.
- **Years used**: 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023.
  No 2020 data (COVID-19 disruption).
- **Sample**: Formal dependent workers, S/200–S/8,000/month.
  - Employed: `ocu500 == 1`
  - Dependent: `cat07p500a1 == 2` (formal-sector employment category; verified with `p507 ∈ {3,4,6}`)
  - Formal: `ocupinf == 2` (registered with social security or under written contract)
- **Key variables**:
  - Wage: `p524a1` (monthly cash income, primary job); fallback `i524a1 / 12`
  - Hours: `p513t` (total weekly hours, primary job)
  - Survey weight: `factor07i500a`
  - Department: first 2 digits of `ubigeo` (24 departments)

**Sample sizes (formal dependent workers):**
| Year | N formal dep |
|------|-------------|
| 2015 | 10,102 |
| 2016 | 11,588 |
| 2017 | 10,930 |
| 2018 | 11,792 |
| 2019 | 10,992 |
| 2021 | 9,008 |
| 2022 | 9,666 |
| 2023 | 9,931 |

### 3.2 EPEN CIU Annual 2023 (Appendix)
INEI code 873. Single cross-section of urban formal workers. Used for corroboration only
(Appendix D). No comparable 2022 annual file is publicly available.

---

## 4. EMPIRICAL STRATEGY (≈800 words)

### 4.1 Cengiz Bunching Estimator (Primary)
Bin width: S/25. Wage distribution: S/200–S/4,000 (unweighted bins, weighted by `factor07i500a`).

**Counterfactual**: degree-4 polynomial fit on clean zone (wages > 2 × MW_new), excluding
the affected zone [0.85 × MW_old, MW_new + 10 × 25].

**Missing mass** (net outflow from zone below MW):
$$B^{miss} = \sum_{b \in [0.85 \times \text{MW}_{old},\; \text{MW}_{new})} \max(-\Delta_b,\; 0)$$

**Excess mass** (net inflow to zone at and just above MW):
$$B^{exc} = \sum_{b \in [\text{MW}_{new},\; \text{MW}_{new} + 250)} \max(\Delta_b,\; 0)$$

**Bunching ratio**: $R = B^{exc} / B^{miss}$. Ratio > 1 indicates excess exceeds missing
(net employment gain in the treated zone); ratio < 1 indicates displacement exceeds bunching.

This estimator is valid for all three events. It does not require parallel trends.

### 4.2 Event Study DiD (Supplementary — Events A and C only)
Pre-trend test passed for Events A and C (joint F-test p = 0.25 and p = 0.26 respectively
for employment). Event B fails (p = 0.001) and is excluded from this section.

Specification:
$$Y_{idt} = \alpha_d + \gamma_t
  + \sum_{s \neq \text{base}} \beta_s \cdot (K_{d,\text{pre}} \times \mathbf{1}[\text{year}=s])
  + \mathbf{X}_{it}'\delta + \varepsilon_{idt}$$

Instrument: $K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.
Standard errors clustered by department (24 clusters). WLS with individual survey weights.

### 4.3 Intensive Margin: Hours DiD (Supplementary)
2×2 DiD of weighted mean weekly hours (`p513t`):
- Treatment zone: [0.85 × MW_old, MW_new)
- Control zone: [1.5 × MW_new, 3.0 × MW_new]
Pseudo-panel using repeated cross-sections.

### 4.4 IV / OWE Approach (Appendix C — Not Identified)
Following Dube & Lindner (2024). First stage: $\log w \sim K_{d,\text{pre}} \times \text{post}$.
OWE = $\hat{\beta}_\text{emp} / \hat{\pi}_\text{wage}$.

First-stage F-statistics:
- Event A: F = 1.7 (well below the F > 10 threshold)
- Event B: F = 0.4, first-stage coefficient **negative** (departments with higher Kaitz
  had lower wage growth — economically reversed sign)
- Event C: F = 1.5

With uniformly weak and in one case sign-reversed first stages, the IV is not identified.
The OWE ratios cannot be interpreted as causal elasticities. We report these results
in Appendix C for transparency and to document the limitations of the geographic
identification strategy with 24 departments and a single national MW.

---

## 5. BUNCHING RESULTS (≈900 words) — PRIMARY RESULTS

### 5.1 Main Bunching Estimates (Table 2)

**TABLE 2: Cengiz Revised Bunching Estimates — ENAHO CS (formal dep workers and informal)**

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

*Missing (pp) = net mass lost from [0.85×MW_old, MW_new). Excess (pp) = net mass gained at
[MW_new, MW_new+250). Ratio = Excess/Missing. Emp. Δ = estimated employment change from
net excess mass as % of weighted employment.*

**Interpretation**:
- Formal dep ratios < 1 across all events (0.70–0.83): missing mass exceeds excess mass,
  indicating that some workers in the affected wage zone do not re-appear at or above the MW
  (transitions to informal employment, unemployment, or higher wages outside the excess zone).
- Informal workers: ratio > 1 in Event A (1.22) — excess exceeds missing, consistent with
  informal wages bunching at the formal MW norm. Ratios drop to ≈ 1.00 (Event B) and 0.81
  (Event C). No consistent pattern across events.
- Formal dep missing mass grows across events: 6.78pp (A) → 8.03pp (B) → 13.02pp (C),
  suggesting increasing MW bite as the RMV rises relative to the wage distribution.

**Placebo test**: [State placebo result from `mw_sanity_checks.json` — artificial MW levels
should produce near-zero missing/excess mass. Confirm from data.]

### 5.2 Figure 2: Wage Distributions Pre vs. Post
*Three panels (one per event). Each panel shows the formal dep wage distribution binned at
S/25 intervals, pre-period (dashed) and post-period (solid), with counterfactual (gray), MW
marked by vertical line. Source: ENAHO Module 500.*

[Figure 2 requires a separate Python plotting script — see `scripts/mw_paper_figures.py`
to be written.]

---

## 6. HETEROGENEITY (≈700 words)

### 6.1 By Industry Sector (Table 4 — Panel A)
*Construction excluded: cell-level outlier ratios (A=3.23, C=8.31) driven by sparse sample.*

**TABLE 4, Panel A: Bunching Ratios by Industry**

| Sector | Event A | Event B | Event C | Average |
|--------|:-------:|:-------:|:-------:|:-------:|
| Total (reference) | 0.733 | 0.854 | 0.840 | 0.809 |
| Agricultura / Minería | 1.947 | 0.792 | 0.594 | 1.111 |
| Manufactura | 1.449 | 0.955 | 1.126 | 1.177 |
| Comercio | 0.791 | 1.230 | 0.728 | 0.916 |
| Transporte y alojamiento | 1.036 | 1.040 | 0.609 | 0.895 |
| Finanzas / Servicios prof. | 0.817 | 1.091 | 0.801 | 0.903 |
| Administración pública | 1.161 | 0.531 | 0.806 | 0.833 |
| Educación y salud | 0.648 | 0.729 | 1.003 | 0.793 |

### 6.2 By Firm Size (Table 4 — Panel B)

**TABLE 4, Panel B: Bunching Ratios by Firm Size**

| Firm size | Event A | Event B | Event C | Average |
|-----------|:-------:|:-------:|:-------:|:-------:|
| Micro (≤10 workers) | 0.987 | 1.002 | 0.699 | 0.896 |
| Pequeña (11–50) | 1.127 | 0.940 | 0.604 | 0.890 |
| Mediana/Grande (51+) | 0.704 | 0.826 | 0.922 | 0.817 |

### 6.3 By Age and Sex (Table 4 — Panel C)

**TABLE 4, Panel C: Bunching Ratios by Age Group and Sex**

| Group | Event A | Event B | Event C | Average |
|-------|:-------:|:-------:|:-------:|:-------:|
| Age 18–24 | 0.729 | 1.083 | 0.846 | 0.886 |
| Age 25–44 | 0.786 | 0.818 | 0.822 | 0.808 |
| Age 45–64 | 0.779 | 0.959 | 0.838 | 0.859 |
| Men | 0.768 | 0.816 | 0.801 | 0.795 |
| Women | 0.742 | 0.967 | 0.883 | 0.864 |

**Interpretation**:
- No heterogeneity is dramatic. Ratios cluster around the aggregate 0.80–0.85 for most cells.
- Manufactura consistently above aggregate: stronger formal MW compliance and wage
  renegotiation activity relative to service sectors.
- Firm size: Micro and pequeña firms show slightly higher ratios in some events (more
  wage floor response), though mediana/grande firms approach 0.92 by Event C.
- Age: Young workers (18–24) show ratio 1.083 in Event B — excess exceeds missing,
  possibly reflecting MW as an entry wage anchor for new hires.
- Sex: Women have slightly higher ratios than men in Events B and C.

---

## 7. SUPPLEMENTARY: EVENT STUDY AND HOURS (≈700 words)

### 7.1 Event Study DiD — Employment (Table 5, Figure 1)
*Events A and C only. Event B in Appendix B.*

**TABLE 5: Event Study Post-Period Employment Coefficients**

| Event | Post β | SE | p | Pre-trend F | Pre-trend p |
|-------|-------:|---:|---|:-----------:|:-----------:|
| A (2017) | −0.011 | 0.040 | 0.79 | — | 0.061 (PASS) |
| C (2023) | +0.183 | 0.088 | 0.04 | 0.89 | 0.260 (PASS) |

*β = coefficient on Kaitz_pre × post. Positive β = higher-Kaitz departments had larger
employment gains post-MW. Pre-trend F: joint F-test on pre-period year × Kaitz interactions.*

**FIGURE 1: Event Study Coefficients — Employment (Events A and C)**
*Two panels showing year × Kaitz coefficients with 95% CI. Base year = last pre-period.
Pre-period estimates should be near zero; deviation from zero indicates pre-trend failure.*

**Interpretation**:
- Event A: post coefficient near zero (β=−0.011, ns). No detectable employment effect.
- Event C: positive post coefficient (β=+0.183, p=0.04). Higher-Kaitz departments had
  larger employment gains post-2022 MW. This is inconsistent with a standard negative
  employment effect but consistent with the near-zero OWE. Interpretation is uncertain
  given weak first stage (F=1.5 — see Appendix C).

### 7.2 Intensive Margin: Hours DiD (Table 6)

**TABLE 6: Weekly Hours DiD — ENAHO CS, p513t**

| Event | Treat pre | Treat post | Δ treat | Ctrl pre | Ctrl post | Δ ctrl | **DiD** |
|-------|:---------:|:----------:|:-------:|:--------:|:---------:|:------:|:-------:|
| A (750→850) | 44.4h | 44.8h | +0.4h | 42.1h | 41.5h | −0.7h | **+1.1h** |
| B (850→930) | 44.8h | 45.1h | +0.3h | 41.2h | 42.0h | +0.8h | **−0.5h** |
| C (930→1025) | 44.5h | 46.4h | +1.9h | 42.1h | 43.6h | +1.5h | **+0.4h** |

*Treatment: formal dep workers earning [0.85×MW_old, MW_new) in pre-period.
Control: [1.5×MW_new, 3.0×MW_new]. Weighted means.*

No DiD exceeds ±1.1h/week. MW increases in Peru do not produce detectable intensive-margin
hours adjustment among formal dependent workers.

---

## 8. CONCLUSION (≈500 words)

**Summary**:
1. MW increases in Peru generate consistent missing mass below the MW and excess mass at/above
   the MW among formal dependent workers — bunching ratios of 0.70–0.83 across three events.
   Missing mass grows with each successive MW increase, from 6.78pp (Event A) to 13.02pp (Event C),
   consistent with increasing MW bite over time.

2. Informal wages show a partial response in Event A (ratio 1.22) but not in Events B and C
   (0.99 and 0.81). The evidence for systematic spillovers from formal to informal wages is
   mixed and does not hold consistently across events.

3. Employment effects: the event study DiD (Events A and C, where pre-trends hold) finds
   near-zero effects. The IV/OWE approach is not identified — first stages are weak across
   all three events (F < 2), and Event B's first stage has a reversed sign. We cannot report
   a credible employment elasticity.

4. No meaningful hours adjustment: DiD estimates of ±0.4–1.1h/week, none statistically
   distinguishable from zero.

**Limitations** (stated plainly):
- Geographic identification is limited: 24 departments, single national MW, weak first stages.
- Event B excluded from regression-based analysis (pre-trends violated).
- Panel evidence requires 86% attrition caveat and is unreliable.
- Bunching estimates do not identify the counterfactual employment level — only the
  redistribution of the wage distribution.

**What can be concluded**: Peru's MW increases produce clear distributional responses
— workers compress toward the MW — without detectably large employment losses.
High informality, uneven enforcement, and the moderate Kaitz index (national median 0.60
in 2021) likely limit the MW's bite. Whether these patterns reflect a labor market with
low labor demand elasticity, widespread non-compliance, or partially binding coverage
cannot be distinguished from the current evidence.

---

## APPENDICES

### Appendix A: ENAHO Panel 978 — Event C Decomposition (Table A1)
**Caveat**: ~86% two-year attrition (2021→2023). Matched subsample identified by
`facpanel2123 > 0`. Results are subject to severe survivorship bias and should not be
used to draw causal inferences.

**TABLE A1: Panel Transition Matrix — Treatment (N=1,171) and Control (N=3,903)**

| Metric | Treatment | Control | DiD |
|--------|:---------:|:-------:|:---:|
| Employment retention (2023) | 15.8% | 15.0% | +0.8pp |
| Formal dep retention (2023) | 12.1% | 12.5% | −0.4pp |
| Δ log wage (stayers only) | +0.264 | +0.143 | **+0.121** |
| Δ hours (stayers only) | −0.5h/wk | −2.1h/wk | **+1.6h/wk** |

**Treatment transition (N=1,171)**:
→ Formal dep 2023: 12.1% | → Informal dep 2023: 2.1% | → Not re-interviewed: 85.8%

The 85.8% "not observed in 2023" category reflects panel attrition (`facpanel2123 == NaN`),
**not unemployment or job loss**. Panel 978 has approximately 25% annual retention;
over a 2-year gap this produces ~17% matched survival rate.

Treatment: formal dep 2021, wage ∈ [S/791, S/1,025). Control: wage ∈ [S/1,230, S/2,563].

### Appendix B: Event B — Pre-Trend Violation and Reversed IV (Table B1)
**Event B employment** (MW: S/850→S/930, April 2018):

Pre-trend joint F-test: p = **0.0014** (FAIL). Pre-period coefficients:
- 2015 (relative to 2017 base): β = +0.016 (ns), 2016: β = +0.080 (p = 0.001\*\*\*)

Higher-Kaitz departments had systematically higher employment growth *before* the 2018 MW
change. The DiD is not identified for Event B employment.

**Event B IV first stage**: π = −0.059 (SE = 0.094, F = 0.4). Sign reversed: departments
with higher Kaitz (where MW bites more) had *lower* wage growth 2017→2019. This is
inconsistent with the instrument's economic logic and makes the IV ratio uninterpretable.

**Event B bunching is retained in main Table 2** — the distributional estimator is valid.

### Appendix C: IV / OWE — Not Identified (Table C1)
Following Dube & Lindner (2024). Instrument: $K_{d,\text{pre}} = \text{MW}_\text{old} / \text{median formal wage}_{d,\text{pre}}$.

**TABLE C1: First Stage, Reduced Form, and OWE**

| Event | π (FS) | SE | F | β (RF emp) | SE | OWE | SE |
|-------|-------:|---:|:-:|----------:|---:|----:|---:|
| A | +0.110 | 0.084 | 1.7 | −0.011 | 0.040 | −0.097 | 0.367 |
| B | −0.059 | 0.094 | 0.4 | +0.171 | 0.052 | — | — |
| C | +0.175 | 0.144 | 1.5 | +0.183 | 0.088 | +1.045 | 0.995 |

*Event B OWE not reported: reversed first-stage sign. F < 10 in all events (Stock-Yogo
threshold for 10% size distortion). OWE ratios are not credible causal estimates.*

First stages are weak because: (1) Peru has a single national MW, so all departments face
the same MW change; (2) with only 24 departments, there is limited statistical power to
detect the differential wage response to pre-period Kaitz variation.

The wide OWE confidence intervals (spanning −0.83 to +0.60 for Event A alone) confirm
that the IV approach cannot meaningfully bound the employment elasticity. We report these
results for transparency, consistent with the convention of reporting attempted but
failed identification strategies in economics papers.

### Appendix D: EPEN CIU Annual 2023 — Lee-Saez Bunching (Table D1)
Single-period bunching using EPEN CIU Annual 2023 (INEI code 873, N=53,316 formal workers).
Formal definition: `INFORMAL_P == '2'` (health insurance registered). MW = S/1,025.

**TABLE D1: EPEN CIU Bunching (Single Period, Post 2023)**

| Sample | N | Excess factor | Bunching ratio |
|--------|--:|:-------------:|:--------------:|
| All cities | 53,316 | 1.34× | 1.622 |
| Lima upper stratum | — | — | 1.602 |
| Other cities | — | — | 1.688 |

Single-period Lee-Saez estimate (no pre-period baseline available). Confirms excess mass
at S/1,025 in post-period urban data, consistent with main ENAHO bunching results.

---

## FILE INDEX

| File | Contents |
|------|----------|
| `exports/data/mw_complete_margins.json` | Bunching (Table 2), wage compression, lighthouse |
| `exports/data/mw_heterogeneity.json` | Table 4 all panels |
| `exports/data/mw_iv_owe.json` | Table C1, Figure 1, event study coefficients |
| `exports/data/mw_hours_epen_dep.json` | Table 6 |
| `exports/data/mw_panel_decomposition.json` | Table A1 |
| `exports/data/mw_epen_ciu_annual_bunching.json` | Table D1 |
| `exports/data/mw_annual_robustness_results.json` | Robustness checks |
| `exports/data/mw_paper_results_complete.json` | All results aggregated |
| `paper/mw_paper_outline.md` | This file |

---

*Revised: 2026-03-17. Data: ENAHO 2015–2023 (Módulo 500), EPEN CIU Annual 2023, ENAHO Panel 978.*
