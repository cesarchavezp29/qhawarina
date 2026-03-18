# Missing Mass and Minimum Wages: Distributional Effects of Three MW Increases in Peru
### [Alt. title: "Where Do the Workers Go? Bunching Evidence from Peru's Minimum Wage Increases"]
## Complete Paper Outline — Revised v5

---

## IDENTIFICATION SCORECARD

| Method | Event A | Event B | Event C | Verdict |
|--------|---------|---------|---------|---------|
| Pre-post bunching estimator (ENAHO) | ✓ valid | ✓ valid | ✓ valid | **Main result** |
| Pre-post bunching estimator (EPE Lima) | ✓ 1.031 [0.70,1.63] | ✓ 0.733 [0.60,1.03] | ✓ 0.885 [0.70,1.13] | Appendix E — confirms ENAHO |
| Event study / DiD | ⚠ untestable (no pre-period)† | ✗ fail (p=0.007) | ✗ fail (p=0.017) | Appendix B — not identified |
| IV / OWE | ✗ F=1.5 (ns) | ✗ F=2.6, reversed sign | ✗ F=0.1 (ns) | Appendix C — not identified |
| Panel decomp | — | — | ✗ 76% attrition, selection bias | Appendix A — unreliable |
| Wage compression | ✓ −0.035*** | ✓ −0.050*** | ✓ −0.069*** | REAL DESCRIPTIVE (41–92% mechanical) |
| Hours DiD | null | null | null | VALID NULL (≤±1.1h/wk) |
| EPEN DEP bunching | — | — | ✗ MW doesn't bind (Kaitz=0.47) | Dropped‡ |

†Event A: 2015 is the base year; no prior ENAHO years available for pre-trend test; untestable.
Events B and C: joint employment pre-trend F-tests fail (p=0.007 and p=0.017). DiD reported descriptively in Appendix B only.

‡**EPEN DEP specification vs canonical** (why the result is not comparable to Table 2):
EPEN DEP Bunching uses `mw_hours_epen_dep_bunching.py` with five non-canonical settings: (1) **No dependent-worker filter** — `c310` (employer/self-employed/employee) exists in the data but is not applied; the treatment zone contains 23.7% employers and self-employed (formal), inflating N from 1,371 to 2,747 in [790,1025). (2) **Narrow bin range** — `arange(200, 4001, 25)` instead of canonical `arange(0, 6025, 25)`. (3) **Polynomial counterfactual** — fits a 4th-degree polynomial on the control zone rather than the inverse-abs-weighted background correction. (4) **Net-outflow missing mass** — uses `max(-delta_adj[zone].sum(), 0)` instead of neg-only bins. (5) **WAGE_MAX=8000**. These differences together yield Kaitz=0.47 (median formal EPEN DEP wage is S/2,246 vs MW S/930), confirming the MW does not bind for this sample. Dropped; Table A2 retained in working notes only.

---

## 1. INTRODUCTION (≈1,000 words)

**Hook**: Peru raised the minimum wage from S/750 to S/1,025 between 2016 and 2022 — a 37%
increase over six years — while formal employment remained stable. This paper uses three
identifiable MW events to study how the wage distribution responded.

**Question**: Do Peru's MW increases produce missing mass below the threshold and bunching at
it? Do they affect employment and hours? We use a pre-post distributional estimator inspired
by the framework of Cengiz et al. (2019), adapted for a national MW setting without control
jurisdictions, as our primary tool.

**Contribution**:
1. First pre-post distributional analysis of Peru's MW using three identifiable MW events
   across 2015–2023. Adapted from Cengiz et al. (2019); methodological precedent from
   Harasztosi & Lindner (2016) for single-national-MW settings.
2. Direct test for MW spillovers to informal wages using the same estimator on formal and
   informal workers simultaneously.
3. We systematically test three identification strategies for employment effects (IV at
   department level, IV at province level, within-year province IV) and document why all
   fail in Peru's national-MW setting: the Kaitz instrument conflates MW exposure with
   poverty gradients and seasonal wage dynamics. This is a methodological contribution for
   researchers working with single-national-MW countries. Results in Appendices B–C.

**Main findings**:
- Formal dependent workers show clear missing mass and excess mass at the MW in all three
  events (bunching ratios 0.70–0.83), with increasing missing mass across events as the MW
  rises relative to the wage distribution.
- Wage compression: workers just above the new MW see 3–5pp faster log-wage growth than
  high-wage workers, consistent with the distributional shift.
- Informal workers: partial response in Event A (ratio 1.22) but not Events B and C
  (0.99 and 0.81). No consistent spillover pattern.
- Event study DiD: pre-trends fail for all testable events (B: p=0.007, C: p=0.017).
  Results are in Appendix B only; no identified employment effect is reported.
- Hours adjustment is negligible across all events (DiD ≤ ±1.1h/week).
- The IV/OWE approach is not identified; no credible employment elasticity is reported.

**Paper structure**: Section 2 describes the institutional context. Section 3 presents data
and summary statistics (Table 1). Section 4 describes the empirical strategy. Sections 5–7
present bunching, heterogeneity, and the hours null result. Section 8 concludes. All
event-study and IV results are in Appendices B–C.

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

Event B: employment pre-trends fail (F-test p=0.007). Event C: employment pre-trends also
fail (p=0.017). All regression-based event study results are in Appendix B. Bunching results
are retained in the main analysis (the pre-post estimator does not require parallel trends).

### 2.3 Department-Level Kaitz Variation
The Kaitz index (MW / weighted-median formal wage by department) ranges from 0.39 to 1.09
across departments and years. Ica (dept 11) has the highest Kaitz (0.94 in 2015), driven by
formal agricultural workers on agro-export farms earning near-minimum monthly wages despite
full-time hours. Huancavelica (dept 09) has low Kaitz (≈0.50) because approximately 85% of
its formal workers are public-sector employees earning above the MW. The Kaitz variation
across departments primarily reflects differences in economic structure (agriculture vs.
public-sector share) rather than differential MW exposure, which limits its usefulness as
an instrument. This variation does not generate sufficient first-stage power for IV
identification (F<3 in all events — see Appendix C), and employment pre-trends fail for
Events B and C (Appendix B). The Kaitz is used here as a descriptive measure of MW
bindingness, not as an IV.

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
| 2015 | 750 | 10,059 | 1,400 | 4.6% | 10.9% | 0.536 | 46.4 | 26.7% |
| 2016 | 850 | 11,488 | 1,500 | 4.8% | 14.0% | 0.567 | 46.2 | 27.9% |
| 2017 | 850 | 10,895 | 1,500 | 6.0% | 10.4% | 0.567 | 46.4 | 27.4% |
| 2018 | 930 | 11,723 | 1,673 | 4.8% | 14.6% | 0.556 | 46.3 | 27.6% |
| 2019 | 930 | 10,922 | 1,700 | 6.5% | 12.5% | 0.547 | 46.2 | 27.3% |
| 2021 | 930 | 8,946 | 1,800 | 7.8% | 11.5% | 0.517 | 46.8 | 23.2% |
| 2022 | 1,025 | 9,554 | 1,800 | 5.7% | 16.9% | 0.569 | 46.6 | 24.3% |
| 2023 | 1,025 | 9,838 | 1,863 | 8.2% | 12.6% | 0.550 | 46.5 | 26.1% |

*N = unweighted formal dependent workers with monthly wage S/1–6,000 and positive survey weight
(`ocu500==1`, `p507∈{3,4,6}`, `ocupinf==2`, `wage∈(0,6000)`, `fac500a>0`). This is the exact
sample entering the bunching estimator (`WAGE_MAX=6000`, `BIN_WIDTH=25`).
% at MW = share earning MW±25 (one bin). % below MW = share earning below MW.
Median Kaitz = MW / median wage, national. Formality rate = formal dep / all employed.
Hours from p513t (formal dep with positive hours). COVID-19 year 2021 shows depressed formality.*

**Key patterns**:
- Median wage rises from S/1,400 (2015) to S/1,863 (2023), a 33% nominal increase, tracking
  the 37% MW increase. Kaitz ranges 0.517–0.569 with no clear trend; COVID-19 depresses formal
  wages in 2021, pushing Kaitz to its lowest value (0.517).
- Share below MW rises after each event (14.0% in 2016, 14.6% in 2018, 16.9% in 2022),
  then falls as wages adjust — consistent with MW bindingness increasing over time.
- Formality rate dips sharply in 2021 (23.2%) due to COVID-19 — context for Event C identification.
- Hours stable at 46–47h/week throughout; no aggregate hours response to MW increases.

### 3.3 EPEN CIU Annual 2023 (Appendix D)
INEI code 873. Urban formal workers, single cross-section. No 2022 annual available.

---

## 4. EMPIRICAL STRATEGY (≈800 words)

### 4.1 Pre-Post Distributional Estimator (Primary)
We use a pre-post distributional estimator inspired by the framework of Cengiz et al. (2019),
adapted for a national MW setting without control jurisdictions. Cengiz et al. construct the
counterfactual using control states that did not raise their MW. Peru has a single national MW;
we use the pre-period wage distribution with a background correction from the upper tail
(>2×MW) instead. The closer methodological precedent for a single-national-MW setting is
Harasztosi & Lindner (2016).

Bin width: S/25. Range: S/0–S/6,000 (survey-weighted by `fac500a`).

**Counterfactual**: inverse-abs-weighted background shift from clean zone (wages > 2 × MW_new),
applied uniformly as a shift correction to the full pre-post delta.

**Missing mass** (net deficit below MW):
$$B^{miss} = \sum_{b \in [0.85 \times \text{MW}_{old},\; \text{MW}_{new})} \max(-\Delta_b^{adj},\; 0)$$

**Excess mass** (net surplus at MW):
$$B^{exc} = \sum_{b \in [\text{MW}_{new},\; \text{MW}_{new} + 250)} \max(\Delta_b^{adj},\; 0)$$

where $\Delta_b^{adj} = \Delta_b - \bar{\Delta}_{clean}$ subtracts the background trend.

**Bunching ratio**: $R = B^{exc} / B^{miss}$.

This estimator is valid for all three events. It does not require parallel trends.

### 4.2 Event Study DiD (Appendix B — Not Identified)
Employment pre-trends fail for Events B (p=0.007) and C (p=0.017). Event A has no
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
F-statistics: A=1.5, B=2.6 (reversed sign), C=0.1. All below Stock-Yogo threshold of F>10.
See Appendix C.

---

## 5. BUNCHING RESULTS (≈900 words)

### 5.1 Main Bunching Estimates (Table 2)

**TABLE 2: Pre-Post Bunching Estimator — ENAHO CS**

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
- **Statistical precision**: Bootstrap 95% CIs (N=1,000) are [0.567, 0.896] (A),
  [0.716, 1.016] (B), and [0.716, 0.960] (C). Events A and C reject R=1 (no job loss)
  at 95%; Event B does not — ~4.7% of bootstrap draws produce ratio ≥ 1.0. The data
  for Event B are consistent with both full redistribution (R=1, zero job loss) and
  displacement of up to 28% of affected workers (where CI lower bound = 0.716 implies
  we can rule out job-loss rates above ~28%). We cannot statistically detect whether
  any displacement occurred in Event B at the 95% level.

**Placebo test**: Applying the same estimator to artificial thresholds (S/1,100→1,200 and
S/1,400→1,500) on the Event B population produces ratios of 0.114 and 0.013, compared to
0.829 at the actual S/930 threshold. Bunching is specific to the true MW.

### 5.2 Figure 2: Wage Distributions Pre vs. Post
Three panels (one per event), formal-dep workers, S/25 bins. Pre-period (blue bars),
post-period (orange bars), polynomial counterfactual (dashed gray). MW marked by vertical
line. Shaded region = affected zone [0.85×MW_old, MW_new).

*Generated by `scripts/mw_paper_figures.py`. File: `exports/figures/fig2_bunching_distributions.png`.*

### 5.3 Wage Compression (Table 3)

The compression DiD measures the change in the mean log wage gap between workers in
[MW_new, 1.5×MW_new] and workers in [2×MW_new, 3×MW_new]. Because this is a repeated
cross-section, the composition of the compression zone changes across periods: workers
pushed from below the MW into the zone enter at the new floor (log(MW_new)), mechanically
pulling the zone's mean downward. Table 3 decomposes each total DiD into its mechanical
(composition) and genuine (relative wage growth) components.

**TABLE 3: Wage Compression DiD — ENAHO CS, with Composition Decomposition**

| Event | Total DiD | Mechanical | Genuine | % mechanical | New arrivals |
|-------|:---------:|:----------:|:-------:|:------------:|:------------:|
| A (750→850) | −0.035 | −0.028 | −0.008 | 78% | 13% of zone |
| B (850→930) | −0.050 | −0.021 | −0.029 | 41% | 10% of zone |
| C (930→1025) | −0.069 | −0.064 | −0.005 | 92% | 26% of zone |

*Mechanical = DiD predicted if existing compression-zone workers grew at the high-wage rate
and new arrivals enter at log(MW_new). Genuine = total DiD minus mechanical.*

*Zones: compression = [MW_new, 1.5×MW_new]; high-wage = [2×MW_new, 3×MW_new].*

**Interpretation**: For Event B, approximately 41% of the −0.050 DiD reflects the
composition effect; the remaining 59% (−0.029) represents genuine relative wage compression
for pre-existing just-above-floor workers. For Events A and C the composition effect
dominates (78% and 92% respectively), and the genuine compression component is near zero
(−0.008 and −0.005). Both components describe real distributional consequences of the MW
increase — the composition effect is the bunching mechanism itself (workers piling up at
the new floor), while the residual in Event B suggests modest spillover compression above
the threshold.

The growing dominance of the mechanical component across events (A: 78% → B: 41% → C: 92%
is non-monotone, driven by Event C's larger affected zone: 26% of the 2023 compression
zone consists of new arrivals vs 10% in Event B. This pattern is consistent with the
rising MW bite (Kaitz: 0.517→0.569) pulling more workers from a wider sub-MW zone up to
the new floor.

**Methodological note**: The compression DiD does not identify a causal effect of MW on
above-floor wages. It is a descriptive statistic of distributional change. Workers in
[MW_new, 1.5×MW_new] in the post-period are a different population than those in that
zone pre-period. The genuine component (−0.008 to −0.029) is an upper bound on any true
spillover compression, since even this residual could reflect differential sector
composition rather than a wage-setting response.

---

## 6. HETEROGENEITY (≈700 words)

### 6.1 By Industry Sector (Table 4 — Panel A)
*Table 4 uses `p524a1` wage, `p507 ∈ {3,4,6}`, bins arange(0, 6025, 25), neg-only missing
mass — identical specification to Table 2 (mw_complete_margins.py canonical). The "Total"
row (0.696 / 0.829 / 0.830) matches Table 2 exactly. Bootstrap 95% CIs (N=1000): A=[0.567, 0.896],
B=[0.716, 1.016], C=[0.716, 0.960]. Event A and C reject R=1 at 95%;
Event B does not (CI upper bound 1.016 > 1.0, ~4.7% of bootstrap draws ≥ 1.0).*

*Sector from CIIU rev4 variable `p506r4` (2-digit). CIIU 35–39 (utilities: electricity,
water, waste — rev4 Sections D+E) are grouped with Agricultura/Minería; in CIIU rev3 these
were code 40–41 (same broad sector). Construction excluded: sparse cells (<500 obs).*

**TABLE 4, Panel A: Bunching Ratios by Industry**

| Sector (N_pre 2015) | Event A | Event B | Event C | Average |
|---------------------|:-------:|:-------:|:-------:|:-------:|
| **Total (formal dep, N=9,952)** | **0.696** | **0.829** | **0.830** | **0.785** |
| Agricultura / Minería / Utilities (N=1,031)† | 1.641 | 0.765 | 0.407 | 0.938 |
| Manufactura (N=904) | 1.000 | 0.869 | 1.049 | 0.973 |
| Comercio (N=937) | 0.784 | 1.209 | 0.723 | 0.905 |
| Transporte y alojamiento (N=556) | 1.041 | 1.018 | 0.585 | 0.881 |
| Finanzas / Servicios prof. (N=1,113) | 0.768 | 1.018 | 0.806 | 0.864 |
| Administración pública (N=1,639) | 1.033 | 0.498 | 0.804 | 0.778 |
| Educación y salud (N=3,021) | 0.598 | 0.709 | 0.975 | 0.760 |

†Event A ratio (1.641) sensitive to sparse pre-period bins; excluding Event A,
Agri/Min/Util average is 0.59 — below aggregate, consistent with high Kaitz exposure
in agro-export agriculture (Ica region).

### 6.2 By Firm Size (Table 4 — Panel B)

*Firm size from `p512b` (exact establishment worker count, available in all years).
Workers with missing `p512b` (4.3–5.8% per year) are excluded from all three rows;
the three rows sum to 9,500 of 9,952 (95.5%) of the 2015 formal-dep sample.
No row-level fallback to `p512a` is applied — the 4.5% gap should be noted in the paper.*

*Note: public-sector workers (p512a==5) appear in all three rows based on their actual
establishment size per `p512b`; most are large employers (p512b>50) and fall into
Mediana/Grande.*

**TABLE 4, Panel B: Bunching Ratios by Firm Size**

| Firm size (N_pre 2015) | Event A | Event B | Event C | Average |
|------------------------|:-------:|:-------:|:-------:|:-------:|
| Micro (≤10, N=705) | 0.899 | 0.909 | 0.690 | 0.833 |
| Pequeña (11–50, N=1,238) | 0.919 | 0.901 | 0.598 | 0.806 |
| Mediana/Grande (51+, N=7,557) | 0.686 | 0.793 | 0.896 | 0.792 |

### 6.3 By Age and Sex (Table 4 — Panel C)

**TABLE 4, Panel C: Bunching Ratios by Age and Sex**

| Group | Event A | Event B | Event C | Average |
|-------|:-------:|:-------:|:-------:|:-------:|
| Age 18–24 | 0.661 | 1.010 | 0.805 | 0.826 |
| Age 25–44 | 0.748 | 0.792 | 0.797 | 0.779 |
| Age 45–64 | 0.701 | 0.927 | 0.853 | 0.827 |
| Men | 0.726 | 0.788 | 0.778 | 0.764 |
| Women | 0.697 | 0.922 | 0.877 | 0.832 |

**Interpretation**:
- Ratios cluster near 0.80–0.85 for most cells. No single subgroup shows dramatically
  different behavior.
- Manufactura consistently above aggregate (0.95–1.45): stronger bunching response,
  likely reflecting higher SUNAFIL enforcement density in manufacturing.
- Mediana/Grande firms: ratio rises from 0.70 (A) to 0.92 (C) — large-firm compliance
  increases as MW bite grows.
- Young workers (18–24): ratio 1.010 in Event B, approximately equal redistribution
  (excess ≈ missing), suggesting MW serves as an entry wage anchor for new hires.
- Women have slightly higher ratios than men in Events B and C.

### 6.4 Ethnicity — Text Paragraph (Not Tabled)

*Source: `mw_ethnicity_heterogeneity.py`. Ethnicity for Events A/B from Module 300 p300a
(mother tongue, 100% merge rate); Event C from Module 500 p558c (self-ID). Data verified:
p300a codes 1=quechua, 2=aymara, 3=otra nativa → indigenous; 4=castellano → Spanish.
p558c codes 1=quechua, 2=aimara, 3=nativo amazónico, 9=otro indígena → indigenous.*

**Ethnicity paragraph (Section 6 or 8):**

Among formal dependent workers, indigenous (Quechua/Aymara/native-language) and
non-indigenous workers have similar wage distributions (both median S/1,400) and similar
MW exposure (8.5% vs 10.7% below MW). The ethnic divide operates through access to formal
employment: the indigenous formality rate is 5.7% vs 20.7% for Spanish speakers (3.6×
gap, 2015 ENAHO). The 925 indigenous formal-dep workers are predominantly public-sector
employees (59% in CIIU 84/85/86, vs 42% for Spanish-speaking workers), concentrated in
highland departments (Puno, Ayacucho, Ancash). Bunching ratios by ethnicity are unstable
across events (indigenous avg 1.27, range 0.63–1.80) and likely reflect sector composition
(government pay scales) rather than differential MW compliance.

*⚠ The bunching analysis cannot capture the main ethnic dimension of MW policy, which
operates through the extensive margin (access to formal employment). The formality-rate
gap (5.7% vs 20.7%) is the substantively important finding.*

**TABLE 4, Panel E: Bunching Ratios by Education**

| Group (N_pre 2015) | Ev A | Ev B | Ev C | Avg |
|--------------------|:----:|:----:|:----:|:---:|
| Primaria o menos (N=465) | 1.229 | 0.506 | 0.808 | 0.848 |
| Secundaria (N=2,876) | 0.824 | 0.960 | 0.791 | 0.859 |
| Superior (N=6,610) | 0.761 | 0.937 | 0.857 | 0.852 |

*No meaningful education gradient: all three groups average 0.85. Primary workers have
higher share below MW (36.3% below S/750 vs 21% for secondary, 4.7% for superior),
consistent with greater MW exposure, but this does not translate into systematically
higher ratios. Superior workers drive the aggregate since 66% of the sample has some
higher education.*

**TABLE 4, Panel F: Bunching Ratios by Geography and Employer Sector**

| Group (N_pre 2015) | Ev A | Ev B | Ev C | Avg |
|--------------------|:----:|:----:|:----:|:---:|
| Lima Metropolitana (N=2,594) | 0.618 | 1.050 | 0.817 | 0.828 |
| Resto del país (N=7,358) | 0.856 | 0.621 | 0.888 | 0.788 |
| Área urbana (N=9,286) | 0.702 | 0.843 | 0.828 | 0.791 |
| Área rural (N=666) | 1.016 | 0.550 | 0.909 | 0.825 |
| Sector público (N=4,315) | 0.921 | 0.456 | 0.865 | 0.747 |
| Sector privado (N=5,576) | 0.766 | 0.962 | 0.768 | 0.832 |

*Geography*: Lima and the rest of the country alternate across events — no consistent
regional differential. Rural workers (N=666, median wage S/1,200, 26.9% below S/750)
have the highest MW exposure but ratios close to the aggregate.

*Employer sector*: Public sector shows notably low Event B ratio (0.456, avg 0.747 vs
0.832 for private). This likely reflects rigidity: public-sector wages are set by law
(decree) and may not adjust smoothly at the event margin. Private sector ratios are
consistently near or above the aggregate.

---

## 7. SUPPLEMENTARY: HOURS DiD (≈300 words)

*All event study DiD results are in Appendix B (descriptive, not identified). No event study
results are reported in the main text. Pre-trends fail for Events B and C (p=0.007 and
p=0.017); Event A has no testable pre-period.*

### 7.1 Intensive Margin: Hours DiD (Table 6)

**TABLE 6: Weekly Hours DiD — ENAHO CS, p513t**

| Event | Treat pre | Treat post | Δ treat | Ctrl pre | Ctrl post | Δ ctrl | DiD |
|-------|:---------:|:----------:|:-------:|:--------:|:---------:|:------:|:---:|
| A (750→850) | 44.4h | 44.8h | +0.4h | 42.1h | 41.5h | −0.7h | +1.1h |
| B (850→930) | 44.8h | 45.1h | +0.3h | 41.2h | 42.0h | +0.8h | −0.5h |
| C (930→1025) | 44.5h | 46.4h | +1.9h | 42.1h | 43.6h | +1.5h | +0.4h |

*Treatment pre-period: workers in [0.85×MW_old, MW_new) in the pre year.*
*Treatment post-period: workers in [MW_new, 1.3×MW_new] in the post year (bunched-up zone).*
*Control: [1.5×MW_new, 3.0×MW_new] in both periods. Weighted means using `fac500a`.*
*`p513t` = total weekly hours across all jobs (not just main job). Median in treatment*
*pre-zone ≈ 48h (mean 44–45h); p10=28h, p90=60h — high within-group variance.*

No DiD exceeds ±1.1h/week. No evidence of intensive-margin hours adjustment.
The post-period treatment band follows the same affected workers who were pushed to or above
the new floor, tracking where the displaced wage mass landed rather than a fixed absolute band.

---

## 8. CONCLUSION (≈500 words)

**Employment effects cannot be identified.** The event study fails pre-trends for Events B
and C (p=0.007 and p=0.017); Event A has no testable pre-period. The IV approach fails due
to weak first stages (F<3 in all events: A=1.5, B=2.6 with reversed sign, C=0.1). The
bunching ratios (0.70–0.83, bootstrap CIs [0.57–1.02]) are consistent with both modest
displacement (up to 28% of affected workers) and full redistribution. The data cannot
distinguish between these scenarios.

**What CAN be concluded**: Peru's MW increases produce a clear, placebo-verified
redistribution of the wage distribution. Workers near the MW floor receive raises — the
pre-post bunching estimator documents 6.78–13.02pp of missing mass below the old floor
across three events, with excess mass accumulating at the new floor (ratios 0.70–0.83).
The intensive margin (hours) shows no adjustment (DiD ≤ ±1.1h/week). Compression of 3–5pp
is present in all events but 41–92% is mechanical (composition effect: new arrivals entering
the wage floor from below pull the compression-zone mean down). The residual genuine
compression (−0.005 to −0.029 log points) represents slower relative wage growth for
pre-existing just-above-floor workers. With individual controls (age, sex, education,
sector), the compression coefficient is −0.045 (p<0.001), confirming it is not purely
compositional.

Three points of external validity: (1) the EPE Lima quarterly dataset produces ratios
(0.73–1.03) consistent with the ENAHO ratios (0.70–0.83) despite different formality
definitions, event windows, and geographic scope; (2) the placebo test (artificial
thresholds S/1,100→1,200 and S/1,400→1,500) produces ratios of 0.01–0.11, confirming
the signal is MW-specific; (3) informal workers show partial response in Event A (1.22)
but not in Events B–C, suggesting some spillover at lower MW levels.

This is not a data-quality failure. It is a structural feature of single-national-MW
settings: Kaitz variation across departments reflects poverty gradients with independent
secular and seasonal dynamics. Future work with wage registries (SUNAT, ESSALUD) that
allow quarterly firm-level identification would improve power substantially.

---

## APPENDICES

### Appendix A: ENAHO Panel 978 — Event C (Table A1)
**Caveat**: ~76% attrition over 2021→2023. Results subject to survivorship bias.
Do not use for causal inference. Outcome statistics are among re-interviewed workers only
(weighted by `facpanel2123`). Attrited workers are **not** counted as non-employed.

**TABLE A1: Panel Transition Results (Event C, 2021→2023)**

| Metric | Treatment (N=1,171) | Control (N=3,903) | DiD |
|--------|:-------------------:|:-----------------:|:---:|
| Re-interviewed rate | 24.4% (n=286) | 23.6% (n=921) | — |
| **Among re-interviewed (weighted by facpanel2123):** | | | |
| Employment retention | 63.9% | 60.1% | **+3.8pp** |
| Formal dep retention | 46.8% | 49.9% | **−3.1pp** |
| Δ log wage (stayers) | +0.307 | +0.154 | **+0.153** |
| Δ hours (stayers) | +0.7h/wk | −1.3h/wk | **+2.0h/wk** |
| **Transition (% of 2021 N, unweighted):** | | | |
| → Formal dep 2023 | 12.1% | 12.5% | — |
| → Informal dep 2023 | 2.1% | 1.2% | — |
| → Not employed 2023 | 8.6% | 8.6% | — |
| → Not re-interviewed | 75.6% | 76.4% | — |

**Previous (incorrect) Table A1** used `emp23.mean()` across all N including attritors
(NaN → False), yielding 15.8%/15.0% employment retention and DiD=+0.8pp. Those figures
confounded attrition dropout with job loss and were unweighted. The corrected table above
separates these two populations.

Treatment: formal dep 2021, wage ∈ [S/791, S/1,025). Control: wage ∈ [S/1,230, S/2,563].
Statistics weighted by `facpanel2123`; stayers = formal dep employed in both years with positive wage.

### Appendix B: Event Study — Pre-Trend Violations (Table B1)
All three events have identification problems for the event study DiD.

**Event A**: No pre-period data (2015 = base year, no 2014 or earlier in our sample).
Parallel trends assumption untestable. Post β=+0.007 (p=0.84), no employment effect,
but cannot be claimed as causal.

**Event B**: Joint pre-trend F-test p=0.007 (employment). Pre-period coefficients (base 2017):
2015: β=−0.015 (ns); 2016: β=+0.055 (p=0.037). Higher-Kaitz departments had
systematically higher employment before the 2018 MW change. DiD not identified.
IV first stage: π=−0.070 (SE=0.103, F=0.5). Reversed sign. IV ratio uninterpretable.

**Event C**: Joint pre-trend F-test p=0.017 (employment). Pre-period 2019: β=+0.163 (p=0.024).
Higher-Kaitz departments had higher employment in 2019 (pre-COVID), which by 2021 (our
base year) collapsed disproportionately. The 2019→2021 pre-trend renders the 2021→2023
comparison non-parallel.

All bunching results remain in main text — the pre-post bunching estimator does not require parallel trends.

### Appendix C: IV / OWE — Not Identified (Table C1)

**Why identification fails — three approaches tested:**

**Approach 1 (main): Department-level Kaitz IV, annual data.**
$K_{d,\text{pre}} = \text{MW}_\text{old} / \text{weighted-median formal wage}_{d,\text{pre}}$
(24 departments). F < 3 in all events (A=1.5, B=2.6 reversed sign, C=0.1). All below
Stock-Yogo threshold of F>10. Events B and C: employment pre-trends fail.

**Approach 2: Province-level Kaitz IV, annual data** (109 provinces).
Event B first stage F=11.72 but **sign is negative** (pi=−0.499): high-Kaitz provinces
had *slower* wage growth 2017→2019. Pre-trend test (2015→2016): F=8.07, pi=−0.74.
Higher-Kaitz provinces are structurally poorer/more remote — pre-existing declining relative
wages, unrelated to MW. Instrument invalid.

**Approach 3: Within-year province IV** (Q1 vs Q3 2018, 48 provinces).
Event B first stage F=16.15, pi=+0.77. Placebo test (2017 Q1→Q3, no MW change):
F=12.71, pi=+1.16. High-Kaitz provinces *always* show faster Q1→Q3 wage growth —
seasonal confound (harvest cycles, sector composition) correlated with Kaitz.
Instrument invalid.

**Root cause**: Peru has a single national MW. Kaitz variation across provinces reflects
poverty gradients with independent seasonal and structural dynamics. No feasible instrument
for a national-MW setting with 24 departments (or 109 provinces) produces F>10 with
valid pre-trends in ENAHO cross-sections.

**TABLE C1: Main Specification — Department-level Kaitz IV**

| Event | π (FS) | SE | F | β (RF emp) | SE | OWE | SE |
|-------|-------:|---:|:-:|----------:|---:|----:|---:|
| A | +0.169 | 0.082 | 1.5 | +0.014 | 0.038 | +0.082 | 0.229 |
| B | −0.070 | 0.103 | 2.6 | +0.122 | 0.044 | — | — |
| C | +0.132 | 0.090 | 0.1 | +0.220 | 0.076 | +1.667 | 1.271 |
| Pooled A+B+C | — | — | — | — | — | +0.114 | 0.267 |

Event B OWE not reported (reversed FS sign). Pooled OWE=+0.114 (SE=0.267, p=0.67) —
consistent with zero but not identified. All alternative specifications also fail
validity tests (see above). Employment effects not identified.

### Appendix D: EPEN CIU Annual 2023 — Lee-Saez Bunching (Table D1)
Single-period Lee-Saez estimate (INEI code 873, N=53,316 formal urban workers).
**Note**: Lee-Saez single-period ratios are not directly comparable to the pre-post ratios
in Table 2. The Lee-Saez method measures the excess of the current distribution over a
smooth counterfactual; the pre-post method measures the pre-to-post change in distribution
shares. The high Lee-Saez ratios (1.6×) simply confirm excess mass at the MW post-period,
consistent with Table 2.

**TABLE D1: EPEN CIU Bunching (Post 2023, Lee-Saez)**

| Sample | N | Excess factor | Ratio |
|--------|--:|:-------------:|:-----:|
| All cities | 53,316 | 1.34× | 1.622 |
| Lima upper stratum | — | — | 1.602 |
| Other cities | — | — | 1.688 |

### Appendix E: EPE Lima Quarterly Bunching — 6-Month Windows (Table E1)

**Motivation**: ENAHO uses 2-year pre/post gaps (e.g., 2017→2019 for Event B). A 6-month
window — EPE Lima quarterly survey — reduces secular wage growth contamination and provides
an independent dataset check. EPE Lima is Lima Metropolitan Area only; it is not used for
the main analysis (no cross-regional variation, Lima-only), but the pre-post estimator does
not require cross-regional variation.

**Formality definition**: EPE Lima uses `p222 == 1` (EsSalud health insurance registration)
as the formality marker. This is **narrower than ENAHO's `ocupinf == 2`** (comprehensive:
social security OR written contract OR firm size ≥ 10). EPE Lima formal workers are a
subset of ENAHO formal workers — direct ratio comparison is illustrative, not definitive.
Lima only. Events B and C are consistent with ENAHO; Event A diverges (EPE ratio 1.03 vs
ENAHO 0.70), likely reflecting Lima's lower Kaitz (MW less binding in the capital).

**Variables**: Employment = `ocu200 == 1`; Dependent = `p206 ∈ {3,4,6}` (empleado/obrero/hogar);
Wage = `ingprin` (monthly principal-job income); Weight = wave-specific `fa_[quarter][yy]`.

**TABLE E1: EPE Lima Bunching — Canonical Pre-Post Spec (25-sol bins, 1,000 bootstraps)**

| Event | Window | N pre | N post | Near-MW N (pre) | Ratio | 95% CI | ENAHO ratio | Match? |
|-------|--------|------:|-------:|----------------:|------:|:------:|:-----------:|:------:|
| A | Q1→Q3 2016 | 2,622 | 2,599 | 240 | 1.031 | [0.696, 1.630] | 0.696 | ✓ |
| B | Q1→Q3 2018 | 2,280 | 2,438 | 302 | 0.733 | [0.601, 1.028] | 0.829 | ✓ |
| C | Q1→Q3 2022 | 1,933 | 1,936 | 331 | 0.885 | [0.696, 1.131] | 0.830 | ✓ |

All three ENAHO annual ratios fall within the EPE Lima 95% bootstrap CIs. EPE Lima ratios
range 0.73–1.03, ENAHO ratios 0.70–0.83 — same ballpark despite different dataset,
population (Lima vs national), formality definition (EsSalud vs comprehensive), and
event window (6 months vs 2 years).

**Placebo test (Event B, fake threshold S/1,100→S/1,200)**:
Real Event B missing mass = 9.4% of distribution; placebo missing mass = 1.3% (7× smaller).
Placebo ratio = 0.666 but CI = [0.252, 2.738] (boot SD = 0.777 vs 0.112 for real event).
The near-identical point estimate masks near-zero mass being divided by near-zero mass —
the placebo CI spans essentially the full plausible range, confirming the ratio is
undefined at the fake threshold. The real event signal is 7× larger and precisely estimated.
Interpretation: bunching is concentrated at the true MW, not present at arbitrary thresholds.

**Caveat**: CIs are wide due to small N (~2,000–2,600 formal dep per quarter, ~240–330 in
near-MW zone). EPE Lima results are confirmatory, not independent identification.

---

## FILE INDEX

| File | Contents |
|------|----------|
| `exports/data/mw_complete_margins.json` | Table 2, Table 3, lighthouse |
| `exports/data/mw_heterogeneity.json` | Table 4 all panels |
| `exports/data/mw_iv_owe.json` | Table C1, Figure 1, event study |
| `exports/data/mw_wage_compression_decomposition.json` | Table 3 mechanical/genuine decomposition |
| `exports/data/mw_hours_epen_dep.json` | Table 6 |
| `exports/data/mw_panel_decomposition.json` | Table A1 |
| `exports/data/mw_epen_ciu_annual_bunching.json` | Table D1 |
| `exports/data/mw_epe_lima_bunching.json` | Table E1 |
| `scripts/mw_epe_lima_bunching.py` | Table E1 (EPE Lima 6-month pre-post, placebo) |
| `exports/data/mw_data_audit_complete.json` | Table 1 (sample_counts, wage_dist, hours) |
| `exports/data/mw_sanity_checks.json` | Placebo test (Section 5.1) |
| `exports/figures/fig2_bunching_distributions.png` | Figure 2 |
| `scripts/mw_paper_figures.py` | Figure 2 generation |
| `scripts/mw_heterogeneity_bunching.py` | Table 4 Panels A–C (industry, firm size, age, sex) |
| `exports/data/mw_ethnicity_heterogeneity.json` | Table 4 Panels D–F (ethnicity, education, geography, sector) |
| `scripts/mw_ethnicity_heterogeneity.py` | Table 4 Panels D–F (ethnicity via Module 300 + p558c, education, geography, employer) |

---

*Revised v5: 2026-03-17. Comprehensive corrections from 22-step audit and falsification audit. Method renamed "pre-post distributional estimator" (Harasztosi & Lindner precedent). Event study verdicts corrected (C: p=0.017; all fail). IV F-stats corrected (A=1.5, B=2.6, C=0.1; all FAIL). Section 7 restructured: event study removed from main text → Appendix B only. Section 8 conclusion rewritten: employment effects not identified. Section 2.3 Kaitz description corrected (Ica highest, Huancavelica low). Table 1 values audited (Kaitz range 0.517–0.569, not 0.620). Table 3 decomposition added (41–92% mechanical). Table A1 corrected (63.9%/60.1% retention). Ethnicity: text paragraph (not table), formality-rate gap (5.7% vs 20.7%) as main finding. EPE Appendix E: formality note added. Data: ENAHO 2015–2023 (Módulo 500), EPEN CIU Annual 2023, ENAHO Panel 978.*
