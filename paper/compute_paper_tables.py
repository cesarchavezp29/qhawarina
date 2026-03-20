#!/usr/bin/env python3
"""
compute_paper_tables.py
=======================
Generates all tables (LaTeX) and figures (PDF) for:

  "Monetary Policy Transmission in Peru:
   A Systematic Comparison of Identification Strategies"

Data sources (all numbers from these files — nothing made up):
  - estimation/full_audit_output.txt
  - exports/data/var_elasticities.json
  - exports/data/lp_elasticities.json
  - data/raw/bcrp/notas_informativas/tone_scores.csv
  - data/processed/national/panel_national_monthly.parquet

Outputs:
  - paper/tables/*.tex
  - paper/figures/*.pdf
"""

import sys, io, json, re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path('D:/Nexus/nexus')
AUDIT      = ROOT / 'estimation' / 'full_audit_output.txt'
VAR_JSON   = ROOT / 'exports' / 'data' / 'var_elasticities.json'
LP_JSON    = ROOT / 'exports' / 'data' / 'lp_elasticities.json'
TONE_CSV   = ROOT / 'data' / 'raw' / 'bcrp' / 'notas_informativas' / 'tone_scores.csv'
PANEL      = ROOT / 'data' / 'processed' / 'national' / 'panel_national_monthly.parquet'
FIG_DIR    = ROOT / 'paper' / 'figures'
TAB_DIR    = ROOT / 'paper' / 'tables'

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib style (shared module) ──────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(ROOT / 'paper'))
from paper_style import apply_style, C, SZ, zero_line, legend_below, legend_outside, stat_box
apply_style()

# Legacy color aliases for table code below
C_BLACK = C["main"]
C_DARK  = C["main"]
C_MED   = C["gray_line"]
C_LIGHT = C["ci_light"]

def savefig(fig, name):
    path = FIG_DIR / f'{name}.pdf'
    fig.savefig(path, format='pdf')
    print(f'  Saved {path.name}')
    plt.close(fig)

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...')

with open(VAR_JSON, encoding='utf-8') as f:
    var_data = json.load(f)

with open(LP_JSON, encoding='utf-8') as f:
    lp_data = json.load(f)

df_panel = pd.read_parquet(PANEL)
df_tone  = pd.read_csv(TONE_CSV, parse_dates=['date'])

# ── Load svar_results from audit (embedded JSON) ───────────────────────────────
audit_text = AUDIT.read_text(encoding='utf-8', errors='replace')

def extract_json_field(text, key):
    """Extract a JSON-like structure following a key: prefix in the audit text."""
    pattern = rf'^\s+{re.escape(key)}:\s+(\{{.*\}}|\[.*\])'
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

# Pull LP-IV data from Section 9
irf_lp_iv = extract_json_field(audit_text, 'irf_lp_iv')

# Pull sign restriction data from embedded svar_results section
sr_med   = extract_json_field(audit_text, 'irf_gdp_med')
sr_lo05  = extract_json_field(audit_text, 'irf_gdp_lo05')
sr_lo16  = extract_json_field(audit_text, 'irf_gdp_lo16')
sr_hi84  = extract_json_field(audit_text, 'irf_gdp_hi84')
sr_hi95  = extract_json_field(audit_text, 'irf_gdp_hi95')
sr_cpi_med = extract_json_field(audit_text, 'irf_cpi_med')

# ── Hard-coded audit values (from full_audit_output.txt) ──────────────────────
# Section 5: Cholesky IRF, h=0..8 (verified against audit bootstrap table)
H8 = list(range(9))

GDP_POINT = [0.000, -0.055, -0.194, -0.195, -0.172, -0.138, -0.104, -0.076, -0.054]
GDP_CI90_LO = [0.000, -1.331, -0.974, -0.698, -0.523, -0.393, -0.281, -0.207, -0.151]
GDP_CI90_HI = [0.000,  0.946,  0.424,  0.271,  0.180,  0.120,  0.082,  0.055,  0.039]
GDP_CI68_LO = [0.000, -0.998, -0.461, -0.219, -0.112, -0.056, -0.028, -0.014, -0.007]
GDP_CI68_HI = [0.000,  0.542,  0.173,  0.072,  0.030,  0.013,  0.006,  0.003,  0.001]

CPI_POINT = [0.000, 0.380, 0.270, 0.206, 0.152, 0.109, 0.077, 0.054, 0.038]
CPI_CI90_LO = [0.000, 0.022, -0.002, 0.001, -0.001, 0.000, 0.000, 0.000, 0.000]
CPI_CI90_HI = [0.000, 0.528, 0.267, 0.142, 0.079, 0.043, 0.024, 0.014, 0.008]
# Approximate 68% CI by scaling 90% CI (ratio 1/1.645 under normality)
_s = 1.0 / 1.645
CPI_CI68_LO = [CPI_POINT[h] + (CPI_CI90_LO[h] - CPI_POINT[h]) * _s for h in range(9)]
CPI_CI68_HI = [CPI_POINT[h] + (CPI_CI90_HI[h] - CPI_POINT[h]) * _s for h in range(9)]

FX_POINT = [0.000, -0.210, -0.139, -0.019, 0.038, 0.051, 0.046, 0.037, 0.028]
FX_CI90_LO = [0.000, -0.940, -0.359, -0.164, -0.079, -0.039, -0.020, -0.010, -0.005]
FX_CI90_HI = [0.000,  0.787,  0.395,  0.208,  0.105,  0.053,  0.027,  0.015,  0.008]
FX_CI68_LO = [FX_POINT[h] + (FX_CI90_LO[h] - FX_POINT[h]) * _s for h in range(9)]
FX_CI68_HI = [FX_POINT[h] + (FX_CI90_HI[h] - FX_POINT[h]) * _s for h in range(9)]

# Section 7: Narrative SR GDP response percentiles (Run 4, no dummy, 119 draws)
NARR_H = [0, 1, 2, 3, 4, 8]
NARR_P05 = [1.221, -0.893, -1.063, -0.811, -0.554, -0.080]
NARR_P50 = [1.619, -0.653, -0.711, -0.549, -0.381, -0.059]
NARR_P95 = [2.189, -0.134, -0.410, -0.369, -0.279, -0.046]
# Approximate 68% CI (P16/P84) by scaling 90% CI under normality
NARR_P16 = [NARR_P50[i] + (NARR_P05[i] - NARR_P50[i]) / 1.645 for i in range(6)]
NARR_P84 = [NARR_P50[i] + (NARR_P95[i] - NARR_P50[i]) / 1.645 for i in range(6)]

# Section 10: Poverty regression data
POV_DATA = [
    (2005, 6.282, -2.700), (2006, 7.555, -5.000), (2007, 8.470, -6.300),
    (2008, 9.185, -5.000), (2009, 1.123, -3.400), (2010, 8.283, -4.500),
    (2011, 6.380, -3.500), (2012, 6.145, -2.300), (2013, 5.827, -1.700),
    (2014, 2.453, -0.400), (2015, 3.223, -1.700), (2016, 3.975, -1.100),
    (2017, 2.515,  0.000), (2018, 3.957, -1.600), (2019, 2.250, -0.600),
    (2022, 2.857,  1.500), (2023, -0.345, 1.500), (2024, 3.473, -2.100),
]
ALPHA_POV = 0.888
BETA_POV  = -0.656
RMSE_POV  = 1.2958
N_POV     = 18
X_MEAN    = 4.645
SXX       = 126.18

BETA_POV_05_14 = -0.461;  SE_POV_05_14 = 0.179
BETA_POV_15_24 = -0.723;  SE_POV_15_24 = 0.287


# =============================================================================
#  FIGURES
# =============================================================================

# ── Figure 1: BCRP Reference Rate Path 2003–2025 ─────────────────────────────
print('\nFigure 1: BCRP Reference Rate Path...')

rate_s = df_panel[df_panel['series_id'] == 'PD04722MM'][['date', 'value_raw']].copy()
rate_s['date'] = pd.to_datetime(rate_s['date'])
rate_s = rate_s.sort_values('date')

fig, ax = plt.subplots(figsize=SZ["single"])

ax.plot(rate_s['date'], rate_s['value_raw'], color=C["main"], lw=2, zorder=3)

EPISODES = [
    ('2008-09-01', '2010-01-01', 'GFC\ncuts'),
    ('2010-04-01', '2011-09-01', 'Hiking\n2010–11'),
    ('2014-09-01', '2017-03-01', 'Gradual\ncuts'),
    ('2020-03-01', '2021-06-01', 'COVID\ncuts'),
    ('2021-07-01', '2023-03-01', 'Hiking\n2021–23'),
    ('2023-04-01', '2025-09-01', 'Easing\n2023–25'),
]
for start, end, label in EPISODES:
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    ax.axvspan(s_ts, e_ts, alpha=0.09, color=C["accent2"], zorder=0)
    mid = s_ts + (e_ts - s_ts) / 2
    ax.annotate(
        label, xy=(mid, 8.2), fontsize=7.5, ha='center', va='top',
        fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
    )

zero_line(ax)
ax.set_ylabel('Reference rate (%)')
ax.set_xlabel('')
ax.set_xlim(pd.Timestamp('2003-01-01'), pd.Timestamp('2026-01-01'))
ax.set_ylim(-0.3, 9.2)
ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

savefig(fig, 'fig1_bcrp_rate_path')


# ── Figure 2: Cholesky IRFs — Rate Shock (3 vertical panels) ─────────────────
print('Figure 2: Cholesky IRFs...')

irf_specs = [
    ('(a) GDP response (pp)', GDP_POINT, GDP_CI90_LO, GDP_CI90_HI, GDP_CI68_LO, GDP_CI68_HI),
    ('(b) CPI response (pp)', CPI_POINT, CPI_CI90_LO, CPI_CI90_HI, CPI_CI68_LO, CPI_CI68_HI),
    ('(c) FX response (pp)',  FX_POINT,  FX_CI90_LO,  FX_CI90_HI,  FX_CI68_LO,  FX_CI68_HI),
]

fig, axes = plt.subplots(3, 1, figsize=(5.5, 6.5), sharex=True,
                          gridspec_kw={'hspace': 0.12})

for i, (ax, (title, point, lo90, hi90, lo68, hi68)) in enumerate(zip(axes, irf_specs)):
    h90 = ax.fill_between(H8, lo90, hi90, alpha=0.18, color=C["ci_light"],
                          edgecolor=C["ci_dark"], linewidth=0.4, label='90% CI')
    h68 = ax.fill_between(H8, lo68, hi68, alpha=0.30, color=C["ci_dark"],
                          edgecolor='none', label='68% CI')
    ax.plot(H8, point, color=C["main"], lw=2, zorder=3)
    zero_line(ax)
    ax.set_ylabel(title.split(') ')[1], labelpad=4)
    ax.text(0.03, 0.95, title.split(')')[0] + ')', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')
    ax.set_xticks(range(0, 9))
    ax.set_xlim(-0.3, 8.3)
    ax.legend(handles=[h90, h68], loc='lower right', fontsize=8, ncol=2)

axes[-1].set_xlabel('Horizon (quarters)')
savefig(fig, 'fig2_cholesky_irfs')


# ── Figure 3: Sign Restriction Identified Set ─────────────────────────────────
print('Figure 3: Sign restriction identified set...')

if sr_med and len(sr_med) >= 9:
    h_sr = list(range(9))
    med  = sr_med[:9]
    lo05 = sr_lo05[:9] if sr_lo05 else None
    lo16 = sr_lo16[:9] if sr_lo16 else None
    hi84 = sr_hi84[:9] if sr_hi84 else None
    hi95 = sr_hi95[:9] if sr_hi95 else None
else:
    h_sr = [0, 1, 2, 3, 4, 8]
    med  = [-2.368, -1.492, -0.877, -0.497, -0.290, -0.049]
    lo05 = [-34.20, -13.53, -7.36,  -3.37,  -1.53,  -0.19]
    lo16 = [-12.75, -4.79,  -2.66,  -1.31,  -0.65,  -0.10]
    hi84 = [5.44,   -0.48,  -0.33,  -0.23,  -0.14,  0.012]
    hi95 = [19.71,  -0.17,  -0.14,  -0.12,  -0.03,  0.132]

fig, ax = plt.subplots(figsize=SZ["single"])

if lo05 and hi95:
    _ymin, _ymax = -15, 8
    lo05_vis = [max(v, _ymin) for v in lo05]
    hi95_vis = [min(v, _ymax) for v in hi95]
    ax.fill_between(h_sr, lo05_vis, hi95_vis, alpha=0.25, color=C["ci_light"],
                    edgecolor=C["ci_dark"], linewidth=0.8, label='90% credible set')
if lo16 and hi84:
    ax.fill_between(h_sr, lo16, hi84, alpha=0.50, color=C["ci_dark"],
                    edgecolor='none', label='68% credible set')
ax.plot(h_sr, med, color=C["main"], lw=2, label='Median', zorder=3)
ax.plot(H8, GDP_POINT, color=C["accent1"], lw=2.5, ls='--', zorder=4, label='Cholesky ($-$0.195\u2009pp)')
zero_line(ax)

ax.set_xlabel('Horizon (quarters)')
ax.set_ylabel('GDP response (pp)')
ax.set_xlim(-0.3, max(h_sr) + 0.3)
ax.set_xticks(h_sr)
ax.set_ylim(-15, 8)
# Annotation arrow pointing to Cholesky line at h=3
ax.annotate('$-$0.195\u2009pp\n(Cholesky)',
            xy=(3, GDP_POINT[3]), xytext=(5.5, 3.5),
            fontsize=7.5, color=C["accent1"],
            arrowprops=dict(arrowstyle='->', color=C["accent1"], lw=1.2),
            ha='center')
stat_box(ax, '80,000 draws · 63,743 accepted (79.7%)', loc='upper right', fontsize=7.5)
legend_below(ax, ncol=2)
savefig(fig, 'fig3_sign_restriction_set')


# ── Figure 4: Narrative SR IRFs vs Cholesky ───────────────────────────────────
print('Figure 4: Narrative SR IRFs...')

fig, ax = plt.subplots(figsize=SZ["single"])

ax.fill_between(NARR_H, NARR_P05, NARR_P95, alpha=0.18, color=C["ci_light"],
                edgecolor=C["ci_dark"], linewidth=0.5, label='90% credible set')
ax.fill_between(NARR_H, NARR_P16, NARR_P84, alpha=0.40, color=C["ci_dark"],
                edgecolor='none', label='68% credible set')
ax.plot(NARR_H, NARR_P50, color=C["main"], lw=2, label='Median (narrative SR)', zorder=3)
ax.plot(H8, GDP_POINT, color=C["accent1"], lw=2.0, ls='--', label='Cholesky point', zorder=2)
zero_line(ax)

ax.set_xlabel('Horizon (quarters)')
ax.set_ylabel('GDP response (pp)')
ax.set_xlim(-0.3, 8.3)
ax.set_xticks(range(9))
stat_box(ax, '119 draws: too few for reliable inference\n(2020Q1\u2013Q2 + 2022Q1\u2013Q4 restrictions)',
         loc='upper right', fontsize=7.5)
legend_below(ax, ncol=2)
savefig(fig, 'fig4_narrative_sr_irfs')


# ── Figure 5: LP-IV with Tone Instrument — Wrong-Sign IRFs ───────────────────
print('Figure 5: LP-IV wrong-sign IRFs...')

if irf_lp_iv:
    lp_beta = irf_lp_iv['gdp']['beta']
    lp_se   = irf_lp_iv['gdp']['se']
    h_lp    = list(range(len(lp_beta)))
else:
    lp_beta = [4.319, 7.359, 5.071, 2.158, 4.213, 6.884, 6.017, 5.172,
               6.625, 5.426, 5.235, 4.113, 4.736, 4.242, 3.751, 4.206,
               4.094, 3.338, 2.886, 3.310, 4.218, 3.936, 2.729, -0.704, 2.643]
    lp_se   = [1.579, 2.882, 2.744, 2.838, 2.884, 4.146, 3.445, 2.664,
               3.610, 3.242, 2.527, 2.607, 2.616, 2.520, 2.476, 2.682,
               2.565, 2.431, 2.470, 2.846, 3.232, 2.984, 3.065, 3.422, 4.270]
    h_lp = list(range(25))

lp_beta = np.array(lp_beta)
lp_se   = np.array(lp_se)
lp_lo90 = lp_beta - 1.645 * lp_se
lp_hi90 = lp_beta + 1.645 * lp_se

mask    = np.array(h_lp) <= 8
hs_plot = np.array(h_lp)[mask]
b_plot  = lp_beta[mask]
lo_plot = lp_lo90[mask]
hi_plot = lp_hi90[mask]

fig, ax = plt.subplots(figsize=SZ["single"])

# Wrong-sign shading above zero
ax.axhspan(0, max(hi_plot) + 2, color=C["accent1"], alpha=0.06, zorder=0)
ax.annotate('Wrong sign (expansionary)', xy=(0.03, 0.88),
            xycoords='axes fraction', fontsize=8, color=C["accent1"],
            fontstyle='italic')
zero_line(ax)

ax.errorbar(hs_plot, b_plot,
            yerr=[b_plot - lo_plot, hi_plot - b_plot],
            fmt='o', color=C["main"], markersize=6,
            ecolor=C["gray_line"], elinewidth=1.5, capsize=4, capthick=1.2,
            zorder=3, label='Point estimate ± 90% CI')

ax.set_xlabel('Horizon (quarters)')
ax.set_ylabel('GDP response (pp)')
ax.set_xlim(-0.5, 8.5)
ax.set_xticks(range(9))
ax.set_ylim(min(lo_plot) - 1, max(hi_plot) + 2)
ax.legend(loc='lower right', fontsize=8)
savefig(fig, 'fig5_lpiv_tone_wrong_sign')


# ── Figure 6: BCRP Tone Time Series ──────────────────────────────────────────
print('Figure 6: Tone time series...')

df_tone['date'] = pd.to_datetime(df_tone['date'])
tone_m = (df_tone.groupby('date')[['dict_tone', 'llm_tone']]
          .mean().reset_index().sort_values('date'))
tone_m = tone_m[tone_m['date'] >= '2001-01-01']

hiking  = [('2010-04-01', '2011-09-01'), ('2021-07-01', '2023-03-01')]
cutting = [('2008-09-01', '2010-03-01'), ('2014-09-01', '2017-03-01'),
           ('2020-03-01', '2021-06-01'), ('2023-04-01', '2026-01-01')]

from matplotlib.patches import Patch as _Patch

fig, axes = plt.subplots(2, 1, figsize=SZ["wide_tall"], sharex=True,
                          gridspec_kw={'hspace': 0.08})

for ax, col, ylabel, ylims in zip(
    axes,
    ['dict_tone', 'llm_tone'],
    ['Dictionary tone', 'LLM tone'],
    [(-1.15, 1.15), (-85, 85)]
):
    for s, e in hiking:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.12, color=C["accent1"], zorder=0)
    for s, e in cutting:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.10, color=C["accent2"], zorder=0)
    zero_line(ax)
    ax.plot(tone_m['date'], tone_m[col], color=C["main"], lw=0.9, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylims)
    ax.set_xlim(pd.Timestamp('2001-01-01'), pd.Timestamp('2026-06-01'))

legend_elements = [
    _Patch(facecolor=C["accent1"], alpha=0.40, label='Hiking episode'),
    _Patch(facecolor=C["accent2"], alpha=0.40, label='Cutting episode'),
]
axes[1].xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0, ha='center')
legend_below(axes[1], ncol=2, handles=legend_elements)
savefig(fig, 'fig6_bcrp_tone_series')


# ── Figure 7: GDP–Poverty Scatter ─────────────────────────────────────────────
print('Figure 7: GDP-Poverty scatter...')

years, gdps, dpovs = zip(*POV_DATA)
gdps   = np.array(gdps)
dpovs  = np.array(dpovs)
outliers = {2009, 2022}
outlier_mask = np.array([yr in outliers for yr in years])

x_line  = np.linspace(-2.5, 11.0, 200)
y_line  = ALPHA_POV + BETA_POV * x_line
se_pred = RMSE_POV * np.sqrt(1/N_POV + (x_line - X_MEAN)**2 / SXX)
ci_lo   = y_line - 1.645 * se_pred
ci_hi   = y_line + 1.645 * se_pred

fig, ax = plt.subplots(figsize=SZ["single_sq"])

ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.15, color=C["ci_light"],
                edgecolor=C["ci_dark"], linewidth=0.5, label='90% prediction interval')
ax.plot(x_line, y_line, color=C["accent1"], lw=2, zorder=2, label='OLS fit')
zero_line(ax)

# Regular points
ax.scatter(gdps[~outlier_mask], dpovs[~outlier_mask],
           color=C["main"], s=40, zorder=3, edgecolors='white', linewidths=0.5)
# Outlier points — open circles
ax.scatter(gdps[outlier_mask], dpovs[outlier_mask],
           color='none', edgecolors=C["accent1"], s=60, lw=1.5, zorder=3)

for yr, g, dp in POV_DATA:
    if yr in outliers:
        ax.annotate(str(yr), xy=(g, dp), xytext=(8, 4),
                    textcoords='offset points', fontsize=8, fontstyle='italic',
                    arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5))

ax.set_xlabel('Annual GDP growth (%)')
ax.set_ylabel('\u0394 Poverty rate (pp)')
ax.set_xlim(-2.0, 10.8)
ax.set_ylim(-7.5, 3.5)
stat_box(ax, r'$\hat{\beta}=-0.656$   $R^2=0.669$   $N=18$', loc='lower left')
legend_below(ax, ncol=2)
savefig(fig, 'fig7_gdp_poverty_scatter')


# ── Figure 8: GDP–Poverty Sub-Period Comparison ───────────────────────────────
print('Figure 8: Sub-period poverty regression...')

early = [(yr, g, dp) for yr, g, dp in POV_DATA if 2005 <= yr <= 2014]
late  = [(yr, g, dp) for yr, g, dp in POV_DATA if 2015 <= yr <= 2024]

def subperiod_ols_full(data):
    """Return alpha, beta, rmse, xbar, sxx, n for OLS regression."""
    gdp_v = np.array([d[1] for d in data])
    pov_v = np.array([d[2] for d in data])
    n = len(gdp_v)
    beta, alpha = np.polyfit(gdp_v, pov_v, 1)
    y_pred = alpha + beta * gdp_v
    rmse = np.sqrt(np.sum((pov_v - y_pred) ** 2) / (n - 2))
    xbar = gdp_v.mean()
    sxx  = np.sum((gdp_v - xbar) ** 2)
    return alpha, beta, rmse, xbar, sxx, n

alpha_e, beta_e, rmse_e, xbar_e, sxx_e, n_e = subperiod_ols_full(early)
alpha_l, beta_l, rmse_l, xbar_l, sxx_l, n_l = subperiod_ols_full(late)

x_e = np.linspace(0.0, 10.5, 100)
x_l = np.linspace(-1.0, 5.0, 100)

# 90% confidence bands
se_e   = rmse_e * np.sqrt(1/n_e + (x_e - xbar_e)**2 / sxx_e)
ci_e_lo = (alpha_e + beta_e * x_e) - 1.645 * se_e
ci_e_hi = (alpha_e + beta_e * x_e) + 1.645 * se_e

se_l   = rmse_l * np.sqrt(1/n_l + (x_l - xbar_l)**2 / sxx_l)
ci_l_lo = (alpha_l + beta_l * x_l) - 1.645 * se_l
ci_l_hi = (alpha_l + beta_l * x_l) + 1.645 * se_l

fig, ax = plt.subplots(figsize=SZ["single_sq"])

# Early sub-period (2005–2014): blue
gdp_e = [d[1] for d in early]; pov_e = [d[2] for d in early]
ax.fill_between(x_e, ci_e_lo, ci_e_hi, alpha=0.10, color=C["accent2"])
ax.plot(x_e, alpha_e + beta_e * x_e, color=C["accent2"], lw=1.5,
        label=rf'2005–2014 ($\hat{{\beta}}={beta_e:.3f}$)')
ax.scatter(gdp_e, pov_e, marker='s', s=40, color=C["accent2"],
           edgecolors='white', linewidths=0.5, zorder=3)

# Late sub-period (2015–2024): red
gdp_l = [d[1] for d in late]; pov_l = [d[2] for d in late]
ax.fill_between(x_l, ci_l_lo, ci_l_hi, alpha=0.10, color=C["accent1"])
ax.plot(x_l, alpha_l + beta_l * x_l, color=C["accent1"], lw=1.5,
        label=rf'2015–2024 ($\hat{{\beta}}={beta_l:.3f}$)')
ax.scatter(gdp_l, pov_l, marker='D', s=40, color=C["accent1"],
           edgecolors='white', linewidths=0.5, zorder=3)

zero_line(ax)
stat_box(ax, r'Chow test: $t = 0.78$, $p \approx 0.45$', loc='upper right')

ax.set_xlabel('Annual GDP growth (%)')
ax.set_ylabel(r'$\Delta$ Poverty rate (pp)')
ax.set_xlim(-2.0, 10.8)
ax.set_ylim(-7.5, 3.5)
legend_below(ax, ncol=2)
savefig(fig, 'fig8_gdp_poverty_subperiod')


# ── Figure 9: Forest Plot — All Strategies ────────────────────────────────────
print('Figure 9: Forest plot...')

# (label, point, ci_lo, ci_hi, is_literature)
# ci_lo/ci_hi = None → no CI whisker; point = None → not identified
FOREST_OWN = [
    ('Cholesky VAR(1)',          -0.195,  -0.698,  0.271,  False),
    ('BVAR (Minnesota)',         -0.209,  -0.653, -0.032,  False),
    ('GIRF',                     -0.917,  -3.510, -0.130,  False),
    ('Narrative SR (full)',      -0.742,  -1.063, -0.410,  False),
    ('Narrative SR (2022 only)', -3.575,  -30.0,  -0.290,  False),
    ('Sign restrictions',        -2.299,  -35.87,  19.71,  False),
    ('LP (endogenous)',          -0.541,  -0.964, -0.117,  False),
    ('Proxy-SVAR: IB rate',       None,    None,   None,   False),
    ('Proxy-SVAR: tone (A)',       None,    None,   None,   False),
    ('Proxy-SVAR: tone (B)',       None,    None,   None,   False),
    ('Taylor-rule IV',             None,    None,   None,   False),
]
FOREST_LIT = [
    ('Pérez Rojo & Rdz. (2024)',  -0.28, None, None, True),
    ('Castillo et al. (2016)',    -0.30, None, None, True),
    ('Portilla et al. (2022)',    -0.25, None, None, True),
]
ALL_FOREST = FOREST_OWN + FOREST_LIT
N_F = len(ALL_FOREST)

# y-positions: top-to-bottom, with a gap between own and literature
n_own = len(FOREST_OWN)
y_own = list(range(n_own - 1, -1, -1))          # n_own-1 … 0
y_lit = [y - 1.8 for y in range(-1, -1 - len(FOREST_LIT), -1)]  # −1.8, −2.8, −3.8

all_y = y_own + y_lit

XLIM_LO = -10.5
XLIM_HI =  3.2

fig, ax = plt.subplots(figsize=SZ["forest"])

# ── Reference bands ──────────────────────────────────────────────────────────
ax.axvspan(-0.30, -0.25, alpha=0.15, color=C["accent2"], zorder=0,
           label='Literature range [−0.30, −0.25]')
ax.axvspan(-0.29, -0.13, alpha=0.15, color=C["accent3"], zorder=0,
           label='Cholesky robustness [−0.29, −0.13]')
ax.axvline(0, color=C["gray_line"], lw=0.7, ls=':', zorder=1)

# ── Divider between own and literature ────────────────────────────────────────
divider_y = (y_own[-1] + y_lit[0]) / 2
ax.axhline(divider_y, color=C["ci_light"], lw=0.8)
ax.annotate('Literature estimates ↓',
            xy=(0.98, divider_y), xycoords=('axes fraction', 'data'),
            fontsize=7.5, ha='right', va='top', color=C["gray_line"],
            fontstyle='italic', annotation_clip=False)

# ── Plot entries ──────────────────────────────────────────────────────────────
for (label, point, lo, hi, is_lit), y in zip(ALL_FOREST, all_y):
    if point is None:
        # Not identified: shaded row, × marker, italic label
        ax.axhspan(y - 0.42, y + 0.42, color='#f2f2f2', zorder=0)
        ax.plot(XLIM_LO + 0.5, y, 'x', color=C["gray_line"], ms=9, mew=2, zorder=3)
        ax.annotate('not identified',
                    xy=(XLIM_LO + 1.2, y), fontsize=7.5,
                    color=C["gray_line"], va='center', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#f2f2f2', ec='none'))
    else:
        if lo is not None and hi is not None:
            lo_clip = max(lo, XLIM_LO + 0.2)
            hi_clip = min(hi, XLIM_HI - 0.1)
            ax.plot([lo_clip, hi_clip], [y, y],
                    color=C["main"] if not is_lit else C["accent2"],
                    lw=2.0, solid_capstyle='butt', zorder=2)
            # Truncation arrows if CI extends beyond plot limits
            if lo < XLIM_LO + 0.2:
                ax.annotate('', xy=(XLIM_LO + 0.05, y),
                            xytext=(XLIM_LO + 0.9, y),
                            arrowprops=dict(arrowstyle='<-',
                                            color=C["main"], lw=1.5,
                                            shrinkA=0, shrinkB=0))
                ax.annotate(f'{lo:.1f}',
                            xy=(XLIM_LO + 0.0, y), xytext=(-4, 0),
                            textcoords='offset points',
                            fontsize=7, ha='right', va='center',
                            color=C["main"])
            if hi > XLIM_HI - 0.1:
                ax.annotate('', xy=(XLIM_HI - 0.05, y),
                            xytext=(XLIM_HI - 0.9, y),
                            arrowprops=dict(arrowstyle='<-',
                                            color=C["main"], lw=1.5,
                                            shrinkA=0, shrinkB=0))

        if is_lit:
            # Literature: open diamonds in accent2
            ax.plot(point, y, 'D', color=C["accent2"], ms=7,
                    markerfacecolor='white', markeredgewidth=1.8, zorder=4)
        else:
            # Own: filled circles in main navy
            ax.plot(point, y, 'o', color=C["main"], ms=7,
                    markerfacecolor=C["main"], markeredgecolor='white',
                    markeredgewidth=0.8, zorder=4)

ax.set_yticks(all_y)
ax.set_yticklabels([r[0] for r in ALL_FOREST], fontsize=8.5)
ax.set_xlabel('Peak GDP response to 100 bp rate hike (pp)', labelpad=6)
ax.set_xlim(XLIM_LO, XLIM_HI)
ax.set_ylim(y_lit[-1] - 0.8, y_own[0] + 0.8)

legend_below(ax, ncol=2)
savefig(fig, 'fig9_forest_plot')


# =============================================================================
#  TABLES (LaTeX)
# =============================================================================

def write_tex(name, content):
    path = TAB_DIR / f'{name}.tex'
    path.write_text(content, encoding='utf-8')
    print(f'  Saved {path.name}')


# ── Table 1: Summary Statistics ───────────────────────────────────────────────
print('\nTable 1: Summary statistics...')
# From audit Section 0 and unit root tests
summary_rows = [
    ('GDP (qoq, pp)',        0.013, 2.810, -10.890,  4.851, '-4.80', '0.000'),
    ('CPI (qoq, pp)',        0.527, 0.574,  -0.680,  2.510, '-5.66', '0.000'),
    ('FX (qoq, $\\%\\Delta$)', 0.085, 2.104,  -6.502,  5.900, '-6.50', '0.000'),
    ('Rate ($\\Delta$ pp)',   0.009, 0.503,  -1.250,  1.250, '-5.90', '0.000'),
    ('ToT (qoq, pp)',        0.170, 4.050, -16.010,  9.220, '-7.51', '0.000'),
]
tex_rows = '\n'.join(
    f'    {var} & {mu:.3f} & {sd:.3f} & {mn:.3f} & {mx:.3f} & {adf} & {p} \\\\'
    for var, mu, sd, mn, mx, adf, p in summary_rows
)
write_tex('tab1_summary_stats', rf"""
\begin{{tabular}}{{lrrrrcc}}
\toprule
Variable & Mean & SD & Min & Max & ADF stat & ADF $p$ \\
\midrule
{tex_rows}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small Sample: 2004Q2--2025Q3 ($T=85$). ADF with intercept; all series stationary.
COVID FWL: 2020Q1--Q2 partialled out in estimation.
\end{{tablenotes}}
""")


# ── Table 2: BCRP Tone Summary ────────────────────────────────────────────────
print('Table 2: Tone summary...')
tone_by_year = df_tone.groupby('year').agg(
    n=('llm_tone', 'count'),
    dict_mean=('dict_tone', 'mean'),
    dict_sd=('dict_tone', 'std'),
    llm_mean=('llm_tone', 'mean'),
    llm_sd=('llm_tone', 'std'),
).reset_index()
tone_by_year = tone_by_year[(tone_by_year['year'] >= 2001) &
                             (tone_by_year['year'] <= 2025)]

tone_tex_rows = '\n'.join(
    f'    {int(r.year)} & {int(r.n)} & {r.dict_mean:.2f} & ({r.dict_sd:.2f}) '
    f'& {r.llm_mean:.1f} & ({r.llm_sd:.1f}) \\\\'
    for r in tone_by_year.itertuples()
)
write_tex('tab2_tone_summary', rf"""
\begin{{tabular}}{{lrcccc}}
\toprule
Year & $N$ & Dict. mean & (SD) & LLM mean & (SD) \\
\midrule
{tone_tex_rows}
\midrule
    Overall & {int(tone_by_year['n'].sum())} &
    {tone_by_year['dict_mean'].mean():.3f} & ({tone_by_year['dict_sd'].mean():.3f}) &
    {tone_by_year['llm_mean'].mean():.1f} & ({tone_by_year['llm_sd'].mean():.1f}) \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small LLM: Claude Haiku, score $-100$ (dovish) to $+100$ (hawkish).
Dict.: dictionary method (Lahura \& Vega 2020 adaptation).
\end{{tablenotes}}
""")


# ── Table 3: Poverty and GDP Data ─────────────────────────────────────────────
print('Table 3: Poverty data...')
def _dpov_fmt(dp):
    if dp == 0.0:
        return r'$\phantom{-}0.0$'
    elif dp > 0:
        return f'${dp:.1f}$'
    else:
        return f'${dp:.1f}$'

def _gdp_fmt(g):
    if g < 0:
        return f'$-{abs(g):.3f}$'
    return f'{g:.3f}'

pov_tex_rows = '\n'.join(
    f'    {yr} & {_gdp_fmt(g)} & {_dpov_fmt(dp)} \\\\'
    for yr, g, dp in POV_DATA
)
write_tex('tab3_poverty_data', rf"""
\begin{{tabular}}{{lrr}}
\toprule
Year & GDP growth (\%) & $\Delta$Poverty (pp) \\
\midrule
{pov_tex_rows}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small GDP: BCRP national accounts, annual average YoY real growth.
Poverty: INEI-ENAHO monetary poverty rate. 2020--2021 excluded (COVID).
2022 and 2023 reflect K-shaped recovery and post-pandemic return to normal.
\end{{tablenotes}}
""")


# ── Table 4: VAR Coefficient Matrix ──────────────────────────────────────────
print('Table 4: VAR coefficient matrix...')
# From audit Section 4, A_1 matrix row by row
# Columns: tot_t-1, gdp_t-1, cpi_t-1, fx_t-1, rate_t-1
var_coef = {
    'ToT':  [0.498, -0.005,  0.151, -0.028,  0.026],
    'GDP':  [-0.025, -0.264,  0.386,  0.028, -0.055],
    'CPI':  [0.001,  0.021,  0.440,  0.018,  0.065],
    'FX':   [0.002, -0.040, -0.105,  0.332, -0.110],
    'Rate': [0.001,  0.011,  0.025, -0.012,  0.614],
}
hdr = '\\thead{ToT$_{t-1}$} & \\thead{GDP$_{t-1}$} & \\thead{CPI$_{t-1}$} & \\thead{FX$_{t-1}$} & \\thead{Rate$_{t-1}$}'
tab4_rows = '\n'.join(
    f'    {eq} & ' + ' & '.join(f'{v:.3f}' for v in coeffs) + ' \\\\'
    for eq, coeffs in var_coef.items()
)
write_tex('tab4_var_coefs', rf"""
\begin{{tabular}}{{l{'r'*5}}}
\toprule
Equation & {hdr} \\
\midrule
{tab4_rows}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small VAR(1), $T=85$, FWL COVID partial-out. Full SEs available upon request.
Ordering: [ToT, GDP, CPI, FX, Rate].
\end{{tablenotes}}
""")


# ── Table 5: Narrative SR Comparison ─────────────────────────────────────────
print('Table 5: Narrative SR comparison...')
write_tex('tab5_narrative_sr', r"""
\begin{tabular}{lrrcc}
\toprule
Specification & Draws & Accepted & Peak GDP (p50) & 68\% CI \\
\midrule
    Sign restr.\ only & 80,000 & 63,743 & $-2.30$ & $[-12.8,\;5.4]$ \\
    Narr.\ SR: 2020+2022, FWL & 80,000 & $\approx4$ & --- & --- \\
    Narr.\ SR: 2022 only, FWL & 80,000 & 1,728 & $-3.57$ & $[-30.0,\;-0.29]$ \\
    Narr.\ SR: 2020+2022, no FWL & 80,000 & 119 & $-0.74$ & $[-1.06,\;-0.38]$ \\
    Narr.\ SR: 2020 only, no FWL & 80,000 & 4,549 & $-0.90$ & --- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small Algorithm: Arias, Rubio-Ram\'{i}rez \& Waggoner (2018) rejection sampling.
Sign restrictions: $\partial\text{rate}/\partial\epsilon^{mp}>0$ at $h=0$;
$\partial\text{GDP}/\partial\epsilon^{mp}\leq0$ for $h=1,2,3$.
FWL: COVID Q1+Q2 2020 partialled out. Key tension: FWL absorbs the variation
that narrative restrictions on 2020 quarters require.
\end{tablenotes}
""")


# ── Table 6: LP Results ───────────────────────────────────────────────────────
print('Table 6: LP results...')
# From audit Section 8
lp_rows = [
    (0,  0.616,  1.604,  0.384, 0.701, 0.639, 84),
    (1,  0.647,  1.490,  0.435, 0.664, 0.507, 83),
    (2,  0.393,  1.231,  0.320, 0.749, 0.535, 82),
    (3, -0.509,  1.178, -0.432, 0.666, 0.537, 81),
    (4, -1.334,  1.287, -1.036, 0.300, 0.514, 80),
    (5, -1.123,  1.251, -0.898, 0.369, 0.496, 79),
    (6, -1.754,  1.216, -1.442, 0.149, 0.465, 78),
    (7, -2.150,  1.332, -1.614, 0.107, 0.427, 77),
    (8, -1.074,  1.579, -0.681, 0.496, 0.409, 76),
]
lp_tex_rows = '\n'.join(
    f'    {h} & {b:.3f} & {se:.3f} & {t:.3f} & {p:.3f} & {r2:.3f} & {n} \\\\'
    for h, b, se, t, p, r2, n in lp_rows
)
write_tex('tab6_lp_results', rf"""
\begin{{tabular}}{{rrrrrrr}}
\toprule
$h$ & $\hat{{\beta}}_h$ & SE (HC3) & $t$-stat & $p$-value & $R^2$ & $N$ \\
\midrule
{lp_tex_rows}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small LP specification: $y_{{t+h}}-y_{{t-1}}=\alpha_h + \beta_h \Delta r_t
+ \gamma_{{1h}} y_{{t-1}} + \gamma_{{2h}} y_{{t-2}}
+ \delta_{{1h}} \Delta r_{{t-1}} + \delta_{{2h}} \Delta r_{{t-2}}
+ \lambda \text{{covid}}_t + \varepsilon_{{t,h}}$.
SE: HC3 (White sandwich, leverage-corrected). Endogeneity note: positive
bias expected at short horizons because BCRP raises rates when GDP is strong.
\end{{tablenotes}}
""")


# ── Table 7: Proxy-SVAR Feasibility (Interbank Rate) ─────────────────────────
print('Table 7: Proxy-SVAR interbank...')
write_tex('tab7_proxy_svar_ib', r"""
\begin{tabular}{lc}
\toprule
Diagnostic & Value \\
\midrule
    Rate-change events (2004--2025) & 72 \\
    Interbank surprise: mean (pp) & $0.040$ \\
    Interbank surprise: SD (pp) & $0.640$ \\
    Correlation surprise--reference rate change & $0.614$ \\
    First-stage $F$-statistic (AR form) & $4.73$ \\
    First-stage $F$-statistic (VAR form) & $4.73$ \\
    Stock--Yogo threshold ($F>10$) & \textbf{Fails} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small Instrument: $z_t = \text{interbank}_{d+1} - \text{interbank}_{d-1}$
around announcement date $d$, aggregated to quarters.
Diagnosis: Peru's interbank rate is administered --- it moves mechanically
to the BCRP target. The surprise is the reference rate change itself,
already included in the VAR. The instrument is endogenous and lacks relevance.
\end{tablenotes}
""")


# ── Table 8: Tone Classification Summary ──────────────────────────────────────
print('Table 8: Tone classification...')
n_total = len(df_tone)
write_tex('tab8_tone_classification', rf"""
\begin{{tabular}}{{lcc}}
\toprule
Statistic & Dictionary & LLM (Claude Haiku) \\
\midrule
    Corpus size (notes) & \multicolumn{{2}}{{c}}{{{n_total}}} \\
    Date range & \multicolumn{{2}}{{c}}{{2001--2026}} \\
    Mean score & {df_tone['dict_tone'].mean():.3f} & {df_tone['llm_tone'].mean():.1f} \\
    SD & {df_tone['dict_tone'].std():.3f} & {df_tone['llm_tone'].std():.1f} \\
    Min & {df_tone['dict_tone'].min():.1f} & {df_tone['llm_tone'].min():.0f} \\
    Max & {df_tone['dict_tone'].max():.1f} & {df_tone['llm_tone'].max():.0f} \\
    Correlation (dict vs LLM) & \multicolumn{{2}}{{c}}{{{df_tone['dict_tone'].corr(df_tone['llm_tone']):.3f}}} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small Dictionary: hawkish/dovish lexicon adapted from Lahura \& Vega (2020 CEMLA).
LLM: Claude Haiku (\texttt{{claude-haiku-4-5-20251001}}), prompt scored each
Nota Informativa on $[-100, +100]$ with chain-of-thought rationale.
\end{{tablenotes}}
""")


# ── Table 9: Tone Instrument Validity ─────────────────────────────────────────
print('Table 9: Tone instrument validity...')
# From audit Section 9
write_tex('tab9_tone_instrument', r"""
\begin{tabular}{lcc}
\toprule
& Construction A & Construction B \\
& (residualize on $\Delta r_t$ + lags) & (residualize on lags only) \\
\midrule
    First-stage $F$ & $4.75$ & $33.7$ \\
    Stock--Yogo threshold & \textbf{Fails} & \textbf{Passes} \\
    GDP response $h=0$ & --- & $+4.32$ (wrong sign) \\
    GDP response $h=1$ & --- & $+7.36$ (wrong sign) \\
    GDP responses $h=0..12$ & --- & All positive \\
    Diagnosis & Lacks relevance & Fails exogeneity \\
\midrule
    Conclusion & \multicolumn{2}{c}{Instrument not valid for identification} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small Construction A: After projecting out the rate decision ($\Delta r_t$),
residual tone has near-zero information about monetary shocks.
Construction B: Tone predicts the rate decision itself (high $F$) but this
represents anticipated policy, violating the exogeneity condition
$E[z_t \varepsilon^j_t]=0$ for $j\neq mp$.
\end{tablenotes}
""")


# ── Table 10: Poverty OLS Full Output ─────────────────────────────────────────
print('Table 10: Poverty OLS...')
write_tex('tab10_poverty_ols', r"""
\begin{tabular}{lrrrrc}
\toprule
Variable & Coeff. & SE & $t$ & $p$ & 95\% CI \\
\midrule
    Constant & $0.888$ & $0.617$ & $1.44$ & $0.169$ & $[-0.385,\;2.161]$ \\
    GDP growth & $-0.656$ & $0.115$ & $-5.69$ & $<0.001$ & $[-0.895,\;-0.417]$ \\
\midrule
    $R^2$ & \multicolumn{5}{l}{$0.669$} \\
    Adj.\ $R^2$ & \multicolumn{5}{l}{$0.648$} \\
    RMSE & \multicolumn{5}{l}{$1.296$ pp} \\
    $N$ & \multicolumn{5}{l}{$18$ (ENAHO 2005--2024, excl.\ 2020--2021)} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small OLS. Dependent variable: annual change in monetary poverty rate (pp).
Independent variable: annual mean of quarterly YoY real GDP growth (\%).
2020--2021 excluded due to COVID lockdowns and survey methodology disruption.
Heteroskedasticity-robust inference (HC3) gives $p<0.001$ for GDP coefficient.
\end{tablenotes}
""")


# ── Table 11: Sub-Period Stability ────────────────────────────────────────────
print('Table 11: Sub-period stability...')
write_tex('tab11_subperiod_stability', r"""
\begin{tabular}{lrrrrrr}
\toprule
Period & $\hat{\beta}$ & SE & $t$ & $p$ & $R^2$ & $N$ \\
\midrule
    Pooled, 2005--2024 & $-0.656$ & $0.115$ & $-5.69$ & $<0.001$ & $0.669$ & 18 \\
    2005--2014 (expansion) & $-0.461$ & $0.179$ & $-2.57$ & $0.033$ & $0.452$ & 10 \\
    2015--2024 excl.\ COVID & $-0.723$ & $0.287$ & $-2.52$ & $0.045$ & $0.514$ & 8 \\
    Chow test ($H_0$: equal slopes) & \multicolumn{3}{l}{$t=0.78$, $p\approx0.45$} &
        \multicolumn{3}{l}{Cannot reject stability} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small Sign is consistently negative across sub-periods. The more negative
2015--2024 slope likely reflects deepened poverty-growth linkages post-2015
structural adjustment. Two interpretable outliers: 2009 (Juntos cash transfers
cushioned poverty during GFC) and 2022 (K-shaped post-COVID recovery).
\end{tablenotes}
""")


# ── Table 12: Master Comparison ───────────────────────────────────────────────
print('Table 12: Master comparison...')
write_tex('tab12_master_comparison', r"""
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc>{\raggedright\arraybackslash}p{2.8cm}}
\toprule
Method & Type & Peak GDP (pp) & 90\% CI & Feasible & Reason \\
\midrule
\multicolumn{6}{l}{\textit{Own estimates, this paper}} \\
    Cholesky VAR(1) & Recursive & $-0.195$ & $[-0.70, 0.27]$ & \cmark & Cholesky timing assumption \\
    Sign restrictions & Set ID & $-2.30$ & $[-35.9, 19.7]$ & \warnmark & Set too wide \\
    Narrative SR (full) & Set ID & $-0.74$ & $[-1.06,-0.41]$ & \warnmark & COVID tension \\
    Narrative SR (2022) & Set ID & $-3.57$ & $[-30.0,-0.29]$ & \warnmark & Single episode \\
    LP (endogenous) & None & $-0.541$ & $[-0.96,-0.12]$ & \warnmark & Endogenous \\
    Proxy-SVAR (IB rate) & IV & --- & --- & \xmark & $F=4.73<10$ \\
    Proxy-SVAR (tone, A) & IV & --- & --- & \xmark & Lacks relevance \\
    Proxy-SVAR (tone, B) & IV & $4.32^*$ & --- & \xmark & Exogeneity fails \\
\midrule
\multicolumn{6}{l}{\textit{Published estimates for Peru}} \\
    P\'{e}rez Rojo \& Rodr\'iguez (2024) & Recursive & $-0.28$ & --- & \cmark & --- \\
    Castillo et al.\ (2016) & Recursive & $-0.30$ & --- & \cmark & --- \\
    Portilla et al.\ (2022) & Recursive & $-0.25$ & --- & \cmark & --- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small \cmark\ = identified; \warnmark\ = feasible with caveats; \xmark\ = not
identified. ${}^*$All LP-IV horizons $h=0$--$8$ positive (exogeneity failure).
Literature: $[-0.30, -0.25]$\,pp per 100\,bp. Cholesky robustness range
$[-0.13, -0.29]$\,pp brackets published estimates. Cholesky CI includes zero.
\end{tablenotes}
""")

# ── Figure 10: Poverty OLS Leave-One-Out Influence Plot ──────────────────────
print('\nFigure 10: Poverty leave-one-out influence plot...')

POV_DATA_LOO = [
    (2005, 6.282, -2.7), (2006, 7.555, -5.0), (2007, 8.470, -6.3),
    (2008, 9.185, -5.0), (2009, 1.123, -3.4), (2010, 8.283, -4.5),
    (2011, 6.380, -3.5), (2012, 6.145, -2.3), (2013, 5.827, -1.7),
    (2014, 2.453, -0.4), (2015, 3.249, -0.1), (2016, 4.017, -1.2),
    (2017, 2.534,  0.2), (2018, 4.019, -1.2), (2019, 2.172, -1.2),
    (2022, 2.681, -0.7), (2023, -0.6,   1.7), (2024,   3.1, -0.4),
]

BETA_FULL_LOO = -0.656

def ols_slope(gdp_v, pov_v):
    gdp_a = np.array(gdp_v); pov_a = np.array(pov_v)
    xbar = gdp_a.mean(); ybar = pov_a.mean()
    return np.sum((gdp_a - xbar) * (pov_a - ybar)) / np.sum((gdp_a - xbar) ** 2)

loo_years, loo_betas = [], []
for i, (yr, g, dp) in enumerate(POV_DATA_LOO):
    subset = [obs for j, obs in enumerate(POV_DATA_LOO) if j != i]
    loo_years.append(yr)
    loo_betas.append(ols_slope([s[1] for s in subset], [s[2] for s in subset]))

loo_betas_arr = np.array(loo_betas)
n_loo = len(loo_years)
y_pos = list(range(n_loo))

fig, ax = plt.subplots(figsize=SZ["single_tall"])

# Connecting lines from reference to each dot (Cleveland dot-plot style)
ax.hlines(y_pos, xmin=min(loo_betas_arr) - 0.02, xmax=loo_betas_arr,
          color=C["ci_light"], lw=0.8, zorder=1)
ax.scatter(loo_betas_arr, y_pos, color=C["main"], s=35,
           edgecolors='white', linewidths=0.5, zorder=3)

# Full-sample reference line in red
ax.axvline(BETA_FULL_LOO, color=C["accent1"], lw=1.5, ls='--', zorder=2,
           label=r'Full-sample $\hat{\beta} = -0.656$')

ax.set_yticks(y_pos)
ax.set_yticklabels([str(yr) for yr in loo_years], fontsize=8)
ax.set_xlabel(r'Leave-one-out OLS slope $\hat{\beta}_i$', labelpad=5)

# Tighten x-axis to actual data range
x_lo = loo_betas_arr.min() - 0.04
x_hi = max(loo_betas_arr.max(), BETA_FULL_LOO) + 0.04
ax.set_xlim(x_lo, x_hi)

ax.legend(loc='lower right', fontsize=8.5)
stat_box(ax, r'Each dot: $\hat{\beta}$ with that year removed',
         loc='upper right', fontsize=7.5)
savefig(fig, 'fig10_poverty_influence')

# ── Table 13: EME Taxonomy ────────────────────────────────────────────────────
print('Table 13: EME taxonomy...')
write_tex('tab13_eme_taxonomy', r"""
\resizebox{\linewidth}{!}{\begin{tabular}{lcccp{5.2cm}}
\toprule
Country & Administered & No rate & Limited & Notes \\
        & interbank & futures & episodes & \\
\midrule
\textit{Same identification constraints as Peru} & & & & \\
\quad Bolivia & \cmark & \cmark & \cmark & Fixed exchange rate \\
\quad Paraguay & \cmark & \cmark & \cmark & Recent IT adopter (2011) \\
\quad Egypt & \cmark & \cmark & \cmark & Managed float, dollarized \\
\quad Pakistan & \cmark & \cmark & \cmark & Administered corridor \\
\quad Bangladesh & \cmark & \cmark & \cmark & Rate-based corridor \\
\midrule
\textit{Administered rate but some market info} & & & & \\
\quad Colombia & \cmark & \warnmark & \warnmark & Some futures; IT since 1999 \\
\quad Uruguay & \cmark & \cmark & \warnmark & Longer IT history \\
\quad Costa Rica & \cmark & \cmark & \warnmark & IT since 2018 only \\
\midrule
\textit{Rate futures exist (Proxy-SVAR feasible)} & & & & \\
\quad Brazil & $\times$ & $\times$ & $\times$ & Deep DI futures market \\
\quad Mexico & $\times$ & $\times$ & \warnmark & TIIE swaps; Banxico surprises \\
\quad Chile & $\times$ & \warnmark & \warnmark & Some OIS; longer IT \\
\bottomrule
\end{tabular}}
\begin{tablenotes}
\small \cmark\ = constraint active; \warnmark\ = partial; $\times$ = absent.
``Administered'' = central bank enforces interbank rate via active liquidity management.
``No futures'' = no liquid OIS/exchange-traded rate futures.
``Limited'' = fewer than 5 unambiguous IT-era shocks.
Brazil's deep DI futures market enables Proxy-SVAR \citep{caldara2019}.
\end{tablenotes}
""")

print('\nDone. All figures saved to:', FIG_DIR)
print('All tables saved to:', TAB_DIR)
