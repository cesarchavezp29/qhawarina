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

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Palatino', 'Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.4,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':       42,   # embed fonts as TrueType (not Type 3)
    'ps.fonttype':        42,
})

# Color palette — grayscale-friendly
C_BLACK    = '#000000'
C_DARK     = '#404040'
C_MED      = '#808080'
C_LIGHT    = '#c0c0c0'
C_BAND1    = '#d0d0d0'   # 90% CI band fill
C_BAND2    = '#b0b0b0'   # 68% CI band fill (darker inside)
C_RED_L    = '#ffd0d0'   # positive zone hint (light red, alpha)
C_BLUE_L   = '#d0d0f0'   # negative zone hint (light blue, alpha)

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

FX_POINT = [0.000, -0.210, -0.139, -0.019, 0.038, 0.051, 0.046, 0.037, 0.028]
FX_CI90_LO = [0.000, -0.940, -0.359, -0.164, -0.079, -0.039, -0.020, -0.010, -0.005]
FX_CI90_HI = [0.000,  0.787,  0.395,  0.208,  0.105,  0.053,  0.027,  0.015,  0.008]

# Section 7: Narrative SR GDP response percentiles (Run 4, no dummy, 119 draws)
NARR_H = [0, 1, 2, 3, 4, 8]
NARR_P05 = [1.221, -0.893, -1.063, -0.811, -0.554, -0.080]
NARR_P50 = [1.619, -0.653, -0.711, -0.549, -0.381, -0.059]
NARR_P95 = [2.189, -0.134, -0.410, -0.369, -0.279, -0.046]

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

fig, ax = plt.subplots(figsize=(6.5, 3.0))

ax.plot(rate_s['date'], rate_s['value_raw'], color=C_BLACK, lw=1.5, zorder=3)

# Shaded episodes (date ranges, label, y-position for annotation, color)
EPISODES = [
    ('2008-09-01', '2010-01-01', 'GFC\ncuts', '#e8e8e8'),
    ('2020-03-01', '2021-06-01', 'COVID\ncuts', '#e8e8e8'),
    ('2021-06-01', '2023-04-01', '2021–22\nhiking', '#d8d8d8'),
]
for start, end, label, col in EPISODES:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               color=col, alpha=0.6, zorder=1)

# Key annotations
annotations = [
    ('2010-06-01', 3.5, 'Hiking\n2010–11'),
    ('2014-09-01', 3.2, 'Gradual\ncuts'),
    ('2023-09-01', 4.5, 'Easing\n2023–25'),
]
for x, y, txt in annotations:
    ax.annotate(txt, xy=(pd.Timestamp(x), y),
                fontsize=7.5, color=C_DARK, ha='center',
                va='bottom', style='italic')

ax.set_ylabel('Reference rate (%)', labelpad=6)
ax.set_xlabel('')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax.set_xlim(pd.Timestamp('2003-01-01'), pd.Timestamp('2026-01-01'))
ax.set_ylim(-0.2, 9.0)

# Tick every 2 years
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

fig.tight_layout()
savefig(fig, 'fig1_bcrp_rate_path')


# ── Figure 2: Cholesky IRFs — Rate Shock (3-panel) ────────────────────────────
print('Figure 2: Cholesky IRFs...')

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=False)

irf_specs = [
    ('GDP', GDP_POINT, GDP_CI90_LO, GDP_CI90_HI, 'GDP response (pp)'),
    ('CPI', CPI_POINT, CPI_CI90_LO, CPI_CI90_HI, 'CPI response (pp)'),
    ('FX',  FX_POINT,  FX_CI90_LO,  FX_CI90_HI,  'FX response (pp)'),
]

for ax, (label, point, lo, hi, ylabel) in zip(axes, irf_specs):
    hs = H8
    ax.fill_between(hs, lo, hi, color=C_BAND1, alpha=0.7, zorder=1, label='90% CI')
    ax.plot(hs, point, color=C_BLACK, lw=1.5, zorder=3)
    ax.axhline(0, color=C_MED, lw=0.7, ls='--', zorder=2)
    ax.set_xlabel('Horizon (quarters)', labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xticks(range(0, 9))
    ax.set_xlim(-0.3, 8.3)
    # Subtle panel label
    ax.annotate(f'({chr(96 + irf_specs.index((label, point, lo, hi, ylabel)) + 1)})',
                xy=(0.04, 0.94), xycoords='axes fraction',
                fontsize=9, fontweight='normal', va='top')

axes[0].legend(loc='lower right', frameon=False, fontsize=8)
fig.tight_layout(w_pad=1.2)
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
    # Fallback from audit percentile table (Section 6)
    h_sr = [0, 1, 2, 3, 4, 8]
    med  = [-2.368, -1.492, -0.877, -0.497, -0.290, -0.049]
    lo05 = [-34.20, -13.53, -7.36,  -3.37,  -1.53,  -0.19]
    lo16 = [-12.75, -4.79,  -2.66,  -1.31,  -0.65,  -0.10]
    hi84 = [5.44,   -0.48,  -0.33,  -0.23,  -0.14,  0.012]
    hi95 = [19.71,  -0.17,  -0.14,  -0.12,  -0.03,  0.132]

fig, ax = plt.subplots(figsize=(4.5, 3.0))

if lo05 and hi95:
    ax.fill_between(h_sr, lo05, hi95, color=C_BAND1, alpha=0.5, label='90% credible set')
if lo16 and hi84:
    ax.fill_between(h_sr, lo16, hi84, color=C_BAND2, alpha=0.6, label='68% credible set')
ax.plot(h_sr, med, color=C_BLACK, lw=1.5, label='Median', zorder=3)
ax.plot(H8, GDP_POINT, color=C_DARK, lw=1.1, ls='--', zorder=2, label='Cholesky point')
ax.axhline(0, color=C_MED, lw=0.7, ls=':', zorder=1)

ax.set_xlabel('Horizon (quarters)', labelpad=4)
ax.set_ylabel('GDP response (pp)', labelpad=4)
ax.set_xlim(-0.3, max(h_sr) + 0.3)
ax.set_xticks(h_sr)
ax.set_ylim(-15, 8)
ax.legend(loc='lower right', frameon=False, fontsize=8, ncol=2)
ax.annotate('80,000 draws · 63,743 accepted (79.7%)', xy=(0.97, 0.96),
            xycoords='axes fraction', fontsize=7.5, ha='right', va='top',
            color=C_DARK, style='italic')

fig.tight_layout()
savefig(fig, 'fig3_sign_restriction_set')


# ── Figure 4: Narrative SR IRFs vs Cholesky ───────────────────────────────────
print('Figure 4: Narrative SR IRFs...')

fig, ax = plt.subplots(figsize=(4.5, 3.0))

h_narr = NARR_H
ax.fill_between(h_narr, NARR_P05, NARR_P95, color=C_BAND1, alpha=0.5,
                label='90% credible set')
ax.plot(h_narr, NARR_P50, color=C_BLACK, lw=1.5, label='Median (narrative SR)')
ax.plot(H8, GDP_POINT, color=C_DARK, lw=1.1, ls='--', label='Cholesky point')
ax.axhline(0, color=C_MED, lw=0.7, ls=':', zorder=1)

ax.set_xlabel('Horizon (quarters)', labelpad=4)
ax.set_ylabel('GDP response (pp)', labelpad=4)
ax.set_xlim(-0.3, 8.3)
ax.set_xticks(range(9))
ax.legend(loc='lower right', frameon=False, fontsize=8)
ax.annotate('119 accepted draws · 2020Q1–Q2 + 2022Q1–Q4 restrictions',
            xy=(0.97, 0.96), xycoords='axes fraction',
            fontsize=7.5, ha='right', va='top', color=C_DARK, style='italic')

fig.tight_layout()
savefig(fig, 'fig4_narrative_sr_irfs')


# ── Figure 5: LP-IV with Tone Instrument — Wrong-Sign IRFs ───────────────────
print('Figure 5: LP-IV wrong-sign IRFs...')

if irf_lp_iv:
    lp_beta = irf_lp_iv['gdp']['beta']
    lp_se   = irf_lp_iv['gdp']['se']
    h_lp    = list(range(len(lp_beta)))
else:
    # From audit section 9 embedded data
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

# Show h=0..12 (most informative range)
mask = np.array(h_lp) <= 12
hs_plot = np.array(h_lp)[mask]
b_plot  = lp_beta[mask]
lo_plot = lp_lo90[mask]
hi_plot = lp_hi90[mask]

fig, ax = plt.subplots(figsize=(7.5, 4.5))

# Light red background (wrong sign zone, above 0)
ax.axhspan(0, 16, color='#fce8e8', alpha=0.35, zorder=0)
ax.axhline(0, color=C_DARK, lw=0.8, ls='--', zorder=2)

for h, b, lo, hi in zip(hs_plot, b_plot, lo_plot, hi_plot):
    ax.plot([h, h], [lo, hi], color=C_MED, lw=0.9, zorder=2)
    ax.plot([h - 0.12, h + 0.12], [lo, lo], color=C_MED, lw=0.9, zorder=2)
    ax.plot([h - 0.12, h + 0.12], [hi, hi], color=C_MED, lw=0.9, zorder=2)
    ax.plot(h, b, 'o', color=C_BLACK, ms=4, zorder=3)

ax.set_xlabel('Horizon (quarters)', labelpad=4)
ax.set_ylabel('GDP response (pp)', labelpad=4)
ax.set_xlim(-0.5, 12.5)
ax.set_xticks(range(13))
ax.set_ylim(-5, 16)

# Annotation in the NEGATIVE region (below zero line) — never overlaps CI bars
# which are all positive
ax.annotate('All estimates positive (wrong sign)\n\u2192 tone instrument fails exogeneity',
            xy=(6, -2.5), xycoords='data',
            fontsize=8.5, ha='center', va='center', color='#800000', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccaaaa', alpha=0.95, lw=0.8))
ax.annotate('Bars: 90% CI  \u00b7  Dot: point estimate',
            xy=(0.97, 0.97), xycoords='axes fraction',
            fontsize=7.5, ha='right', va='top', color=C_DARK)

fig.tight_layout()
savefig(fig, 'fig5_lpiv_tone_wrong_sign')


# ── Figure 6: BCRP Tone Time Series ──────────────────────────────────────────
print('Figure 6: Tone time series...')

# Aggregate to monthly (take mean within month-year for duplicates)
df_tone['date'] = pd.to_datetime(df_tone['date'])
tone_m = (df_tone.groupby('date')[['dict_tone', 'llm_tone']]
          .mean().reset_index().sort_values('date'))
tone_m = tone_m[tone_m['date'] >= '2001-01-01']

# Rate for background shading
rate_q = df_panel[df_panel['series_id'] == 'PD04722MM'][['date', 'value_raw']].copy()
rate_q['date'] = pd.to_datetime(rate_q['date'])

# Rate-hiking vs cutting episodes for background shading
hiking  = [('2010-04-01', '2011-09-01'), ('2021-07-01', '2023-03-01')]
cutting = [('2008-09-01', '2010-03-01'), ('2014-09-01', '2017-03-01'),
           ('2020-03-01', '2021-06-01'), ('2023-04-01', '2026-01-01')]

fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)

from matplotlib.patches import Patch
_first = True
for ax, col, ylabel, ylims in zip(
    axes,
    ['dict_tone', 'llm_tone'],
    ['Dictionary tone\n(\u22121 to +1)', 'LLM tone\n(\u2212100 to +100)'],
    [(-1.2, 1.2), (-80, 80)]
):
    for s, e in hiking:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   facecolor='none', edgecolor=C_DARK, hatch='/', lw=0,
                   alpha=0.3, zorder=0)
    for s, e in cutting:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   facecolor='none', edgecolor=C_MED, hatch='..', lw=0,
                   alpha=0.3, zorder=0)

    ax.axhline(0, color=C_MED, lw=0.6, ls='--', zorder=1)
    ax.plot(tone_m['date'], tone_m[col], color=C_BLACK, lw=0.9, zorder=2)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_ylim(*ylims)
    ax.set_xlim(pd.Timestamp('2001-01-01'), pd.Timestamp('2026-06-01'))
    _first = False

# Legend patches
legend_elements = [
    Patch(facecolor='none', edgecolor=C_DARK, hatch='/', label='Hiking episode'),
    Patch(facecolor='none', edgecolor=C_MED,  hatch='..', label='Cutting episode'),
]
leg = axes[0].legend(handles=legend_elements, loc='lower left', frameon=True,
                     fontsize=8, framealpha=0.95, edgecolor='none')

axes[1].xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0, ha='center')

fig.tight_layout(h_pad=0.8)
savefig(fig, 'fig6_bcrp_tone_series')


# ── Figure 7: GDP–Poverty Scatter ─────────────────────────────────────────────
print('Figure 7: GDP-Poverty scatter...')

years, gdps, dpovs = zip(*POV_DATA)
gdps   = np.array(gdps)
dpovs  = np.array(dpovs)
outliers = {2009, 2022}

x_line = np.linspace(-2.5, 11.0, 200)
y_line = ALPHA_POV + BETA_POV * x_line
se_pred = RMSE_POV * np.sqrt(1/N_POV + (x_line - X_MEAN)**2 / SXX)
ci_lo = y_line - 1.645 * se_pred
ci_hi = y_line + 1.645 * se_pred

fig, ax = plt.subplots(figsize=(4.5, 3.6))

ax.fill_between(x_line, ci_lo, ci_hi, color=C_BAND1, alpha=0.6, zorder=1,
                label='90% prediction interval')
ax.plot(x_line, y_line, color=C_DARK, lw=1.2, ls='--', zorder=2, label='OLS fit')
ax.axhline(0, color=C_LIGHT, lw=0.5, zorder=0)

# Points
for yr, g, dp in POV_DATA:
    if yr in outliers:
        ax.plot(g, dp, 'o', color=C_DARK, ms=5.5, zorder=4, fillstyle='none')
        ax.annotate(str(yr), xy=(g, dp), xytext=(6, 3),
                    textcoords='offset points', fontsize=8, color=C_DARK)
    else:
        ax.plot(g, dp, 'o', color=C_BLACK, ms=4.5, zorder=3)

ax.set_xlabel('Annual GDP growth (%)', labelpad=4)
ax.set_ylabel('Δ Poverty rate (pp)', labelpad=4)
ax.set_xlim(-2.0, 10.8)
ax.set_ylim(-7.5, 3.5)
ax.legend(loc='upper right', frameon=False, fontsize=8)
ax.annotate(r'$\hat{\beta}=-0.656$  $R^2=0.669$  $N=18$',
            xy=(0.04, 0.05), xycoords='axes fraction',
            fontsize=8.5, color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8))

fig.tight_layout()
savefig(fig, 'fig7_gdp_poverty_scatter')


# ── Figure 8: GDP–Poverty Sub-Period Comparison ───────────────────────────────
print('Figure 8: Sub-period poverty regression...')

early = [(yr, g, dp) for yr, g, dp in POV_DATA if 2005 <= yr <= 2014]
late  = [(yr, g, dp) for yr, g, dp in POV_DATA if 2015 <= yr <= 2024]

def subperiod_ols(data):
    gdp_v = np.array([d[1] for d in data])
    pov_v = np.array([d[2] for d in data])
    beta, alpha = np.polyfit(gdp_v, pov_v, 1)
    return alpha, beta

alpha_e, beta_e = subperiod_ols(early)
alpha_l, beta_l = subperiod_ols(late)

x_e = np.linspace(0, 10.5, 100)
x_l = np.linspace(-1.0, 5.0, 100)

fig, ax = plt.subplots(figsize=(4.5, 3.6))

# Early: solid
ax.plot(x_e, alpha_e + beta_e * x_e, color=C_BLACK, lw=1.4,
        ls='-', label=f'2005–2014  β={beta_e:.3f}')
for _, g, dp in early:
    ax.plot(g, dp, 's', color=C_BLACK, ms=5, zorder=3)

# Late: dashed
ax.plot(x_l, alpha_l + beta_l * x_l, color=C_DARK, lw=1.4,
        ls='--', label=f'2015–2024  β={beta_l:.3f}')
for yr, g, dp in late:
    marker = 'D' if yr in {2022, 2023} else '^'
    ax.plot(g, dp, marker, color=C_DARK, ms=5, zorder=3)

ax.axhline(0, color=C_LIGHT, lw=0.5)
ax.set_xlabel('Annual GDP growth (%)', labelpad=4)
ax.set_ylabel('Δ Poverty rate (pp)', labelpad=4)
ax.set_xlim(-2.0, 10.8)
ax.set_ylim(-7.5, 3.5)
ax.legend(loc='upper right', frameon=False, fontsize=8)
ax.annotate('Both slopes negative; no structural break (Chow p > 0.40)',
            xy=(0.04, 0.05), xycoords='axes fraction',
            fontsize=7.5, color=C_DARK, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8))

fig.tight_layout()
savefig(fig, 'fig8_gdp_poverty_subperiod')


# ── Figure 9: Forest Plot — All Strategies ────────────────────────────────────
print('Figure 9: Forest plot...')

# (strategy, peak_gdp_point, ci_lo, ci_hi, feasibility)
# NaN ci = not identified / shown as a marker only
STRATEGIES = [
    # Own estimates
    ('Cholesky VAR(1)',          -0.195,  -0.698,  0.271,  True),
    ('Sign restrictions (SR)',   -2.299,  -35.87, 19.71,   True),
    ('Narrative SR (full)',      -0.742,  -1.063, -0.410,  True),
    ('Narrative SR (2022 only)', -3.575,  -30.0,  -0.29,   True),
    ('LP (endogenous)',          -0.541,  -0.964, -0.117,  True),
    ('Proxy-SVAR: IB rate',       None,   None,   None,    False),
    ('Proxy-SVAR: tone',          None,   None,   None,    False),
    # Literature
    ('Pérez Rojo & Rodríguez (2024)',  -0.28,  None, None, True),
    ('Castillo et al. (2016)',         -0.30,  None, None, True),
    ('Portilla et al. (2022)',         -0.25,  None, None, True),
]

Y_MULT = 2.2   # generous vertical spacing — no row crowding
N = len(STRATEGIES)

# Shorten long labels so they fit the y-axis without crowding
SHORT_LABELS = [
    'Cholesky VAR(1)',
    'Sign restrictions',
    'Narrative SR (full)',
    'Narrative SR (2022)',
    'LP (endogenous)',
    'Proxy-SVAR: IB rate',
    'Proxy-SVAR: tone',
    'Pérez Rojo & Rdz. (2024)',
    'Castillo et al. (2016)',
    'Portilla et al. (2022)',
]

fig, ax = plt.subplots(figsize=(9.0, 7.0))
fig.subplots_adjust(left=0.26, right=0.88, top=0.96, bottom=0.09)

y_positions = [i * Y_MULT for i in range(N)]

# Literature range reference band
ax.axvspan(-0.30, -0.25, color='#e8e8f8', alpha=0.8, zorder=0)
ax.axvline(0, color=C_MED, lw=0.7, ls='--', zorder=1)

# Divider between own estimates and literature
divider_y = (y_positions[6] + y_positions[7]) / 2
ax.axhline(divider_y, color=C_LIGHT, lw=0.8, ls='-')
# "Literature estimates" label — far right, anchored to divider, clear of data
ax.annotate('Literature\nestimates \u2193',
            xy=(0.91, divider_y), xycoords=('axes fraction', 'data'),
            fontsize=7.5, ha='left', va='top', color=C_MED, style='italic',
            annotation_clip=False)

XLIM_LO = -11.0
XLIM_HI =  3.5
for i, (label, point, lo, hi, feasible) in enumerate(STRATEGIES):
    y = y_positions[N - 1 - i]   # top-to-bottom

    if point is None:
        # Not identified — shaded row + × marker + text with bbox
        ax.axhspan(y - 0.7, y + 0.7, color='#f0f0f0', zorder=0)
        ax.plot(XLIM_LO + 0.6, y, 'x', color=C_MED, ms=8, mew=1.8, zorder=3)
        ax.annotate('not identified',
                    xy=(XLIM_LO + 1.5, y), fontsize=7.5,
                    color=C_MED, va='center', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#f0f0f0', ec='none', alpha=0.9))
    else:
        if lo is not None and hi is not None:
            ci_lo_clipped = max(lo, XLIM_LO + 0.5)
            ci_hi_clipped = min(hi, XLIM_HI - 0.2)
            ax.plot([ci_lo_clipped, ci_hi_clipped], [y, y],
                    color=C_MED, lw=1.5, zorder=2)
            if lo < XLIM_LO + 0.5:
                ax.annotate('', xy=(XLIM_LO + 0.2, y),
                            xytext=(ci_lo_clipped, y),
                            arrowprops=dict(arrowstyle='->', color=C_DARK,
                                            lw=1.2, shrinkA=0, shrinkB=0))
                ax.annotate(f'{lo:.1f}',
                            xy=(XLIM_LO + 0.15, y), xytext=(-6, 0),
                            textcoords='offset points',
                            fontsize=7, ha='right', va='center', color=C_DARK)
        is_lit = (i >= 7)
        marker = 'D' if is_lit else 'o'
        color  = C_DARK if is_lit else C_BLACK
        ax.plot(point, y, marker, color=color, ms=6.5, zorder=4,
                markerfacecolor='none' if is_lit else color,
                markeredgewidth=1.8 if is_lit else 1.0)

ax.set_yticks(y_positions)
ax.set_yticklabels(list(reversed(SHORT_LABELS)), fontsize=9)
ax.set_xlabel('Peak GDP response to 100bp rate hike (pp)', labelpad=6)
ax.set_xlim(XLIM_LO, XLIM_HI)
ax.set_ylim(-Y_MULT * 0.7, y_positions[-1] + Y_MULT * 0.7)

# Shaded band label — below the bottom row, in empty space, left-anchored
ax.annotate('Shaded band: literature range [\u22120.30, \u22120.25] pp',
            xy=(0.02, 0.01), xycoords='axes fraction',
            fontsize=7.5, ha='left', va='bottom', color=C_MED, style='italic')

fig.savefig(str(FIG_DIR / 'fig9_forest_plot.pdf'), bbox_inches='tight')
plt.close(fig)
print('  Saved fig9_forest_plot.pdf')


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
\small VAR(1), $T=85$, FWL COVID partial-out. Full SEs in Appendix A.
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
    Cholesky VAR(1) & Recursive & $-0.195$ & $[-0.70, 0.27]$ & \cmark & Standard timing \\
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
identified. ${}^*$All LP-IV horizons $h=0$--$12$ positive (exogeneity failure).
Literature: $[-0.30, -0.25]$pp per 100bp. Cholesky CI includes zero.
\end{tablenotes}
""")

# ── Figure 10: Poverty OLS Leave-One-Out Influence Plot ──────────────────────
print('\nFigure 10: Poverty leave-one-out influence plot...')

POV_DATA_LOO = [
    (2005, 6.282, -2.7), (2006, 7.555, -5.0), (2007, 8.470, -6.3),
    (2008, 9.185, -5.0), (2009, 1.123, -3.4), (2010, 8.283, -4.5),
    (2011, 6.380, -3.5), (2012, 6.145, -2.3), (2013, 5.827, -1.7),
    (2014, 2.453, -0.4), (2015, 3.249, -0.1), (2016, 4.017, -1.2),
    (2017, 2.534, 0.2),  (2018, 4.019, -1.2), (2019, 2.172, -1.2),
    (2022, 2.681, -0.7), (2023, -0.6,  1.7), (2024, 3.1,   -0.4),
]

BETA_FULL_LOO = -0.656  # full-sample OLS slope

def ols_slope(gdp_v, pov_v):
    """Return OLS slope (no intercept extraction needed)."""
    gdp_a = np.array(gdp_v)
    pov_a = np.array(pov_v)
    xbar = gdp_a.mean()
    ybar = pov_a.mean()
    beta = np.sum((gdp_a - xbar) * (pov_a - ybar)) / np.sum((gdp_a - xbar) ** 2)
    return beta

loo_years = []
loo_betas = []
for i, (yr, g, dp) in enumerate(POV_DATA_LOO):
    subset = [obs for j, obs in enumerate(POV_DATA_LOO) if j != i]
    gdp_sub = [s[1] for s in subset]
    pov_sub = [s[2] for s in subset]
    beta_i = ols_slope(gdp_sub, pov_sub)
    loo_years.append(yr)
    loo_betas.append(beta_i)

fig, ax = plt.subplots(figsize=(5.5, 4.5))

n_loo = len(loo_years)
y_pos = list(range(n_loo))

ax.scatter(loo_betas, y_pos, color=C_BLACK, s=30, zorder=3)
ax.axvline(BETA_FULL_LOO, color=C_DARK, lw=1.2, ls='--', zorder=2,
           label=r'Full-sample $\hat{\beta}=-0.656$')

ax.set_yticks(y_pos)
ax.set_yticklabels([str(yr) for yr in loo_years], fontsize=8)
ax.set_xlabel(r'Leave-one-out OLS slope $\hat{\beta}_i$', labelpad=5)
ax.set_xlim(min(loo_betas) - 0.05, max(loo_betas) + 0.05)

ax.legend(loc='lower right', frameon=False, fontsize=8)
ax.annotate('Each dot: $\\hat{\\beta}$ with that year removed',
            xy=(0.97, 0.97), xycoords='axes fraction',
            fontsize=8, ha='right', va='top', color=C_DARK, style='italic')

fig.tight_layout()
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
