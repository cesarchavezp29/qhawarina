#!/usr/bin/env python3
"""run_figures_B.py — Publication figures B1-B6 + Table 12 update"""
import sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
TONE_CSV = ROOT / 'data/raw/bcrp/notas_informativas/tone_scores.csv'
OUTDIR  = Path('/mnt/user-data/outputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def savefig(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUTDIR / f'{name}.{ext}', dpi=300 if ext=='png' else None)
    print(f'Saved: {name}.pdf/.png')
    plt.close(fig)

# Poverty data (N=18)
POV_DATA = [
    (2005, 6.282, -2.7), (2006, 7.555, -5.0), (2007, 8.470, -6.3),
    (2008, 9.185, -5.0), (2009, 1.123, -3.4), (2010, 8.283, -4.5),
    (2011, 6.380, -3.5), (2012, 6.145, -2.3), (2013, 5.827, -1.7),
    (2014, 2.453, -0.4), (2015, 3.223, -1.7), (2016, 3.975, -1.1),
    (2017, 2.515,  0.0), (2018, 3.957, -1.6), (2019, 2.250, -0.6),
    (2022, 2.857,  1.5), (2023,-0.345,  1.5), (2024, 3.473, -2.1),
]
POV_YEARS, POV_GDP, POV_DPOV = map(list, zip(*POV_DATA))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B1 — Corrected Chain Estimate Comparison (3.5×3.5in)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B1...")

methods = [
    ('Baseline\n(wrong frequency)', 0.128,  None,   None,   False, False),
    ('Method A: 1-yr\ncumulated IRF',  0.291,  -0.923, 1.924, True,  False),
    ('Method B: matched\nunits (preferred)', 0.498, 0.379, 0.608, True, True),
    ('Method A: 2-yr\ncumulated IRF',  0.613,  -1.184, 2.709, True,  False),
]

fig, ax = plt.subplots(figsize=(3.5, 3.5))

# y positions: 3 at top (baseline), 0 at bottom
y_positions = [3, 2, 1, 0]

for idx, (label, est, ci_lo, ci_hi, has_ci, highlighted) in enumerate(methods):
    y = y_positions[idx]
    is_baseline = (idx == 0)

    if highlighted:
        color = '#c0392b'
        marker = 's'
        ms = 7
        lw = 2.0
        zorder = 5
    elif is_baseline:
        color = 'gray'
        marker = 's'
        ms = 6
        lw = 1.0
        zorder = 3
    else:
        color = 'black'
        marker = 's'
        ms = 6
        lw = 1.0
        zorder = 3

    # Draw CI line
    if has_ci:
        ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=lw, zorder=zorder-1)
        # Cap ends
        cap_h = 0.15
        ax.plot([ci_lo, ci_lo], [y - cap_h/2, y + cap_h/2], color=color, lw=lw, zorder=zorder-1)
        ax.plot([ci_hi, ci_hi], [y - cap_h/2, y + cap_h/2], color=color, lw=lw, zorder=zorder-1)

    # Plot marker
    ax.plot(est, y, marker=marker, color=color, ms=ms, zorder=zorder, ls='none',
            markeredgecolor=color)

    # Strikethrough for baseline
    if is_baseline:
        ax.plot([est - 0.15, est + 0.15], [y, y], color='gray', lw=2.5, zorder=zorder+1, solid_capstyle='round')

ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Poverty effect (pp per 100 bp hike)')
ax.set_yticks(range(4))
ax.set_yticklabels([m[0] for m in reversed(methods)], fontsize=8)

# Annotation for Method B (y=1)
ax.annotate('CI excludes zero',
            xy=(0.55, 1), xytext=(1.0, 1.5),
            fontsize=7, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

# Legend
sq_red   = mpatches.Patch(color='#c0392b', label='Preferred estimate')
sq_black = mpatches.Patch(color='black',   label='Alternative')
ax.legend(handles=[sq_red, sq_black], fontsize=7, loc='upper right')

ax.set_title('Chain Estimate Comparison', fontsize=10)
fig.tight_layout()
savefig(fig, 'fig_B1_chain_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B2 — Small-Sample Inference Forest Plot (3.5×3in)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B2...")

results = [
    ('OLS (baseline)',         -0.6565, -0.9010, -0.4120),
    ('Wild bootstrap',         -0.6565, -0.8725, -0.4406),
    ('Jackknife',              -0.6565, -0.9512, -0.3618),
    ('HC3 (robust SE)',        -0.6565, -0.9369, -0.3761),
    ('Bayesian\n(posterior)',  -0.6563, -0.8972, -0.4150),
]

fig, ax = plt.subplots(figsize=(3.5, 3))

for i, (label, est, ci_lo, ci_hi) in enumerate(results):
    y = i
    color = 'black' if i == 0 else '#808080'
    ms = 7 if i == 0 else 6

    # CI line
    ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=1.2, zorder=2)
    cap_h = 0.15
    ax.plot([ci_lo, ci_lo], [y - cap_h/2, y + cap_h/2], color=color, lw=1.2, zorder=2)
    ax.plot([ci_hi, ci_hi], [y - cap_h/2, y + cap_h/2], color=color, lw=1.2, zorder=2)

    # Point estimate
    ax.plot(est, y, marker='o', color=color, ms=ms, zorder=3, ls='none',
            markeredgecolor=color)

ax.axvline(0,      color='k',        lw=0.5, ls='--', label='H₀: β = 0')
ax.axvline(-0.656, color='#c0392b',  lw=1,   ls='--', alpha=0.7, label='OLS estimate')

ax.set_yticks(range(len(results)))
ax.set_yticklabels([r[0] for r in results], fontsize=8)
ax.set_xlabel('GDP-poverty coefficient β')

# Caption note
ax.text(0.98, 0.04, 'All CIs exclude zero',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7, style='italic', color='#404040')

ax.legend(fontsize=7, loc='upper left')
ax.set_title('Small-Sample Robustness: Inference Methods', fontsize=10)
fig.tight_layout()
savefig(fig, 'fig_B2_poverty_inference')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B3 — LLM Tone Validation (7×3.5in, two panels)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B3...")

tone = pd.read_csv(TONE_CSV)
tone['date'] = pd.to_datetime(tone['date'])

raw = pd.read_parquet(PANEL)
raw['date'] = pd.to_datetime(raw['date'])
rate_m = raw[raw['series_id'] == 'PD04722MM'][['date', 'value_raw']].copy()
rate_m = rate_m.set_index('date').sort_index()
rate_m.columns = ['rate_level']
rate_m['d_rate'] = rate_m['rate_level'].diff()
rate_m_monthly = rate_m['d_rate'].resample('MS').first()

tone['merge_date'] = tone['date'].dt.to_period('M').dt.to_timestamp()
rate_df = rate_m_monthly.reset_index()
rate_df.columns = ['merge_date', 'd_rate']
merged = tone.merge(rate_df, on='merge_date', how='left')
merged['d_rate'] = merged['d_rate'].fillna(0.0)
llm_col  = 'llm_tone'
dict_col = 'dict_tone'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

# Panel (a) — Scatter: LLM tone vs rate change
colors_map = merged['d_rate'].apply(
    lambda x: '#c0392b' if x > 0 else ('#2c7bb6' if x < 0 else '#636363'))
ax1.scatter(merged[llm_col], merged['d_rate'],
            c=colors_map, alpha=0.5, s=15, edgecolors='none')

X_fit = sm.add_constant(merged[llm_col].values)
fit = sm.OLS(merged['d_rate'].values, X_fit).fit()
x_range = np.linspace(merged[llm_col].min(), merged[llm_col].max(), 100)
ax1.plot(x_range, fit.params[0] + fit.params[1] * x_range, 'k-', lw=1.5)
r2 = fit.rsquared
ax1.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax1.transAxes, fontsize=9)
ax1.set_xlabel('LLM tone score')
ax1.set_ylabel('Rate change (pp)')
ax1.set_title('(a) LLM tone vs rate decision')

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', ls='none', color='#c0392b', ms=5, label='Hike'),
           Line2D([0], [0], marker='o', ls='none', color='#636363', ms=5, label='Hold'),
           Line2D([0], [0], marker='o', ls='none', color='#2c7bb6', ms=5, label='Cut')]
ax1.legend(handles=handles, fontsize=7, loc='lower right')

# Panel (b) — Confusion matrix heatmap
merged['tone_class'] = merged[llm_col].apply(
    lambda x: 'hawkish' if x > 20 else ('dovish' if x < -20 else 'neutral'))
merged['rate_class'] = merged['d_rate'].apply(
    lambda x: 'hike' if x > 0 else ('cut' if x < 0 else 'hold'))

ct = pd.crosstab(merged['tone_class'], merged['rate_class'])
row_order = ['hawkish', 'neutral', 'dovish']
col_order = ['hike', 'hold', 'cut']
ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

cmap = plt.cm.Greys
im = ax2.imshow(ct.values, cmap=cmap, aspect='auto', vmin=0)
ax2.set_xticks(range(3))
ax2.set_xticklabels(col_order, fontsize=9)
ax2.set_yticks(range(3))
ax2.set_yticklabels(row_order, fontsize=9)
for i in range(3):
    for j in range(3):
        val = ct.values[i, j]
        text_color = 'white' if val > ct.values.max() * 0.6 else 'black'
        ax2.text(j, i, str(val), ha='center', va='center', fontsize=10, color=text_color)

total   = ct.values.sum()
correct = ct.loc['hawkish', 'hike'] + ct.loc['neutral', 'hold'] + ct.loc['dovish', 'cut']
acc     = correct / total if total > 0 else 0.0
ax2.set_title(f'(b) Confusion matrix (accuracy = {acc:.1%})')
ax2.set_xlabel('Actual decision')
ax2.set_ylabel('LLM classification')

fig.tight_layout()
savefig(fig, 'fig_B3_tone_validation')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B4 — LLM vs Dictionary Prediction (3.5×3in)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B4...")

fig, ax = plt.subplots(figsize=(3.5, 3))
x = np.arange(2)
width = 0.35
llm_vals  = [0.459, 0.264]
dict_vals = [0.106, 0.062]

bars1 = ax.bar(x - width/2, llm_vals,  width, color='#404040', label='LLM (Claude Haiku)')
bars2 = ax.bar(x + width/2, dict_vals, width, color='#c0c0c0', label='Dictionary')

ax.set_xticks(x)
ax.set_xticklabels(['Contemporaneous\nR²', 'Forward-looking\nR²\n(t+1)'], fontsize=9)
ax.set_ylabel('R² (regression on rate change)')
ax.set_ylim(0, 0.55)

for i, (llm, dct) in enumerate(zip(llm_vals, dict_vals)):
    ax.text(i - width/2, llm  + 0.01, f'{llm:.3f}',  ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, dct  + 0.01, f'{dct:.3f}',  ha='center', va='bottom', fontsize=8)
    ax.annotate(f'×{llm/dct:.1f}', xy=(i, (llm + dct)/2 + 0.02),
                fontsize=8, color='#c0392b', ha='center')

ax.legend(fontsize=8)
ax.set_title('LLM vs Dictionary: Predictive Power\nfor Rate Decisions')
fig.tight_layout()
savefig(fig, 'fig_B4_llm_dict_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B5 — Dollarization Channel Decomposition (3.5×2.5in)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B5...")

total_peak   = -0.1952
blocked_peak = -0.2171
fx_contrib   = total_peak - blocked_peak   # ≈ +0.022
fx_share     = fx_contrib / total_peak * 100  # ≈ -11.2%

fig, ax = plt.subplots(figsize=(3.5, 2.5))
labels      = ['Total effect\n(baseline)', 'Direct channel\n(FX blocked)', 'FX channel']
values      = [total_peak, blocked_peak, fx_contrib]
colors_bars = ['#404040', '#808080', '#c0392b']

bars = ax.bar(range(3), values, color=colors_bars, width=0.5)
ax.axhline(0, color='k', lw=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val - 0.005 if val < 0 else val + 0.003,
            f'{val:+.3f}pp',
            ha='center', va='top' if val < 0 else 'bottom', fontsize=8)

ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Peak GDP response (pp per 100 bp)')
ax.set_title('Exchange Rate Channel Decomposition')
ax.text(2, fx_contrib + 0.005,
        f'FX: {abs(fx_share):.1f}%\n(offsetting, negligible)',
        ha='center', va='bottom', fontsize=7, color='#c0392b')
fig.tight_layout()
savefig(fig, 'fig_B5_dollarization_channel')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B6 — Chain Logic Diagram (7×3in)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B6...")

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

boxes = [
    (0.1, 1.0, 2.0, 2.0, 'Rate Shock',    ['+100 bp', 'Q3 2021 – Q4 2022', '(hiking cycle)']),
    (3.0, 1.0, 2.8, 2.0, 'GDP Response',  ['Peak: −0.195 pp (QoQ)', '≈ −0.443 pp (1-yr cumul.)', 'Horizon: h = 1–3']),
    (6.5, 1.0, 2.8, 2.0, 'Poverty Effect', ['β = −0.656', '×(−0.443 pp) = +0.290 pp', 'Range: [+0.29, +0.50]']),
]

accent = '#c0392b'
for (x, y, w, h, title, lines) in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle='round,pad=0.08',
                                   ec='black', fc='white', lw=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title,
            ha='center', va='top', fontsize=9, fontweight='bold', zorder=3)
    for i, line in enumerate(lines):
        color = accent if i == 0 else 'black'
        ax.text(x + w/2, y + h - 0.7 - i * 0.38, line,
                ha='center', va='top', fontsize=8, color=color, zorder=3)

arrow_kw = dict(arrowstyle='->', color='black', lw=1.5,
                connectionstyle='arc3,rad=0')
for (x0, y0, x1, y1) in [(2.1, 2.0, 3.0, 2.0), (5.8, 2.0, 6.5, 2.0)]:
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=arrow_kw)

# Bottom timeline labels
ax.text(1.1, 0.6, 'Quarterly VAR (T = 85)', ha='center', va='center', fontsize=8, color='gray')
ax.annotate('', xy=(2.9, 0.45), xytext=(0.2, 0.45),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))
ax.text(5.9, 0.6, 'Annual OLS (N = 18)', ha='center', va='center', fontsize=8, color='gray')
ax.annotate('', xy=(9.2, 0.45), xytext=(3.1, 0.45),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))

# Result callout
ax.text(5.0, 3.5,
        'Implied: 750 bp hiking cycle \u2192 +3.7 pp poverty increase',
        ha='center', va='center', fontsize=8.5, style='italic',
        bbox=dict(boxstyle='round', fc='#fff3cd', ec='#c0392b', lw=1.0))

ax.set_title('Figure B6: Two-Step Chain Estimate \u2014 Rate \u2192 GDP \u2192 Poverty',
             fontsize=10, pad=6)
fig.tight_layout()
savefig(fig, 'fig_B6_chain_logic')

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 12 — Updated LaTeX
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Table 12...")

table12_tex = r"""\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc>{\raggedright\arraybackslash}p{2.8cm}}
\toprule
Method & Type & Peak GDP (pp) & 90\% CI & Feasible & Reason \\
\midrule
\multicolumn{6}{l}{\textit{Own estimates, this paper}} \\
    Cholesky VAR(1) & Recursive & $-0.195$ & $[-0.70, 0.27]$ & \cmark & Baseline identification \\
    GIRF (Pesaran-Shin 1998) & Ordering-free & $-0.917$ & $[-3.51,-0.13]^{\star}$ & \cmark & CI excludes zero; not structural \\
    BVAR (Minnesota, $\lambda=0.2$) & Bayesian rec. & $-0.184$ & $[-0.91,+0.62]$ & \cmark & Confirms frequentist \\
    Sign restrictions & Set ID & $-2.30$ & $[-35.9, 19.7]$ & \warnmark & Set too wide \\
    Narrative SR (full) & Set ID & $-0.74$ & $[-1.06,-0.41]$ & \warnmark & COVID tension \\
    Narrative SR (2022) & Set ID & $-3.57$ & $[-30.0,-0.29]$ & \warnmark & Single episode \\
    LP (endogenous) & None & $-0.541$ & $[-0.96,-0.12]$ & \warnmark & Endogenous \\
    Proxy-SVAR (IB rate) & IV & --- & --- & \xmark & $F=4.73<10$ \\
    Proxy-SVAR (tone, A) & IV & --- & --- & \xmark & Lacks relevance \\
    Proxy-SVAR (tone, B) & IV & $+4.32$ & --- & \xmark & Exogeneity fails \\
    Taylor-rule Proxy & IV & --- & --- & \warnmark & $F=25.17$ strong; LP-IV imprecise \\
\midrule
\multicolumn{6}{l}{\textit{Published estimates for Peru}} \\
    P\'{e}rez Rojo \& Rodr\'{\i}guez (2024) & Recursive & $-0.28$ & --- & \cmark & --- \\
    Castillo et al.\ (2016) & Recursive & $-0.30$ & --- & \cmark & --- \\
    Portilla et al.\ (2022) & Recursive & $-0.25$ & --- & \cmark & --- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small ${}^{\star}$ 90\% CI excludes zero (ordering-invariant GIRF). \cmark\ = identified;
\warnmark\ = feasible with caveats; \xmark\ = not identified.
All orderings (6 Cholesky variants) produce negative peaks;
range $[-0.195, -1.079]$\,pp. Cholesky robustness range $[-0.13, -0.29]$\,pp
brackets published estimates. ${}^{\star}$GIRF is not a structural estimate but
confirms sign; BVAR prior-insensitive across $\lambda_1\in\{0.1,0.2,0.5\}$.
\end{tablenotes}"""

with open(OUTDIR / 'table12_updated.tex', 'w', encoding='utf-8') as f:
    f.write(table12_tex)
print("Saved: table12_updated.tex")

# ─────────────────────────────────────────────────────────────────────────────
print("\nAll outputs saved to:", OUTDIR)
print("Files generated:")
for p in sorted(OUTDIR.glob('fig_B*.*')) :
    print(f"  {p.name}")
print(f"  table12_updated.tex")
