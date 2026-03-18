"""
Figure 2: Wage Distributions Pre vs. Post — Three MW Events
Cengiz bunching visual. One panel per event, formal dependent workers.
Saves: exports/figures/fig2_bunching_distributions.png (and .pdf)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

CS_BASE = 'D:/Nexus/nexus/data/raw/enaho/cross_section'
OUT_DIR = 'D:/Nexus/nexus/exports/figures'
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS = [
    ('A', 2015, 2017, 750,  850,  'Event A: S/750→850 (May 2016)'),
    ('B', 2017, 2019, 850,  930,  'Event B: S/850→930 (Apr 2018)'),
    ('C', 2021, 2023, 930, 1025,  'Event C: S/930→1,025 (May 2022)'),
]
BIN_W = 25
WAGE_LO, WAGE_HI = 300, 2500


def load_formal_dep(year):
    import os
    for stem in [f'enaho01a-{year}-500.dta', f'enaho01a-{year}_500.dta']:
        path = f'{CS_BASE}/modulo_05_{year}/{stem}'
        if os.path.exists(path):
            break
    else:
        return None
    if not HAS_PYREADSTAT:
        return None
    df, _ = pyreadstat.read_dta(path)
    df.columns = [c.lower() for c in df.columns]

    emp   = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    dep_v = 'cat07p500a1' if 'cat07p500a1' in df.columns else 'p507'
    dep_n = pd.to_numeric(df.get(dep_v, pd.Series(np.nan, index=df.index)), errors='coerce')
    dep   = (dep_n == 2) if dep_v == 'cat07p500a1' else dep_n.isin([3, 4, 6])
    form  = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce') == 2
    w_p   = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    w_i   = pd.to_numeric(df.get('i524a1', pd.Series(np.nan, index=df.index)), errors='coerce') / 12.0
    wage  = w_p.where(w_p > 0, w_i)
    wt    = pd.to_numeric(df.get('factor07i500a', df.get('fac500a', pd.Series(1, index=df.index))), errors='coerce')

    mask = emp & dep & form & (wage > 0) & wage.notna() & wt.notna()
    return wage[mask].values, wt[mask].values


def make_hist(wages, weights, bins):
    counts, _ = np.histogram(wages, bins=bins, weights=weights)
    total = counts.sum()
    return counts / total * 100 if total > 0 else counts


def counterfactual(shares, bc, mw_new, degree=4):
    """Fit polynomial on clean zone (bc > 2 * mw_new), apply everywhere."""
    clean = bc > 2 * mw_new
    if clean.sum() < 5:
        return np.full_like(shares, np.nan)
    try:
        cf = np.polyval(np.polyfit(bc[clean], shares[clean], degree), bc)
        return np.maximum(cf, 0)
    except Exception:
        return np.full_like(shares, np.nan)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.patch.set_facecolor('white')

colors = {'pre': '#4472C4', 'post': '#ED7D31', 'cf': '#666666'}

for ax, (ev, pre_yr, post_yr, mw_old, mw_new, title) in zip(axes, EVENTS):
    print(f"Processing Event {ev}...")
    pre_data  = load_formal_dep(pre_yr)
    post_data = load_formal_dep(post_yr)
    if pre_data is None or post_data is None:
        ax.text(0.5, 0.5, f'Data not available\n({pre_yr}, {post_yr})',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        continue

    pre_wages, pre_wts   = pre_data
    post_wages, post_wts = post_data

    bins = np.arange(WAGE_LO, WAGE_HI + BIN_W, BIN_W)
    bc   = bins[:-1] + BIN_W / 2

    pre_shares  = make_hist(pre_wages,  pre_wts,  bins)
    post_shares = make_hist(post_wages, post_wts, bins)

    # Counterfactual from post-period clean zone
    cf = counterfactual(post_shares, bc, mw_new)

    # Affected zone shading
    aff_lo = 0.85 * mw_old
    ax.axvspan(aff_lo, mw_new, alpha=0.08, color='red', zorder=0, label='_nolegend_')

    # Bars: pre (blue), post (orange)
    width = BIN_W * 0.45
    ax.bar(bc - width/2, pre_shares,  width=width, color=colors['pre'],
           alpha=0.75, label=f'Pre ({pre_yr})', zorder=2)
    ax.bar(bc + width/2, post_shares, width=width, color=colors['post'],
           alpha=0.75, label=f'Post ({post_yr})', zorder=2)

    # Counterfactual line
    valid_cf = ~np.isnan(cf)
    ax.plot(bc[valid_cf], cf[valid_cf], color=colors['cf'], linewidth=1.5,
            linestyle='--', label='Counterfactual', zorder=3)

    # MW vertical line
    ax.axvline(mw_new, color='black', linewidth=1.8, linestyle='-', zorder=4)
    ax.text(mw_new + 10, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1.5,
            f'S/{mw_new}', fontsize=8, ha='left', va='top', color='black')

    ax.set_xlim(WAGE_LO, min(WAGE_HI, 3 * mw_new))
    ax.set_xlabel('Monthly wage (S/.)', fontsize=9)
    ax.set_ylabel('Share of workers (%)' if ev == 'A' else '', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate MW line after setting xlim
    ylim = ax.get_ylim()
    ax.text(mw_new + 12, ylim[1] * 0.90, f'MW = S/{mw_new}',
            fontsize=8, ha='left', va='top', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.suptitle(
    'Wage Distribution of Formal Dependent Workers — Pre and Post MW Increases\n'
    'ENAHO Module 500. Bin width = S/25. Shaded zone = affected range [0.85\u00d7MW_old, MW_new).',
    fontsize=9, y=1.02
)
plt.tight_layout()

out_png = f'{OUT_DIR}/fig2_bunching_distributions.png'
out_pdf = f'{OUT_DIR}/fig2_bunching_distributions.pdf'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
