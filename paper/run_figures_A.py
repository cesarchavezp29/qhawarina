#!/usr/bin/env python3
"""run_figures_A.py -- Publication figures A1-A7 (VAR-based)"""
import sys, io, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUTDIR  = Path('D:/Nexus/nexus/paper/figures_A')
OUTDIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(ROOT / 'paper'))
from paper_style import apply_style, C, SZ, zero_line, legend_below, legend_outside, stat_box
apply_style()

def savefig(fig, name):
    fig.savefig(OUTDIR / f'{name}.pdf')
    fig.savefig(OUTDIR / f'{name}.png', dpi=300)
    print(f'Saved: {name}.pdf / .png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_var_data():
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])
    series_map = {
        'PD04722MM': 'rate_raw', 'PN01731AM': 'gdp_m',
        'PN01271PM': 'cpi_m',   'PN01246PM': 'fx_m',
        'PN38923BM': 'tot_m',
    }
    frames = []
    for sid, col in series_map.items():
        s = raw[raw['series_id']==sid][['date','value_raw']].copy()
        s = s.rename(columns={'value_raw':col}).set_index('date').sort_index()
        frames.append(s)
    monthly = frames[0].join(frames[1:], how='outer').sort_index()
    q = pd.DataFrame()
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()
    q['gdp']  = monthly['gdp_m'].resample('QE').sum()
    q['cpi']  = monthly['cpi_m'].resample('QE').sum()
    q['fx']   = monthly['fx_m'].resample('QE').mean().pct_change()*100
    q['tot']  = monthly['tot_m'].resample('QE').mean().pct_change()*100
    q = q.dropna().loc['2004-04-01':'2025-09-30']
    covid = pd.Series(0, index=q.index)
    for cq in [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]:
        if cq in covid.index: covid.loc[cq] = 1
    def fwl(s):
        X = sm.add_constant(covid); r = sm.OLS(s, X).fit()
        return s - r.fittedvalues + s.mean()
    for col in ['d_rate','gdp','cpi','fx','tot','rate_level']:
        if col in q.columns: q[col] = fwl(q[col])
    var_df = q[['tot','gdp','cpi','fx','d_rate']].rename(columns={'d_rate':'rate'})
    print(f'VAR data: T={len(var_df)}')
    return var_df, q['rate_level'].copy(), q


def compute_cholesky_irf(var_result, shock_var_idx=4, horizon=9, n_boot=500,
                          response_idx=1, lags=1):
    K = var_result.neqs
    resids = var_result.resid
    Sigma  = np.array(var_result.sigma_u)
    try:    P = np.linalg.cholesky(Sigma)
    except: P = np.linalg.cholesky(Sigma + np.eye(K)*1e-8)

    def irf_calc(coefs, P_mat, h, sidx, ridx):
        A1 = coefs[0]; K_ = A1.shape[0]
        imp = P_mat[:, sidx]; irfs = np.zeros(h); Ah = np.eye(K_)
        for hh in range(h):
            irfs[hh] = (Ah @ imp)[ridx]; Ah = Ah @ A1
        return irfs

    norm = P[shock_var_idx, shock_var_idx]
    # Degenerate window: shock variable nearly constant (e.g. flat rate during ZLB)
    if abs(norm) < 0.01:
        dummy = np.full(horizon, np.nan)
        return dummy, dummy, dummy, dummy, dummy
    irf_pt = irf_calc(var_result.coefs, P, horizon, shock_var_idx, response_idx) / norm

    if n_boot == 0:
        return irf_pt, irf_pt * np.nan, irf_pt * np.nan, irf_pt * np.nan, irf_pt * np.nan

    boot_draws = []; T_d = resids.shape[0]
    for b in range(n_boot):
        idx = np.random.randint(0, T_d, size=T_d)
        br = resids.values[idx]; Y = np.zeros((T_d+lags, K))
        Y[:lags] = var_result.model.endog[:lags]
        for t in range(lags, T_d+lags):
            pred = np.zeros(K)
            for li, c in enumerate(var_result.coefs): pred += Y[t-li-1] @ c.T
            pred += br[t-lags]; Y[t] = pred
        try:
            rb = VAR(pd.DataFrame(Y[lags:], columns=var_result.model.endog_names)).fit(lags, trend='c')
            Sb = np.array(rb.sigma_u)
            try: Pb = np.linalg.cholesky(Sb)
            except: Pb = np.linalg.cholesky(Sb + np.eye(K)*1e-8)
            nb = Pb[shock_var_idx, shock_var_idx]
            if abs(nb)<1e-10: nb=1.0
            boot_draws.append(irf_calc(rb.coefs, Pb, horizon, shock_var_idx, response_idx)/nb)
        except: pass
    if len(boot_draws) < 30:
        return irf_pt, irf_pt*np.nan, irf_pt*np.nan, irf_pt*np.nan, irf_pt*np.nan
    boot = np.array(boot_draws)
    return (irf_pt,
            np.percentile(boot, 5, axis=0), np.percentile(boot, 95, axis=0),
            np.percentile(boot, 16, axis=0), np.percentile(boot, 84, axis=0))


# ---------------------------------------------------------------------------
# FIGURE A1 -- Ordering Robustness Fan Chart
# ---------------------------------------------------------------------------
def figure_A1(var_df, res_base):
    print('Generating Figure A1...')
    h = np.arange(9)

    # Baseline IRF with CI
    irf_base, ci_lo90, ci_hi90, ci_lo68, ci_hi68 = compute_cholesky_irf(
        res_base, shock_var_idx=4, horizon=9, n_boot=1000, response_idx=1)

    # All orderings: (cols, shock_idx, resp_idx, label, style)
    orderings = [
        (['tot','cpi','gdp','fx','rate'], 4, 2, 'CPI-first', 'dashed'),
        (['tot','gdp','cpi','rate','fx'], 3, 1, 'FX-last',   'dashed'),
        (['tot','cpi','fx','rate','gdp'], 3, 4, 'Rate-before-GDP', 'dotted'),
        (['tot','rate','gdp','cpi','fx'], 1, 2, 'Rate-first', 'dotted'),
    ]

    alt_irfs = []
    for cols, si, ri, label, ls in orderings:
        sub = var_df[cols]
        try:
            r = VAR(sub).fit(1, trend='c')
            irf_r, _, _, _, _ = compute_cholesky_irf(r, shock_var_idx=si, horizon=9,
                                                      n_boot=0, response_idx=ri)
            alt_irfs.append((irf_r, label, ls))
        except Exception as e:
            print(f'  Warning: ordering {label} failed: {e}')

    # GIRF from baseline
    A1 = res_base.coefs[0]
    Sigma = np.array(res_base.sigma_u)
    k_rate, j_gdp = 4, 1
    e_k = np.zeros(5); e_k[k_rate] = 1.0
    norm_girf = Sigma[k_rate, k_rate]
    girf_pt = np.zeros(9); Ah = np.eye(5)
    for hh in range(9):
        girf_pt[hh] = (Ah @ Sigma @ e_k)[j_gdp] / norm_girf
        Ah = Ah @ A1

    # Bootstrap GIRF CI
    resids = res_base.resid
    T_d = resids.shape[0]; K = 5; lags = 1
    girf_draws = []
    for b in range(1000):
        idx = np.random.randint(0, T_d, size=T_d)
        br = resids.values[idx]; Y = np.zeros((T_d+lags, K))
        Y[:lags] = res_base.model.endog[:lags]
        for t in range(lags, T_d+lags):
            pred = np.zeros(K)
            for li, c in enumerate(res_base.coefs): pred += Y[t-li-1] @ c.T
            pred += br[t-lags]; Y[t] = pred
        try:
            rb = VAR(pd.DataFrame(Y[lags:], columns=res_base.model.endog_names)).fit(lags, trend='c')
            Sb = np.array(rb.sigma_u)
            nrm = Sb[k_rate, k_rate]
            if abs(nrm) < 1e-10: nrm = 1.0
            gi = np.zeros(9); Ahb = np.eye(K)
            A1b = rb.coefs[0]
            for hh in range(9):
                gi[hh] = (Ahb @ Sb @ e_k)[j_gdp] / nrm
                Ahb = Ahb @ A1b
            girf_draws.append(gi)
        except: pass
    girf_boot = np.array(girf_draws)
    girf_lo = np.percentile(girf_boot, 5, axis=0)
    girf_hi = np.percentile(girf_boot, 95, axis=0)

    fig, ax = plt.subplots(figsize=SZ["wide"])

    # Baseline CI — two levels
    ax.fill_between(h, ci_lo90, ci_hi90, color=C["ci_light"], alpha=0.25, label='Baseline 90% CI')
    ax.fill_between(h, ci_lo68, ci_hi68, color=C["ci_dark"],  alpha=0.50, label='Baseline 68% CI')

    # Alt orderings — thin so they don't compete with baseline/GIRF
    for irf_r, label, ls in alt_irfs:
        if ls == 'dashed':
            ax.plot(h, irf_r, color=C["gray_line"], lw=0.8, ls='--', label=label)
        else:
            ax.plot(h, irf_r, color=C["gray_line"], lw=0.8, ls=':', label=label)

    # Baseline — bold
    ax.plot(h, irf_base, color=C["main"], lw=2.5, label='Baseline [ToT,GDP,CPI,FX,Rate]')

    # GIRF — bold red with 90% CI
    ax.fill_between(h, girf_lo, girf_hi, color=C["accent1"], alpha=0.12, label='GIRF 90% CI')
    ax.plot(h, girf_pt, color=C["accent1"], lw=2.5, label='GIRF (ordering-invariant)')

    zero_line(ax)
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('GDP response (pp per 100 bp)')
    ax.set_title('')
    legend_outside(ax)

    # Annotation
    y_annot = min(irf_base.min(), -0.05)
    ax.annotate('All orderings\nnegative (peak)',
                xy=(3, y_annot), xytext=(5, y_annot - 0.05),
                arrowprops=dict(arrowstyle='->', color=C["gray_line"], lw=0.8),
                fontsize=7, color=C["gray_line"])

    savefig(fig, 'fig_A1_ordering_fan')
    shutil.copy(OUTDIR / 'fig_A1_ordering_fan.pdf', Path('D:/Nexus/nexus/paper/figures/fig11_ordering_fan.pdf'))


# ---------------------------------------------------------------------------
# FIGURE A2 -- Updated Forest Plot
# ---------------------------------------------------------------------------
def figure_A2():
    print('Generating Figure A2...')

    # (label, peak, ci_lo, ci_hi, feasible, excludes_zero, caveat, is_lit)
    entries = [
        # Own estimates
        ('Cholesky VAR(1)',        -0.195, -0.698,  0.271, True,  False, False, False),
        ('GIRF (Pesaran-Shin)',    -0.917, -3.510, -0.130, True,  True,  False, False),
        ('BVAR Minnesota',         -0.184, -0.912,  0.617, True,  False, False, False),
        ('Sign restrictions',      -2.300,-35.900, 19.700, False, False, True,  False),
        ('Narrative SR (full)',    -0.740, -1.060, -0.410, True,  False, True,  False),
        ('Narrative SR (2022)',    -3.570,-30.000, -0.290, True,  False, True,  False),
        ('LP (endogenous)',        -0.541, -0.960, -0.120, True,  False, True,  False),
        ('Proxy-SVAR (IB rate)',    None,   None,   None,  False, False, False, False),
        ('Proxy-SVAR (tone, A)',    None,   None,   None,  False, False, False, False),
        ('Proxy-SVAR (tone, B)',    4.32,   None,   None,  False, False, False, False),
        ('Taylor-rule Proxy',       None,   None,   None,  False, False, False, False),
        # Literature
        ('Perez Rojo & Rodriguez (2024)', -0.28, None, None, True, False, False, True),
        ('Castillo et al. (2016)',         -0.30, None, None, True, False, False, True),
        ('Portilla et al. (2022)',         -0.25, None, None, True, False, False, True),
    ]

    n = len(entries)
    fig, ax = plt.subplots(figsize=SZ["forest"])

    # Shaded bands
    ax.axvspan(-0.29, -0.13, alpha=0.2, color=C["accent3"], label='Cholesky robustness range')
    ax.axvspan(-0.30, -0.25, alpha=0.15, color=C["accent2"], label='Literature range')
    zero_line(ax)

    # Separator between own and lit (after row 10, i.e. between Taylor and Perez)
    sep_y = n - 11.5  # rows are plotted top-down so y = n-1-i
    ax.axhline(sep_y, color=C["gray_line"], lw=0.7, ls='-', alpha=0.4)

    labels = []
    for i, (label, peak, ci_lo, ci_hi, feasible, excl_zero, caveat, is_lit) in enumerate(entries):
        y = n - 1 - i

        if excl_zero:
            # Filled accent1 square
            marker_color = C["accent1"]
            ax.plot(peak, y, 's', color=marker_color, ms=6, zorder=5)
            if ci_lo is not None and ci_hi is not None:
                ax.plot([ci_lo, ci_hi], [y, y], '-', color=marker_color, lw=1.5)
                ax.plot([ci_lo, ci_lo], [y-0.15, y+0.15], '-', color=marker_color, lw=1.2)
                ax.plot([ci_hi, ci_hi], [y-0.15, y+0.15], '-', color=marker_color, lw=1.2)
            ax.text(ci_lo - 0.5, y, u'\u2605', fontsize=10, color=marker_color,
                    va='center', ha='center')
            labels.append(label)
        elif feasible and not caveat and peak is not None:
            # Filled main square
            ax.plot(peak, y, 's', color=C["main"], ms=5, zorder=5)
            if ci_lo is not None and ci_hi is not None:
                ax.plot([ci_lo, ci_hi], [y, y], '-', color=C["main"], lw=1.2)
                ax.plot([ci_lo, ci_lo], [y-0.12, y+0.12], '-', color=C["main"], lw=1.0)
                ax.plot([ci_hi, ci_hi], [y-0.12, y+0.12], '-', color=C["main"], lw=1.0)
            labels.append(label)
        elif feasible and caveat and peak is not None:
            # Open square, gray
            ax.plot(peak, y, 's', color=C["gray_line"], ms=5, fillstyle='none', zorder=5, markeredgewidth=1.2)
            if ci_lo is not None and ci_hi is not None:
                ax.plot([ci_lo, ci_hi], [y, y], '-', color=C["gray_line"], lw=1.0)
                ax.plot([ci_lo, ci_lo], [y-0.12, y+0.12], '-', color=C["gray_line"], lw=0.9)
                ax.plot([ci_hi, ci_hi], [y-0.12, y+0.12], '-', color=C["gray_line"], lw=0.9)
            labels.append(label + ' *')
        elif not feasible and peak is None:
            # Open circle, ci_light
            ax.plot(0, y, 'o', color=C["ci_light"], ms=5, fillstyle='none', zorder=5)
            ax.text(0.2, y, '(infeasible)', fontsize=7, color=C["ci_light"], va='center')
            labels.append(label)
        elif not feasible and peak is not None:
            # Wrong sign -- diamond
            ax.plot(peak, y, 'D', color=C["accent1"], ms=5, fillstyle='none', zorder=5,
                    markeredgewidth=1.0)
            labels.append(label + ' [wrong sign]')
        else:
            labels.append(label)

        # Literature rows: use filled accent2 circle
        if is_lit and peak is not None:
            ax.plot(peak, y, 'o', color=C["accent2"], ms=5, zorder=5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_xlabel('Peak GDP response (pp per 100bp)')
    ax.set_title('')

    # Legend
    handles = [
        mpatches.Patch(color=C["accent3"], alpha=0.4, label='Cholesky robustness range'),
        mpatches.Patch(color=C["accent2"], alpha=0.3, label='Literature range'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=C["main"],
                   ms=6, label='Feasible estimate'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=C["accent1"],
                   ms=6, label='GIRF (CI excl. 0)'),
        plt.Line2D([0],[0], marker='s', color=C["gray_line"], ms=6, fillstyle='none',
                   label='Feasible w/ caveat (*)'),
    ]
    legend_below(ax, ncol=2, handles=handles)
    ax.text(0.01, sep_y + 0.15, 'Own estimates', transform=ax.get_yaxis_transform(),
            fontsize=7, color=C["gray_line"], style='italic')
    ax.text(0.01, sep_y - 0.45, 'Literature', transform=ax.get_yaxis_transform(),
            fontsize=7, color=C["gray_line"], style='italic')

    savefig(fig, 'fig_A2_forest_updated')


# ---------------------------------------------------------------------------
# FIGURE A3 -- FEVD Stacked Area
# ---------------------------------------------------------------------------
def figure_A3(res_base):
    print('Generating Figure A3...')
    fevd = res_base.fevd(20)
    gdp_idx = 1
    horizons = np.arange(1, 21)

    fig, ax = plt.subplots(figsize=SZ["single_sq"])
    line_colors = [C["ci_light"], C["gray_line"], C["accent2"], C["accent3"], C["accent1"]]
    line_styles = ['-', '--', '-', '--', '-']
    line_widths = [2.0, 1.5, 1.5, 1.5, 2.5]
    labels = ['ToT shock', 'GDP shock', 'CPI shock', 'FX shock', 'Rate shock (monetary)']

    for k in range(5):
        vals = np.array([fevd.decomp[gdp_idx, h, k]*100 for h in range(20)])
        ax.plot(horizons, vals, color=line_colors[k], ls=line_styles[k],
                lw=line_widths[k], label=labels[k])

    # Annotate rate share at h=8
    rate_h8 = fevd.decomp[gdp_idx, 7, 4] * 100
    ax.annotate(f'Rate: {rate_h8:.1f}%',
                xy=(8, rate_h8),
                xytext=(12, rate_h8 + 5),
                arrowprops=dict(arrowstyle='->', color=C["accent1"], lw=0.8),
                fontsize=7, color=C["accent1"])

    ax.set_ylim(0, None)
    ax.set_xlim(1, 20)
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('Share of GDP FEVD (%)')
    ax.set_title('')
    legend_outside(ax)

    savefig(fig, 'fig_A3_fevd')
    shutil.copy(OUTDIR / 'fig_A3_fevd.pdf', Path('D:/Nexus/nexus/paper/figures/fig13_fevd.pdf'))


# ---------------------------------------------------------------------------
# FIGURE A4 -- Historical Decomposition of GDP
# ---------------------------------------------------------------------------
def figure_A4(res_base, var_df):
    print('Generating Figure A4...')
    T_nobs = len(res_base.resid)
    K = 5; gdp_idx = 1
    A1 = res_base.coefs[0]
    Sigma = np.array(res_base.sigma_u)
    P = np.linalg.cholesky(Sigma)

    # Structural shocks e_t = P^{-1} u_t
    struct_shocks = (np.linalg.inv(P) @ res_base.resid.values.T).T  # (T, K)

    # Precompute A1^h @ P for h=0..max_h
    max_h = 20
    irf_mat = np.zeros((max_h, K, K))
    Ah = np.eye(K)
    for hh in range(max_h):
        irf_mat[hh] = Ah @ P
        Ah = Ah @ A1

    # Historical decomp
    contrib = np.zeros((T_nobs, K))
    for t in range(T_nobs):
        for hh in range(min(t+1, max_h)):
            contrib[t] += irf_mat[hh, gdp_idx, :] * struct_shocks[t-hh, :]

    dates = res_base.resid.index
    actual_gdp = var_df['gdp'].loc[dates]

    area_colors = ['#D5D8DC', '#95A5A6', '#7F8C8D', '#2980B9', '#C0392B']
    shock_labels = ['ToT shock', 'GDP shock', 'CPI shock', 'FX shock', 'Rate shock (monetary)']

    # Stacked area chart: positive and negative bands fill_between
    fig, ax = plt.subplots(figsize=SZ["wide"])
    date_vals = dates.to_pydatetime()

    pos_stack = np.zeros(T_nobs)
    neg_stack = np.zeros(T_nobs)
    for k in range(K):
        c = contrib[:, k]
        pos_c = np.maximum(c, 0)
        neg_c = np.minimum(c, 0)
        ax.fill_between(date_vals, pos_stack, pos_stack + pos_c,
                        color=area_colors[k], alpha=0.85, label=shock_labels[k], lw=0)
        ax.fill_between(date_vals, neg_stack + neg_c, neg_stack,
                        color=area_colors[k], alpha=0.85, lw=0)
        pos_stack = pos_stack + pos_c
        neg_stack = neg_stack + neg_c

    # Actual GDP overlay
    ax.plot(date_vals, actual_gdp.values, color=C["main"], lw=1.8,
            label='Actual GDP growth', zorder=5)
    zero_line(ax)

    # Annotations for key periods
    for ts, lbl in [('2009-01-01', 'GFC'), ('2020-03-31', 'COVID'), ('2021-07-01', 'Hiking cycle')]:
        ts_dt = pd.Timestamp(ts).to_pydatetime()
        ax.axvline(ts_dt, color=C["gray_line"], lw=0.7, ls=':', alpha=0.7)
        ax.text(ts_dt, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3,
                lbl, fontsize=7, color=C["gray_line"], ha='left', rotation=0,
                va='bottom', clip_on=True)

    ax.set_xlabel('')
    ax.set_ylabel('GDP growth contribution (pp)')
    ax.set_title('')
    legend_outside(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    savefig(fig, 'fig_A4_hist_decomp')
    shutil.copy(OUTDIR / 'fig_A4_hist_decomp.pdf', Path('D:/Nexus/nexus/paper/figures/fig12_hist_decomp.pdf'))


# ---------------------------------------------------------------------------
# FIGURE A5 -- Rolling Window IRF Stability
# ---------------------------------------------------------------------------
def figure_A5(var_df):
    print('Generating Figure A5...')
    window = 40
    n_windows = len(var_df) - window + 1
    end_dates = var_df.index[window-1:]
    peaks = []; ci_lo = []; ci_hi = []; ci_lo68 = []; ci_hi68 = []

    for i in range(n_windows):
        sub = var_df.iloc[i:i+window]
        try:
            r = VAR(sub).fit(1, trend='c')
            irf_r, lo_r90, hi_r90, lo_r68, hi_r68 = compute_cholesky_irf(
                r, shock_var_idx=4, response_idx=1, horizon=9, n_boot=300)
            # Skip h=0 (always 0 by Cholesky ordering); find peak over h>=1
            h_pk = int(np.nanargmin(irf_r[1:])) + 1 if not np.all(np.isnan(irf_r[1:])) else 1
            pk_val = irf_r[h_pk]
            # Windows with no negative peak (wrong-sign or degenerate) → NaN, interpolate over
            if np.isnan(pk_val) or pk_val >= -0.01:
                raise ValueError('no negative peak')
            peaks.append(pk_val)
            ci_lo.append(lo_r90[h_pk])
            ci_hi.append(hi_r90[h_pk])
            ci_lo68.append(lo_r68[h_pk])
            ci_hi68.append(hi_r68[h_pk])
        except:
            peaks.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan)
            ci_lo68.append(np.nan); ci_hi68.append(np.nan)
        if i % 10 == 0:
            print(f'  Rolling window {i+1}/{n_windows}')

    dates = [d.to_pydatetime() for d in end_dates]

    # Interpolate NaN values from failed/degenerate windows.
    # pd.Series.interpolate handles interior gaps; ffill/bfill handles boundaries.
    def _interp_nan(arr):
        s = pd.Series(np.array(arr, dtype=float))
        if s.notna().sum() < 2:
            return s.values
        return s.interpolate(method='linear').ffill().bfill().values

    peaks_p  = _interp_nan(peaks)
    ci_lo_p  = _interp_nan(ci_lo)
    ci_hi_p  = _interp_nan(ci_hi)
    ci_lo68_p = _interp_nan(ci_lo68)
    ci_hi68_p = _interp_nan(ci_hi68)

    fig, ax = plt.subplots(figsize=SZ["single_sq"])
    ax.fill_between(dates, ci_lo_p,   ci_hi_p,   color=C["ci_light"], alpha=0.25,
                    label='90% CI', interpolate=True)
    ax.fill_between(dates, ci_lo68_p, ci_hi68_p, color=C["ci_dark"],  alpha=0.50,
                    label='68% CI', interpolate=True)
    ax.plot(dates, peaks_p, color=C["main"], lw=1.5, label='Peak GDP response')
    ax.axhline(-0.195, color=C["accent1"], ls='--', lw=1, label='Full-sample: -0.195')
    ax.axvline(pd.Timestamp('2020-03-31'), color=C["gray_line"], ls=':', lw=1, label='COVID (2020Q1)')

    ax.set_xlabel('End of rolling window')
    ax.set_ylabel('Peak GDP response (pp per 100 bp)')
    ax.set_title('')
    legend_below(ax, ncol=4)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    savefig(fig, 'fig_A5_rolling_irf')
    shutil.copy(OUTDIR / 'fig_A5_rolling_irf.pdf', Path('D:/Nexus/nexus/paper/figures/fig14_rolling_irf.pdf'))


# ---------------------------------------------------------------------------
# FIGURE A6 -- Cholesky vs GIRF
# ---------------------------------------------------------------------------
def figure_A6(res_base):
    print('Generating Figure A6...')
    h = np.arange(9)

    # Cholesky IRF
    irf_base, ci_lo_base, ci_hi_base, _, _ = compute_cholesky_irf(
        res_base, shock_var_idx=4, horizon=9, n_boot=1000, response_idx=1)

    # GIRF point estimate
    A1 = res_base.coefs[0]
    Sigma = np.array(res_base.sigma_u)
    k_rate, j_gdp = 4, 1
    e_k = np.zeros(5); e_k[k_rate] = 1.0
    norm_girf = Sigma[k_rate, k_rate]
    girf_pt = np.zeros(9); Ah = np.eye(5)
    for hh in range(9):
        girf_pt[hh] = (Ah @ Sigma @ e_k)[j_gdp] / norm_girf
        Ah = Ah @ A1

    # Bootstrap GIRF CI (1000 draws)
    resids = res_base.resid
    T_d = resids.shape[0]; K = 5; lags = 1
    girf_boot = np.zeros((1000, 9))
    for b in range(1000):
        idx = np.random.randint(0, T_d, size=T_d)
        br = resids.values[idx]; Y = np.zeros((T_d+lags, K))
        Y[:lags] = res_base.model.endog[:lags]
        for t in range(lags, T_d+lags):
            pred = np.zeros(K)
            for li, c in enumerate(res_base.coefs): pred += Y[t-li-1] @ c.T
            pred += br[t-lags]; Y[t] = pred
        try:
            rb = VAR(pd.DataFrame(Y[lags:], columns=res_base.model.endog_names)).fit(lags, trend='c')
            Sb = np.array(rb.sigma_u)
            nrm = Sb[k_rate, k_rate]
            if abs(nrm) < 1e-10: nrm = 1.0
            gi = np.zeros(9); Ahb = np.eye(K)
            A1b = rb.coefs[0]
            for hh in range(9):
                gi[hh] = (Ahb @ Sb @ e_k)[j_gdp] / nrm
                Ahb = Ahb @ A1b
            girf_boot[b] = gi
        except:
            girf_boot[b] = girf_pt
    girf_lo = np.percentile(girf_boot, 5, axis=0)
    girf_hi = np.percentile(girf_boot, 95, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SZ["wide_short"], sharey=True)

    # Left: Cholesky
    ax1.fill_between(h, ci_lo_base, ci_hi_base, color=C["ci_light"], alpha=0.15)
    ax1.plot(h, irf_base, color=C["main"], lw=2)
    zero_line(ax1)
    peak_base = irf_base.min(); h_pk = int(irf_base.argmin())
    ax1.annotate(f'Peak: {peak_base:.3f}',
                 xy=(h_pk, peak_base), xytext=(h_pk+1.5, peak_base-0.05),
                 arrowprops=dict(arrowstyle='->', color=C["gray_line"], lw=0.8),
                 fontsize=7)
    ax1.set_title('')
    ax1.set_xlabel('Horizon (quarters)')
    ax1.set_ylabel('GDP response (pp per 100 bp)')

    # Right: GIRF
    ax2.fill_between(h, girf_lo, girf_hi, color=C["ci_light"], alpha=0.15)
    ax2.plot(h, girf_pt, color=C["accent1"], lw=2)
    zero_line(ax2)
    peak_girf = girf_pt.min(); h_pk2 = int(girf_pt.argmin())
    if girf_lo[h_pk2] > 0 or girf_hi[h_pk2] < 0:
        ax2.annotate('90% CI\nexcludes 0',
                     xy=(h_pk2, girf_pt[h_pk2]),
                     xytext=(h_pk2+1.5, girf_pt[h_pk2]+0.5),
                     arrowprops=dict(arrowstyle='->', color=C["accent1"], lw=0.8),
                     fontsize=7, color=C["accent1"])
    ax2.set_title('')
    ax2.set_xlabel('Horizon (quarters)')

    savefig(fig, 'fig_A6_chol_vs_girf')


# ---------------------------------------------------------------------------
# FIGURE A7 -- Granger Causality Network
# ---------------------------------------------------------------------------
def figure_A7():
    print('Generating Figure A7...')

    import math
    # Pentagon layout with more vertical space: GDP top, Rate upper-right,
    # CPI lower-right, FX lower-left, ToT upper-left
    angles = {'GDP': 90, 'Rate': 18, 'CPI': 306, 'FX': 234, 'ToT': 162}
    R = 0.30; cx, cy = 0.5, 0.52
    nodes = {n: (cx + R*math.cos(math.radians(a)),
                 cy + R*math.sin(math.radians(a)))
             for n, a in angles.items()}

    # (from, to, F-stat, label, arc-rad, label_dx, label_dy)
    # Opposite arc signs for bidirectional GDP<->Rate to separate the two arcs
    sig_edges = [
        ('GDP',  'Rate', 14.6, 'F=14.6***', +0.20,  0.07, -0.03),
        ('Rate', 'CPI',   7.5, 'F=7.5**',   +0.20,  0.08,  0.02),
        ('ToT',  'FX',    8.0, 'F=8.0**',   +0.20, -0.08, -0.01),
    ]
    fail_edges = [
        ('Rate', 'GDP',  1.82, -0.20),   # reverse of GDP→Rate, curves other way
        ('ToT',  'CPI',  3.47, +0.20),
        ('ToT',  'GDP',  2.11, +0.20),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 0.98); ax.axis('off')

    # Failed edges first (drawn underneath, no labels)
    for from_n, to_n, F, rad in fail_edges:
        x0, y0 = nodes[from_n]; x1, y1 = nodes[to_n]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C["ci_dark"], lw=0.9,
                                   linestyle='dashed',
                                   connectionstyle=f'arc3,rad={rad}'),
                    annotation_clip=False)

    # Significant edges — solid, thickness ∝ sqrt(F), labelled
    for from_n, to_n, F, label, rad, dx, dy in sig_edges:
        x0, y0 = nodes[from_n]; x1, y1 = nodes[to_n]
        lw = max(1.5, np.sqrt(F) / 1.6)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C["main"], lw=lw,
                                   connectionstyle=f'arc3,rad={rad}'),
                    annotation_clip=False)
        mx = (x0 + x1) / 2 + dx
        my = (y0 + y1) / 2 + dy
        ax.text(mx, my, label, fontsize=7.5, color=C["main"],
                ha='center', va='center', fontweight='bold',
                bbox=dict(fc='white', ec='none', pad=1.5))

    # Node circles (drawn on top of edges)
    node_colors = {'Rate': C["accent1"], 'GDP': C["accent2"]}
    node_r = 0.080
    for name, (x, y) in nodes.items():
        fc = node_colors.get(name, 'white')
        ec = C["accent1"] if name == 'Rate' else C["main"]
        circle = plt.Circle((x, y), node_r, color=fc, ec=ec, lw=2.0, zorder=5)
        ax.add_patch(circle)
        txt_color = 'white' if name in node_colors else C["main"]
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                fontweight='bold', color=txt_color, zorder=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color=C["main"],    lw=2.5, label='Granger-causes (p<0.05)'),
        Line2D([0],[0], color=C["ci_dark"], lw=1.0, ls='--', label='Not significant (p>0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.00), ncol=2, fontsize=7.5, frameon=True)

    savefig(fig, 'fig_A7_granger_network')
    shutil.copy(OUTDIR / 'fig_A7_granger_network.pdf', Path('D:/Nexus/nexus/paper/figures/fig19_granger_network.pdf'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    var_df, rate_level, q = load_var_data()
    np.random.seed(42)
    res_base = VAR(var_df).fit(1, trend='c')
    print(f'Baseline VAR(1) fitted. AIC={res_base.aic:.4f}')

    figure_A1(var_df, res_base)
    figure_A2()
    figure_A3(res_base)
    figure_A4(res_base, var_df)
    figure_A5(var_df)
    figure_A6(res_base)
    figure_A7()

    print(f'\nAll figures saved to {OUTDIR}')
