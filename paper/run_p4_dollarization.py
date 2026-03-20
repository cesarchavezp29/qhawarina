#!/usr/bin/env python3
import sys, io, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUT_DIR = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(ROOT / 'paper'))
from paper_style import apply_style, C, SZ, zero_line, legend_below, legend_outside, stat_box
apply_style()

BASELINE_PEAK_GDP = -0.195
BASELINE_POV_BETA = -0.656
BASELINE_POV_SE   =  0.115

POV_DATA = [
    (2005, 6.282, -2.7), (2006, 7.555, -5.0), (2007, 8.470, -6.3),
    (2008, 9.185, -5.0), (2009, 1.123, -3.4), (2010, 8.283, -4.5),
    (2011, 6.380, -3.5), (2012, 6.145, -2.3), (2013, 5.827, -1.7),
    (2014, 2.453, -0.4), (2015, 3.223, -1.7), (2016, 3.975, -1.1),
    (2017, 2.515,  0.0), (2018, 3.957, -1.6), (2019, 2.250, -0.6),
    (2022, 2.857,  1.5), (2023,-0.345,  1.5), (2024, 3.473, -2.1),
]
POV_YEARS, POV_GDP, POV_DPOV = map(list, zip(*POV_DATA))

def load_var_data():
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])
    series_map = {
        'PD04722MM': 'rate_raw',
        'PN01731AM': 'gdp_m',
        'PN01271PM': 'cpi_m',
        'PN01246PM': 'fx_m',
        'PN38923BM': 'tot_m',
    }
    frames = []
    for sid, col in series_map.items():
        s = raw[raw['series_id'] == sid][['date','value_raw']].copy()
        s = s.rename(columns={'value_raw': col}).set_index('date').sort_index()
        frames.append(s)
    monthly = frames[0].join(frames[1:], how='outer').sort_index()
    q = pd.DataFrame()
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()
    q['gdp']  = monthly['gdp_m'].resample('QE').sum()
    q['cpi']  = monthly['cpi_m'].resample('QE').sum()
    q['fx']   = monthly['fx_m'].resample('QE').mean().pct_change() * 100
    q['tot']  = monthly['tot_m'].resample('QE').mean().pct_change() * 100
    q = q.dropna().loc['2004-04-01':'2025-09-30']
    covid_dummy = pd.Series(0, index=q.index, name='covid')
    for cq in [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1
    def fwl_partial(series, dummy):
        X = sm.add_constant(dummy)
        res = sm.OLS(series, X).fit()
        return series - res.fittedvalues + series.mean()
    for col in ['d_rate','gdp','cpi','fx','tot','rate_level']:
        if col in q.columns:
            q[col] = fwl_partial(q[col], covid_dummy)
    var_df = q[['tot','gdp','cpi','fx','d_rate']].copy()
    var_df.columns = ['tot','gdp','cpi','fx','rate']
    print(f"VAR data: T={len(var_df)}")
    return var_df, q['rate_level'].copy(), covid_dummy, q


# =============================================================================
# HELPER: compute IRF from companion matrix
# =============================================================================

def compute_irf(A1, P, n_ahead, shock_idx, resp_idx):
    """
    Compute IRF of variable resp_idx to structural shock shock_idx.
    A1: (K x K) coefficient matrix (VAR(1))
    P:  (K x K) Cholesky factor of Sigma_u  (lower triangular)
    n_ahead: number of periods
    """
    K = A1.shape[0]
    # Normalize to 1pp (100bp) shock: divide by P[shock_idx, shock_idx]
    norm = P[shock_idx, shock_idx]
    if abs(norm) < 1e-10:
        norm = 1.0
    irf = np.zeros(n_ahead + 1)
    Phi_h = np.eye(K)
    for h in range(n_ahead + 1):
        response_mat = Phi_h @ P
        irf[h] = response_mat[resp_idx, shock_idx] / norm
        Phi_h = Phi_h @ A1
    return irf


# =============================================================================
# STEP 0 — Data availability check
# =============================================================================

def step0_data_check(raw):
    print("=" * 70)
    print("STEP 0 — Data Availability Check")
    print("=" * 70)
    all_series = raw['series_id'].unique()
    print(f"Total series in panel: {len(all_series)}")

    credit_candidates = [s for s in all_series if any(
        k in str(s).upper() for k in ['PN006', 'PN007', 'PD047', 'CREDIT'])]
    print(f"Candidate credit/rate series: {credit_candidates[:30]}")

    pn00 = sorted([s for s in all_series if str(s).startswith('PN00')])
    print(f"PN00* series: {pn00[:50]}")

    pd05 = sorted([s for s in all_series if str(s).startswith('PD04')])
    print(f"PD04* series: {pd05[:50]}")

    # Search more broadly for credit/lending related series
    broader = [s for s in all_series if any(
        k in str(s).upper() for k in ['661', '662', '663', 'TAMN', 'TAME', 'ACTIV'])]
    print(f"Broader credit search: {broader[:30]}")

    credit_series_found = {}
    for candidate in ['PN00661MM', 'PN00662MM']:
        if candidate in all_series:
            credit_series_found[candidate] = candidate
            print(f"  FOUND: {candidate}")
        else:
            print(f"  NOT FOUND: {candidate}")

    return credit_series_found


# =============================================================================
# STEP 1 — Historical Decomposition of FX
# =============================================================================

def step1_historical_decomposition(var_df, output_lines):
    print("\n" + "=" * 70)
    print("STEP 1 — Historical Decomposition of FX")
    print("=" * 70)
    output_lines.append("=" * 70)
    output_lines.append("STEP 1 — Historical Decomposition of FX")
    output_lines.append("=" * 70)

    K = var_df.shape[1]  # 5
    var_names = list(var_df.columns)  # ['tot','gdp','cpi','fx','rate']
    fx_idx = var_names.index('fx')

    # Estimate VAR(1)
    model = VAR(var_df)
    results = model.fit(1, trend='c')

    # Coefficient matrix A1 (K x K), drop constant
    # statsmodels VAR: params shape (K*lags + 1, K), rows = [const, lag1_var1, ..., lag1_varK]
    A1 = results.coefs[0]  # (K x K)
    Sigma_u = results.sigma_u  # (K x K)

    # Cholesky decomposition
    P = np.linalg.cholesky(Sigma_u)  # lower triangular

    # Residuals from VAR (T-1 x K, since VAR(1) loses first obs)
    resid_mat = results.resid.values  # (T_eff x K)
    T_eff = resid_mat.shape[0]

    # Structural shocks: u_struct = P^{-1} @ residuals.T  => (K x T_eff)
    P_inv = np.linalg.inv(P)
    u_struct = (P_inv @ resid_mat.T)  # (K x T_eff)

    # Historical decomposition for FX variable
    # contrib_jk(t) = sum_{h=0}^{t} [Phi_h @ P]_{j,k} * u_struct_k(t-h)
    # We work with t=0..T_eff-1
    contrib = np.zeros((K, T_eff))  # contrib[k, t] = contribution of shock k to fx at t

    for t in range(T_eff):
        Phi_h = np.eye(K)
        for h in range(t + 1):
            response_mat = Phi_h @ P  # (K x K)
            # contribution of each shock to fx at time t
            for k in range(K):
                contrib[k, t] += response_mat[fx_idx, k] * u_struct[k, t - h]
            Phi_h = Phi_h @ A1

    # Demeaned actual FX series (aligned with residual index)
    fx_actual = var_df['fx'].iloc[1:].values - var_df['fx'].iloc[1:].mean()
    fx_reconstructed = contrib.sum(axis=0)

    recon_corr = np.corrcoef(fx_actual, fx_reconstructed)[0, 1]
    print(f"  Reconstruction check: corr(actual, reconstructed) = {recon_corr:.4f}")
    output_lines.append(f"  Reconstruction check: corr(actual, reconstructed) = {recon_corr:.4f}")

    # Variance shares
    var_total = np.var(fx_actual)
    shock_labels = ['ToT shock', 'GDP shock', 'CPI shock', 'FX shock', 'Rate shock']
    var_shares = []
    for k in range(K):
        share = np.var(contrib[k]) / var_total * 100
        var_shares.append(share)
        msg = f"  {shock_labels[k]:<15}: {share:6.2f}% of FX variance"
        print(msg)
        output_lines.append(msg)

    output_lines.append("")

    # Dates for plotting
    dates = var_df.index[1:]

    # Figure: stacked area chart
    fig, ax = plt.subplots(figsize=SZ["wide_tall"])

    colors = [C["accent2"], C["accent3"], C["accent1"], C["gray_line"], C["main"]]
    contrib_pos = np.maximum(contrib, 0)
    contrib_neg = np.minimum(contrib, 0)

    # Positive contributions
    bottom_pos = np.zeros(T_eff)
    for k in range(K):
        ax.fill_between(dates, bottom_pos, bottom_pos + contrib_pos[k],
                        alpha=0.75, color=colors[k], label=shock_labels[k])
        bottom_pos += contrib_pos[k]

    # Negative contributions
    bottom_neg = np.zeros(T_eff)
    for k in range(K):
        ax.fill_between(dates, bottom_neg + contrib_neg[k], bottom_neg,
                        alpha=0.75, color=colors[k])
        bottom_neg += contrib_neg[k]

    ax.plot(dates, fx_actual, color=C["main"], lw=1.5, label='Actual FX (demeaned)', zorder=10)
    zero_line(ax)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('FX change contribution (pp)')
    ax.set_title('')
    legend_below(ax, ncol=3)

    fig_path = OUT_DIR / 'p4_fx_decomp.pdf'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure saved to: {fig_path}")

    return A1, P, Sigma_u, results, var_names, var_shares, shock_labels


# =============================================================================
# STEP 2 — Channel Decomposition (FX blocking)
# =============================================================================

def step2_channel_decomposition(A1, P, var_names, output_lines):
    print("\n" + "=" * 70)
    print("STEP 2 — Channel Decomposition (FX blocking)")
    print("=" * 70)
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("STEP 2 — Channel Decomposition (FX blocking)")
    output_lines.append("=" * 70)

    K = A1.shape[0]
    fx_idx = var_names.index('fx')
    gdp_idx = var_names.index('gdp')
    rate_idx = var_names.index('rate')
    n_ahead = 20

    # Baseline IRF: GDP response to rate shock
    irf_baseline = compute_irf(A1, P, n_ahead, rate_idx, gdp_idx)

    # Blocked FX channel: zero out FX row and column in A1
    A1_blocked = A1.copy()
    A1_blocked[fx_idx, :] = 0.0   # FX equation: FX no longer responds to anything
    A1_blocked[:, fx_idx] = 0.0   # Other equations: no longer respond to FX

    irf_blocked = compute_irf(A1_blocked, P, n_ahead, rate_idx, gdp_idx)

    # FX channel contribution
    irf_fx_channel = irf_baseline - irf_blocked

    # Peak GDP response
    peak_baseline = irf_baseline[np.argmin(irf_baseline)]
    peak_blocked = irf_blocked[np.argmin(irf_blocked)]
    peak_h_base = int(np.argmin(irf_baseline))
    peak_h_block = int(np.argmin(irf_blocked))

    # Share of peak attributable to FX channel
    if peak_baseline != 0:
        fx_share_peak = (peak_baseline - peak_blocked) / peak_baseline * 100
    else:
        fx_share_peak = 0.0

    lines = [
        f"  Baseline peak GDP response: {peak_baseline:.4f} at h={peak_h_base}",
        f"  Blocked FX peak GDP response: {peak_blocked:.4f} at h={peak_h_block}",
        f"  FX channel contribution to peak: {fx_share_peak:.1f}%",
        f"  (Baseline BCRP paper estimate: {BASELINE_PEAK_GDP:.4f})",
    ]
    for line in lines:
        print(line)
        output_lines.append(line)

    # Variance share of FX channel
    var_base = np.sum(irf_baseline[1:]**2)
    var_fx_ch = np.sum(irf_fx_channel[1:]**2)
    if var_base > 0:
        fx_var_share = var_fx_ch / var_base * 100
    else:
        fx_var_share = 0.0
    msg = f"  FX channel: {fx_var_share:.1f}% of cumulative squared GDP IRF"
    print(msg)
    output_lines.append(msg)
    output_lines.append("")

    # Figure: IRF comparison
    h_range = np.arange(n_ahead + 1)
    fig, axes = plt.subplots(1, 2, figsize=SZ["wide_tall"])

    ax = axes[0]
    ax.plot(h_range, irf_baseline, color=C["accent2"], marker='o', ms=4, lw=2, label='Baseline VAR')
    ax.plot(h_range, irf_blocked, color=C["accent1"], marker='s', ms=4, lw=2, ls='--', label='FX channel blocked')
    ax.fill_between(h_range, irf_blocked, irf_baseline,
                    alpha=0.15, color=C["accent3"], label='FX channel contribution')
    zero_line(ax)
    ax.set_xlabel('Quarters after rate shock')
    ax.set_ylabel('GDP response (pp)')
    ax.set_title('')
    legend_below(ax, ncol=3)
    stat_box(ax, f'FX channel: {fx_share_peak:.1f}% of peak\n({fx_var_share:.1f}% of cum. variance)',
             loc='lower right')

    ax2 = axes[1]
    ax2.bar(h_range[1:], irf_fx_channel[1:], color=C["accent3"], alpha=0.8,
            edgecolor=C["accent3"], label='FX channel (baseline - blocked)')
    zero_line(ax2)
    ax2.set_xlabel('Quarters after rate shock')
    ax2.set_ylabel('GDP response attributable to FX (pp)')
    ax2.set_title('')
    legend_below(ax2, ncol=1)

    fig_path = OUT_DIR / 'p4_channel_decomp.pdf'
    fig.savefig(fig_path)
    shutil.copy(fig_path, ROOT / 'paper' / 'figures' / 'fig18_fx_channel.pdf')
    plt.close(fig)
    print(f"\n  Figure saved to: {fig_path}")

    return irf_baseline, irf_blocked, irf_fx_channel, peak_baseline, peak_blocked, fx_share_peak


# =============================================================================
# STEP 3 — Augmented Poverty Regression with FX
# =============================================================================

def step3_augmented_poverty(var_df, q_full, output_lines):
    print("\n" + "=" * 70)
    print("STEP 3 — Augmented Poverty Regression with FX")
    print("=" * 70)
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("STEP 3 — Augmented Poverty Regression with FX")
    output_lines.append("=" * 70)

    pov_years = list(POV_YEARS)
    pov_gdp = np.array(POV_GDP)
    pov_dpov = np.array(POV_DPOV)

    # Compute annual FX change from quarterly FX data
    # var_df has 'fx' = quarterly pct change of FX level
    # Annual FX = mean of 4 quarterly pct changes for each year
    fx_q = var_df['fx'].copy()
    fx_q.index = pd.to_datetime(fx_q.index)

    annual_fx = {}
    for yr in range(2004, 2026):
        yr_data = fx_q[fx_q.index.year == yr]
        if len(yr_data) >= 3:
            annual_fx[yr] = yr_data.mean()
        elif len(yr_data) > 0:
            annual_fx[yr] = yr_data.mean()

    # Build aligned dataset
    fx_vals = []
    valid_mask = []
    for yr in pov_years:
        if yr in annual_fx and not np.isnan(annual_fx[yr]):
            fx_vals.append(annual_fx[yr])
            valid_mask.append(True)
        else:
            fx_vals.append(np.nan)
            valid_mask.append(False)

    fx_arr = np.array(fx_vals)
    valid = np.array(valid_mask) & ~np.isnan(fx_arr)

    gdp_v = pov_gdp[valid]
    dpov_v = pov_dpov[valid]
    fx_v = fx_arr[valid]
    years_v = np.array(pov_years)[valid]
    N_aug = valid.sum()

    print(f"  Matched observations: {N_aug} (out of {len(pov_years)})")
    output_lines.append(f"  Matched observations: {N_aug} (out of {len(pov_years)})")

    if N_aug < 5:
        msg = "  Insufficient FX data for augmented regression."
        print(msg)
        output_lines.append(msg)
        return None

    # Augmented OLS
    X_aug = np.column_stack([np.ones(N_aug), gdp_v, fx_v])
    ols_aug = sm.OLS(dpov_v, X_aug).fit()
    beta_gdp_aug = ols_aug.params[1]
    beta_fx_aug = ols_aug.params[2]
    se_gdp_aug = ols_aug.bse[1]
    se_fx_aug = ols_aug.bse[2]
    r2_aug = ols_aug.rsquared
    r2_adj_aug = ols_aug.rsquared_adj
    p_gdp_aug = ols_aug.pvalues[1]
    p_fx_aug = ols_aug.pvalues[2]

    # Univariate baseline (on same sample)
    X_uni = sm.add_constant(gdp_v)
    ols_uni = sm.OLS(dpov_v, X_uni).fit()
    r2_uni = ols_uni.rsquared

    lines = [
        f"",
        f"  Augmented regression: DPoverty = a + b_gdp*GDP + b_fx*FX",
        f"  N = {N_aug}",
        f"  b_gdp  = {beta_gdp_aug:.4f}  (SE={se_gdp_aug:.4f}, p={p_gdp_aug:.4f})",
        f"  b_fx   = {beta_fx_aug:.4f}  (SE={se_fx_aug:.4f}, p={p_fx_aug:.4f})",
        f"  R² (augmented) = {r2_aug:.4f}  (adj. R² = {r2_adj_aug:.4f})",
        f"  R² (univariate GDP, same sample) = {r2_uni:.4f}",
        f"  Baseline R² (full N=18) = 0.669",
        f"  Delta R² from adding FX = {r2_aug - r2_uni:.4f}",
    ]
    for line in lines:
        print(line)
        output_lines.append(line)

    # Figure: augmented regression
    fig, axes = plt.subplots(1, 2, figsize=SZ["wide_tall"])

    # Panel 1: partial regression plot — GDP after partialling out FX
    X_fx_only = sm.add_constant(fx_v)
    resid_dpov_on_fx = sm.OLS(dpov_v, X_fx_only).fit().resid
    resid_gdp_on_fx = sm.OLS(gdp_v, X_fx_only).fit().resid

    ax = axes[0]
    ax.scatter(resid_gdp_on_fx, resid_dpov_on_fx, color=C["accent2"], s=50, alpha=0.85, zorder=3)
    for i, yr in enumerate(years_v):
        ax.annotate(str(int(yr)), (resid_gdp_on_fx[i], resid_dpov_on_fx[i]),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    xfit = np.linspace(resid_gdp_on_fx.min()-0.3, resid_gdp_on_fx.max()+0.3, 200)
    ax.plot(xfit, beta_gdp_aug * xfit, color=C["accent1"], lw=2,
            label=f'β_gdp = {beta_gdp_aug:.3f} (SE={se_gdp_aug:.3f})')
    zero_line(ax)
    ax.axvline(0, color=C["zero"], lw=0.7, ls='--', zorder=0)
    ax.set_xlabel('GDP growth | FX (partial residual)')
    ax.set_ylabel('ΔPoverty | FX (partial residual)')
    ax.set_title('')
    legend_below(ax, ncol=1)

    # Panel 2: partial regression plot — FX after partialling out GDP
    X_gdp_only = sm.add_constant(gdp_v)
    resid_dpov_on_gdp = sm.OLS(dpov_v, X_gdp_only).fit().resid
    resid_fx_on_gdp = sm.OLS(fx_v, X_gdp_only).fit().resid

    ax2 = axes[1]
    ax2.scatter(resid_fx_on_gdp, resid_dpov_on_gdp, color=C["accent3"], s=50, alpha=0.85, zorder=3)
    for i, yr in enumerate(years_v):
        ax2.annotate(str(int(yr)), (resid_fx_on_gdp[i], resid_dpov_on_gdp[i]),
                     textcoords='offset points', xytext=(4, 3), fontsize=7)
    xfit2 = np.linspace(resid_fx_on_gdp.min()-0.3, resid_fx_on_gdp.max()+0.3, 200)
    ax2.plot(xfit2, beta_fx_aug * xfit2, color=C["accent1"], lw=2,
             label=f'β_fx = {beta_fx_aug:.3f} (SE={se_fx_aug:.3f})')
    zero_line(ax2)
    ax2.axvline(0, color=C["zero"], lw=0.7, ls='--', zorder=0)
    ax2.set_xlabel('Annual FX change | GDP (partial residual)')
    ax2.set_ylabel('ΔPoverty | GDP (partial residual)')
    ax2.set_title('')
    legend_below(ax2, ncol=1)
    stat_box(ax2, f'R² = {r2_aug:.3f}\nN={N_aug}', loc='upper right')

    fig_path = OUT_DIR / 'p4_poverty_fx.pdf'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure saved to: {fig_path}")

    return {
        'beta_gdp': beta_gdp_aug, 'se_gdp': se_gdp_aug,
        'beta_fx': beta_fx_aug, 'se_fx': se_fx_aug,
        'r2': r2_aug, 'r2_adj': r2_adj_aug, 'r2_uni': r2_uni,
        'N': N_aug
    }


# =============================================================================
# STEP 4 — Credit data (conditional on availability)
# =============================================================================

def step4_credit_var(raw, var_df, credit_series_found, output_lines):
    print("\n" + "=" * 70)
    print("STEP 4 — Credit Channel VAR (conditional on data availability)")
    print("=" * 70)
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("STEP 4 — Credit Channel VAR")
    output_lines.append("=" * 70)

    if not credit_series_found:
        msg = [
            "  Credit series not found in panel — augmented VAR skipped.",
            "  To add this analysis, download the following BCRP series:",
            "  - PN00661MM: Credit in domestic currency (MN), monthly",
            "  - PN00662MM: Credit in foreign currency (ME), monthly",
            "  Then re-run this script.",
        ]
        for m in msg:
            print(m)
            output_lines.append(m)
        return

    # Load credit series
    credit_frames = {}
    for sid, label in credit_series_found.items():
        s = raw[raw['series_id'] == sid][['date', 'value_raw']].copy()
        s['date'] = pd.to_datetime(s['date'])
        s = s.rename(columns={'value_raw': sid}).set_index('date').sort_index()
        credit_frames[sid] = s

    # Aggregate to quarterly, compute growth rates
    # Use first available credit series as credit_sol
    first_sid = list(credit_frames.keys())[0]
    credit_monthly = credit_frames[first_sid]
    credit_q = credit_monthly.resample('QE').mean().pct_change() * 100
    credit_q.columns = ['credit_sol']

    # FWL partial out COVID
    covid_dummy = pd.Series(0, index=credit_q.index, name='covid')
    for cq in [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1

    def fwl_partial(series, dummy):
        aligned_dummy = dummy.reindex(series.index, fill_value=0)
        X = sm.add_constant(aligned_dummy)
        res = sm.OLS(series.dropna(), X.loc[series.dropna().index]).fit()
        out = series.copy()
        out[series.dropna().index] = res.resid + series.mean()
        return out

    credit_q['credit_sol'] = fwl_partial(credit_q['credit_sol'], covid_dummy)

    # Merge with baseline VAR data
    var6_df = var_df.join(credit_q, how='inner').dropna()
    var6_df = var6_df[['tot', 'gdp', 'cpi', 'fx', 'credit_sol', 'rate']]
    print(f"  6-variable VAR data: T={len(var6_df)}")
    output_lines.append(f"  6-variable VAR data: T={len(var6_df)}")

    if len(var6_df) < 20:
        msg = "  Insufficient observations for 6-variable VAR."
        print(msg)
        output_lines.append(msg)
        return

    # Estimate VAR(1)
    model6 = VAR(var6_df)
    results6 = model6.fit(1, trend='c')

    K6 = 6
    A1_6 = results6.coefs[0]
    Sigma6 = results6.sigma_u
    P6 = np.linalg.cholesky(Sigma6)

    var_names_6 = list(var6_df.columns)
    rate_idx_6 = var_names_6.index('rate')
    gdp_idx_6 = var_names_6.index('gdp')
    credit_idx_6 = var_names_6.index('credit_sol')
    n_ahead = 20

    irf_gdp_6 = compute_irf(A1_6, P6, n_ahead, rate_idx_6, gdp_idx_6)
    irf_credit_6 = compute_irf(A1_6, P6, n_ahead, rate_idx_6, credit_idx_6)

    # Compare with baseline 5-var GDP IRF
    model5 = VAR(var_df)
    results5 = model5.fit(1, trend='c')
    A1_5 = results5.coefs[0]
    P5 = np.linalg.cholesky(results5.sigma_u)
    var_names_5 = list(var_df.columns)
    rate_idx_5 = var_names_5.index('rate')
    gdp_idx_5 = var_names_5.index('gdp')
    irf_gdp_5 = compute_irf(A1_5, P5, n_ahead, rate_idx_5, gdp_idx_5)

    peak_6 = irf_gdp_6[np.argmin(irf_gdp_6)]
    peak_5 = irf_gdp_5[np.argmin(irf_gdp_5)]
    peak_credit = irf_credit_6[np.argmin(irf_credit_6)]

    lines = [
        f"  Peak GDP IRF (5-var baseline):    {peak_5:.4f}",
        f"  Peak GDP IRF (6-var with credit): {peak_6:.4f}",
        f"  Peak credit IRF:                  {peak_credit:.4f}",
    ]
    for line in lines:
        print(line)
        output_lines.append(line)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=SZ["wide_tall"])
    h_range = np.arange(n_ahead + 1)

    ax = axes[0]
    ax.plot(h_range, irf_gdp_5, color=C["accent2"], marker='o', ms=4, lw=2, label='5-var baseline')
    ax.plot(h_range, irf_gdp_6, color=C["accent1"], marker='s', ms=4, lw=2, ls='--', label='6-var (with credit)')
    zero_line(ax)
    ax.set_xlabel('Quarters after rate shock')
    ax.set_ylabel('GDP response (pp)')
    ax.set_title('')
    legend_below(ax, ncol=2)

    ax2 = axes[1]
    ax2.plot(h_range, irf_credit_6, color=C["accent3"], marker='o', ms=4, lw=2, label='Credit (MN) response')
    zero_line(ax2)
    ax2.set_xlabel('Quarters after rate shock')
    ax2.set_ylabel('Credit growth response (pp)')
    ax2.set_title('')
    legend_below(ax2, ncol=1)

    fig_path = OUT_DIR / 'p4_credit_var.pdf'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure saved to: {fig_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(42)

    output_lines = []
    output_lines.append("P4 DOLLARIZATION AND EXCHANGE RATE CHANNELS")
    output_lines.append("Peru Monetary Policy Paper — Robustness Analysis")
    output_lines.append(f"Run date: 2026-03-20")
    output_lines.append("")

    # Load raw data once
    print("Loading panel data...")
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])
    print(f"Panel loaded: {len(raw)} rows, {raw['series_id'].nunique()} series")

    # Step 0: data check
    credit_series_found = step0_data_check(raw)

    # Load VAR data
    var_df, rate_level, covid_dummy, q_full = load_var_data()

    # Step 1: Historical decomposition
    (A1, P, Sigma_u, var_results, var_names,
     var_shares, shock_labels) = step1_historical_decomposition(var_df, output_lines)

    # Step 2: Channel decomposition
    (irf_baseline, irf_blocked, irf_fx_channel,
     peak_baseline, peak_blocked,
     fx_share_peak) = step2_channel_decomposition(A1, P, var_names, output_lines)

    # Step 3: Augmented poverty regression
    aug_results = step3_augmented_poverty(var_df, q_full, output_lines)

    # Step 4: Credit VAR
    step4_credit_var(raw, var_df, credit_series_found, output_lines)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("SUMMARY")
    output_lines.append("=" * 70)

    summary = [
        f"1. Historical decomposition of FX:",
        f"   Dominant shock: {shock_labels[np.argmax(var_shares)]} ({max(var_shares):.1f}% of FX variance)",
        f"",
        f"2. Channel decomposition:",
        f"   Baseline peak GDP response: {peak_baseline:.4f}",
        f"   FX-blocked peak GDP response: {peak_blocked:.4f}",
        f"   FX channel contributes ~{fx_share_peak:.1f}% of peak GDP effect",
        f"",
    ]

    if aug_results is not None:
        summary += [
            f"3. Augmented poverty regression (N={aug_results['N']}):",
            f"   GDP coefficient: {aug_results['beta_gdp']:.4f} (SE={aug_results['se_gdp']:.4f})",
            f"   FX coefficient:  {aug_results['beta_fx']:.4f} (SE={aug_results['se_fx']:.4f})",
            f"   R² = {aug_results['r2']:.4f} vs. univariate R² = {aug_results['r2_uni']:.4f}",
            f"   Adding FX raises R² by {aug_results['r2'] - aug_results['r2_uni']:.4f}",
        ]
    else:
        summary.append("3. Augmented poverty regression: insufficient matched FX data.")

    for line in summary:
        print(line)
        output_lines.append(line)

    # Save all text output
    out_txt = OUT_DIR / 'p4_dollarization_results.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print(f"\nAll text output saved to: {out_txt}")
    print(f"\nDone. Figures and results written to: {OUT_DIR}")


if __name__ == '__main__':
    main()
