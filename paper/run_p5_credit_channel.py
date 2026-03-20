#!/usr/bin/env python3
"""
run_p5_credit_channel.py — Credit channel, FEVD, Granger causality, stability
"""
import sys, io, warnings
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

BASELINE_PEAK_GDP = -0.195

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Palatino','Georgia','Times New Roman','DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

lines = []
def log(s=''):
    print(s)
    lines.append(str(s))

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
    log(f"VAR data: T={len(var_df)}")
    return var_df, q['rate_level'].copy(), covid_dummy, q

def compute_cholesky_irf(var_result, shock_var_idx, horizon=9, n_boot=500,
                          response_idx=1, lags=1):
    K = var_result.neqs
    resids = var_result.resid
    Sigma  = np.array(var_result.sigma_u)
    try:
        P = np.linalg.cholesky(Sigma)
    except Exception:
        P = np.linalg.cholesky(Sigma + np.eye(K) * 1e-8)

    def irf_from_coefs_and_P(coefs_arr, P_mat, h, shock_idx, resp_idx):
        K_ = coefs_arr[0].shape[0]
        A1 = coefs_arr[0]
        impact = P_mat[:, shock_idx]
        irfs = np.zeros(h)
        Ah = np.eye(K_)
        for hh in range(h):
            irfs[hh] = (Ah @ impact)[resp_idx]
            Ah = Ah @ A1
        return irfs

    norm_factor = P[shock_var_idx, shock_var_idx]
    if abs(norm_factor) < 1e-10: norm_factor = 1.0
    irf_point = irf_from_coefs_and_P(var_result.coefs, P, horizon, shock_var_idx, response_idx) / norm_factor

    boot_irfs = np.zeros((n_boot, horizon))
    T_data = resids.shape[0]
    for b in range(n_boot):
        idx = np.random.randint(0, T_data, size=T_data)
        boot_resids = resids.values[idx]
        coefs0 = var_result.coefs
        Y_boot = np.zeros((T_data + lags, K))
        Y_boot[:lags] = var_result.model.endog[:lags]
        for t in range(lags, T_data + lags):
            pred = np.zeros(K)
            for lag_i, c in enumerate(coefs0):
                pred += Y_boot[t - lag_i - 1] @ c.T
            pred += boot_resids[t - lags]
            Y_boot[t] = pred
        try:
            Y_df = pd.DataFrame(Y_boot[lags:], columns=var_result.model.endog_names)
            res_b = VAR(Y_df).fit(lags, trend='c')
            Sigma_b = np.array(res_b.sigma_u)
            try:
                P_b = np.linalg.cholesky(Sigma_b)
            except Exception:
                P_b = np.linalg.cholesky(Sigma_b + np.eye(K) * 1e-8)
            norm_b = P_b[shock_var_idx, shock_var_idx]
            if abs(norm_b) < 1e-10: norm_b = 1.0
            boot_irfs[b] = irf_from_coefs_and_P(res_b.coefs, P_b, horizon, shock_var_idx, response_idx) / norm_b
        except Exception:
            boot_irfs[b] = irf_point
    ci_lo = np.percentile(boot_irfs, 5, axis=0)
    ci_hi = np.percentile(boot_irfs, 95, axis=0)
    return irf_point, ci_lo, ci_hi


def main():
    log('='*70)
    log('SCRIPT 5: CREDIT CHANNEL, FEVD, GRANGER, STABILITY')
    log('='*70)

    var_df, rate_level, covid_dummy, q = load_var_data()
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])
    all_series = sorted(raw['series_id'].unique())

    # ─── Step 0: Data availability ────────────────────────────────────────────
    log('\n' + '='*70)
    log('STEP 0: SERIES AVAILABILITY IN PANEL')
    log('='*70)
    log(f'Total series: {len(all_series)}')
    for prefix in ['PN00','PN01','PN02','PN03','PN04','PN05','PN06','PN07','PN08','PD04','PD05']:
        matches = [s for s in all_series if s.startswith(prefix)]
        if matches:
            log(f'  {prefix}*: {matches[:15]}')

    target_series = ['PN00661MM','PN00662MM','PN02069MM','PD04718MM',
                     'PD04719MM','PD04720MM','PD04721MM','PD04723MM','PD04724MM']
    found_extra = []
    for code in target_series:
        found = code in all_series
        log(f'  {code}: {"FOUND" if found else "not found"}')
        if found:
            found_extra.append(code)

    # ─── Baseline VAR ─────────────────────────────────────────────────────────
    log('\n' + '='*70)
    log('STEP 1: BASELINE VAR(1) — GRANGER CAUSALITY')
    log('='*70)

    var_model = VAR(var_df)
    var_result = var_model.fit(1, trend='c')
    col_names = list(var_df.columns)
    log(f'VAR(1) estimated. T={var_result.nobs}, K={var_result.neqs}')

    # Pairwise Granger causality
    log('\nPairwise Granger causality (H0: X does not Granger-cause Y):')
    log(f"  {'Causing':<8} {'Caused':<8}  F-stat    p-value")
    log('  ' + '-'*45)
    for causing in col_names:
        for caused in col_names:
            if causing == caused:
                continue
            try:
                res_g = var_result.test_causality(caused=caused, causing=causing, kind='f')
                F = res_g.test_statistic
                p = res_g.pvalue
                sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
                log(f'  {causing:<8} → {caused:<8}  {F:8.3f}  {p:8.4f}  {sig}')
            except Exception as e:
                log(f'  {causing:<8} → {caused:<8}  (error: {e})')

    # ─── Step 2: FEVD ─────────────────────────────────────────────────────────
    log('\n' + '='*70)
    log('STEP 2: FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)')
    log('='*70)

    fevd = var_result.fevd(20)
    # fevd.decomp shape: (K, H, K) — [variable, horizon, shock]
    gdp_idx = col_names.index('gdp')
    rate_idx = col_names.index('rate')

    log(f'\nFEVD for GDP (% variance explained by each shock):')
    header = f"  {'h':>3}" + ''.join(f"  {c:>8}" for c in col_names)
    log(header)
    for h in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
        h_idx = h - 1
        row = f"  {h:>3}"
        for k in range(var_result.neqs):
            row += f"  {fevd.decomp[gdp_idx, h_idx, k]*100:>7.1f}%"
        log(row)

    log(f'\nShare of GDP variance explained by RATE shock:')
    for h in [1, 4, 8, 12, 20]:
        h_idx = h - 1
        share = fevd.decomp[gdp_idx, h_idx, rate_idx] * 100
        log(f'  h={h:>2}: {share:.1f}%')

    # Plot FEVD
    fig, ax = plt.subplots(figsize=(8, 4))
    horizons = list(range(1, 21))
    bottom = np.zeros(20)
    colors = ['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c']
    for k, (col, color) in enumerate(zip(col_names, colors)):
        shares = [fevd.decomp[gdp_idx, h-1, k] * 100 for h in horizons]
        ax.bar(horizons, shares, bottom=bottom, color=color, label=col, alpha=0.85)
        bottom += np.array(shares)
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('% of forecast error variance')
    ax.set_title('GDP Forecast Error Variance Decomposition')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0.5, 20.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'p5_fevd.pdf')
    plt.close(fig)
    log(f'\nFigure saved: {OUT_DIR / "p5_fevd.pdf"}')

    # ─── Step 3: Lending spread (if available) ────────────────────────────────
    log('\n' + '='*70)
    log('STEP 3: LENDING RATE / SPREAD (CONDITIONAL)')
    log('='*70)

    lending_series_found = None
    for code in ['PD04718MM','PD04719MM','PD04720MM','PD04721MM','PD04723MM','PD04724MM']:
        if code in all_series:
            lending_series_found = code
            break

    if lending_series_found:
        log(f'Lending rate series found: {lending_series_found}')
        s = raw[raw['series_id'] == lending_series_found][['date','value_raw']].copy()
        s = s.set_index('date').sort_index()
        lending_q = s['value_raw'].resample('QE').mean()
        lending_q = lending_q.reindex(var_df.index)
        spread = lending_q - rate_level.reindex(var_df.index)
        spread = spread.dropna()
        common_idx = var_df.index.intersection(spread.index)
        var_df6 = var_df.loc[common_idx].copy()
        var_df6['spread'] = spread.loc[common_idx]
        var_df6 = var_df6[['tot','gdp','cpi','fx','spread','rate']]
        log(f'6-var VAR with spread: T={len(var_df6)}')
        try:
            res6 = VAR(var_df6).fit(1, trend='c')
            irf6, ci6_lo, ci6_hi = compute_cholesky_irf(res6, shock_var_idx=5,
                                                         response_idx=1, horizon=9)
            log(f'6-var VAR GDP IRF peak: {irf6.min():.4f} at h={irf6.argmin()}')
            log(f'Baseline 5-var peak: {BASELINE_PEAK_GDP}')
            irf6_spread, _, _ = compute_cholesky_irf(res6, shock_var_idx=5,
                                                      response_idx=4, horizon=9)
            log(f'Spread IRF (response to rate shock): {irf6_spread}')
        except Exception as e:
            log(f'6-var VAR error: {e}')
    else:
        log('No lending rate series found in panel.')
        log('To add: download TAMN (PD04718MM) from BCRP estadísticas.')
        log('Falling back to baseline 5-var analysis only.')

    # ─── Step 4: Credit VAR (if available) ────────────────────────────────────
    log('\n' + '='*70)
    log('STEP 4: CREDIT CHANNEL VAR (CONDITIONAL)')
    log('='*70)

    credit_found = [c for c in ['PN00661MM','PN00662MM','PN02069MM'] if c in all_series]
    if credit_found:
        log(f'Credit series found: {credit_found}')
        for code in credit_found:
            s = raw[raw['series_id'] == code][['date','value_raw']].copy()
            s = s.set_index('date').sort_index()
            credit_q = s['value_raw'].resample('QE').mean().pct_change() * 100
            credit_q = credit_q.dropna()
            common_idx = var_df.index.intersection(credit_q.index)
            if len(common_idx) < 30:
                log(f'  {code}: too few obs ({len(common_idx)}), skipping')
                continue
            var_df6 = var_df.loc[common_idx].copy()
            var_df6['credit'] = credit_q.loc[common_idx]
            var_df6 = var_df6[['tot','gdp','cpi','fx','credit','rate']]
            log(f'  {code}: T={len(var_df6)}')
            try:
                res6 = VAR(var_df6).fit(1, trend='c')
                irf6_gdp, ci6_lo, ci6_hi = compute_cholesky_irf(res6, shock_var_idx=5,
                                                                  response_idx=1, horizon=9)
                irf6_cred, _, _ = compute_cholesky_irf(res6, shock_var_idx=5,
                                                        response_idx=4, horizon=9)
                log(f'  GDP IRF peak: {irf6_gdp.min():.4f} (baseline: {BASELINE_PEAK_GDP})')
                log(f'  Credit IRF peak: {irf6_cred.min():.4f} at h={irf6_cred.argmin()}')
            except Exception as e:
                log(f'  VAR error: {e}')
    else:
        log('Credit series not found in panel — augmented VAR skipped.')
        log('Download: PN00661MM (MN credit), PN00662MM (ME credit) from BCRP.')

    # ─── Step 5: Stability ────────────────────────────────────────────────────
    log('\n' + '='*70)
    log('STEP 5: VAR STABILITY & RECURSIVE IRF')
    log('='*70)

    A1 = var_result.coefs[0]
    eigenvalues = np.linalg.eigvals(A1)
    moduli = np.abs(eigenvalues)
    log(f'Eigenvalue moduli of A1: {np.sort(moduli)[::-1].round(4).tolist()}')
    log(f'Max modulus: {moduli.max():.4f} (< 1 → stationary)')

    # Unit circle plot
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.5)
    for ev in eigenvalues:
        ax.plot(ev.real, ev.imag, 'bo', ms=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('VAR(1) Eigenvalues (Unit Circle)')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'p5_stability.pdf')
    plt.close(fig)
    log(f'Stability plot saved: {OUT_DIR / "p5_stability.pdf"}')

    # Recursive expanding-window IRF
    log('\nRecursive expanding-window peak GDP IRF:')
    log(f"  {'T_end':<8}  {'Peak':>8}  {'h*':>4}")
    recursive_peaks = []
    for T_end in [60, 65, 70, 75, 80, 85, len(var_df)]:
        T_end = min(T_end, len(var_df))
        sub = var_df.iloc[:T_end]
        try:
            res_r = VAR(sub).fit(1, trend='c')
            irf_r, _, _ = compute_cholesky_irf(res_r, shock_var_idx=4,
                                                response_idx=1, horizon=9, n_boot=200)
            peak_r = irf_r.min()
            h_r = irf_r.argmin()
            log(f'  T={T_end:<6}  {peak_r:>8.4f}  h={h_r}')
            recursive_peaks.append((T_end, peak_r))
        except Exception as e:
            log(f'  T={T_end}: error: {e}')

    if recursive_peaks:
        Ts, peaks = zip(*recursive_peaks)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ts, peaks, 'ko-', ms=6)
        ax.axhline(BASELINE_PEAK_GDP, color='gray', ls='--', lw=1, label=f'Baseline {BASELINE_PEAK_GDP}')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Sample size T')
        ax.set_ylabel('Peak GDP IRF (pp)')
        ax.set_title('Recursive Expanding-Window Peak GDP IRF')
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'p5_recursive_irf.pdf')
        plt.close(fig)
        log(f'Recursive IRF plot saved: {OUT_DIR / "p5_recursive_irf.pdf"}')

    # ─── Save output ─────────────────────────────────────────────────────────
    out_txt = OUT_DIR / 'p5_credit_results.txt'
    out_txt.write_text('\n'.join(lines), encoding='utf-8')
    log(f'\nResults saved: {out_txt}')
    log('Done.')


if __name__ == '__main__':
    main()
