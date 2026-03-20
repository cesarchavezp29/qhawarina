#!/usr/bin/env python3
"""
run_p1_frequency_chain.py — Fix QoQ vs YoY frequency mismatch in the poverty chain estimate.

Three methods:
  A — Cumulate QoQ IRF to get YoY-equivalent, then chain with annual β
  B — Re-estimate poverty regression using annual-avg-of-QoQ GDP measure
  C — Direct annual rate→poverty regression with GDP control
"""
import sys, io, warnings, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')

# ─── Paths & constants ──────────────────────────────────────────────────────
ROOT     = Path('D:/Nexus/nexus')
PANEL    = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUT_DIR  = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(ROOT / 'paper'))
from paper_style import apply_style, C, SZ, zero_line, legend_below, legend_outside, stat_box
apply_style()

BASELINE_PEAK_GDP  = -0.195   # pp at h=3
BASELINE_CI_LO     = -0.698
BASELINE_CI_HI     = +0.271
BASELINE_POV_BETA  = -0.656
BASELINE_POV_SE    =  0.115

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
        s = raw[raw['series_id'] == sid][['date', 'value_raw']].copy()
        s = s.rename(columns={'value_raw': col})
        s = s.set_index('date').sort_index()
        frames.append(s)
    monthly = frames[0].join(frames[1:], how='outer').sort_index()
    q = pd.DataFrame()
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()
    q['gdp'] = monthly['gdp_m'].resample('QE').sum()
    q['cpi'] = monthly['cpi_m'].resample('QE').sum()
    q['fx']  = monthly['fx_m'].resample('QE').mean().pct_change() * 100
    q['tot'] = monthly['tot_m'].resample('QE').mean().pct_change() * 100
    q = q.dropna()
    q = q.loc['2004-04-01':'2025-09-30']
    covid_dummy = pd.Series(0, index=q.index, name='covid')
    for cq in [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1
    def fwl_partial(series, dummy):
        X = sm.add_constant(dummy)
        res = sm.OLS(series, X).fit()
        return series - res.fittedvalues + series.mean()
    for col in ['d_rate', 'gdp', 'cpi', 'fx', 'tot', 'rate_level']:
        if col in q.columns:
            q[col] = fwl_partial(q[col], covid_dummy)
    var_df = q[['tot', 'gdp', 'cpi', 'fx', 'd_rate']].copy()
    var_df.columns = ['tot', 'gdp', 'cpi', 'fx', 'rate']
    print(f"VAR data: T={len(var_df)}")
    return var_df, q['rate_level'].copy(), covid_dummy, q


def compute_cholesky_irf(var_result, shock_var_idx=4, horizon=9, n_boot=500,
                          response_idx=1, data=None, lags=1):
    K = var_result.neqs
    nobs = var_result.nobs
    resids = var_result.resid
    Sigma  = var_result.sigma_u
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        P = np.linalg.cholesky(Sigma + np.eye(K) * 1e-8)

    def irf_from_coefs_and_P(coefs_arr, P_mat, h, shock_idx, resp_idx):
        K_ = coefs_arr[0].shape[0]
        p  = len(coefs_arr)
        if p == 1:
            A1 = coefs_arr[0]
            impact = P_mat[:, shock_idx]
            irfs = np.zeros(h)
            Ah = np.eye(K_)
            for hh in range(h):
                irfs[hh] = (Ah @ impact)[resp_idx]
                Ah = Ah @ A1
        else:
            A = np.zeros((K_*p, K_*p))
            for i, c in enumerate(coefs_arr):
                A[:K_, i*K_:(i+1)*K_] = c
            A[K_:, :-K_] = np.eye(K_ * (p-1))
            e_shock = np.zeros(K_*p)
            e_shock[:K_] = P_mat[:, shock_idx]
            irfs = np.zeros(h)
            Ah = np.eye(K_*p)
            for hh in range(h):
                irfs[hh] = (Ah @ e_shock)[resp_idx]
                Ah = Ah @ A
        return irfs

    norm_factor = P[shock_var_idx, shock_var_idx]
    if abs(norm_factor) < 1e-10:
        norm_factor = 1.0
    irf_point = irf_from_coefs_and_P(var_result.coefs, P, horizon, shock_var_idx, response_idx)
    irf_point = irf_point / norm_factor

    boot_irfs = np.zeros((n_boot, horizon))
    T_data = resids.shape[0]
    for b in range(n_boot):
        idx = np.random.randint(0, T_data, size=T_data)
        boot_resids = resids.values[idx]
        coefs0 = var_result.coefs
        Y_boot = np.zeros((T_data + lags, K))
        act_vals = var_result.model.endog
        Y_boot[:lags] = act_vals[:lags]
        for t in range(lags, T_data + lags):
            pred = np.zeros(K)
            for lag_i, c in enumerate(coefs0):
                pred += Y_boot[t - lag_i - 1] @ c.T
            pred += boot_resids[t - lags]
            Y_boot[t] = pred
        try:
            Y_df = pd.DataFrame(Y_boot[lags:], columns=var_result.model.endog_names)
            res_b = VAR(Y_df).fit(lags, trend='c')
            Sigma_b = res_b.sigma_u
            try:
                P_b = np.linalg.cholesky(Sigma_b)
            except:
                P_b = np.linalg.cholesky(Sigma_b + np.eye(K) * 1e-8)
            norm_b = P_b[shock_var_idx, shock_var_idx]
            if abs(norm_b) < 1e-10: norm_b = 1.0
            boot_irfs[b] = irf_from_coefs_and_P(res_b.coefs, P_b, horizon, shock_var_idx, response_idx) / norm_b
        except:
            boot_irfs[b] = irf_point
    ci_lo = np.percentile(boot_irfs, 5, axis=0)
    ci_hi = np.percentile(boot_irfs, 95, axis=0)
    return irf_point, ci_lo, ci_hi


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("=" * 70)
    print("P1: Frequency-chain robustness check")
    print("=" * 70)

    var_df, rate_level, covid_dummy, q_full = load_var_data()

    # Fit baseline VAR(1)
    var_model = VAR(var_df)
    var_res   = var_model.fit(1, trend='c')
    print(f"VAR(1) fitted: {var_res.neqs} variables, {var_res.nobs} observations")

    # ── Baseline chain (for comparison) ──────────────────────────────────────
    baseline_chain = BASELINE_PEAK_GDP * BASELINE_POV_BETA
    print(f"\nBaseline chain: {BASELINE_PEAK_GDP:.3f} x {BASELINE_POV_BETA:.3f} = {baseline_chain:.4f} pp poverty")

    # =========================================================================
    # METHOD A: Cumulate QoQ IRF → YoY-equivalent, chain with annual β
    # =========================================================================
    print("\n--- METHOD A: Cumulative QoQ IRF ---")

    N_BOOT_A = 2000
    irf_point, ci_lo_raw, ci_hi_raw = compute_cholesky_irf(
        var_res, shock_var_idx=4, horizon=9, n_boot=N_BOOT_A, response_idx=1, lags=1
    )

    # YoY impact = sum of first 4 quarters (h=0..3)
    gdp_yoy_1yr  = np.sum(irf_point[0:4])
    gdp_yoy_2yr  = np.sum(irf_point[0:8])

    print(f"  Point IRF (h=0..8): {np.round(irf_point, 4)}")
    print(f"  YoY impact (h=0..3, 1yr cumul): {gdp_yoy_1yr:.4f} pp")
    print(f"  YoY impact (h=0..7, 2yr cumul): {gdp_yoy_2yr:.4f} pp")

    # Chain with annual β
    chain_A_1yr = gdp_yoy_1yr * BASELINE_POV_BETA
    chain_A_2yr = gdp_yoy_2yr * BASELINE_POV_BETA
    print(f"  Chain A (1yr): {gdp_yoy_1yr:.4f} x {BASELINE_POV_BETA:.3f} = {chain_A_1yr:.4f} pp poverty")
    print(f"  Chain A (2yr): {gdp_yoy_2yr:.4f} x {BASELINE_POV_BETA:.3f} = {chain_A_2yr:.4f} pp poverty")

    # Bootstrap CI for Method A
    # Re-run bootstrap collecting all IRF paths, then compute cumulative sums
    resids  = var_res.resid
    coefs0  = var_res.coefs
    K       = var_res.neqs
    T_data  = resids.shape[0]
    lags    = 1

    Sigma = var_res.sigma_u
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        P = np.linalg.cholesky(Sigma + np.eye(K) * 1e-8)
    norm_factor = P[4, 4]
    if abs(norm_factor) < 1e-10:
        norm_factor = 1.0

    def irf_from_A1(A1, P_mat, h=9, shock_idx=4, resp_idx=1):
        impact = P_mat[:, shock_idx]
        irfs   = np.zeros(h)
        Ah     = np.eye(K)
        for hh in range(h):
            irfs[hh] = (Ah @ impact)[resp_idx]
            Ah = Ah @ A1
        return irfs

    boot_cum_1yr = np.zeros(N_BOOT_A)
    boot_cum_2yr = np.zeros(N_BOOT_A)
    boot_chain_1yr = np.zeros(N_BOOT_A)
    boot_chain_2yr = np.zeros(N_BOOT_A)

    for b in range(N_BOOT_A):
        idx_b = np.random.randint(0, T_data, size=T_data)
        boot_resids_b = resids.values[idx_b]
        Y_boot = np.zeros((T_data + lags, K))
        Y_boot[:lags] = var_res.model.endog[:lags]
        for t in range(lags, T_data + lags):
            pred = np.zeros(K)
            for lag_i, c in enumerate(coefs0):
                pred += Y_boot[t - lag_i - 1] @ c.T
            pred += boot_resids_b[t - lags]
            Y_boot[t] = pred
        try:
            Y_df_b = pd.DataFrame(Y_boot[lags:], columns=var_res.model.endog_names)
            res_b  = VAR(Y_df_b).fit(lags, trend='c')
            Sigma_b = res_b.sigma_u
            try:
                P_b = np.linalg.cholesky(Sigma_b)
            except:
                P_b = np.linalg.cholesky(Sigma_b + np.eye(K) * 1e-8)
            norm_b = P_b[4, 4]
            if abs(norm_b) < 1e-10: norm_b = 1.0
            irf_b = irf_from_A1(res_b.coefs[0], P_b) / norm_b
        except:
            irf_b = irf_point

        cum1 = np.sum(irf_b[0:4])
        cum2 = np.sum(irf_b[0:8])
        boot_cum_1yr[b] = cum1
        boot_cum_2yr[b] = cum2

        # Draw β from N(-0.656, 0.115^2)
        beta_draw = np.random.normal(BASELINE_POV_BETA, BASELINE_POV_SE)
        boot_chain_1yr[b] = cum1 * beta_draw
        boot_chain_2yr[b] = cum2 * beta_draw

    ci_A_1yr = (np.percentile(boot_chain_1yr, 5), np.percentile(boot_chain_1yr, 95))
    ci_A_2yr = (np.percentile(boot_chain_2yr, 5), np.percentile(boot_chain_2yr, 95))
    print(f"  90% CI chain A (1yr): [{ci_A_1yr[0]:.4f}, {ci_A_1yr[1]:.4f}]")
    print(f"  90% CI chain A (2yr): [{ci_A_2yr[0]:.4f}, {ci_A_2yr[1]:.4f}]")

    # =========================================================================
    # METHOD B: Re-estimate poverty regression using annual-avg-of-QoQ GDP
    # =========================================================================
    print("\n--- METHOD B: Annual-avg-of-QoQ GDP regression ---")

    # Compute annual average of quarterly GDP growth from VAR data
    # var_df['gdp'] is quarterly GDP (QoQ-equivalent after FWL)
    gdp_q = var_df['gdp'].copy()
    gdp_q.index = pd.to_datetime(gdp_q.index)

    # Annual average: group quarters by calendar year, take mean of 4 quarterly values
    gdp_annual_avg = gdp_q.groupby(gdp_q.index.year).mean()
    gdp_annual_avg.index.name = 'year'

    # Match to POV_DATA years
    pov_years_arr = np.array(POV_YEARS)
    pov_gdp_arr   = np.array(POV_GDP)     # annual YoY GDP from official sources
    pov_dpov_arr  = np.array(POV_DPOV)    # ΔPoverty rate

    # Build regression dataset for Method B
    b_rows = []
    for i, yr in enumerate(POV_YEARS):
        if yr in gdp_annual_avg.index:
            b_rows.append({
                'year': yr,
                'gdp_qqq': gdp_annual_avg.loc[yr],   # annual avg of QoQ quarters
                'dpov': POV_DPOV[i],
            })

    df_B = pd.DataFrame(b_rows).dropna()
    print(f"  Method B: {len(df_B)} matched years")
    print(f"  Years: {df_B['year'].tolist()}")

    if len(df_B) >= 4:
        X_B = sm.add_constant(df_B['gdp_qqq'])
        res_B = sm.OLS(df_B['dpov'], X_B).fit()
        beta_B = res_B.params['gdp_qqq']
        se_B   = res_B.bse['gdp_qqq']
        r2_B   = res_B.rsquared
        print(f"  β_B = {beta_B:.4f}  SE = {se_B:.4f}  R² = {r2_B:.4f}")

        chain_B = BASELINE_PEAK_GDP * beta_B
        print(f"  Chain B: {BASELINE_PEAK_GDP:.3f} x {beta_B:.4f} = {chain_B:.4f} pp poverty")

        # Bootstrap CI for Method B (resample years)
        N_BOOT_B = 2000
        boot_chain_B = np.zeros(N_BOOT_B)
        n_obs_B = len(df_B)
        for b in range(N_BOOT_B):
            idx_b = np.random.randint(0, n_obs_B, size=n_obs_B)
            df_boot = df_B.iloc[idx_b].copy()
            try:
                Xb = sm.add_constant(df_boot['gdp_qqq'])
                rb = sm.OLS(df_boot['dpov'], Xb).fit()
                beta_b = rb.params.get('gdp_qqq', beta_B)
            except:
                beta_b = beta_B
            # Also draw the IRF peak from bootstrap distribution
            irf_b  = boot_cum_1yr[b % N_BOOT_A] / 4  # reuse, convert cumul to avg quarter
            # Use point IRF for simplicity in the chain
            boot_chain_B[b] = BASELINE_PEAK_GDP * beta_b
        ci_B = (np.percentile(boot_chain_B, 5), np.percentile(boot_chain_B, 95))
        print(f"  90% CI chain B: [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")
    else:
        print("  WARNING: insufficient matched years for Method B regression.")
        beta_B, se_B, r2_B, chain_B = np.nan, np.nan, np.nan, np.nan
        ci_B = (np.nan, np.nan)

    # =========================================================================
    # METHOD C: Direct annual rate → poverty (with GDP control)
    # =========================================================================
    print("\n--- METHOD C: Direct annual rate → poverty ---")

    # Construct annual monetary stance from VAR data
    rate_annual = rate_level.groupby(rate_level.index.year).mean()
    stance_neutral = 3.5   # neutral rate assumption (approximate sample mean)
    stance_annual  = rate_annual - stance_neutral

    # Annual cumulative Δrate (sum of quarterly changes)
    d_rate_q = var_df['rate'].copy()
    d_rate_q.index = pd.to_datetime(d_rate_q.index)
    d_rate_annual = d_rate_q.groupby(d_rate_q.index.year).sum()

    # Build regression dataset for Method C
    c_rows = []
    for i, yr in enumerate(POV_YEARS):
        row = {'year': yr, 'gdp_yoy': POV_GDP[i], 'dpov': POV_DPOV[i]}
        if yr in stance_annual.index:
            row['stance'] = stance_annual.loc[yr]
        else:
            row['stance'] = np.nan
        if yr in d_rate_annual.index:
            row['d_rate_cum'] = d_rate_annual.loc[yr]
        else:
            row['d_rate_cum'] = np.nan
        c_rows.append(row)

    df_C = pd.DataFrame(c_rows).dropna(subset=['gdp_yoy', 'dpov', 'stance'])
    print(f"  Method C: {len(df_C)} years with stance data")

    # Regression A: ΔPoverty = α + β_gdp × GDP_yoy + β_rate × stance + ε
    if len(df_C) >= 5:
        X_Ca = sm.add_constant(df_C[['gdp_yoy', 'stance']])
        res_Ca = sm.OLS(df_C['dpov'], X_Ca).fit()
        beta_gdp_Ca   = res_Ca.params['gdp_yoy']
        beta_rate_Ca  = res_Ca.params['stance']
        se_gdp_Ca     = res_Ca.bse['gdp_yoy']
        se_rate_Ca    = res_Ca.bse['stance']
        t_rate_Ca     = res_Ca.tvalues['stance']
        p_rate_Ca     = res_Ca.pvalues['stance']
        r2_Ca         = res_Ca.rsquared
        print(f"  Reg C-stance:")
        print(f"    β_gdp  = {beta_gdp_Ca:.4f}  SE = {se_gdp_Ca:.4f}")
        print(f"    β_rate = {beta_rate_Ca:.4f}  SE = {se_rate_Ca:.4f}  t = {t_rate_Ca:.3f}  p = {p_rate_Ca:.4f}")
        print(f"    R² = {r2_Ca:.4f}")
        rate_has_power = p_rate_Ca < 0.10
        print(f"    Rate has independent power beyond GDP: {'YES (p<0.10)' if rate_has_power else 'NO (p>=0.10)'}")
    else:
        print("  WARNING: insufficient data for Method C regression.")
        beta_rate_Ca, se_rate_Ca, t_rate_Ca, p_rate_Ca, r2_Ca = [np.nan]*5
        rate_has_power = False

    # Also try with cumulative d_rate
    df_Cb = pd.DataFrame(c_rows).dropna(subset=['gdp_yoy', 'dpov', 'd_rate_cum'])
    if len(df_Cb) >= 5:
        X_Cb = sm.add_constant(df_Cb[['gdp_yoy', 'd_rate_cum']])
        res_Cb = sm.OLS(df_Cb['dpov'], X_Cb).fit()
        t_drate   = res_Cb.tvalues.get('d_rate_cum', np.nan)
        p_drate   = res_Cb.pvalues.get('d_rate_cum', np.nan)
        print(f"  Reg C-d_rate_cum: t = {t_drate:.3f}  p = {p_drate:.4f}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Chain Estimates")
    print("=" * 70)
    print(f"{'Method':<35} {'Chain Est':>10} {'CI Lo':>9} {'CI Hi':>9}")
    print("-" * 70)
    print(f"{'Baseline (QoQ IRF x annual β)':<35} {baseline_chain:>10.4f} {'-':>9} {'-':>9}")
    print(f"{'Method A — Cumul h=0..3 (1yr)':<35} {chain_A_1yr:>10.4f} {ci_A_1yr[0]:>9.4f} {ci_A_1yr[1]:>9.4f}")
    print(f"{'Method A — Cumul h=0..7 (2yr)':<35} {chain_A_2yr:>10.4f} {ci_A_2yr[0]:>9.4f} {ci_A_2yr[1]:>9.4f}")
    if not np.isnan(chain_B):
        print(f"{'Method B — Re-est β (QoQ units)':<35} {chain_B:>10.4f} {ci_B[0]:>9.4f} {ci_B[1]:>9.4f}")
    else:
        print(f"{'Method B — Re-est β (QoQ units)':<35} {'n/a':>10}")
    print("=" * 70)

    # =========================================================================
    # Figure: side-by-side bar chart
    # =========================================================================
    labels  = ['Baseline\n(QoQ×annual β)', 'Method A\n1yr cumul', 'Method A\n2yr cumul']
    centers = [baseline_chain, chain_A_1yr, chain_A_2yr]
    lo_errs = [0, chain_A_1yr - ci_A_1yr[0], chain_A_2yr - ci_A_2yr[0]]
    hi_errs = [0, ci_A_1yr[1] - chain_A_1yr, ci_A_2yr[1] - chain_A_2yr]

    if not np.isnan(chain_B):
        labels.append('Method B\nre-est β')
        centers.append(chain_B)
        lo_errs.append(chain_B - ci_B[0])
        hi_errs.append(ci_B[1] - chain_B)

    colors = [C["ci_light"]] + [C["main"]] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=SZ["wide"])
    x = np.arange(len(labels))
    bars = ax.bar(x, centers, width=0.5, color=colors, edgecolor=C["main"], linewidth=0.7)
    ax.errorbar(
        x[1:], centers[1:],
        yerr=[lo_errs[1:], hi_errs[1:]],
        fmt='none', color=C["main"], capsize=5, linewidth=1.2
    )
    zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Poverty rate change (pp per 100bp shock)')
    ax.set_title('')
    # Annotate bars with values — place above the error bar cap (or above bar if no error bar)
    for i, (b_val, rect) in enumerate(zip(centers, bars)):
        cap_top = b_val + hi_errs[i] if i > 0 else b_val
        ypos = cap_top + 0.04 if cap_top >= 0 else b_val - 0.06
        ax.text(rect.get_x() + rect.get_width()/2, ypos,
                f'{b_val:.3f}', ha='center', va='bottom', fontsize=8)

    out_fig = OUT_DIR / 'p1_frequency_chain.pdf'
    fig.savefig(out_fig)
    PAPER_FIGS = ROOT / 'paper' / 'figures'
    shutil.copy(out_fig, PAPER_FIGS / 'fig15_chain_comparison.pdf')
    print(f"\nFigure saved: {out_fig}")
    plt.close(fig)

    # =========================================================================
    # Save results text
    # =========================================================================
    lines = []
    lines.append("P1: FREQUENCY-CHAIN ROBUSTNESS CHECK")
    lines.append("=" * 70)
    lines.append("")
    lines.append("MOTIVATION")
    lines.append("  Baseline chain: peak QoQ IRF x annual ΔPoverty/GDP elasticity")
    lines.append(f"  Baseline peak GDP IRF (h=3): {BASELINE_PEAK_GDP:.3f} pp (QoQ)")
    lines.append(f"  Poverty elasticity β̂: {BASELINE_POV_BETA:.3f}  SE: {BASELINE_POV_SE:.3f}")
    lines.append(f"  Baseline chain estimate: {baseline_chain:.4f} pp poverty reduction per 100bp hike")
    lines.append("")
    lines.append("METHOD A — Cumulate QoQ IRF to YoY-equivalent")
    lines.append(f"  GDP IRF path (h=0..8): {np.round(irf_point, 4).tolist()}")
    lines.append(f"  Cumulative 1yr (h=0..3): {gdp_yoy_1yr:.4f} pp")
    lines.append(f"  Cumulative 2yr (h=0..7): {gdp_yoy_2yr:.4f} pp")
    lines.append(f"  Chain A (1yr): {chain_A_1yr:.4f} pp  90% CI: [{ci_A_1yr[0]:.4f}, {ci_A_1yr[1]:.4f}]")
    lines.append(f"  Chain A (2yr): {chain_A_2yr:.4f} pp  90% CI: [{ci_A_2yr[0]:.4f}, {ci_A_2yr[1]:.4f}]")
    lines.append(f"  Bootstrap draws: {N_BOOT_A} (VAR) x {N_BOOT_A} (β ~ N)")
    lines.append("")
    lines.append("METHOD B — Re-estimate poverty regression using annual-avg-of-QoQ GDP")
    if not np.isnan(chain_B):
        lines.append(f"  Matched years: {df_B['year'].tolist()}")
        lines.append(f"  β̂_B = {beta_B:.4f}  SE = {se_B:.4f}  R² = {r2_B:.4f}")
        lines.append(f"  Chain B: {BASELINE_PEAK_GDP:.3f} x {beta_B:.4f} = {chain_B:.4f} pp")
        lines.append(f"  90% CI: [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")
    else:
        lines.append("  Insufficient matched years.")
    lines.append("")
    lines.append("METHOD C — Direct annual rate → poverty (with GDP control)")
    if not np.isnan(beta_rate_Ca):
        lines.append(f"  Dataset years: {df_C['year'].tolist()}")
        lines.append(f"  ΔPoverty = α + β_gdp×GDP_yoy + β_rate×stance + ε")
        lines.append(f"  β_gdp  = {beta_gdp_Ca:.4f}  SE = {se_gdp_Ca:.4f}")
        lines.append(f"  β_rate = {beta_rate_Ca:.4f}  SE = {se_rate_Ca:.4f}")
        lines.append(f"  t_rate = {t_rate_Ca:.3f}  p_rate = {p_rate_Ca:.4f}")
        lines.append(f"  R² = {r2_Ca:.4f}")
        lines.append(f"  Rate has independent power beyond GDP: {'YES (p<0.10)' if rate_has_power else 'NO (p>=0.10)'}")
    else:
        lines.append("  Insufficient data.")
    lines.append("")
    lines.append("COMPARISON TABLE")
    lines.append(f"  {'Method':<35} {'Chain Est':>10} {'CI Lo':>9} {'CI Hi':>9}")
    lines.append(f"  {'-'*64}")
    lines.append(f"  {'Baseline (QoQ x annual β)':<35} {baseline_chain:>10.4f} {'-':>9} {'-':>9}")
    lines.append(f"  {'Method A — 1yr cumul':<35} {chain_A_1yr:>10.4f} {ci_A_1yr[0]:>9.4f} {ci_A_1yr[1]:>9.4f}")
    lines.append(f"  {'Method A — 2yr cumul':<35} {chain_A_2yr:>10.4f} {ci_A_2yr[0]:>9.4f} {ci_A_2yr[1]:>9.4f}")
    if not np.isnan(chain_B):
        lines.append(f"  {'Method B — re-est β':<35} {chain_B:>10.4f} {ci_B[0]:>9.4f} {ci_B[1]:>9.4f}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  The baseline chain estimate conflates QoQ and annual frequencies.")
    lines.append("  Corrected estimates (Method A) use cumulated IRF in YoY-equivalent units,")
    lines.append("  which is the appropriate operand for the annual poverty elasticity β.")
    lines.append("  Method B provides a fully frequency-consistent regression.")
    lines.append("  Method C tests whether monetary policy has independent direct effects on")
    lines.append("  poverty beyond the GDP transmission channel.")

    out_txt = OUT_DIR / 'p1_frequency_chain_results.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Results saved: {out_txt}")
    print("\nDone.")


if __name__ == '__main__':
    main()
