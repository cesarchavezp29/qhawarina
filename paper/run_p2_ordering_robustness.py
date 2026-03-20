#!/usr/bin/env python3
"""
run_p2_ordering_robustness.py — Test whether Cholesky ordering matters for the
identified GDP response to a monetary policy shock.

6 orderings tested + GIRF (ordering-invariant) for comparison.
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


# ─── GIRF ────────────────────────────────────────────────────────────────────
def compute_girf(var_result, shock_var_idx=4, response_idx=1, horizon=9):
    """
    Generalized IRF (ordering-invariant).
    GIRF_j(h) = [A1^h × Sigma × e_k] / Sigma[k,k]
    Normalizes shock to 1pp (100bp) unit.
    """
    A1    = var_result.coefs[0]                    # VAR(1) coefficient matrix, shape (K,K)
    Sigma = np.array(var_result.sigma_u)           # residual covariance as numpy array
    K     = A1.shape[0]

    e_k   = np.zeros(K)
    e_k[shock_var_idx] = 1.0

    norm  = Sigma[shock_var_idx, shock_var_idx]   # Var(ε_k)
    if abs(norm) < 1e-10:
        norm = 1.0

    girf = np.zeros(horizon)
    Ah   = np.eye(K)
    for h in range(horizon):
        # GIRF_j(h) = [Phi_h Sigma e_k]_j / Sigma[k,k]
        girf[h] = (Ah @ Sigma @ e_k)[response_idx] / norm
        Ah = Ah @ A1

    return girf


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("=" * 70)
    print("P2: Cholesky ordering robustness check")
    print("=" * 70)

    var_df, rate_level, covid_dummy, q_full = load_var_data()

    HORIZON  = 9
    N_BOOT   = 2000

    # Variable names in baseline order
    VARS = ['tot', 'gdp', 'cpi', 'fx', 'rate']

    # ── Define orderings ─────────────────────────────────────────────────────
    # Each entry: (label, description, column_order, shock_var_idx, response_idx)
    orderings = [
        (
            'Baseline',
            'tot, gdp, cpi, fx, rate',
            ['tot', 'gdp', 'cpi', 'fx', 'rate'],
            4,   # shock = rate (last)
            1,   # response = gdp (2nd)
        ),
        (
            'Rate-before-GDP',
            'tot, cpi, fx, rate, gdp',
            ['tot', 'cpi', 'fx', 'rate', 'gdp'],
            3,   # shock = rate (4th)
            4,   # response = gdp (last)
        ),
        (
            'Rate-first-domestic',
            'tot, rate, gdp, cpi, fx',
            ['tot', 'rate', 'gdp', 'cpi', 'fx'],
            1,   # shock = rate (2nd)
            2,   # response = gdp (3rd)
        ),
        (
            'CPI-first',
            'tot, cpi, gdp, fx, rate',
            ['tot', 'cpi', 'gdp', 'fx', 'rate'],
            4,   # shock = rate (last)
            2,   # response = gdp (3rd)
        ),
        (
            'FX-last',
            'tot, gdp, cpi, rate, fx',
            ['tot', 'gdp', 'cpi', 'rate', 'fx'],
            3,   # shock = rate (4th)
            1,   # response = gdp (2nd)
        ),
        (
            'Rate-last-GDP-last',
            'tot, cpi, fx, rate, gdp',
            ['tot', 'cpi', 'fx', 'rate', 'gdp'],
            3,   # shock = rate (4th)
            4,   # response = gdp (last)
        ),
    ]

    results = []
    irf_paths = []

    for label, desc, col_order, shock_idx, resp_idx in orderings:
        print(f"\n  Ordering: {label}  ({desc})")
        df_ord = var_df[col_order].copy()
        var_mod = VAR(df_ord)
        var_res = var_mod.fit(1, trend='c')

        irf_pt, ci_lo, ci_hi = compute_cholesky_irf(
            var_res,
            shock_var_idx=shock_idx,
            horizon=HORIZON,
            n_boot=N_BOOT,
            response_idx=resp_idx,
            lags=1,
        )

        peak_val  = irf_pt.min()
        peak_h    = int(np.argmin(irf_pt))
        ci_lo_pk  = ci_lo[peak_h]
        ci_hi_pk  = ci_hi[peak_h]

        print(f"    Peak GDP = {peak_val:.4f} at h={peak_h}  90%CI: [{ci_lo_pk:.4f}, {ci_hi_pk:.4f}]")

        results.append({
            'label':    label,
            'desc':     desc,
            'peak':     peak_val,
            'peak_h':   peak_h,
            'ci_lo':    ci_lo_pk,
            'ci_hi':    ci_hi_pk,
            'irf':      irf_pt,
            'ci_lo_path': ci_lo,
            'ci_hi_path': ci_hi,
        })
        irf_paths.append((label, irf_pt, ci_lo, ci_hi))

    # ── GIRF (ordering-invariant, baseline var_df) ────────────────────────────
    print(f"\n  GIRF (ordering-invariant, baseline ordering)")
    var_baseline = VAR(var_df).fit(1, trend='c')
    girf_pt = compute_girf(var_baseline, shock_var_idx=4, response_idx=1, horizon=HORIZON)

    girf_peak   = girf_pt.min()
    girf_peak_h = int(np.argmin(girf_pt))
    print(f"    GIRF Peak GDP = {girf_peak:.4f} at h={girf_peak_h}")

    # Bootstrap CI for GIRF
    resids   = var_baseline.resid
    coefs0   = var_baseline.coefs
    K_base   = var_baseline.neqs
    T_data   = resids.shape[0]
    boot_girf_peaks = np.zeros(N_BOOT)

    for b in range(N_BOOT):
        idx_b = np.random.randint(0, T_data, size=T_data)
        boot_resids_b = resids.values[idx_b]
        Y_boot = np.zeros((T_data + 1, K_base))
        Y_boot[:1] = var_baseline.model.endog[:1]
        for t in range(1, T_data + 1):
            pred = np.zeros(K_base)
            for lag_i, c in enumerate(coefs0):
                pred += Y_boot[t - lag_i - 1] @ c.T
            pred += boot_resids_b[t - 1]
            Y_boot[t] = pred
        try:
            Y_df_b = pd.DataFrame(Y_boot[1:], columns=var_baseline.model.endog_names)
            res_b  = VAR(Y_df_b).fit(1, trend='c')
            girf_b = compute_girf(res_b, shock_var_idx=4, response_idx=1, horizon=HORIZON)
            boot_girf_peaks[b] = girf_b.min()
        except:
            boot_girf_peaks[b] = girf_peak

    girf_ci_lo = np.percentile(boot_girf_peaks, 5)
    girf_ci_hi = np.percentile(boot_girf_peaks, 95)
    print(f"    GIRF 90% CI at peak: [{girf_ci_lo:.4f}, {girf_ci_hi:.4f}]")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 80)
    print("ORDERING ROBUSTNESS TABLE")
    print("=" * 80)
    hdr = f"  {'Ordering':<24} {'Variables':<34} {'Peak GDP':>9} {'h*':>4} {'CI Lo':>8} {'CI Hi':>8}"
    print(hdr)
    print("  " + "-" * 78)
    for r in results:
        print(f"  {r['label']:<24} {r['desc']:<34} {r['peak']:>9.4f} {r['peak_h']:>4d} {r['ci_lo']:>8.4f} {r['ci_hi']:>8.4f}")
    print(f"  {'GIRF (invariant)':<24} {'baseline vars (ordering-free)':<34} {girf_peak:>9.4f} {girf_peak_h:>4d} {girf_ci_lo:>8.4f} {girf_ci_hi:>8.4f}")
    print("=" * 80)

    # =========================================================================
    # Figure: 6 IRF paths overlaid + GIRF
    # =========================================================================
    horizons = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=SZ["wide_tall"])

    # Gray paths for non-baseline orderings
    for i, (label, irf_pt, ci_lo, ci_hi) in enumerate(irf_paths):
        if label == 'Baseline':
            continue
        ax.plot(horizons, irf_pt, color=C["gray_line"], linewidth=1.0,
                alpha=0.75, label=label if i == 1 else '_nolegend_')

    # Baseline in main color thick
    base_irf = irf_paths[0][1]
    ax.plot(horizons, base_irf, color=C["main"], linewidth=2.0, label='Baseline')

    # Shaded 90% CI for baseline
    ax.fill_between(
        horizons,
        irf_paths[0][2], irf_paths[0][3],
        color=C["ci_light"], alpha=0.15, label='Baseline 90% CI'
    )

    # GIRF as dashed accent1
    ax.plot(horizons, girf_pt, color=C["accent1"], linewidth=1.6,
            linestyle='--', label='GIRF (ordering-invariant)')

    zero_line(ax)
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('GDP response (pp per 100bp shock)')
    ax.set_title('')
    ax.set_xticks(horizons)

    # Legend
    gray_line_h = plt.Line2D([0], [0], color=C["gray_line"], linewidth=1.0, label='Alternative orderings (5)')
    main_line   = plt.Line2D([0], [0], color=C["main"],      linewidth=2.0, label='Baseline')
    ci_fill     = plt.matplotlib.patches.Patch(facecolor=C["ci_light"], alpha=0.5, label='Baseline 90% CI')
    red_dashed  = plt.Line2D([0], [0], color=C["accent1"],   linewidth=1.6, linestyle='--', label='GIRF (invariant)')
    legend_below(ax, ncol=4, handles=[main_line, ci_fill, gray_line_h, red_dashed])

    out_fig = OUT_DIR / 'p2_ordering_robustness.pdf'
    fig.savefig(out_fig)
    PAPER_FIGS = ROOT / 'paper' / 'figures'
    shutil.copy(out_fig, PAPER_FIGS / 'fig11_ordering_fan.pdf')
    print(f"\nFigure saved: {out_fig}")
    plt.close(fig)

    # =========================================================================
    # Save table text
    # =========================================================================
    lines = []
    lines.append("P2: CHOLESKY ORDERING ROBUSTNESS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("QUESTION")
    lines.append("  Does the identified GDP response to a 100bp rate shock depend on the")
    lines.append("  Cholesky ordering of variables in the VAR?")
    lines.append("")
    lines.append("SETUP")
    lines.append("  VAR(1) with trend='c', quarterly data 2004Q2–2025Q3")
    lines.append("  Variables: tot (terms of trade), gdp, cpi, fx, rate (policy rate)")
    lines.append(f"  Bootstrap replications: {N_BOOT} (residual bootstrap)")
    lines.append("  Confidence interval: 5th–95th percentile of bootstrap distribution")
    lines.append("")
    lines.append("RESULTS")
    lines.append("")
    hdr = f"  {'Ordering':<24} {'Variables':<34} {'Peak GDP':>9} {'h*':>4} {'CI Lo':>8} {'CI Hi':>8}"
    lines.append(hdr)
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(
            f"  {r['label']:<24} {r['desc']:<34} {r['peak']:>9.4f} {r['peak_h']:>4d} "
            f"{r['ci_lo']:>8.4f} {r['ci_hi']:>8.4f}"
        )
    lines.append(
        f"  {'GIRF (invariant)':<24} {'baseline vars (ordering-free)':<34} "
        f"{girf_peak:>9.4f} {girf_peak_h:>4d} {girf_ci_lo:>8.4f} {girf_ci_hi:>8.4f}"
    )
    lines.append("")
    lines.append("GIRF FORMULA")
    lines.append("  GIRF_j(h) = [A1^h × Sigma × e_k]_j / Sigma[k,k]")
    lines.append("  where A1 = VAR(1) companion matrix, Sigma = residual covariance,")
    lines.append("  e_k = unit vector for rate variable (k=4 in baseline order).")
    lines.append("  Shock normalized to 1pp (100bp). Ordering-invariant by construction.")
    lines.append("")
    lines.append("IRF PATHS BY ORDERING")
    lines.append(f"  {'Horizon':<10} " + "  ".join(f"{r['label'][:12]:>13}" for r in results) + f"  {'GIRF':>8}")
    lines.append("  " + "-" * (10 + 15 * len(results) + 10))
    for h in range(HORIZON):
        row = f"  {h:<10} "
        for r in results:
            row += f"  {r['irf'][h]:>13.4f}"
        row += f"  {girf_pt[h]:>8.4f}"
        lines.append(row)
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  Stability of peak GDP response across orderings indicates the result is")
    lines.append("  not an artifact of identification assumptions. The GIRF provides a")
    lines.append("  fully identification-free benchmark.")

    out_txt = OUT_DIR / 'p2_ordering_results.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Results saved: {out_txt}")
    print("\nDone.")


if __name__ == '__main__':
    main()
