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
# MAIN
# =============================================================================

def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    x = np.array(POV_GDP, dtype=float)
    y = np.array(POV_DPOV, dtype=float)
    N = len(x)
    years = np.array(POV_YEARS)

    # -------------------------------------------------------------------------
    # Baseline OLS
    # -------------------------------------------------------------------------
    X_ols = sm.add_constant(x)
    ols_res = sm.OLS(y, X_ols).fit()
    beta_hat = ols_res.params[1]
    alpha_hat = ols_res.params[0]
    se_ols = ols_res.bse[1]
    t_obs = ols_res.tvalues[1]
    p_ols = ols_res.pvalues[1]
    resid = ols_res.resid
    ci_raw = ols_res.conf_int()
    ci_ols = np.array(ci_raw)[1] if hasattr(ci_raw, '__len__') else ci_raw[1]  # [low, high]
    yhat = ols_res.fittedvalues

    print("=" * 70)
    print("P3 — Small-Sample Inference for Poverty Regression (N=18)")
    print("=" * 70)
    print(f"Baseline OLS: beta={beta_hat:.4f}, SE={se_ols:.4f}, t={t_obs:.3f}, p={p_ols:.4f}")
    print(f"95% CI: [{ci_ols[0]:.4f}, {ci_ols[1]:.4f}]")
    print()

    # -------------------------------------------------------------------------
    # Method 1 — Wild Bootstrap (Rademacher)
    # -------------------------------------------------------------------------
    B = 9999
    y_bar = y.mean()

    # Restricted residuals under H0: beta=0
    resid_0 = y - y_bar

    t_boot = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=N)
        y_star = y_bar + resid_0 * w
        X_b = sm.add_constant(x)
        res_b = sm.OLS(y_star, X_b).fit()
        t_boot[b] = res_b.tvalues[1]

    wild_p = np.mean(np.abs(t_boot) >= np.abs(t_obs))

    # Unrestricted wild bootstrap CI
    beta_boot_unrestricted = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=N)
        y_star = alpha_hat + beta_hat * x + resid * w
        X_b = sm.add_constant(x)
        res_b = sm.OLS(y_star, X_b).fit()
        beta_boot_unrestricted[b] = res_b.params[1]

    wild_ci = np.percentile(beta_boot_unrestricted, [2.5, 97.5])
    wild_se = np.std(beta_boot_unrestricted, ddof=1)

    print(f"Method 1 — Wild Bootstrap (B={B}):")
    print(f"  p-value = {wild_p:.4f}, SE = {wild_se:.4f}, CI = [{wild_ci[0]:.4f}, {wild_ci[1]:.4f}]")

    # -------------------------------------------------------------------------
    # Method 2 — Permutation Test
    # -------------------------------------------------------------------------
    beta_perm = np.empty(B)
    for b in range(B):
        y_perm = rng.permutation(y)
        X_b = sm.add_constant(x)
        res_b = sm.OLS(y_perm, X_b).fit()
        beta_perm[b] = res_b.params[1]

    perm_p = np.mean(np.abs(beta_perm) >= np.abs(beta_hat))
    print(f"\nMethod 2 — Permutation Test (B={B}):")
    print(f"  p-value = {perm_p:.4f}")

    # -------------------------------------------------------------------------
    # Method 3 — Exact Sign Test
    # -------------------------------------------------------------------------
    n_pos = np.sum(resid > 0)
    n_neg = np.sum(resid < 0)
    n_nonzero = n_pos + n_neg
    # Two-sided exact binomial test
    try:
        sign_result = stats.binomtest(n_pos, n_nonzero, 0.5, alternative='two-sided')
        sign_p = sign_result.pvalue
    except AttributeError:
        # older scipy fallback
        sign_p = stats.binom_test(n_pos, n_nonzero, 0.5)

    print(f"\nMethod 3 — Exact Sign Test:")
    print(f"  n_pos={n_pos}, n_neg={n_neg}, n_total={n_nonzero}")
    print(f"  p-value = {sign_p:.4f}")

    # -------------------------------------------------------------------------
    # Method 4a — Jackknife SE
    # -------------------------------------------------------------------------
    beta_jack = np.empty(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X_j = sm.add_constant(x[mask])
        res_j = sm.OLS(y[mask], X_j).fit()
        beta_jack[i] = res_j.params[1]

    beta_jack_mean = beta_jack.mean()
    jack_se = np.sqrt(((N - 1) / N) * np.sum((beta_jack - beta_jack_mean) ** 2))
    jack_t = beta_hat / jack_se
    jack_p = 2 * stats.t.sf(np.abs(jack_t), df=N - 2)
    t_crit = stats.t.ppf(0.975, df=N - 2)
    jack_ci = [beta_hat - t_crit * jack_se, beta_hat + t_crit * jack_se]

    print(f"\nMethod 4a — Jackknife SE:")
    print(f"  SE = {jack_se:.4f}, t = {jack_t:.3f}, p = {jack_p:.4f}")
    print(f"  95% CI = [{jack_ci[0]:.4f}, {jack_ci[1]:.4f}]")

    # -------------------------------------------------------------------------
    # Method 4b — HC3 Robust SE
    # -------------------------------------------------------------------------
    ols_hc3 = sm.OLS(y, X_ols).fit(cov_type='HC3')
    hc3_se = ols_hc3.bse[1]
    hc3_t = ols_hc3.tvalues[1]
    hc3_p = ols_hc3.pvalues[1]
    hc3_ci = np.array(ols_hc3.conf_int())[1]

    print(f"\nMethod 4b — HC3 Robust SE:")
    print(f"  SE = {hc3_se:.4f}, t = {hc3_t:.3f}, p = {hc3_p:.4f}")
    print(f"  95% CI = [{hc3_ci[0]:.4f}, {hc3_ci[1]:.4f}]")

    # -------------------------------------------------------------------------
    # Method 5 — Bayesian Gibbs Sampler (conjugate normal-inverse-gamma)
    # -------------------------------------------------------------------------
    print(f"\nMethod 5 — Bayesian Gibbs Sampler (12000 iters, 2000 burnin):")

    X_mat = np.array(X_ols)  # N x 2
    y_vec = np.array(y).copy()

    # Priors
    beta0_prior = np.zeros(2)
    Lambda0 = np.eye(2) / 100.0   # prior precision (diffuse: variance=100)
    a0 = 0.001
    b0 = 0.001

    n_iter = 12000
    burnin = 2000
    n_keep = n_iter - burnin

    beta_samples = np.empty((n_keep, 2))
    sigma2_samples = np.empty(n_keep)

    # Initialize
    sigma2 = 1.0
    beta_curr = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]

    XtX = X_mat.T @ X_mat
    Xty = X_mat.T @ y_vec

    for it in range(n_iter):
        # Sample beta | sigma2, y
        post_prec = XtX / sigma2 + Lambda0
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (Xty / sigma2 + Lambda0 @ beta0_prior)
        L = np.linalg.cholesky(post_cov)
        beta_curr = post_mean + L @ rng.standard_normal(2)

        # Sample sigma2 | beta, y
        resid_b = y_vec - X_mat @ beta_curr
        rss = resid_b @ resid_b
        a_post = a0 + N / 2.0
        b_post = b0 + 0.5 * rss
        # sigma2 ~ InvGamma(a_post, b_post): sample via 1/Gamma(a_post, 1/b_post)
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        if it >= burnin:
            idx = it - burnin
            beta_samples[idx] = beta_curr
            sigma2_samples[idx] = sigma2

    bayes_beta_median = np.median(beta_samples[:, 1])
    bayes_beta_sd = np.std(beta_samples[:, 1], ddof=1)
    bayes_ci = np.percentile(beta_samples[:, 1], [2.5, 97.5])
    p_neg = np.mean(beta_samples[:, 1] < 0)

    print(f"  Posterior median(beta) = {bayes_beta_median:.4f}")
    print(f"  Posterior SD = {bayes_beta_sd:.4f}")
    print(f"  95% Credible Interval = [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]")
    print(f"  P(beta < 0) = {p_neg:.4f}")

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    def fmt_p(p):
        if p < 0.001:
            return '<0.001'
        return f'{p:.4f}'

    lines = []
    lines.append("=" * 80)
    lines.append("SMALL-SAMPLE INFERENCE SUMMARY — Poverty Regression (N=18)")
    lines.append("Dependent variable: Annual change in poverty headcount ratio (pp)")
    lines.append("Regressor: Annual real GDP growth (%)")
    lines.append("=" * 80)
    header = f"{'METHOD':<28} | {'beta/Median':>11} | {'SE/SD':>9} | {'p-value':>9} | {'95% CI'}"
    lines.append(header)
    lines.append("-" * 80)

    def row(method, beta_str, se_str, p_str, ci_str):
        return f"{method:<28} | {beta_str:>11} | {se_str:>9} | {p_str:>9} | {ci_str}"

    lines.append(row("OLS (baseline)",
                     f"{beta_hat:.4f}", f"{se_ols:.4f}", fmt_p(p_ols),
                     f"[{ci_ols[0]:.4f}, {ci_ols[1]:.4f}]"))
    lines.append(row("Wild bootstrap",
                     f"{beta_hat:.4f}", f"{wild_se:.4f}", fmt_p(wild_p),
                     f"[{wild_ci[0]:.4f}, {wild_ci[1]:.4f}]"))
    lines.append(row("Permutation test",
                     f"{beta_hat:.4f}", "—", fmt_p(perm_p), "—"))
    lines.append(row("Sign test",
                     "—", "—", fmt_p(sign_p), "—"))
    lines.append(row("Jackknife",
                     f"{beta_hat:.4f}", f"{jack_se:.4f}", fmt_p(jack_p),
                     f"[{jack_ci[0]:.4f}, {jack_ci[1]:.4f}]"))
    lines.append(row("HC3",
                     f"{beta_hat:.4f}", f"{hc3_se:.4f}", fmt_p(hc3_p),
                     f"[{hc3_ci[0]:.4f}, {hc3_ci[1]:.4f}]"))
    lines.append(row("Bayesian (posterior)",
                     f"{bayes_beta_median:.4f}", f"{bayes_beta_sd:.4f}",
                     f"P<0={p_neg:.4f}",
                     f"[{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]"))
    lines.append("=" * 80)

    # Conclusion
    all_p = [p_ols, wild_p, perm_p, sign_p, jack_p, hc3_p]
    all_sig = all(p < 0.05 for p in all_p)
    lines.append("")
    lines.append("CONCLUSION:")
    if all_sig and p_neg > 0.99:
        lines.append("  beta < 0 is confirmed at 5% significance across all six frequentist")
        lines.append("  methods, and posterior probability P(beta<0) > 0.99.")
        lines.append("  The negative growth-poverty link is robust to small-sample concerns.")
    else:
        lines.append(f"  Not all methods significant at 5%. Check individual p-values above.")
        lines.append(f"  Bayesian P(beta<0) = {p_neg:.4f}.")
    lines.append("")

    table_str = "\n".join(lines)
    print("\n" + table_str)

    # Save table
    out_txt = OUT_DIR / 'p3_poverty_inference_results.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(table_str)
    print(f"\nTable saved to: {out_txt}")

    # -------------------------------------------------------------------------
    # Figure: Posterior distribution of beta
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=SZ["wide_tall"])

    # Left panel: posterior density
    ax = axes[0]
    ax.hist(beta_samples[:, 1], bins=80, density=True,
            color=C["accent2"], alpha=0.25, edgecolor='none', label='Posterior draws')
    # KDE overlay
    kde = stats.gaussian_kde(beta_samples[:, 1])
    xgrid = np.linspace(beta_samples[:, 1].min(), beta_samples[:, 1].max(), 400)
    ax.plot(xgrid, kde(xgrid), color=C["main"], lw=2, label='KDE')
    ax.axvline(beta_hat, color=C["accent1"], lw=2, ls='--', label=f'OLS estimate ({beta_hat:.3f})')
    ax.axvline(bayes_ci[0], color=C["gray_line"], lw=1, ls=':', label='95% credible interval')
    ax.axvline(bayes_ci[1], color=C["gray_line"], lw=1, ls=':')
    ax.axvline(0, color=C["zero"], lw=1, ls='-', alpha=0.4, label='Zero')
    ax.set_xlabel(r'$\hat{\beta}$ (GDP growth coefficient)')
    ax.set_ylabel('Density')
    ax.set_title('')
    legend_below(ax, ncol=2)
    stat_box(ax, f'P(β<0) = {p_neg:.4f}', loc='upper left')

    # Right panel: data scatter with OLS fit
    ax2 = axes[1]
    ax2.scatter(x, y, color=C["accent2"], s=50, zorder=3, alpha=0.85)
    for i, yr in enumerate(years):
        ax2.annotate(str(yr), (x[i], y[i]), textcoords='offset points',
                     xytext=(4, 3), fontsize=7, color=C["gray_line"])
    xfit = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    ax2.plot(xfit, alpha_hat + beta_hat * xfit, color=C["accent1"], lw=2,
             label=f'OLS: β={beta_hat:.3f} (SE={se_ols:.3f})')
    # Bayesian credible band: use posterior samples
    sample_idx = rng.choice(n_keep, size=500, replace=False)
    for idx in sample_idx:
        a_s, b_s = beta_samples[idx, 0], beta_samples[idx, 1]
        ax2.plot(xfit, a_s + b_s * xfit, color=C["accent2"], lw=0.3, alpha=0.06)
    ax2.set_xlabel('Annual GDP Growth (%)')
    ax2.set_ylabel('Annual Change in Poverty Rate (pp)')
    ax2.set_title('')
    legend_below(ax2, ncol=1)
    stat_box(ax2, f'N=18, R²={ols_res.rsquared:.3f}', loc='lower right')

    fig_path = OUT_DIR / 'p3_poverty_inference.pdf'
    fig.savefig(fig_path)
    shutil.copy(fig_path, ROOT / 'paper' / 'figures' / 'fig16_small_sample.pdf')
    plt.close(fig)
    print(f"Figure saved to: {fig_path}")

    # -------------------------------------------------------------------------
    # Also save a bootstrap distribution figure for completeness
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=SZ["wide_tall"])

    ax = axes2[0]
    ax.hist(t_boot, bins=60, density=True, color=C["accent2"], alpha=0.25,
            edgecolor='none', label='Wild bootstrap t*')
    ax.axvline(t_obs, color=C["accent1"], lw=2, ls='--', label=f't_obs = {t_obs:.2f}')
    ax.axvline(-np.abs(t_obs), color=C["accent1"], lw=2, ls='--')
    xg = np.linspace(-6, 6, 300)
    ax.plot(xg, stats.t.pdf(xg, df=N-2), color=C["main"], lw=1.5, alpha=0.7, label=f't({N-2}) distribution')
    ax.set_xlabel('t-statistic')
    ax.set_ylabel('Density')
    ax.set_title('')
    legend_below(ax, ncol=2)

    ax2 = axes2[1]
    ax2.hist(beta_perm, bins=60, density=True, color=C["accent2"], alpha=0.25,
             edgecolor='none', label='Permutation β*')
    ax2.axvline(beta_hat, color=C["accent1"], lw=2, ls='--', label=f'β_obs = {beta_hat:.3f}')
    ax2.axvline(-np.abs(beta_hat), color=C["accent1"], lw=2, ls='--')
    ax2.set_xlabel('β coefficient')
    ax2.set_ylabel('Density')
    ax2.set_title('')
    legend_below(ax2, ncol=2)

    fig2_path = OUT_DIR / 'p3_poverty_bootstrap_permutation.pdf'
    fig2.savefig(fig2_path)
    plt.close(fig2)
    print(f"Bootstrap/permutation figure saved to: {fig2_path}")

    print("\nDone. All outputs written to:", OUT_DIR)


if __name__ == '__main__':
    main()
