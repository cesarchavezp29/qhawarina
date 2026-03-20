#!/usr/bin/env python3
"""
run_option4_bvar.py — Bayesian VAR with Minnesota Prior
Option 4 for Peru Monetary Policy Paper

Minnesota prior BVAR(1) for 5-variable system [tot, gdp, cpi, fx, rate].
Normal-Inverse-Wishart conjugate posterior, implemented directly in numpy/scipy.
"""
import sys, io, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, linalg
import statsmodels.api as sm
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT   = Path('D:/Nexus/nexus')
PANEL  = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUT_DIR = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ───────────────────────────────────────────────────────────────────
C_BLACK  = '#000000'
C_DARK   = '#404040'
C_MED    = '#808080'
C_LIGHT  = '#c0c0c0'
C_BAND1  = '#d0d0d0'
C_BAND2  = '#e8e8e8'

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Palatino', 'Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ─── Frequentist baseline ─────────────────────────────────────────────────────
FREQ_PEAK   = -0.195
FREQ_CI_LO  = -0.698
FREQ_CI_HI  = +0.271


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (copied verbatim from run_robustness.py lines 91-173)
# ════════════════════════════════════════════════════════════════════════════

def load_var_data():
    """
    Build quarterly VAR dataset from monthly panel.
    Returns DataFrame with columns [tot, gdp, cpi, fx, rate] and DatetimeIndex.
    Also returns rate_level (not differenced).
    """
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])

    series_map = {
        'PD04722MM': 'rate_raw',   # level
        'PN01731AM': 'gdp_m',      # monthly % change SA
        'PN01271PM': 'cpi_m',      # monthly % change SA
        'PN01246PM': 'fx_m',       # PEN/USD level
        'PN38923BM': 'tot_m',      # ToT index
    }

    # Extract each series
    frames = []
    for sid, col in series_map.items():
        s = raw[raw['series_id'] == sid][['date', 'value_raw']].copy()
        s = s.rename(columns={'value_raw': col})
        s = s.set_index('date').sort_index()
        frames.append(s)

    monthly = frames[0].join(frames[1:], how='outer')
    monthly = monthly.sort_index()

    # ── Quarterly aggregation ──────────────────────────────────────────────
    q = pd.DataFrame()
    # rate: quarterly mean of level, then diff
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()

    # gdp: sum of monthly % changes
    q['gdp'] = monthly['gdp_m'].resample('QE').sum()

    # cpi: sum of monthly % changes
    q['cpi'] = monthly['cpi_m'].resample('QE').sum()

    # fx: quarterly mean, pct_change*100
    q['fx'] = monthly['fx_m'].resample('QE').mean().pct_change() * 100

    # tot: quarterly mean, pct_change*100
    q['tot'] = monthly['tot_m'].resample('QE').mean().pct_change() * 100

    # Drop first row (NaN from diff/pct_change)
    q = q.dropna()

    # Keep 2004Q2 through 2025Q3
    q = q.loc['2004-04-01':'2025-09-30']

    # ── FWL COVID treatment ───────────────────────────────────────────────
    covid_dummy = pd.Series(0, index=q.index, name='covid')
    covid_quarters = [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]
    for cq in covid_quarters:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1

    def fwl_partial(series, dummy):
        X = sm.add_constant(dummy)
        res = sm.OLS(series, X).fit()
        return series - res.fittedvalues + series.mean()

    cols_to_fwl = ['d_rate', 'gdp', 'cpi', 'fx', 'tot', 'rate_level']
    for col in cols_to_fwl:
        if col in q.columns:
            q[col] = fwl_partial(q[col], covid_dummy)

    # Build final VAR DataFrame: [tot, gdp, cpi, fx, d_rate]
    var_df = q[['tot', 'gdp', 'cpi', 'fx', 'd_rate']].copy()
    var_df.columns = ['tot', 'gdp', 'cpi', 'fx', 'rate']

    t0 = var_df.index[0]
    t1 = var_df.index[-1]
    print(f"VAR data: {t0.year}Q{t0.quarter} to {t1.year}Q{t1.quarter}, T={len(var_df)}")
    print(f"  Columns: {list(var_df.columns)}")
    print(f"  Means:   {var_df.mean().round(3).to_dict()}")

    # Also return rate_level separately
    rate_level = q['rate_level'].copy()

    return var_df, rate_level, covid_dummy, q


# ════════════════════════════════════════════════════════════════════════════
#  MINNESOTA PRIOR BVAR(1)
#  Normal-Inverse-Wishart conjugate posterior
# ════════════════════════════════════════════════════════════════════════════

def build_minnesota_prior(K, p, lambda1, lambda2, lambda3,
                          sigma2_ols, random_walk_idx):
    """
    Build Minnesota prior moments for BVAR(p).

    Variables are ordered: [tot, gdp, cpi, fx, rate]
    Indices:                [  0,   1,   2,  3,    4]

    Parameters
    ----------
    K           : number of variables
    p           : number of lags
    lambda1     : overall tightness  (e.g. 0.2)
    lambda2     : cross-variable shrinkage  (e.g. 0.5)
    lambda3     : lag decay  (e.g. 1.0, higher = faster decay)
    sigma2_ols  : array (K,) of OLS residual variances, used for scaling
    random_walk_idx : set of variable indices that get random-walk prior mean

    Returns
    -------
    beta0       : prior mean  (K*p + 1, K)  — includes constant column
    Omega0_diag : prior variance diagonal for vec(B)  (K*(K*p+1),)
                  (we use a diagonal prior covariance)
    """
    # Total regressors per equation: K*p lags + 1 constant
    M = K * p + 1  # number of regressors per equation

    # Prior mean matrix B0: shape (M, K)
    # Row layout: [lag1_var0, lag1_var1, ..., lag1_varK, lag2_var0, ..., const]
    # i.e. rows 0..(K-1) are lag-1 coefficients, rows K..(2K-1) are lag-2, etc.
    # Last row is the constant.
    B0 = np.zeros((M, K))

    # Random-walk prior: diagonal of lag-1 block = 1 for random-walk variables
    for j in range(K):
        if j in random_walk_idx:
            # AR(1) coefficient on own lag at lag 1
            B0[j, j] = 1.0  # row = lag1_var_j, col = equation j

    # Prior variance (diagonal Omega0), stored as a vector of length M*K
    # For coefficient on variable i at lag l in equation j:
    #   var = (lambda1 * l^{-lambda3})^2        if i == j  (own lag)
    #   var = (lambda1 * lambda2 * l^{-lambda3})^2 * sigma2_j / sigma2_i  if i != j
    # For the constant: large variance (uninformative)

    Omega0_diag = np.zeros((M, K))

    for j in range(K):        # equation index
        for l in range(1, p + 1):   # lag index
            row_start = (l - 1) * K
            for i in range(K):      # variable index (which lag)
                row = row_start + i
                lag_decay = float(l) ** lambda3
                if i == j:
                    # own lag
                    var_ij = (lambda1 / lag_decay) ** 2
                else:
                    # cross-variable
                    var_ij = (lambda1 * lambda2 / lag_decay) ** 2 * (sigma2_ols[j] / max(sigma2_ols[i], 1e-12))
                Omega0_diag[row, j] = var_ij
        # constant: uninformative
        Omega0_diag[M - 1, j] = 1e6

    return B0, Omega0_diag


def prepare_ols_matrices(Y, p):
    """
    Prepare OLS data matrices for VAR(p).

    Y : (T, K) data matrix
    p : number of lags

    Returns
    -------
    Y_dep  : (T-p, K)  dependent variable matrix (rows t = p..T-1)
    X_reg  : (T-p, K*p+1)  regressor matrix [Y_{t-1}, ..., Y_{t-p}, 1]
    """
    T, K = Y.shape
    T_eff = T - p
    M = K * p + 1

    Y_dep = Y[p:, :]           # (T_eff, K)
    X_reg = np.zeros((T_eff, M))

    for l in range(1, p + 1):
        col_start = (l - 1) * K
        X_reg[:, col_start:col_start + K] = Y[p - l:T - l, :]

    X_reg[:, M - 1] = 1.0     # constant

    return Y_dep, X_reg


def ols_residual_variances(Y, p):
    """Equation-by-equation OLS residual variances for Minnesota scaling."""
    T, K = Y.shape
    Y_dep, X_reg = prepare_ols_matrices(Y, p)
    sigma2 = np.zeros(K)
    for j in range(K):
        y_j = Y_dep[:, j]
        beta_j, _, _, _ = np.linalg.lstsq(X_reg, y_j, rcond=None)
        resid_j = y_j - X_reg @ beta_j
        sigma2[j] = np.dot(resid_j, resid_j) / max(len(resid_j) - X_reg.shape[1], 1)
    return sigma2


def compute_niw_posterior(Y_dep, X_reg, B0, Omega0_diag, nu0, S0):
    """
    Compute Normal-Inverse-Wishart posterior for BVAR.

    Model:  Y_dep = X_reg @ B + E,   E_t ~ N(0, Sigma)
    Prior:  vec(B) | Sigma ~ N(vec(B0), Sigma (x) Omega0)   (Kronecker structure)
            Sigma ~ IW(S0, nu0)

    This is the standard Minnesota/Litterman conjugate prior.
    We use the equation-by-equation approximation (treating Sigma as diagonal
    in the prior) which is standard Minnesota practice, then update the
    full IW posterior for Sigma.

    Parameters
    ----------
    Y_dep      : (T, K) dependent variables
    X_reg      : (T, M) regressors
    B0         : (M, K) prior mean
    Omega0_diag: (M, K) prior variances (diagonal blocks, per-equation)
    nu0        : prior degrees of freedom for IW (scalar)
    S0         : (K, K) prior scale matrix for IW

    Returns posterior moments: B_post (M,K), Omega_post_list, nu_post, S_post
    """
    T, K = Y_dep.shape
    T_, M = X_reg.shape
    assert T == T_

    # Equation-by-equation posterior for B (Minnesota diagonal-Omega treatment)
    B_post = np.zeros((M, K))
    # We store per-equation posterior precisions for sampling
    Omega_post_list = []   # list of K arrays, each (M, M)

    for j in range(K):
        omega0_j = Omega0_diag[:, j]          # (M,) prior variance diagonal
        Omega0_j_inv = np.diag(1.0 / np.maximum(omega0_j, 1e-15))  # (M, M) prior precision
        XtX = X_reg.T @ X_reg                 # (M, M)
        Xty = X_reg.T @ Y_dep[:, j]           # (M,)
        b0_j = B0[:, j]                       # (M,)

        # Posterior precision = prior precision + X'X (scaled, but for Minnesota
        # we treat sigma_jj = sigma2_ols[j] as known in the prior covariance)
        # Standard result: Omega_post_j^{-1} = Omega0_j^{-1} + X'X
        Omega_post_j_inv = Omega0_j_inv + XtX
        # Regularize
        Omega_post_j_inv += np.eye(M) * 1e-10

        try:
            Omega_post_j = np.linalg.inv(Omega_post_j_inv)
        except np.linalg.LinAlgError:
            Omega_post_j = np.linalg.pinv(Omega_post_j_inv)

        b_post_j = Omega_post_j @ (Omega0_j_inv @ b0_j + Xty)
        B_post[:, j] = b_post_j
        Omega_post_list.append(Omega_post_j)

    # IW posterior for Sigma
    # nu_post = nu0 + T
    nu_post = nu0 + T

    # S_post = S0 + (Y - X @ B_post)' (Y - X @ B_post)
    #          + (B_post - B0)' Omega0^{-1} (B_post - B0)  [cross term]
    # Standard NIW update: use OLS residuals + prior correction
    E_post = Y_dep - X_reg @ B_post     # (T, K) posterior residuals
    S_post = S0 + E_post.T @ E_post

    # Add prior correction term (diagonal Omega0 approximation)
    for j in range(K):
        diff_j = B_post[:, j] - B0[:, j]
        omega0_j = Omega0_diag[:, j]
        Omega0_j_inv = np.diag(1.0 / np.maximum(omega0_j, 1e-15))
        correction = diff_j @ Omega0_j_inv @ diff_j
        S_post[j, j] += correction

    # Symmetrize
    S_post = 0.5 * (S_post + S_post.T)

    return B_post, Omega_post_list, nu_post, S_post


def draw_niw_posterior(B_post, Omega_post_list, nu_post, S_post, n_draws):
    """
    Draw (B, Sigma) pairs from the NIW posterior.

    For each draw:
      1. Draw Sigma ~ IW(S_post, nu_post)
      2. For each equation j: draw beta_j | Sigma ~ N(b_post_j, sigma_jj * Omega_post_j)
         (equation-by-equation, consistent with Minnesota diagonal Omega structure)

    Returns
    -------
    B_draws    : (n_draws, M, K)
    Sigma_draws: (n_draws, K, K)
    """
    K = S_post.shape[0]
    M = B_post.shape[0]

    B_draws     = np.zeros((n_draws, M, K))
    Sigma_draws = np.zeros((n_draws, K, K))

    # Precompute Cholesky of Omega_post per equation for sampling
    L_Omega = []
    for j in range(K):
        try:
            L_j = np.linalg.cholesky(Omega_post_list[j])
        except np.linalg.LinAlgError:
            reg = Omega_post_list[j] + np.eye(M) * 1e-10
            L_j = np.linalg.cholesky(reg)
        L_Omega.append(L_j)

    # Cholesky of S_post for IW sampling
    try:
        L_S = np.linalg.cholesky(S_post)
    except np.linalg.LinAlgError:
        L_S = np.linalg.cholesky(S_post + np.eye(K) * 1e-8)

    for d in range(n_draws):
        # ── Draw Sigma ~ IW(S_post, nu_post) ──────────────────────────────
        # IW(S, nu): if X ~ Wishart(S^{-1}, nu) then X^{-1} ~ IW(S, nu)
        # Draw from Wishart using Bartlett decomposition
        # Wishart(scale=S_post^{-1}, df=nu_post) — scipy convention: scale matrix
        try:
            S_post_inv = np.linalg.inv(S_post)
            S_post_inv = 0.5 * (S_post_inv + S_post_inv.T)
            # scipy.stats.wishart: scale = S_post_inv, df = nu_post
            W = stats.wishart.rvs(df=nu_post, scale=S_post_inv)
            Sigma_d = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            Sigma_d = np.diag(np.diag(S_post) / max(nu_post - K - 1, 1))

        Sigma_d = 0.5 * (Sigma_d + Sigma_d.T)
        Sigma_draws[d] = Sigma_d

        # ── Draw B | Sigma equation by equation ───────────────────────────
        sigma_diag = np.diag(Sigma_d)
        for j in range(K):
            sig_jj = max(sigma_diag[j], 1e-12)
            # beta_j | Sigma ~ N(b_post_j, sig_jj * Omega_post_j)
            z = np.random.standard_normal(M)
            beta_j_draw = B_post[:, j] + np.sqrt(sig_jj) * L_Omega[j] @ z
            B_draws[d, :, j] = beta_j_draw

    return B_draws, Sigma_draws


# ════════════════════════════════════════════════════════════════════════════
#  CHOLESKY IRF FROM A SINGLE DRAW
# ════════════════════════════════════════════════════════════════════════════

def irf_from_draw(B_draw, Sigma_draw, K, p, horizon,
                  shock_idx=4, response_idx=1):
    """
    Compute Cholesky IRF for one posterior draw.

    B_draw    : (M, K) coefficient matrix  [lag1_var0..lag1_varK-1, ..., const] x K
    Sigma_draw: (K, K) covariance matrix
    shock_idx : index of shock variable (rate = 4)
    response_idx: index of response variable (gdp = 1)

    Returns irfs: array of length `horizon`
    """
    M = B_draw.shape[0]
    # Extract lag coefficient matrix (exclude constant, last row)
    # Rows 0..K*p-1 are lag coefficients, row K*p is constant
    A1 = B_draw[:K, :].T   # (K, K) — lag-1 coefficient matrix
    # A1[i,j] = coefficient of variable j at lag 1 in equation i

    # Cholesky decomposition of Sigma
    try:
        P = np.linalg.cholesky(Sigma_draw)  # lower-triangular
    except np.linalg.LinAlgError:
        try:
            P = np.linalg.cholesky(Sigma_draw + np.eye(K) * 1e-8)
        except np.linalg.LinAlgError:
            return np.zeros(horizon)

    # Normalization: 100bp shock to rate
    norm = P[shock_idx, shock_idx]
    if abs(norm) < 1e-10:
        norm = 1.0

    # Impact vector: P-th column normalized
    impact = P[:, shock_idx] / norm   # (K,)

    # Compute IRF: irf_h = A1^h @ impact,  extract response_idx
    irfs = np.zeros(horizon)
    current = impact.copy()
    for h in range(horizon):
        irfs[h] = current[response_idx]
        current = A1 @ current

    return irfs


# ════════════════════════════════════════════════════════════════════════════
#  MAIN BVAR ROUTINE
# ════════════════════════════════════════════════════════════════════════════

def run_bvar(Y, lambda1=0.2, lambda2=0.5, lambda3=1.0,
             p=1, B_draws_n=5000, horizon=9,
             shock_idx=4, response_idx=1,
             random_walk_vars=None, label=''):
    """
    Full BVAR workflow: prior setup, posterior, sampling, IRFs.

    Y           : (T, K) numpy array
    lambda1/2/3 : Minnesota hyperparameters
    p           : VAR lag order
    B_draws_n   : number of posterior draws
    horizon     : IRF horizon (h=0..horizon-1)
    shock_idx   : index of shock variable
    response_idx: index of response variable
    random_walk_vars: set of variable indices with RW prior mean (default: {shock_idx})

    Returns dict with IRF posterior draws and summary statistics.
    """
    T, K = Y.shape
    if random_walk_vars is None:
        random_walk_vars = {shock_idx}   # rate gets RW prior

    print(f"\n{'─'*60}")
    print(f"BVAR (lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3})")
    print(f"  T={T}, K={K}, p={p}, draws={B_draws_n}")

    # ── OLS for Minnesota scaling ─────────────────────────────────────────
    sigma2_ols = ols_residual_variances(Y, p)
    print(f"  OLS residual variances: {np.round(sigma2_ols, 4)}")

    # ── Build Minnesota prior ─────────────────────────────────────────────
    B0, Omega0_diag = build_minnesota_prior(
        K=K, p=p,
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        sigma2_ols=sigma2_ols,
        random_walk_idx=random_walk_vars
    )

    # IW prior: diffuse
    # nu0 = K + 2 (minimum proper, weakly informative)
    # S0 = diagonal with OLS variances (standard Minnesota choice)
    nu0 = K + 2
    S0  = np.diag(sigma2_ols * nu0)   # E[Sigma] = S0/(nu0 - K - 1) ≈ sigma2_ols

    # ── Prepare OLS matrices ──────────────────────────────────────────────
    Y_dep, X_reg = prepare_ols_matrices(Y, p)
    print(f"  Effective obs after lags: T_eff={Y_dep.shape[0]}")

    # ── Compute posterior ─────────────────────────────────────────────────
    B_post, Omega_post_list, nu_post, S_post = compute_niw_posterior(
        Y_dep, X_reg, B0, Omega0_diag, nu0, S0
    )

    print(f"  Posterior nu = {nu_post}")
    print(f"  Posterior B (lag-1 block, own-lag diagonal):")
    for j in range(K):
        print(f"    Equation {j}: own-lag coef = {B_post[j, j]:.4f}")

    # ── Draw from posterior ───────────────────────────────────────────────
    B_samp, Sigma_samp = draw_niw_posterior(
        B_post, Omega_post_list, nu_post, S_post, n_draws=B_draws_n
    )
    print(f"  Posterior draws: B {B_samp.shape}, Sigma {Sigma_samp.shape}")

    # ── Compute IRFs for each draw ────────────────────────────────────────
    irf_draws = np.zeros((B_draws_n, horizon))
    valid = 0
    for d in range(B_draws_n):
        irf_d = irf_from_draw(
            B_samp[d], Sigma_samp[d], K, p, horizon,
            shock_idx=shock_idx, response_idx=response_idx
        )
        # Check for explosive draw (max eigenvalue of A1 > 2, discard)
        A1_d = B_samp[d, :K, :].T
        eigvals = np.abs(np.linalg.eigvals(A1_d))
        if eigvals.max() < 2.0:
            irf_draws[d] = irf_d
            valid += 1
        else:
            irf_draws[d] = np.nan

    print(f"  Valid (non-explosive) draws: {valid}/{B_draws_n}")

    # ── Posterior summary ─────────────────────────────────────────────────
    # Use nanpercentile to handle explosive draws
    irf_median = np.nanpercentile(irf_draws, 50, axis=0)
    irf_p16    = np.nanpercentile(irf_draws, 16, axis=0)   # 68% CI lower
    irf_p84    = np.nanpercentile(irf_draws, 84, axis=0)   # 68% CI upper
    irf_p05    = np.nanpercentile(irf_draws,  5, axis=0)   # 90% CI lower
    irf_p95    = np.nanpercentile(irf_draws, 95, axis=0)   # 90% CI upper

    # Peak GDP response (median)
    peak_h    = np.argmin(np.abs(irf_median))  # horizon of minimum absolute (could go either way)
    # Actually we want the most negative (contractionary) response
    peak_h    = np.argmin(irf_median)
    peak_val  = irf_median[peak_h]

    print(f"\n  ── Posterior IRF Summary ──")
    print(f"  Horizon:                {list(range(horizon))}")
    print(f"  Median:                 {np.round(irf_median, 4).tolist()}")
    print(f"  68% CI lower (p16):     {np.round(irf_p16, 4).tolist()}")
    print(f"  68% CI upper (p84):     {np.round(irf_p84, 4).tolist()}")
    print(f"  90% CI lower (p05):     {np.round(irf_p05, 4).tolist()}")
    print(f"  90% CI upper (p95):     {np.round(irf_p95, 4).tolist()}")
    print(f"\n  PEAK GDP response:")
    print(f"    Horizon:              h={peak_h}")
    print(f"    Posterior median:     {peak_val:.4f} pp")
    print(f"    68% CI:               [{irf_p16[peak_h]:.4f}, {irf_p84[peak_h]:.4f}]")
    print(f"    90% CI:               [{irf_p05[peak_h]:.4f}, {irf_p95[peak_h]:.4f}]")
    print(f"\n  Frequentist baseline:")
    print(f"    Peak:                 {FREQ_PEAK:.4f} pp  (h=3)")
    print(f"    90% CI:               [{FREQ_CI_LO:.4f}, {FREQ_CI_HI:.4f}]")

    return {
        'lambda1':    lambda1,
        'irf_draws':  irf_draws,
        'irf_median': irf_median,
        'irf_p16':    irf_p16,
        'irf_p84':    irf_p84,
        'irf_p05':    irf_p05,
        'irf_p95':    irf_p95,
        'peak_h':     peak_h,
        'peak_val':   peak_val,
        'peak_ci68':  (irf_p16[peak_h], irf_p84[peak_h]),
        'peak_ci90':  (irf_p05[peak_h], irf_p95[peak_h]),
        'B_post':     B_post,
        'nu_post':    nu_post,
        'S_post':     S_post,
    }


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════════════════════════

def make_figure(results_list, out_path):
    """
    Plot BVAR IRF with credible bands + frequentist overlay.
    results_list: list of result dicts from run_bvar() for different lambda1.
    Main result is lambda1=0.2.
    """
    # Find baseline (lambda1=0.2)
    baseline = next(r for r in results_list if r['lambda1'] == 0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: Main BVAR result ─────────────────────────────────────
    ax = axes[0]
    h_ax = np.arange(len(baseline['irf_median']))

    ax.fill_between(h_ax, baseline['irf_p05'], baseline['irf_p95'],
                    alpha=0.25, color=C_LIGHT, label='90% credible band')
    ax.fill_between(h_ax, baseline['irf_p16'], baseline['irf_p84'],
                    alpha=0.45, color=C_MED, label='68% credible band')
    ax.plot(h_ax, baseline['irf_median'],
            color=C_BLACK, lw=2, label='Posterior median')

    # Frequentist point estimate at h=3
    ax.scatter([3], [FREQ_PEAK], color='red', s=60, zorder=5,
               label=f'Frequentist peak ({FREQ_PEAK:.3f})', marker='D')
    ax.axhline(0, color=C_MED, lw=0.8, ls='--')
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('GDP response (pp)')
    ax.set_title('BVAR Minnesota Prior — GDP response to\n100bp monetary policy shock (λ₁=0.2)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks(h_ax)

    # Annotate peak
    pk = baseline['peak_h']
    ax.annotate(
        f"Peak h={pk}\nMedian={baseline['peak_val']:.3f}",
        xy=(pk, baseline['peak_val']),
        xytext=(pk + 1.2, baseline['peak_val'] - 0.08),
        fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color=C_DARK, lw=0.8),
        color=C_DARK
    )

    # ── Right panel: Sensitivity to lambda1 ──────────────────────────────
    ax2 = axes[1]
    colors_sens = {0.1: '#2060a0', 0.2: '#000000', 0.5: '#a03020'}
    ls_sens     = {0.1: '--',      0.2: '-',        0.5: '-.'}

    for r in results_list:
        lam = r['lambda1']
        ax2.plot(h_ax, r['irf_median'],
                 color=colors_sens.get(lam, C_MED),
                 ls=ls_sens.get(lam, '-'),
                 lw=1.8,
                 label=f"λ₁={lam}  90%CI=[{r['peak_ci90'][0]:.3f},{r['peak_ci90'][1]:.3f}]")
        ax2.fill_between(h_ax, r['irf_p05'], r['irf_p95'],
                         alpha=0.12, color=colors_sens.get(lam, C_MED))

    ax2.axhline(0, color=C_MED, lw=0.8, ls='--')
    ax2.scatter([3], [FREQ_PEAK], color='red', s=60, zorder=5,
                marker='D', label=f'Freq. peak ({FREQ_PEAK:.3f})')
    ax2.set_xlabel('Horizon (quarters)')
    ax2.set_ylabel('GDP response (pp)')
    ax2.set_title('Sensitivity to Minnesota tightness λ₁\n(90% credible bands shown)')
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.set_xticks(h_ax)

    fig.suptitle('Bayesian VAR — Monetary Policy Shock → GDP\nPeru, 2004Q2–2025Q3',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format='pdf', bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("OPTION 4: BAYESIAN VAR WITH MINNESOTA PRIOR")
    print("Peru Monetary Policy — GDP response to 100bp rate shock")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    var_df, rate_level, covid_dummy, q = load_var_data()
    Y = var_df.values.astype(float)     # (T, 5)
    K = Y.shape[1]
    var_names = list(var_df.columns)    # [tot, gdp, cpi, fx, rate]
    print(f"\nVariable ordering: {var_names}")
    print(f"Shock:    {var_names[4]} (index 4)")
    print(f"Response: {var_names[1]} (index 1)")

    # ── Main BVAR: lambda1=0.2 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Main BVAR  (lambda1=0.2, lambda2=0.5, lambda3=1.0)")
    print("=" * 70)

    res_main = run_bvar(
        Y,
        lambda1=0.2, lambda2=0.5, lambda3=1.0,
        p=1, B_draws_n=5000, horizon=9,
        shock_idx=4, response_idx=1,
        random_walk_vars={4},
        label='main'
    )

    # ── Sensitivity analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Sensitivity to lambda1 in [0.1, 0.2, 0.5]")
    print("=" * 70)

    results_all = []
    for lam1 in [0.1, 0.2, 0.5]:
        r = run_bvar(
            Y,
            lambda1=lam1, lambda2=0.5, lambda3=1.0,
            p=1, B_draws_n=5000, horizon=9,
            shock_idx=4, response_idx=1,
            random_walk_vars={4},
            label=f'lambda1_{lam1}'
        )
        results_all.append(r)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL NUMERICAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Parameter':<20}{'Peak h':<10}{'Median':<12}{'68% CI':<26}{'90% CI':<26}")
    print("-" * 90)
    for r in results_all:
        pk = r['peak_h']
        ci68 = f"[{r['peak_ci68'][0]:.4f}, {r['peak_ci68'][1]:.4f}]"
        ci90 = f"[{r['peak_ci90'][0]:.4f}, {r['peak_ci90'][1]:.4f}]"
        print(f"lambda1={r['lambda1']:<12.1f}{pk:<10}{r['peak_val']:<12.4f}{ci68:<26}{ci90:<26}")

    print(f"\nFrequentist baseline (OLS/bootstrap):")
    print(f"  Peak h=3:  {FREQ_PEAK:.4f} pp")
    print(f"  90% CI:    [{FREQ_CI_LO:.4f}, {FREQ_CI_HI:.4f}]")

    # Print full IRF tables for lambda1=0.2
    print("\n" + "─" * 70)
    print("Full IRF table — lambda1=0.2 (main result)")
    print("─" * 70)
    r = res_main
    print(f"\n{'h':<5}{'Median':<12}{'p16':<12}{'p84':<12}{'p05':<12}{'p95':<12}")
    print("-" * 60)
    for h in range(9):
        print(f"{h:<5}{r['irf_median'][h]:<12.4f}{r['irf_p16'][h]:<12.4f}"
              f"{r['irf_p84'][h]:<12.4f}{r['irf_p05'][h]:<12.4f}{r['irf_p95'][h]:<12.4f}")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("COMPARISON: BVAR vs Frequentist")
    print("─" * 70)
    r = res_main
    print(f"\n  BVAR (lambda1=0.2):")
    print(f"    Peak GDP response (median):  {r['peak_val']:.4f} pp  (h={r['peak_h']})")
    print(f"    68% credible interval:       [{r['peak_ci68'][0]:.4f}, {r['peak_ci68'][1]:.4f}]")
    print(f"    90% credible interval:       [{r['peak_ci90'][0]:.4f}, {r['peak_ci90'][1]:.4f}]")
    freq_width = FREQ_CI_HI - FREQ_CI_LO
    bvar_width = r['peak_ci90'][1] - r['peak_ci90'][0]
    print(f"\n  Frequentist (OLS/bootstrap):")
    print(f"    Peak GDP response:           {FREQ_PEAK:.4f} pp  (h=3)")
    print(f"    90% CI:                      [{FREQ_CI_LO:.4f}, {FREQ_CI_HI:.4f}]")
    print(f"    90% CI width:                {freq_width:.4f}")
    print(f"\n  BVAR 90% CI width:             {bvar_width:.4f}")
    print(f"  Width ratio (BVAR/Freq):       {bvar_width/freq_width:.3f}")

    # Sign consistency
    bvar_neg = r['peak_ci90'][1] < 0
    bvar_contains_freq = (r['peak_ci90'][0] <= FREQ_PEAK <= r['peak_ci90'][1])
    print(f"\n  BVAR 90% CI entirely negative: {bvar_neg}")
    print(f"  Frequentist peak in BVAR 90% CI: {bvar_contains_freq}")

    # ── Make figure ───────────────────────────────────────────────────────
    out_fig = OUT_DIR / 'option4_bvar_irf.pdf'
    make_figure(results_all, out_fig)

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == '__main__':
    main()
