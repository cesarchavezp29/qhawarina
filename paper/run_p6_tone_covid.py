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
TONE_CSV = ROOT / 'data/raw/bcrp/notas_informativas/tone_scores.csv'
OUT_DIR = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(ROOT / 'paper'))
from paper_style import apply_style, C, SZ, zero_line, legend_below, legend_outside, stat_box
apply_style()

BASELINE_PEAK_GDP = -0.195

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


def compute_cholesky_irf(var_result, shock_var_idx, horizon=9, n_boot=2000,
                          response_idx=1, lags=1):
    K = var_result.neqs
    resids = var_result.resid
    Sigma  = var_result.sigma_u
    try:
        P = np.linalg.cholesky(Sigma)
    except:
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_column(df, candidates):
    """Return the first column name in df that matches any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
results_lines = []

def log(msg=''):
    print(msg)
    results_lines.append(str(msg))

log("=" * 70)
log("SCRIPT 6: TONE VALIDATION & COVID STRUCTURAL BREAK")
log("=" * 70)

# ─── Load VAR data ────────────────────────────────────────────────────────────
var_df, rate_level, covid_dummy, q_full = load_var_data()
K         = 5
col_names = ['tot','gdp','cpi','fx','rate']
rate_idx  = 4
gdp_idx   = 1

# ─────────────────────────────────────────────────────────────────────────────
# PART A — TONE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("PART A: LLM TONE VALIDATION")
log("=" * 70)

part_a_ok = False
tone = None

try:
    # Try primary path
    try:
        tone = pd.read_csv(TONE_CSV)
        log(f"Tone data loaded from: {TONE_CSV}")
    except FileNotFoundError:
        TONE_CSV2 = ROOT / 'paper/output/bcrp_tone_dataset.csv'
        tone = pd.read_csv(TONE_CSV2)
        log(f"Loaded from alternate path: {TONE_CSV2}")

    log(f"Tone data shape: {tone.shape}")
    log(f"Columns: {tone.columns.tolist()}")
    log(tone.head(3).to_string())

    # ── Identify column names defensively ────────────────────────────────────
    date_col  = find_column(tone, ['date','fecha','period','month'])
    dict_col  = find_column(tone, ['dict_tone','dict_score','dictionary_score','score_dict'])
    llm_col   = find_column(tone, ['llm_tone','llm_score','tone_llm','score_llm','tone_score'])
    nchar_col = find_column(tone, ['n_chars','word_count','nchars','length','n_words'])

    log(f"\nColumn mapping: date={date_col}, dict={dict_col}, llm={llm_col}, length={nchar_col}")

    if date_col is None:
        raise ValueError("No date column found in tone CSV.")

    tone[date_col] = pd.to_datetime(tone[date_col])
    tone = tone.sort_values(date_col).reset_index(drop=True)

    # ── Merge with monthly rate changes ──────────────────────────────────────
    raw_panel = pd.read_parquet(PANEL)
    raw_panel['date'] = pd.to_datetime(raw_panel['date'])
    rate_m = raw_panel[raw_panel['series_id'] == 'PD04722MM'][['date','value_raw']].copy()
    rate_m = rate_m.set_index('date').sort_index()
    rate_m.columns = ['rate_level_m']
    rate_m['d_rate_m'] = rate_m['rate_level_m'].diff()

    # Align tone to month-start for merging
    tone_merge = tone.copy()
    tone_merge['merge_date'] = tone_merge[date_col].dt.to_period('M').dt.to_timestamp()
    rate_m_reset = rate_m.copy()
    rate_m_reset.index = rate_m_reset.index.to_period('M').to_timestamp()

    rate_m_for_merge = rate_m_reset[['d_rate_m']].reset_index()
    rate_m_for_merge = rate_m_for_merge.rename(columns={rate_m_for_merge.columns[0]: 'merge_date'})
    merged = tone_merge.merge(rate_m_for_merge, on='merge_date', how='left')
    merged['d_rate_m'] = merged['d_rate_m'].fillna(0.0)

    log(f"\nMerged tone+rate rows: {len(merged)}")
    log(f"Non-zero rate changes: {(merged['d_rate_m'] != 0).sum()}")

    # ── Descriptive stats ────────────────────────────────────────────────────
    log("\nDescriptive statistics:")
    desc_cols = [c for c in [dict_col, llm_col, 'd_rate_m'] if c is not None]
    log(merged[desc_cols].describe().round(4).to_string())

    if dict_col and llm_col:
        corr_dl = merged[[dict_col, llm_col]].corr().iloc[0,1]
        log(f"\nCorrelation (dict_tone, llm_tone): {corr_dl:.4f}")

    # ── 1. Predictive validity regressions ───────────────────────────────────
    log("\n--- 1. Predictive Validity Regressions ---")

    for score_col, score_label in [(llm_col, 'LLM'), (dict_col, 'Dict')]:
        if score_col is None:
            log(f"{score_label} score column not found — skipped.")
            continue

        sub = merged[[score_col, 'd_rate_m']].dropna()
        if len(sub) < 20:
            log(f"{score_label}: too few observations ({len(sub)}).")
            continue

        # Contemporaneous: Δrate_t = α + β × tone_t
        X = sm.add_constant(sub[score_col])
        res_contemp = sm.OLS(sub['d_rate_m'], X).fit(cov_type='HC3')
        log(f"\n  {score_label} → Δrate (contemporaneous):")
        log(f"    β = {res_contemp.params[score_col]:.5f}  p = {res_contemp.pvalues[score_col]:.4f}  "
            f"R² = {res_contemp.rsquared:.4f}  N = {len(sub)}")

        # Forward-looking: Δrate_{t+1} = α + β × tone_t
        sub2 = merged[[score_col, 'd_rate_m']].copy()
        sub2['d_rate_fwd'] = sub2['d_rate_m'].shift(-1)
        sub2 = sub2.dropna()
        if len(sub2) >= 20:
            X2 = sm.add_constant(sub2[score_col])
            res_fwd = sm.OLS(sub2['d_rate_fwd'], X2).fit(cov_type='HC3')
            log(f"  {score_label} → Δrate(t+1) (forward-looking):")
            log(f"    β = {res_fwd.params[score_col]:.5f}  p = {res_fwd.pvalues[score_col]:.4f}  "
                f"R² = {res_fwd.rsquared:.4f}  N = {len(sub2)}")

    # ── 2. Confusion matrix ───────────────────────────────────────────────────
    log("\n--- 2. Confusion Matrix: Tone vs Rate Decision ---")

    if llm_col is not None:
        def classify_tone(score):
            if score > 20:   return 'hawkish'
            elif score < -20: return 'dovish'
            else:             return 'neutral'

        def classify_decision(delta):
            if delta > 0:    return 'hike'
            elif delta < 0:  return 'cut'
            else:            return 'hold'

        merged['tone_class']  = merged[llm_col].apply(classify_tone)
        merged['rate_class']  = merged['d_rate_m'].apply(classify_decision)

        xtab = pd.crosstab(merged['tone_class'], merged['rate_class'],
                           margins=True, margins_name='Total')
        log("\nCross-tabulation (rows=tone, cols=decision):")
        log(xtab.to_string())

        # Precision/recall on non-zero rate changes
        nonzero = merged[merged['d_rate_m'] != 0].copy()
        if len(nonzero) > 0:
            hawk_hike = ((nonzero['tone_class']=='hawkish') & (nonzero['rate_class']=='hike')).sum()
            hawk_tot  = (nonzero['tone_class']=='hawkish').sum()
            dove_cut  = ((nonzero['tone_class']=='dovish')  & (nonzero['rate_class']=='cut')).sum()
            dove_tot  = (nonzero['tone_class']=='dovish').sum()
            log(f"\nOn non-zero rate changes (N={len(nonzero)}):")
            log(f"  Hawkish→Hike accuracy: {hawk_hike}/{hawk_tot} = "
                f"{hawk_hike/hawk_tot:.2%}" if hawk_tot > 0 else "  Hawkish→Hike: no hawkish months in active periods")
            log(f"  Dovish→Cut accuracy:   {dove_cut}/{dove_tot} = "
                f"{dove_cut/dove_tot:.2%}" if dove_tot > 0 else "  Dovish→Cut: no dovish months in active periods")
    else:
        log("LLM score column not found — confusion matrix skipped.")

    # ── 3. Text length confound ───────────────────────────────────────────────
    log("\n--- 3. Text Length Confound ---")

    if nchar_col is not None and llm_col is not None:
        sub_len = merged[[nchar_col, llm_col]].dropna()
        if len(sub_len) > 10:
            corr_len = sub_len.corr().iloc[0,1]
            log(f"Correlation (n_chars/word_count, LLM score): {corr_len:.4f}")
            _, pval_len = stats.pearsonr(sub_len[nchar_col], sub_len[llm_col])
            log(f"Pearson r = {corr_len:.4f}, p = {pval_len:.4f}")
            if abs(corr_len) > 0.3:
                log("WARNING: moderate correlation with text length — potential confound.")
            else:
                log("Low correlation with text length — confound unlikely.")
    else:
        # Try n_chars column (present in actual data)
        nchar_col2 = find_column(merged, ['n_chars','nchars','n_words','word_count','length'])
        if nchar_col2 and llm_col:
            sub_len = merged[[nchar_col2, llm_col]].dropna()
            corr_len = sub_len.corr().iloc[0,1]
            _, pval_len = stats.pearsonr(sub_len[nchar_col2], sub_len[llm_col])
            log(f"Correlation ({nchar_col2}, LLM score): {corr_len:.4f}, p = {pval_len:.4f}")
        else:
            log("Word count not available in CSV.")

    # ── 4. Spot-check disagreements ───────────────────────────────────────────
    log("\n--- 4. Spot-Check: Largest Dict vs LLM Disagreements ---")

    if dict_col and llm_col:
        sub_dis = merged[[date_col, dict_col, llm_col, 'd_rate_m']].dropna(
            subset=[dict_col, llm_col])
        if len(sub_dis) > 0:
            # Normalize to z-scores
            dict_z = (sub_dis[dict_col] - sub_dis[dict_col].mean()) / (sub_dis[dict_col].std() + 1e-10)
            llm_z  = (sub_dis[llm_col]  - sub_dis[llm_col].mean())  / (sub_dis[llm_col].std()  + 1e-10)
            sub_dis = sub_dis.copy()
            sub_dis['dict_z'] = dict_z.values
            sub_dis['llm_z']  = llm_z.values
            sub_dis['disagreement'] = (sub_dis['dict_z'] - sub_dis['llm_z']).abs()
            top10 = sub_dis.nlargest(10, 'disagreement')
            log(f"\nTop 10 dict vs LLM disagreements (z-score normalized):")
            cols_show = [date_col, dict_col, llm_col, 'd_rate_m', 'dict_z', 'llm_z', 'disagreement']
            cols_show = [c for c in cols_show if c in top10.columns]
            log(top10[cols_show].to_string(index=False))
    else:
        log("Dict or LLM column not found — disagreement analysis skipped.")

    # ── Part A figure ─────────────────────────────────────────────────────────
    fig_a, axes_a = plt.subplots(1, 2, figsize=SZ["wide_tall"])

    # Panel 1: Scatter LLM tone vs rate change
    if llm_col:
        sub_sc = merged[[llm_col, 'd_rate_m']].dropna()
        axes_a[0].scatter(sub_sc[llm_col], sub_sc['d_rate_m'],
                          alpha=0.5, s=18, color=C["accent2"], edgecolors='none')
        if len(sub_sc) >= 3:
            m, b, r, p, se = stats.linregress(sub_sc[llm_col], sub_sc['d_rate_m'])
            xr = np.linspace(sub_sc[llm_col].min(), sub_sc[llm_col].max(), 100)
            axes_a[0].plot(xr, m*xr + b, color=C["accent1"], lw=1.5,
                           label=f'β={m:.4f}, p={p:.3f}, R²={r**2:.3f}')
        zero_line(axes_a[0])
        axes_a[0].set_xlabel('LLM Tone Score')
        axes_a[0].set_ylabel('Δ Policy Rate (pp)')
        axes_a[0].set_title('')
        legend_below(axes_a[0], ncol=1)

    # Panel 2: Confusion matrix heatmap
    if llm_col and 'rate_class' in merged.columns:
        orders_t = ['dovish','neutral','hawkish']
        orders_d = ['cut','hold','hike']
        ctab = pd.crosstab(merged['tone_class'], merged['rate_class'])
        ctab = ctab.reindex(index=[o for o in orders_t if o in ctab.index],
                            columns=[o for o in orders_d if o in ctab.columns],
                            fill_value=0)
        im = axes_a[1].imshow(ctab.values, aspect='auto', cmap='Blues')
        axes_a[1].set_xticks(range(len(ctab.columns)))
        axes_a[1].set_yticks(range(len(ctab.index)))
        axes_a[1].set_xticklabels(ctab.columns)
        axes_a[1].set_yticklabels(ctab.index)
        axes_a[1].set_xlabel('Rate Decision')
        axes_a[1].set_ylabel('LLM Tone Classification')
        axes_a[1].set_title('')
        for i in range(ctab.shape[0]):
            for j in range(ctab.shape[1]):
                axes_a[1].text(j, i, str(ctab.values[i,j]),
                               ha='center', va='center', fontsize=10,
                               color='white' if ctab.values[i,j] > ctab.values.max()*0.6 else 'black')
        plt.colorbar(im, ax=axes_a[1], shrink=0.8)

    fig_a.savefig(OUT_DIR / 'p6a_tone_validation.pdf')
    shutil.copy(OUT_DIR / 'p6a_tone_validation.pdf', ROOT / 'paper' / 'figures' / 'fig17_tone_validation.pdf')
    plt.close(fig_a)
    log("\nSaved: p6a_tone_validation.pdf")

    part_a_ok = True

except Exception as e:
    log(f"\nPart A failed: {type(e).__name__}: {e}")
    log("Continuing with Part B...")
    import traceback
    log(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# PART B — COVID STRUCTURAL BREAK
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("PART B: COVID STRUCTURAL BREAK ANALYSIS")
log("=" * 70)

# ── Estimate full-sample baseline VAR ────────────────────────────────────────
var_model  = VAR(var_df)
var_result = var_model.fit(1, trend='c')
log(f"\nBaseline VAR(1): T={len(var_df)}")

# ── 1. Split-sample VAR ───────────────────────────────────────────────────────
log("\n--- 1. Split-Sample VAR ---")

pre_df  = var_df.loc[:'2019-12-31'].copy()
post_df = var_df.loc['2021-01-01':].copy()

log(f"Pre-COVID sample:  {pre_df.index[0].date()} to {pre_df.index[-1].date()}  T={len(pre_df)}")
log(f"Post-COVID sample: {post_df.index[0].date()} to {post_df.index[-1].date()}  T={len(post_df)}")

split_results = {}

for label, sub_df, flag_short in [('Pre-COVID', pre_df, False), ('Post-COVID', post_df, True)]:
    if len(sub_df) < 10:
        log(f"\n{label}: too few observations (T={len(sub_df)}) — skipped.")
        continue
    if flag_short:
        log(f"\nWARNING: Post-COVID T={len(sub_df)} is very short for a VAR(1). "
            f"Estimates will be unreliable. Interpret with extreme caution.")
    try:
        res_s = VAR(sub_df).fit(1, trend='c')
        irf_s, ci_lo_s, ci_hi_s = compute_cholesky_irf(
            res_s, shock_var_idx=rate_idx, horizon=9,
            n_boot=500, response_idx=gdp_idx, lags=1
        )
        peak_s = float(np.min(irf_s))
        split_results[label] = {
            'result': res_s, 'irf': irf_s,
            'ci_lo': ci_lo_s, 'ci_hi': ci_hi_s, 'peak': peak_s,
        }
        log(f"\n{label} VAR(1):")
        log(f"  Peak GDP IRF: {peak_s:.4f} at h={int(np.argmin(irf_s))}")
        log(f"  90% CI: [{ci_lo_s[np.argmin(irf_s)]:.4f}, {ci_hi_s[np.argmin(irf_s)]:.4f}]")
    except Exception as e:
        log(f"\n{label} VAR estimation failed: {e}")

# ── 2. Rolling window analysis ────────────────────────────────────────────────
log("\n--- 2. Rolling Window Analysis (40-quarter window) ---")

WINDOW = 40
roll_results = []
T_total = len(var_df)

for t_end in range(WINDOW, T_total + 1):
    sub = var_df.iloc[t_end - WINDOW : t_end]
    end_date = sub.index[-1]
    try:
        res_r = VAR(sub).fit(1, trend='c')
        irf_r, _, _ = compute_cholesky_irf(
            res_r, shock_var_idx=rate_idx, horizon=9,
            n_boot=100, response_idx=gdp_idx, lags=1
        )
        # Bootstrap SD for uncertainty band
        boot_peaks = []
        resids_r = res_r.resid.values
        T_r = len(resids_r)
        for _ in range(100):
            idx_b = np.random.randint(0, T_r, T_r)
            br = resids_r[idx_b]
            Yb = np.zeros((T_r + 1, K))
            Yb[0] = res_r.model.endog[0]
            for t_ in range(1, T_r + 1):
                Yb[t_] = Yb[t_-1] @ res_r.coefs[0].T + br[t_-1]
            try:
                rb = VAR(pd.DataFrame(Yb[1:], columns=var_df.columns)).fit(1, trend='c')
                ib, _, _ = compute_cholesky_irf(rb, shock_var_idx=rate_idx, horizon=9,
                                                 n_boot=0, response_idx=gdp_idx, lags=1)
                boot_peaks.append(float(np.min(ib)))
            except:
                pass
        sd_peak = float(np.std(boot_peaks)) if boot_peaks else 0.0
        roll_results.append({
            'end_date': end_date,
            'peak': float(np.min(irf_r)),
            'sd_peak': sd_peak,
            'can_compute': True,
        })
    except Exception as e:
        roll_results.append({'end_date': end_date, 'peak': np.nan, 'sd_peak': 0.0, 'can_compute': False})

roll_df = pd.DataFrame(roll_results).set_index('end_date')
valid_roll = roll_df[roll_df['can_compute'] & roll_df['peak'].notna()]
log(f"Rolling windows computed: {len(valid_roll)} / {len(roll_df)}")
log(f"Peak GDP IRF range: [{valid_roll['peak'].min():.4f}, {valid_roll['peak'].max():.4f}]")

# Rolling IRF figure
fig_roll, ax_roll = plt.subplots(figsize=SZ["wide"])
ax_roll.plot(valid_roll.index, valid_roll['peak'], color=C["accent2"], lw=1.5, label='Rolling peak GDP IRF')
ax_roll.fill_between(valid_roll.index,
                     valid_roll['peak'] - valid_roll['sd_peak'],
                     valid_roll['peak'] + valid_roll['sd_peak'],
                     alpha=0.15, color=C["ci_light"], label='±1 bootstrap SD')
ax_roll.axhline(BASELINE_PEAK_GDP, color=C["accent1"], ls='--', lw=1.0, label=f'Full-sample baseline ({BASELINE_PEAK_GDP:.3f})')
covid_line = pd.Timestamp('2020-03-31')
if covid_line >= valid_roll.index.min() and covid_line <= valid_roll.index.max():
    ax_roll.axvline(covid_line, color=C["gray_line"], ls=':', lw=1.5, label='COVID start (2020Q1)')
ax_roll.set_xlabel('Window end date')
ax_roll.set_ylabel('Peak GDP IRF')
ax_roll.set_title('')
legend_below(ax_roll, ncol=4)
fig_roll.savefig(OUT_DIR / 'p6b_rolling_irf.pdf')
plt.close(fig_roll)
log("Saved: p6b_rolling_irf.pdf")

# ── 3. Chow-type LR test ──────────────────────────────────────────────────────
log("\n--- 3. Chow-Type Likelihood Ratio Test ---")

def var_loglik(res):
    """Extract log-likelihood from a fitted VAR result."""
    try:
        return res.llf
    except AttributeError:
        # Manually compute: sum of equation log-likelihoods
        T = res.resid.shape[0]
        K_ = res.neqs
        Sigma = res.sigma_u
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return np.nan
        return -0.5 * T * (K_ * np.log(2 * np.pi) + logdet + K_)

logL_full = var_loglik(var_result)

logL_pre  = np.nan
logL_post = np.nan

if 'Pre-COVID' in split_results:
    logL_pre = var_loglik(split_results['Pre-COVID']['result'])
if 'Post-COVID' in split_results:
    logL_post = var_loglik(split_results['Post-COVID']['result'])

log(f"Log-likelihood (full sample):  {logL_full:.4f}")
log(f"Log-likelihood (pre-COVID):    {logL_pre:.4f}")
log(f"Log-likelihood (post-COVID):   {logL_post:.4f}")

if not np.isnan(logL_pre) and not np.isnan(logL_post) and not np.isnan(logL_full):
    LR_stat = -2.0 * (logL_full - logL_pre - logL_post)
    # df = K² × p × (number of equations) — for K=5, p=1: 5 coefs × 5 equations = 25
    # plus intercepts: 5 intercepts × equations = 5
    # Total restricted: K*(K*p + 1) = 5*(5+1) = 30
    df_chow = K * (K * 1 + 1)   # 30
    pval_chow = 1 - stats.chi2.cdf(LR_stat, df=df_chow)
    log(f"\nLR statistic: {LR_stat:.4f}")
    log(f"Degrees of freedom: {df_chow}")
    log(f"p-value: {pval_chow:.4f}")
    sig_chow = '***' if pval_chow < 0.01 else ('**' if pval_chow < 0.05 else ('*' if pval_chow < 0.10 else ''))
    log(f"Significance: {sig_chow if sig_chow else 'not significant at 10%'}")
    log("\nCAVEAT: Post-COVID sub-sample T=19 makes this test unreliable.")
    log("Low power; LR test result should be interpreted cautiously.")
else:
    log("LR test could not be computed (missing log-likelihoods).")

# ── 4. Interacted VAR ─────────────────────────────────────────────────────────
log("\n--- 4. Interacted VAR (COVID Regime Switching) ---")

# Create post-COVID indicator
D_post = pd.Series(0, index=var_df.index, name='D_post')
D_post.loc['2021-01-01':] = 1
log(f"D_post = 1 for {D_post.sum()} quarters (2021Q1 onward)")

# Build augmented dataset: Y, Y_lag, D, D*Y_lag
Y = var_df.copy()
Y_lag = Y.shift(1)
Y_lag.columns = [c + '_L1' for c in col_names]

D_ser = D_post.reindex(Y.index)
interact_cols = {}
for c in col_names:
    interact_cols[c + '_post'] = Y_lag[c + '_L1'] * D_ser

df_int = pd.concat([Y, Y_lag, D_ser, pd.DataFrame(interact_cols, index=Y.index)], axis=1).dropna()
log(f"Interacted VAR dataset T={len(df_int)}")

# Estimate equation-by-equation OLS
int_coefs = {}   # stores dict {'A1_pre': array(K,K), 'A1_post_delta': array(K,K)}
A1_pre_int   = np.zeros((K, K))
A1_delta_int = np.zeros((K, K))
all_pvals_interaction = []

for i, dep in enumerate(col_names):
    rhs_base    = ['const'] + [c + '_L1' for c in col_names] + ['D_post']
    rhs_inter   = [c + '_post' for c in col_names]
    rhs_all     = rhs_base + rhs_inter

    X_full = sm.add_constant(df_int[[c for c in rhs_all if c != 'const'] + []])
    # Build manually
    X_mat = pd.DataFrame({'const': 1.0}, index=df_int.index)
    for c in [c + '_L1' for c in col_names]:
        X_mat[c] = df_int[c]
    X_mat['D_post'] = df_int['D_post']
    for c in [c + '_post' for c in col_names]:
        X_mat[c] = df_int[c]

    res_int_eq = sm.OLS(df_int[dep], X_mat).fit(cov_type='HC3')

    for j, pred in enumerate(col_names):
        A1_pre_int[i, j]   = res_int_eq.params.get(pred + '_L1', 0.0)
        A1_delta_int[i, j] = res_int_eq.params.get(pred + '_post', 0.0)
        pval_ij = res_int_eq.pvalues.get(pred + '_post', 1.0)
        all_pvals_interaction.append(pval_ij)

    int_coefs[dep] = res_int_eq

# F-test: H0: all A1* = 0
# Use joint Wald test across all interaction terms
interaction_terms = [c + '_post' for c in col_names]
log("\nInteraction term p-values (A1* = 0 tests, per equation):")
header = f"{'Equation':<12}" + ''.join(f"  {it[:12]:>12}" for it in interaction_terms)
log(header)
for dep, res_eq in int_coefs.items():
    row = f"  {dep:<10}" + ''.join(
        f"  {res_eq.pvalues.get(it, np.nan):>10.3f}{'*' if res_eq.pvalues.get(it,1)<0.10 else ' ':>2}"
        for it in interaction_terms
    )
    log(row)

# Joint F-test: all interaction terms = 0 in GDP equation
try:
    gdp_eq_res = int_coefs[col_names[gdp_idx]]
    hyp_str = ', '.join([f"({it} = 0)" for it in interaction_terms])
    f_test = gdp_eq_res.f_test(np.eye(len(gdp_eq_res.params))[
        [list(gdp_eq_res.params.index).index(it) for it in interaction_terms
         if it in gdp_eq_res.params.index]
    ])
    log(f"\nJoint F-test (GDP equation, all interaction terms = 0):")
    log(f"  F = {f_test.fvalue:.4f}  df_num = {f_test.df_num}  p = {f_test.pvalue:.4f}")
except Exception as e:
    log(f"Joint F-test failed: {e}")

# Pre and post IRFs from interacted VAR
def compute_interacted_irf(A1, horizon=9):
    """Compute Cholesky IRF from a given A1 matrix (Σ from pre-COVID residuals)."""
    irf_raw = np.zeros((horizon, K))
    Ah = np.eye(K)
    # Use identity shock for rate (column rate_idx) normalized to 1
    e_rate = np.zeros(K)
    e_rate[rate_idx] = 1.0
    for h in range(horizon):
        irf_raw[h] = Ah @ e_rate
        Ah = Ah @ A1
    return irf_raw[:, gdp_idx]

irf_pre_int  = compute_interacted_irf(A1_pre_int)
irf_post_int = compute_interacted_irf(A1_pre_int + A1_delta_int)
peak_pre_int  = float(np.min(irf_pre_int))
peak_post_int = float(np.min(irf_post_int))

log(f"\nInteracted VAR IRFs (simplified, unit shock to rate):")
log(f"  Pre-COVID  peak GDP: {peak_pre_int:.4f}")
log(f"  Post-COVID peak GDP: {peak_post_int:.4f}")

# Split-sample IRF figure
fig_split, ax_split = plt.subplots(figsize=SZ["wide"])
horizons = list(range(9))

ax_split.plot(horizons, irf_pre_int, 'o-', color=C["accent2"], lw=1.5, ms=4, label='Pre-COVID (interacted)')
ax_split.plot(horizons, irf_post_int, 's--', color=C["accent1"], lw=1.5, ms=4, label='Post-COVID (interacted)')

if 'Pre-COVID' in split_results:
    ax_split.plot(horizons, split_results['Pre-COVID']['irf'], '^:', color=C["accent3"], lw=1.2, ms=3,
                  label='Pre-COVID split-sample')
if 'Post-COVID' in split_results:
    ax_split.plot(horizons, split_results['Post-COVID']['irf'], 'v:', color=C["gray_line"], lw=1.2, ms=3,
                  label=f"Post-COVID split-sample (T={len(post_df)}, caution)")

zero_line(ax_split)
ax_split.set_xlabel('Horizon (quarters)')
ax_split.set_ylabel('GDP response to rate shock')
ax_split.set_title('')
legend_below(ax_split, ncol=4)
fig_split.savefig(OUT_DIR / 'p6b_split_irf.pdf')
plt.close(fig_split)
log("Saved: p6b_split_irf.pdf")

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("SUMMARY: COVID STRUCTURAL BREAK")
log("=" * 70)
log(f"Full-sample baseline peak GDP IRF: {BASELINE_PEAK_GDP:.4f}")
if 'Pre-COVID' in split_results:
    log(f"Pre-COVID split-sample peak:       {split_results['Pre-COVID']['peak']:.4f}")
if 'Post-COVID' in split_results:
    log(f"Post-COVID split-sample peak:      {split_results['Post-COVID']['peak']:.4f}  "
        f"(T={len(post_df)}, UNRELIABLE)")
log(f"Interacted VAR pre-COVID peak:     {peak_pre_int:.4f}")
log(f"Interacted VAR post-COVID peak:    {peak_post_int:.4f}")
log(f"\nRolling window peak range: [{valid_roll['peak'].min():.4f}, {valid_roll['peak'].max():.4f}]")
log("Interpretation: stability assessed via rolling windows and interacted VAR.")
log("Post-COVID split subsample (T~19) is too short for reliable inference.")

# ── Save results text ─────────────────────────────────────────────────────────
out_txt = OUT_DIR / 'p6_tone_covid_results.txt'
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results_lines))
log(f"\nAll results saved to: {out_txt}")
log("Script 6 complete.")
