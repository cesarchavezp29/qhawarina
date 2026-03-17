"""
Strategy 4: Annual ENAHO — Robustness check.
Uses existing corrected results from mw_canonical_results.json.
Adds IVW pooling across events A, B, C (excl. C from employment due to
COVID recovery confound; flagged as caveat).
Saves to mw_annual_robustness_results.json.

NOTE: This is a ROBUSTNESS check. Primary results are Cengiz bunching
(Strategy 1) and quarterly DiD (Strategy 2). Annual data has large gaps
between pre/post periods (2+ years) making causal identification weaker.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import os
import shutil
import numpy as np
from scipy import stats

# Load existing corrected results
CANON = json.load(open('D:/Nexus/nexus/exports/data/mw_canonical_results.json'))
MULTI = json.load(open('D:/Nexus/nexus/exports/data/mw_multi_dataset_results.json'))

EVENTS_ORDER = ['A', 'B', 'C']
OUTCOMES = ['employed', 'formal', 'log_wage']

OUTCOME_LABELS = {
    'employed': 'Employment',
    'formal': 'Formalization',
    'log_wage': 'Log Wages',
}

CAVEATS = {
    'C': {
        'employed': 'Event C employment may reflect COVID-19 recovery (pre=2021, post=2022-23). '
                    'Not included in employment pooled estimate.',
    }
}


def ivw_pool(betas, ses, labels=None):
    """Inverse-variance weighted pooled estimate."""
    valid = [(b, s, l) for b, s, l in zip(betas, ses, labels or [''] * len(betas))
             if not np.isnan(b) and not np.isnan(s) and s > 0]
    if len(valid) < 2:
        return None

    betas_v = [x[0] for x in valid]
    ses_v = [x[1] for x in valid]
    labels_v = [x[2] for x in valid]

    weights = [1 / s**2 for s in ses_v]
    total_w = sum(weights)
    beta_pool = sum(b * w for b, w in zip(betas_v, weights)) / total_w
    se_pool = 1 / np.sqrt(total_w)
    z = beta_pool / se_pool
    p_pool = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_low = beta_pool - 1.96 * se_pool
    ci_high = beta_pool + 1.96 * se_pool

    Q = sum(w * (b - beta_pool)**2 for b, w in zip(betas_v, weights))
    df_Q = len(betas_v) - 1
    I2 = max(0.0, (Q - df_Q) / Q * 100) if Q > 0 else 0.0
    p_Q = float(1 - stats.chi2.cdf(Q, df_Q))

    return {
        'beta': float(beta_pool), 'se': float(se_pool),
        'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'p': p_pool, 'Q': float(Q), 'I2': float(I2), 'p_Q': p_Q,
        'n_events': len(betas_v), 'events_included': labels_v,
    }


def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '   '))


if __name__ == '__main__':
    print('=' * 70)
    print('STRATEGY 4: ANNUAL ENAHO ROBUSTNESS CHECK')
    print('=' * 70)
    print('NOTE: Annual data gaps (2+ years pre/post) weaken causal identification.')
    print('      Primary results: Cengiz bunching (S1) + Quarterly DiD (S2).')
    print()

    events = CANON.get('events', {})
    pooled_results = {}

    for outcome in OUTCOMES:
        print(f'\n{OUTCOME_LABELS[outcome].upper()}:')
        print(f'  {"Event":<35} {"beta":>8} {"SE":>7} {"CI":>20} {"p":>7} {"N":>8}')
        print('  ' + '-' * 90)

        betas, ses, labels = [], [], []
        event_results = {}

        for eid in EVENTS_ORDER:
            ev = events.get(eid, {})
            res = ev.get(outcome, {})
            b = res.get('beta', np.nan)
            s = res.get('se', np.nan)
            p = res.get('p', np.nan)
            n = res.get('n', '?')
            ci_low = res.get('ci_low', b - 1.96 * s if not np.isnan(b) else np.nan)
            ci_high = res.get('ci_high', b + 1.96 * s if not np.isnan(b) else np.nan)

            label = ev.get('label', f'Event {eid}')[:35]
            caveat = CAVEATS.get(eid, {}).get(outcome, '')
            caveat_flag = ' [CAVEAT]' if caveat else ''

            print(f'  {label:<35} {b:>+8.4f} {s:>7.4f} '
                  f'[{ci_low:>+7.4f},{ci_high:>+7.4f}] '
                  f'{p:>7.3f}{stars(p)}{caveat_flag}')
            if caveat:
                print(f'    ** {caveat}')

            event_results[eid] = {'beta': b, 'se': s, 'ci_low': ci_low,
                                   'ci_high': ci_high, 'p': p, 'n': n}

            # Only include in pool if no caveat (or outcome not caveated)
            if not caveat:
                betas.append(b)
                ses.append(s)
                labels.append(eid)

        # Placebo check
        plac = events.get('placebo', {}).get(outcome, {})
        if plac:
            pb = plac.get('beta', np.nan)
            ps = plac.get('se', np.nan)
            pp = plac.get('p', np.nan)
            print(f'  {"Placebo (2016 vs 2017, no MW change)":<35} {pb:>+8.4f} {ps:>7.4f} '
                  f'p={pp:.3f}{stars(pp)}')
            if abs(pb) > 2 * ps:
                print(f'    ** PLACEBO FAIL: significant result in no-change window')

        # IVW Pool (excluding caveated events)
        pool = ivw_pool(betas, ses, labels)
        if pool:
            print(f'\n  {"POOLED IVW (" + "/".join(labels) + ")":<35} '
                  f'{pool["beta"]:>+8.4f} {pool["se"]:>7.4f} '
                  f'[{pool["ci_low"]:>+7.4f},{pool["ci_high"]:>+7.4f}] '
                  f'{pool["p"]:>7.3f}{stars(pool["p"])}  '
                  f'Q={pool["Q"]:.2f} I2={pool["I2"]:.1f}%')
            pooled_results[outcome] = pool
        else:
            print(f'  POOLED: insufficient events (n={len(betas)})')

    # Event study coefficients (from canonical results)
    print(f'\n\n{"="*70}')
    print('EVENT STUDY COEFFICIENTS (annual)')
    print(f'{"="*70}')
    for eid in EVENTS_ORDER:
        ev = events.get(eid, {})
        for outcome in ['employed', 'formal', 'log_wage']:
            es = ev.get(f'{outcome}_event_study', [])
            if es:
                print(f'\n  {eid} — {OUTCOME_LABELS[outcome]}:')
                for pt in es:
                    yr = pt.get('year_offset', '?')
                    b = pt.get('beta', np.nan)
                    s = pt.get('se', np.nan)
                    p = pt.get('p', np.nan)
                    print(f'    Year+{yr}: beta={b:+.4f} SE={s:.4f} p={p:.3f}{stars(p)}')

    # Final summary table
    print(f'\n\n{"="*70}')
    print('ROBUSTNESS SUMMARY (Annual ENAHO, Strategy 4)')
    print(f'{"="*70}')
    print(f'{"Outcome":<20} {"Pooled beta":>12} {"SE":>8} {"p":>8} {"I2":>6}')
    print('-' * 60)
    for outcome in OUTCOMES:
        pool = pooled_results.get(outcome)
        if pool:
            print(f'{OUTCOME_LABELS[outcome]:<20} {pool["beta"]:>+12.4f} '
                  f'{pool["se"]:>8.4f} {pool["p"]:>8.3f}{stars(pool["p"])} '
                  f'{pool["I2"]:>5.1f}%')

    print('\nConsistency check:')
    emp_pool = pooled_results.get('employed', {})
    frm_pool = pooled_results.get('formal', {})
    lw_pool = pooled_results.get('log_wage', {})

    if emp_pool:
        emp_b = emp_pool['beta']
        if abs(emp_b) < 2 * emp_pool['se']:
            print('  Employment: CONSISTENT with S1/S2 (zero, no destruction)')
        else:
            print(f'  Employment: {emp_b:+.4f} — NOTE high heterogeneity (I2={emp_pool["I2"]:.0f}%)')

    if frm_pool and frm_pool['p'] < 0.1:
        print(f'  Formalization: {frm_pool["beta"]:+.4f}* — consistent with quarterly DiD (+0.015***)')
    elif frm_pool:
        print(f'  Formalization: {frm_pool["beta"]:+.4f} (ns) — note quarterly DiD shows +0.015***')

    if lw_pool and lw_pool['p'] < 0.1:
        print(f'  Log wages: {lw_pool["beta"]:+.4f}* — consistent with S2 (+0.100***)')

    # Save
    output = {
        'strategy': 'Strategy 4: Annual ENAHO robustness',
        'note': 'Annual data has 2+ year gaps between pre/post periods. '
                'Weaker identification than quarterly (Strategy 2). '
                'Primary results are Cengiz bunching (S1) and quarterly DiD (S2).',
        'data_source': 'ENAHO cross-section annual (Modules 500), 2015-2023',
        'specification': 'Y_idt = alpha_d + gamma_t + beta*(Post_t x Kaitz_d_pre) + controls',
        'formality_definition': 'V4 (contrato + EsSalud + AFP/ONP + planilla)',
        'caveats': {
            'C_employed': 'Event C (2022) employment may reflect COVID-19 recovery bias. '
                          'Pre=2021 (pandemic year), Post=2022-23. Excluded from employment pool.'
        },
        'events': {
            eid: {
                'label': events[eid].get('label', ''),
                'mw_old': events[eid].get('mw_old'),
                'mw_new': events[eid].get('mw_new'),
                'kaitz_range': events[eid].get('kaitz_range'),
                'n_departments': events[eid].get('n_departments'),
                'employed': events[eid].get('employed'),
                'formal': events[eid].get('formal'),
                'log_wage': events[eid].get('log_wage'),
                'employed_event_study': events[eid].get('employed_event_study'),
                'formal_event_study': events[eid].get('formal_event_study'),
                'log_wage_event_study': events[eid].get('log_wage_event_study'),
            }
            for eid in EVENTS_ORDER if eid in events
        },
        'placebo': events.get('placebo', {}),
        'pooled': pooled_results,
    }

    out_path = 'D:/Nexus/nexus/exports/data/mw_annual_robustness_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out_path}')

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_annual_robustness_results.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f'Copied to {qhaw_path}')
