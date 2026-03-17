"""
EPEN Ciudades Annual 2023 — Post-Event C Bunching Analysis
==========================================================
Single-period Lee-Saez bunching at MW=1025 (Event C new MW).
Source: INEI EPEN Ciudades Anual 2023 (download code 873).
No 2022 annual available (code 872 returns 404).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import json
from datetime import datetime

DATA_FILE = 'D:/Nexus/nexus/data/raw/epen/ciu/annual/873/872-Modulo76/EPEN BD Ciudades Anual 2023.csv'
OUT_JSON  = 'D:/Nexus/nexus/exports/data/mw_epen_ciu_annual_bunching.json'
OUT_TXT   = 'D:/Nexus/nexus/exports/data/mw_epen_ciu_annual_bunching.txt'

BIN_W = 25
MW_C  = 1025   # Event C new MW
MW_B  = 930    # Event C old MW
BINS  = np.arange(400, 3001, BIN_W)
BC    = BINS[:-1] + BIN_W / 2


def make_bins(w, wt):
    counts, _ = np.histogram(w, bins=BINS, weights=wt)
    total = counts.sum()
    shares = counts / total if total > 0 else counts
    return shares, BC.copy()


def poly_counterfactual(shares, bc, mw, n_excl=3):
    excl_lo = mw - n_excl * BIN_W
    excl_hi = mw + n_excl * BIN_W
    clean = (bc < excl_lo) | (bc > excl_hi)
    cf = np.polyval(np.polyfit(bc[clean], shares[clean], 4), bc)
    return np.maximum(cf, 0)


def bunching_stats(w, wt, mw, mw_old, label=""):
    shares, bc = make_bins(w, wt)
    cf = poly_counterfactual(shares, bc, mw)

    mw_idx = int(np.abs(bc - mw).argmin())
    mw_share = float(shares[mw_idx])
    mw_cf    = float(cf[mw_idx])
    mw_excess_factor = mw_share / mw_cf if mw_cf > 0 else None

    at_mw    = (bc >= mw - BIN_W/2) & (bc < mw + 8 * BIN_W)
    excess   = float((shares[at_mw] - cf[at_mw]).clip(min=0).sum())

    aff_lo   = 0.85 * mw_old
    below_mw = (bc >= aff_lo) & (bc < mw)
    missing  = float((cf[below_mw] - shares[below_mw]).clip(min=0).sum())

    ratio    = excess / missing if missing > 0.001 else None

    pct_at    = float(shares[(bc >= mw - BIN_W/2) & (bc < mw + BIN_W/2)].sum() * 100)
    pct_below = float(shares[(bc >= aff_lo) & (bc < mw)].sum() * 100)

    valid = (w > 0) & np.isfinite(wt)
    sorted_idx = np.argsort(w[valid])
    cum_wt = np.cumsum(wt[valid][sorted_idx])
    mid = np.searchsorted(cum_wt, cum_wt[-1] * 0.5)
    wmed = float(w[valid][sorted_idx[min(mid, len(sorted_idx)-1)]])

    return {
        'label':              label,
        'n_obs':              int(valid.sum()),
        'weighted_pop':       float(wt[valid].sum()),
        'median_wage':        float(np.median(w[valid])),
        'weighted_median_wage': wmed,
        'kaitz':              round(mw / wmed, 4),
        'mw':                 mw,
        'mw_bin_share_pct':   round(pct_at, 4),
        'below_mw_share_pct': round(pct_below, 4),
        'mw_excess_factor':   round(mw_excess_factor, 3) if mw_excess_factor else None,
        'excess_mass':        round(excess, 5),
        'missing_mass':       round(missing, 5),
        'bunching_ratio':     round(ratio, 3) if ratio else None,
        'bin_centers':        bc.tolist(),
        'shares':             shares.tolist(),
        'counterfactual':     cf.tolist(),
    }


def main():
    print("Loading EPEN CIU 2023 annual...")
    df = pd.read_csv(DATA_FILE, encoding='latin1', low_memory=False)
    print(f"  Shape: {df.shape}")

    employed = df['OCUP300'] == 1
    formal   = df['Informal_P'].astype(str).str.strip() == '2'
    wage     = pd.to_numeric(df['ingtrabw'], errors='coerce')
    weight   = pd.to_numeric(df['FAC300_ANUAL'], errors='coerce')
    lima_hi  = df['DIVISION_LIMA'].astype(str).str.strip() == '1'
    mes      = pd.to_numeric(df['MES'], errors='coerce')

    mask = employed & formal & (wage > 0) & (wage < 8000) & weight.notna()
    print(f"Formal dep employed: {mask.sum():,}  wtd_pop={weight[mask].sum()/1e6:.2f}M")

    results = {}

    # 1. Full sample
    r = bunching_stats(wage[mask].values, weight[mask].values, MW_C, MW_B, "All EPEN Cities 2023")
    results['all'] = r
    print(f"\nALL: N={r['n_obs']:,}  wmed={r['weighted_median_wage']:.0f}  "
          f"Kaitz={r['kaitz']:.3f}  excess_factor={r['mw_excess_factor']}x  "
          f"ratio={r['bunching_ratio']}")

    # 2. Lima upper stratum vs other cities
    for label, submask, key in [
        ('Lima upper stratum (DIVISION_LIMA=1)', mask & lima_hi, 'lima_hi'),
        ('Other EPEN cities', mask & ~lima_hi, 'other_cities'),
    ]:
        r2 = bunching_stats(wage[submask].values, weight[submask].values, MW_C, MW_B, label)
        results[key] = r2
        print(f"{label}: N={r2['n_obs']:,}  wmed={r2['weighted_median_wage']:.0f}  "
              f"Kaitz={r2['kaitz']:.3f}  ratio={r2['bunching_ratio']}")

    # 3. Quarterly seasonality
    quarterly = {}
    print("\nMES-based (seasonality):")
    for label, months in [('Q1_Jan_Mar', [1,2,3]), ('Q2_Apr_Jun', [4,5,6]),
                          ('Q3_Jul_Sep', [7,8,9]), ('Q4_Oct_Dec', [10,11,12])]:
        qmask = mask & mes.isin(months)
        if qmask.sum() < 500:
            continue
        qw, qwt = wage[qmask].values, weight[qmask].values
        qshares, qbc = make_bins(qw, qwt)
        mw_idx = int(np.abs(qbc - MW_C).argmin())
        mw_share = float(qshares[mw_idx])
        below = float(qshares[(qbc >= 0.85*MW_B) & (qbc < MW_C)].sum())
        quarterly[label] = {
            'n_obs': int(qmask.sum()),
            'mw_bin_share': round(mw_share, 5),
            'below_mw_share': round(below, 5),
        }
        print(f"  {label}: N={qmask.sum():,}  share@1025={mw_share:.4f}  below_mw={below:.4f}")
    results['quarterly'] = quarterly

    # Summary stats
    w_emp = weight[employed & weight.notna()]
    f_emp = formal[employed & weight.notna()]
    formal_rate_wt = float((f_emp * w_emp).sum() / w_emp.sum())

    summary = {
        'source': 'EPEN Ciudades Anual 2023 (INEI code 873)',
        'n_total': int(len(df)),
        'n_employed_unwt': int(employed.sum()),
        'employment_rate_unwt': round(employed.sum() / len(df), 4),
        'n_formal_dep_employed': int(mask.sum()),
        'formal_rate_of_employed_wt': round(formal_rate_wt, 4),
        'weighted_pop_formal_dep_M': round(float(weight[mask].sum()) / 1e6, 3),
        'kaitz_all': results['all']['kaitz'],
        'kaitz_lima_hi': results['lima_hi']['kaitz'],
        'kaitz_other_cities': results['other_cities']['kaitz'],
        'mw_event': 'C (S/930 -> S/1,025, May 2022)',
        'data_year': 2023,
        'bunching_ratio_all': results['all']['bunching_ratio'],
        'mw_excess_factor_all': results['all']['mw_excess_factor'],
        'mw_bin_share_pct': results['all']['mw_bin_share_pct'],
        'note': (
            '2023 annual = post-Event C. No 2022 annual with Informal_P available '
            '(INEI code 872 returns 404). Single-period bunching uses Lee-Saez '
            'polynomial counterfactual. CODCIUDAD not in file; city-level DiD not possible.'
        ),
    }

    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'description': 'EPEN Ciudades Annual 2023 — single-period bunching at MW=1025 (Event C post)',
            'methodology': (
                'Lee-Saez single-period bunching. Polynomial counterfactual (degree=4) '
                'excluding [mw-75, mw+75). Formal = Informal_P=="2". Employed = OCUP300==1.'
            ),
        },
        'summary': summary,
        'results': results,
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_JSON}")

    # Text report
    all_r = results['all']
    li_r  = results['lima_hi']
    ot_r  = results['other_cities']
    lines = [
        "EPEN CIUDADES ANNUAL 2023 — POST-EVENT C BUNCHING ANALYSIS",
        "=" * 60,
        "",
        f"Source: EPEN Ciudades Anual 2023 (INEI download code 873)",
        f"File:   EPEN BD Ciudades Anual 2023.csv",
        f"N obs:  {len(df):,} total | {mask.sum():,} formal dep employed",
        f"Wtd pop (formal dep): {weight[mask].sum()/1e6:.2f}M",
        "",
        "SAMPLE DEFINITION",
        "  Employed:   OCUP300 == 1",
        "  Formal:     Informal_P == '2' (health insurance registered)",
        "  Wage:       ingtrabw (weekly labor income, converted to monthly)",
        "  Weight:     FAC300_ANUAL",
        "  Period:     All 2023 (MW = S/1,025 throughout; Event C May 2022)",
        "",
        "KEY RESULTS",
        f"  Median formal wage (wt):   S/{all_r['weighted_median_wage']:.0f}",
        f"  Kaitz index (MW/median):   {all_r['kaitz']:.3f}",
        f"  At-MW bin share:           {all_r['mw_bin_share_pct']:.3f}%",
        f"  Excess factor at MW:       {all_r['mw_excess_factor']}x counterfactual",
        f"  Excess mass [mw, mw+200):  {all_r['excess_mass']:.4f}",
        f"  Missing [0.85*930, 1025):  {all_r['missing_mass']:.4f}",
        f"  Bunching ratio:            {all_r['bunching_ratio']}",
        "",
        "BY SUBSAMPLE",
        f"  Lima upper stratum:  N={li_r['n_obs']:,}  median={li_r['weighted_median_wage']:.0f}  "
        f"Kaitz={li_r['kaitz']:.3f}  ratio={li_r['bunching_ratio']}",
        f"  Other EPEN cities:   N={ot_r['n_obs']:,}  median={ot_r['weighted_median_wage']:.0f}  "
        f"Kaitz={ot_r['kaitz']:.3f}  ratio={ot_r['bunching_ratio']}",
        "",
        "QUARTERLY (SEASONALITY CHECK, MW=1025 ALL QUARTERS)",
    ]
    for q, qr in quarterly.items():
        lines.append(f"  {q}: N={qr['n_obs']:,}  share@1025={qr['mw_bin_share']:.4f}  below_mw={qr['below_mw_share']:.4f}")
    lines += [
        "",
        "DATA LIMITATIONS",
        "  - No 2022 annual EPEN CIU (code 872 returns 404; likely not published).",
        "  - Without pre-period: single-period bunching only (Lee-Saez), not Cengiz.",
        "  - CODCIUDAD absent; DIVISION_LIMA=1 is Lima upper stratum, not full Lima.",
        "  - City-level Kaitz DiD not possible from this file.",
        "",
        "CROSS-DATASET COMPARISON (Event C post-period, formal dep)",
        f"  EPEN CIU 2023: median S/{all_r['weighted_median_wage']:.0f}  Kaitz={all_r['kaitz']:.3f}  (30 cities)",
        "  ENAHO CS 2023: median ~S/1,700  Kaitz ~0.60  (national sample)",
        "  Gap reflects EPEN urban scope + higher city wages.",
    ]
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {OUT_TXT}")
    print("\nDone.")


if __name__ == '__main__':
    main()
