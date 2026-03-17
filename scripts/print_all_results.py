import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

m  = json.load(open('D:/Nexus/nexus/exports/data/mw_complete_margins.json'))
h  = json.load(open('D:/Nexus/nexus/exports/data/mw_heterogeneity.json'))
ep = json.load(open('D:/Nexus/nexus/exports/data/mw_epen_results.json'))
ar = json.load(open('D:/Nexus/nexus/exports/data/mw_annual_robustness_results.json'))
ec = json.load(open('D:/Nexus/nexus/exports/data/mw_epen_ciu_annual_bunching.json'))
au = json.load(open('D:/Nexus/nexus/exports/data/mw_data_audit_complete.json'))

def fmt(v, decimals=3):
    if v is None or v == 'N/A': return '—'
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)

def sig(p):
    if not isinstance(p, (int, float)): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

SEP = '=' * 72

# ─── 1. CENGIZ BUNCHING ────────────────────────────────────────────────────────
br = m['bunching_revised']
print(SEP)
print("1. CENGIZ BUNCHING  —  ENAHO Cross-Section (Strategies 1-3)")
print(SEP)
events_meta = {
    'A': ('S/750→850', 'May 2016', 2015, 2017),
    'B': ('S/850→930', 'Apr 2018', 2017, 2019),
    'C': ('S/930→1025', 'May 2022', 2021, 2023),
}
for ev, (name, month, pre, post) in events_meta.items():
    print(f"\nEvent {ev}: {name} ({month}) | pre-yr={pre}  post-yr={post}")
    print(f"  {'Sample':<16} {'Missing(pp)':<14} {'Excess(pp)':<13} {'Ratio':<9} {'Emp%chg'}")
    print(f"  {'-'*16} {'-'*13} {'-'*12} {'-'*8} {'-'*8}")
    for sg in ['all', 'formal_dep', 'informal', 'dependent']:
        r = br[ev].get(sg, {})
        if not isinstance(r, dict): continue
        miss = r.get('missing_mass_pp')
        exc  = r.get('excess_mass_pp')
        rat  = r.get('ratio')
        emp  = r.get('employment_change_pct')
        emp_s = f"{emp:+.2f}%" if isinstance(emp, (int, float)) else '—'
        print(f"  {sg:<16} {fmt(miss)+'pp':<14} {fmt(exc)+'pp':<13} {fmt(rat):<9} {emp_s}")

print("\n  NOTE: Counterfactual = polynomial fit on clean zone (>2*mw_new).")
print("  Missing mass = net outflow from [0.85*mw_old, mw_new).")
print("  Excess mass  = positive deltas in [mw_new, mw_new + 10*BIN).")
print("  Employment change = (post_employed - pre_employed) / pre_employed.")

# ─── 2. WAGE COMPRESSION ──────────────────────────────────────────────────────
wc = m.get('wage_compression', {})
print("\n" + SEP)
print("2. WAGE COMPRESSION  —  ENAHO CS")
print(SEP)
if isinstance(wc, dict):
    for k, v in wc.items():
        if k != 'series':
            print(f"  {k}: {v}")
    ser = wc.get('series', {})
    if ser:
        print("  Year-by-year series:")
        for yr, vals in ser.items():
            if isinstance(vals, dict):
                p25 = vals.get('p25_wage', '?')
                p50 = vals.get('p50_wage', '?')
                p75 = vals.get('p75_wage', '?')
                print(f"    {yr}: p25=S/{p25}  p50=S/{p50}  p75=S/{p75}")

# ─── 3. INFORMAL DECOMPOSITION ─────────────────────────────────────────────────
inf = m.get('informal_decomp', {})
print("\n" + SEP)
print("3. INFORMAL SECTOR DECOMPOSITION  —  ENAHO CS")
print(SEP)
if isinstance(inf, dict):
    for k, v in inf.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}:")
            for k2, v2 in v.items():
                if k2 != 'series':
                    print(f"    {k2}: {v2}")

# ─── 4. HETEROGENEITY ─────────────────────────────────────────────────────────
print("\n" + SEP)
print("4. HETEROGENEITY BUNCHING  —  ENAHO CS, formal dep only")
print(SEP)
print("  Method: Cengiz revised, ocupinf==2, wage=p524a1, BIN=S/25")
print("  MIN_MISSING_PP guard = 0.003 (ratio marked unreliable if below)")
print()
print(f"  {'Subgroup':<28} {'Ev.A':>7} {'Ev.B':>7} {'Ev.C':>7} {'Avg':>7} {'N_pre':>8} {'Stable'}")
print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
for sg in h.get('subgroups', []):
    if not isinstance(sg, dict): continue
    label = sg.get('label', sg.get('name', '?'))
    n_pre = sg.get('n_pre_2015', sg.get('n_pre', '?'))
    ea = sg.get('event_A', {}); eb = sg.get('event_B', {}); ec2 = sg.get('event_C', {})
    ra = ea.get('bunching_ratio'); rb = eb.get('bunching_ratio'); rc = ec2.get('bunching_ratio')
    ua = ea.get('ratio_unreliable', False); ub = eb.get('ratio_unreliable', False); uc = ec2.get('ratio_unreliable', False)
    valid = [r for r, u in zip([ra,rb,rc],[ua,ub,uc]) if r is not None and not u]
    avg = sum(valid)/len(valid) if valid else None
    sd  = (sum((x-avg)**2 for x in valid)/len(valid))**0.5 if avg and len(valid)>1 else None
    stable = 'YES' if sd is not None and sd < 0.25 else ('NO' if sd is not None else '—')
    ra_s = (fmt(ra)+'*' if ua else fmt(ra)) if ra is not None else '—'
    rb_s = (fmt(rb)+'*' if ub else fmt(rb)) if rb is not None else '—'
    rc_s = (fmt(rc)+'*' if uc else fmt(rc)) if rc is not None else '—'
    avg_s = fmt(avg) if avg else '—'
    n_s   = f"{n_pre:,}" if isinstance(n_pre, int) else str(n_pre)
    print(f"  {label:<28} {ra_s:>7} {rb_s:>7} {rc_s:>7} {avg_s:>7} {n_s:>8} {stable}")
print("  (* = ratio_unreliable: missing mass < 0.3pp)")

# ─── 5. ANNUAL ROBUSTNESS ─────────────────────────────────────────────────────
print("\n" + SEP)
print("5. ANNUAL ROBUSTNESS  —  Strategy 4 (ENAHO Annual, dept-level DiD)")
print(SEP)
print(f"  Spec: {ar.get('specification', '?')}")
print(f"  Formality: {ar.get('formality_definition', '?')}")
print(f"  Source: {ar.get('data_source', '?')}")
print()
for ev, ev_data in ar.get('events', {}).items():
    cfg = ev_data.get('config', {})
    print(f"  Event {ev}: {cfg.get('name', ev)}  pre={cfg.get('pre_year')} post={cfg.get('post_year')}")
    print(f"    {'Outcome':<14} {'Coef':<11} {'SE':<11} {'p':<8} {'N_dept'}")
    for outcome in ['employment', 'wages', 'formality']:
        od = ev_data.get(outcome, {})
        if not isinstance(od, dict): continue
        coef   = od.get('coefficient', od.get('coef'))
        se     = od.get('se')
        pv     = od.get('p_value', od.get('p'))
        ndept  = od.get('n_dept', od.get('n_obs', '?'))
        flag   = sig(pv)
        print(f"    {outcome:<14} {fmt(coef):<11} {fmt(se):<11} {fmt(pv):<8} {ndept}  {flag}")
    print()

pl = ar.get('placebo', {})
if pl:
    print(f"  Placebo: {pl.get('note', pl)}")

po = ar.get('pooled', {})
if po:
    print("\n  Pooled IVW (Events A+B, Event C excluded from employment):")
    for outcome, vals in po.items():
        if isinstance(vals, dict):
            coef = vals.get('pooled_coef', vals.get('coef', '?'))
            i2   = vals.get('I2', '?')
            pv   = vals.get('p_value', '?')
            flag = sig(pv) if isinstance(pv, (int, float)) else ''
            print(f"    {outcome:<14} coef={fmt(coef):<10} I2={i2}  {flag}")

# ─── 6. EPEN DEP — STRATEGY 5 ─────────────────────────────────────────────────
print("\n" + SEP)
print("6. EPEN DEPARTAMENTOS  —  Strategy 5 (dept DiD, Event C)")
print(SEP)
s5 = ep.get('strategy_5', {})
vars5 = ep.get('variables', {})
if vars5:
    print("  Variables:")
    for k, v in vars5.items():
        print(f"    {k}: {v}")
    print()
for spec, spec_data in (s5.items() if isinstance(s5, dict) else []):
    if not isinstance(spec_data, dict): continue
    print(f"  Spec: {spec}")
    print(f"    {'Outcome':<14} {'Coef':<11} {'SE':<11} {'p':<8}")
    for outcome, od in spec_data.items():
        if not isinstance(od, dict): continue
        coef = od.get('coefficient', od.get('coef'))
        se   = od.get('se')
        pv   = od.get('p_value', od.get('p'))
        flag = sig(pv)
        print(f"    {outcome:<14} {fmt(coef):<11} {fmt(se):<11} {fmt(pv):<8} {flag}")
    print()

# ─── 7. EPEN CIU — STRATEGY 6 ─────────────────────────────────────────────────
print("\n" + SEP)
print("7. EPEN CIUDADES QUARTERLY  —  Strategy 6 (bunching, Event C)")
print(SEP)
s6 = ep.get('strategy_6', {})
ev_c = ep.get('event_c', {})
if ev_c:
    print(f"  Event C config: {ev_c}")
    print()
for spec, spec_data in (s6.items() if isinstance(s6, dict) else []):
    if not isinstance(spec_data, dict): continue
    ratio = spec_data.get('bunching_ratio', spec_data.get('ratio'))
    miss  = spec_data.get('missing_mass', spec_data.get('missing'))
    exc   = spec_data.get('excess_mass',  spec_data.get('excess'))
    print(f"  {spec}:")
    print(f"    ratio={fmt(ratio)}  missing={fmt(miss)}  excess={fmt(exc)}")

# ─── 8. EPEN CIU ANNUAL ─────────────────────────────────────────────────────
print("\n" + SEP)
print("8. EPEN CIUDADES ANNUAL 2023  —  Single-period Lee-Saez bunching")
print(SEP)
sm = ec.get('summary', {})
ra = ec.get('results', {}).get('all', {})
rl = ec.get('results', {}).get('lima_hi', {})
ro = ec.get('results', {}).get('other_cities', {})
qt = ec.get('results', {}).get('quarterly', {})

print(f"  Source: {sm.get('source')}")
print(f"  N formal dep: {sm.get('n_formal_dep_employed', '?'):,}" if isinstance(sm.get('n_formal_dep_employed'), int) else f"  N: {sm.get('n_formal_dep_employed')}")
print(f"  Formality rate (wt): {sm.get('formal_rate_of_employed_wt', 0):.1%}")
print(f"  Wtd pop formal dep:  {sm.get('weighted_pop_formal_dep_M', '?')}M")
print()
print(f"  {'Sample':<22} {'N_obs':<9} {'Median wage':<14} {'Kaitz':<8} {'Excess factor':<15} {'Ratio'}")
for label, r in [('All cities', ra), ('Lima upper stratum', rl), ('Other EPEN cities', ro)]:
    n    = r.get('n_obs', '?')
    med  = r.get('weighted_median_wage', '?')
    kai  = r.get('kaitz', '?')
    ef   = r.get('mw_excess_factor', '?')
    rat  = r.get('bunching_ratio', '?')
    n_s  = f"{n:,}" if isinstance(n, int) else str(n)
    med_s = f"S/{med:.0f}" if isinstance(med, float) else str(med)
    print(f"  {label:<22} {n_s:<9} {med_s:<14} {fmt(kai):<8} {fmt(ef)+'x':<15} {fmt(rat)}")

print()
print("  Quarterly (seasonality check):")
for q, qr in qt.items():
    if isinstance(qr, dict):
        print(f"    {q}: N={qr.get('n_obs','?'):,}  share@MW={qr.get('mw_bin_share',0):.4f}  below_mw={qr.get('below_mw_share',0):.4f}")
print(f"  Note: {sm.get('note','')}")

# ─── 9. DATA AUDIT SUMMARY ────────────────────────────────────────────────────
print("\n" + SEP)
print("9. DATA AUDIT SUMMARY")
print(SEP)

sc = au.get('sample_counts', {})
wd = au.get('wage_dist', {})
fm = au.get('formality', {})
hr = au.get('hours', {})

print("\n  ENAHO CS — formal dep employed, by year:")
print(f"  {'Year':<6} {'N_obs':<9} {'Med_wage':<12} {'Formal_rate_wt':<17} {'Med_hrs'}")
years_cs = sorted(set(list(sc.keys()) + list(wd.keys()) + list(fm.keys()) + list(hr.keys())))
for yr in years_cs:
    n   = sc.get(yr, '?')
    med = wd.get(yr, {}).get('median_wage', '?') if isinstance(wd.get(yr), dict) else '?'
    frt = fm.get(yr, {}).get('formal_rate_wt', '?') if isinstance(fm.get(yr), dict) else '?'
    hrs = hr.get(yr, {}).get('median_hours', '?') if isinstance(hr.get(yr), dict) else '?'
    n_s   = f"{n:,}" if isinstance(n, int) else str(n)
    med_s = f"S/{med}" if isinstance(med, (int, float)) else str(med)
    frt_s = f"{frt:.1%}" if isinstance(frt, float) else str(frt)
    print(f"  {yr:<6} {n_s:<9} {med_s:<12} {frt_s:<17} {hrs}")

epe = au.get('epe', {})
print("\n  EPE Lima:")
if isinstance(epe, dict):
    for k, v in epe.items():
        if not isinstance(v, dict):
            print(f"    {k}: {v}")

epen_dep = au.get('epen_dep', {})
print("\n  EPEN DEP (annual files):")
for entry in (epen_dep if isinstance(epen_dep, list) else []):
    if not isinstance(entry, dict): continue
    yr  = entry.get('year', '?')
    n   = entry.get('n_rows', '?')
    fw  = entry.get('formal_rate_wt', '?')
    mw  = entry.get('median_formal_wage', '?')
    n_s = f"{n:,}" if isinstance(n, int) else str(n)
    fw_s = f"{fw:.1%}" if isinstance(fw, float) else str(fw)
    mw_s = f"S/{mw}" if isinstance(mw, (int, float)) else str(mw)
    print(f"    {yr}: N={n_s}  formal_rate={fw_s}  med_formal_wage={mw_s}")

epen_ciu = au.get('epen_ciu', [])
print("\n  EPEN CIU (quarterly files + annual):")
for entry in epen_ciu:
    if not isinstance(entry, dict): continue
    src  = entry.get('folder', entry.get('source', '?'))
    n    = entry.get('n_rows', entry.get('n_total', '?'))
    fw   = entry.get('formal_rate_wt', entry.get('formal_rate_of_employed_wt', '?'))
    mww  = entry.get('wage_stats', {}).get('median', entry.get('weighted_pop_formal_dep_M', '?'))
    n_s  = f"{n:,}" if isinstance(n, int) else str(n)
    fw_s = f"{fw:.1%}" if isinstance(fw, float) else str(fw)
    print(f"    {str(src)[:40]:<40} N={n_s}  formal_rate={fw_s}")

panel = au.get('panel', {})
print("\n  ENAHO Panel 978:")
if isinstance(panel, dict):
    for k, v in panel.items():
        if not isinstance(v, (dict, list)):
            print(f"    {k}: {v}")

epe_match = au.get('epe_matching', {})
print("\n  EPE Panel matching (rotating design):")
if isinstance(epe_match, dict):
    for k, v in epe_match.items():
        if not isinstance(v, (dict, list)):
            print(f"    {k}: {v}")

cross = au.get('cross_consistency_2022', {})
print("\n  Cross-dataset consistency (2022):")
if isinstance(cross, dict):
    for dataset, vals in cross.items():
        if isinstance(vals, dict):
            med = vals.get('median_formal_wage', '?')
            frt = vals.get('formal_rate', '?')
            frt_s = f"{frt:.1%}" if isinstance(frt, float) else str(frt)
            print(f"    {dataset:<20} med_formal=S/{med}  formal_rate={frt_s}")

print("\n" + SEP)
print("Done.")
