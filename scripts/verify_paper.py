import json, sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load(path):
    return json.load(open(path, encoding='utf-8'))

margins       = load('D:/Nexus/nexus/exports/data/mw_complete_margins.json')
heterogeneity = load('D:/Nexus/nexus/exports/data/mw_heterogeneity.json')
iv_owe        = load('D:/Nexus/nexus/exports/data/mw_iv_owe.json')
power         = load('D:/Nexus/nexus/exports/data/mw_power_analysis.json')
panel         = load('D:/Nexus/nexus/exports/data/mw_panel_decomposition.json')
hours         = load('D:/Nexus/nexus/exports/data/mw_hours_epen_dep.json')
epe_lima      = load('D:/Nexus/nexus/exports/data/mw_epe_lima_bunching.json')
sanity        = load('D:/Nexus/nexus/exports/data/mw_sanity_checks.json')
audit         = load('D:/Nexus/nexus/exports/data/mw_data_audit_complete.json')

ok = 0; fail = 0; hardcoded = 0
rows = []

def chk(table, cell, paper_val, json_val, tol=0.002, note=''):
    global ok, fail
    if json_val is None:
        rows.append((table, cell, paper_val, 'NOT IN JSON', 'MISSING', note))
        return
    try:
        pv = float(paper_val); jv = float(json_val)
        match = abs(pv - jv) <= tol
        rows.append((table, cell, str(pv), '{:.4g}'.format(jv), 'OK' if match else 'MISMATCH', note))
        if match: ok += 1
        else: fail += 1
    except:
        rows.append((table, cell, str(paper_val), str(json_val), 'CHECK', note))

def hardcode(table, cell, val, note=''):
    global hardcoded
    rows.append((table, cell, str(val), 'NO JSON SOURCE', 'HARDCODED', note))
    hardcoded += 1

# ── Build subgroup lookup by name ──────────────────────────────────────────────
sg = {s['name']: s for s in heterogeneity['subgroups']}

def ratio(name, ev):
    s = sg.get(name, {})
    return s.get('event_' + ev, {}).get('bunching_ratio')

def miss(name, ev):
    s = sg.get(name, {})
    return s.get('event_' + ev, {}).get('missing_pp')

def exc(name, ev):
    s = sg.get(name, {})
    return s.get('event_' + ev, {}).get('excess_pp')

# ── Bunching from margins JSON (has ratio, missing_mass_pp, excess_mass_pp) ────
br = margins['bunching_revised']

def br_ratio(ev, seg):
    return br[ev][seg].get('ratio')

def br_miss(ev, seg):
    return br[ev][seg].get('missing_mass_pp')

def br_exc(ev, seg):
    return br[ev][seg].get('excess_mass_pp')

print('Subgroup names in heterogeneity JSON:')
for s in heterogeneity['subgroups']:
    print('  ' + s['name'])
print()

# ── TABLE 2: Formal dep rows (ALL subgroup in heterogeneity = formal_dep) ─────
# Note: heterogeneity['subgroups'] ALL = formal_dep sample from margins
# Cross-check: heterogeneity ALL ratio A = 0.6963, matches margins formal_dep A = 0.6963
chk('T2','A formal dep ratio',   0.696, ratio('ALL','A'), tol=0.002)
chk('T2','A formal dep missing', 6.78,  br_miss('A','formal_dep'), tol=0.01)
chk('T2','A formal dep excess',  4.72,  br_exc('A','formal_dep'), tol=0.01)
chk('T2','B formal dep ratio',   0.829, ratio('ALL','B'), tol=0.002)
chk('T2','B formal dep missing', 8.03,  br_miss('B','formal_dep'), tol=0.01)
chk('T2','B formal dep excess',  6.66,  br_exc('B','formal_dep'), tol=0.01)
chk('T2','C formal dep ratio',   0.830, ratio('ALL','C'), tol=0.002)
chk('T2','C formal dep missing', 13.02, br_miss('C','formal_dep'), tol=0.01)
chk('T2','C formal dep excess',  10.80, br_exc('C','formal_dep'), tol=0.01)

# Informal subgroup — from margins JSON
print('Informal subgroup (from margins bunching_revised):')
for ev in ['A','B','C']:
    print('  Event {}: ratio={}, miss={}, exc={}'.format(
        ev, br_ratio(ev,'informal'), br_miss(ev,'informal'), br_exc(ev,'informal')))
chk('T2','A informal ratio',  1.218, br_ratio('A','informal'), tol=0.005)
chk('T2','B informal ratio',  0.987, br_ratio('B','informal'), tol=0.005)
chk('T2','C informal ratio',  0.811, br_ratio('C','informal'), tol=0.005)

# All workers
m_all_A = br_ratio('A','all')
m_all_B = br_ratio('B','all')
m_all_C = br_ratio('C','all')
chk('T2','A all workers ratio',  0.872, m_all_A, tol=0.005)
chk('T2','B all workers ratio',  0.901, m_all_B, tol=0.005)
chk('T2','C all workers ratio',  0.824, m_all_C, tol=0.005)
print('margins all: A={}, B={}, C={}'.format(m_all_A, m_all_B, m_all_C))

# Bootstrap CIs
boot = power['bootstrap_cis']
for ev, ci_lo_paper, ci_hi_paper in [('A',0.567,0.896),('B',0.716,1.016),('C',0.716,0.960)]:
    fd = boot.get('Event_' + ev, {}).get('formal_dep', {})
    chk('T2', ev + ' CI lo', ci_lo_paper, fd.get('ci_lo'), tol=0.005)
    chk('T2', ev + ' CI hi', ci_hi_paper, fd.get('ci_hi'), tol=0.005)
    print('  Boot Event {}: JSON lo={}, hi={}'.format(ev, fd.get('ci_lo'), fd.get('ci_hi')))

# Placebo
pl = sanity['check6_placebo']
chk('T2','Placebo 1100->1200', 0.114, pl['PLACEBO_1100\u21921200']['ratio'], tol=0.002)
chk('T2','Placebo 1400->1500', 0.013, pl['PLACEBO_1400\u21921500']['ratio'], tol=0.002)

# ── TABLE 3: Wage compression DiD ─────────────────────────────────────────────
wc = margins['wage_compression']
chk('T3','A compression DiD', -0.035, wc['A']['compression_did'], tol=0.001)
chk('T3','B compression DiD', -0.050, wc['B']['compression_did'], tol=0.001)
chk('T3','C compression DiD', -0.069, wc['C']['compression_did'], tol=0.001)

hardcode('T3','A mechanical',-0.028,'Decomposition from Step 11 session work - not in JSON')
hardcode('T3','A genuine',   -0.008,'Decomposition from Step 11 session work - not in JSON')
hardcode('T3','B mechanical',-0.021,'Decomposition from Step 11 session work - not in JSON')
hardcode('T3','B genuine',   -0.029,'Decomposition from Step 11 session work - not in JSON')
hardcode('T3','C mechanical',-0.064,'Decomposition from Step 11 session work - not in JSON')
hardcode('T3','C genuine',   -0.005,'Decomposition from Step 11 session work - not in JSON')

# ── TABLE 4 Panel A (sector) ──────────────────────────────────────────────────
sector_map = {
    'sector_agri_mining':    ('Agri/Min/Util', 1.641, 0.765, 0.407),
    'sector_manufacturing':  ('Manufactura',   1.000, 0.869, 1.049),
    'sector_commerce':       ('Comercio',      0.784, 1.209, 0.723),
    'sector_transport_accom':('Transport',     1.041, 1.018, 0.585),
    'sector_finance_prof':   ('Finance',       0.768, 1.018, 0.806),
    'sector_public_admin':   ('Admin pub',     1.033, 0.498, 0.804),
    'sector_edu_health':     ('Edu/health',    0.598, 0.709, 0.975),
}
print('\nSector subgroup matching:')
for key, (label, pA, pB, pC) in sector_map.items():
    jA = ratio(key,'A')
    jB = ratio(key,'B')
    jC = ratio(key,'C')
    print('  {}: A={}, B={}, C={}'.format(key, jA, jB, jC))
    chk('T4A', label + ' A', pA, jA, tol=0.003)
    chk('T4A', label + ' B', pB, jB, tol=0.003)
    chk('T4A', label + ' C', pC, jC, tol=0.003)

# ── TABLE 4 Panel B (firm size) ───────────────────────────────────────────────
firm_map = {
    'size_micro':       ('Micro',         0.899, 0.909, 0.690),
    'size_small':       ('Pequenya',      0.919, 0.901, 0.598),
    'size_medium_plus': ('Mediana/Grande',0.686, 0.793, 0.896),
}
print('\nFirm size subgroup matching:')
for key, (label, pA, pB, pC) in firm_map.items():
    jA = ratio(key,'A')
    jB = ratio(key,'B')
    jC = ratio(key,'C')
    print('  {}: A={}, B={}, C={}'.format(key, jA, jB, jC))
    chk('T4B', label + ' A', pA, jA, tol=0.003)
    chk('T4B', label + ' B', pB, jB, tol=0.003)
    chk('T4B', label + ' C', pC, jC, tol=0.003)

# ── TABLE 4 Panel C (age/sex) ─────────────────────────────────────────────────
demo_map = {
    'age_18_24': ('18-24',   0.661, 1.010, 0.805),
    'age_25_44': ('25-44',   0.748, 0.792, 0.797),
    'age_45_64': ('45-64',   0.701, 0.927, 0.853),
    'sex_male':  ('Hombres', 0.726, 0.788, 0.778),
    'sex_female':('Mujeres', 0.697, 0.922, 0.877),
}
print('\nAge/sex subgroup matching:')
for key, (label, pA, pB, pC) in demo_map.items():
    jA = ratio(key,'A')
    jB = ratio(key,'B')
    jC = ratio(key,'C')
    print('  {}: A={}, B={}, C={}'.format(key, jA, jB, jC))
    chk('T4C', label + ' A', pA, jA, tol=0.003)
    chk('T4C', label + ' B', pB, jB, tol=0.003)
    chk('T4C', label + ' C', pC, jC, tol=0.003)

# ── TABLE 5 / Event study ─────────────────────────────────────────────────────
es = iv_owe['event_study']
chk('T5','A post beta', 0.007, es['A']['employment']['2017']['beta'], tol=0.001)
chk('T5','A post SE',   0.034, es['A']['employment']['2017']['se'],   tol=0.001)
chk('T5','B post beta', 0.120, es['B']['employment']['2019']['beta'], tol=0.001)
chk('T5','B post SE',   0.044, es['B']['employment']['2019']['se'],   tol=0.001)
chk('T5','B pretrend p',0.010, es['B']['employment']['pretrend_f_pvalue'], tol=0.001)
chk('T5','C post beta', 0.224, es['C']['employment']['2023']['beta'], tol=0.001)
chk('T5','C post SE',   0.078, es['C']['employment']['2023']['se'],   tol=0.001)
chk('T5','C pretrend p',0.034, es['C']['employment']['pretrend_f_pvalue'], tol=0.001)

# ── TABLE 6 hours ─────────────────────────────────────────────────────────────
h = hours['hours_intensive_margin']
chk('T6','A treat pre',  44.4, h['A']['treat_pre_hrs'],  tol=0.1)
chk('T6','A treat post', 44.8, h['A']['treat_post_hrs'], tol=0.1)
chk('T6','A ctrl pre',   42.1, h['A']['ctrl_pre_hrs'],   tol=0.1)
chk('T6','A ctrl post',  41.5, h['A']['ctrl_post_hrs'],  tol=0.1)
chk('T6','A DiD',         1.1, h['A']['did'],            tol=0.05)
chk('T6','B treat pre',  44.8, h['B']['treat_pre_hrs'],  tol=0.1)
chk('T6','B DiD',        -0.5, h['B']['did'],            tol=0.05)
chk('T6','C DiD',         0.4, h['C']['did'],            tol=0.05)

# ── TABLE C1 IV/OWE ───────────────────────────────────────────────────────────
ow = iv_owe['owe']
chk('TC1','A pi',    0.124, ow['A']['pi_wage'],    tol=0.001)
chk('TC1','A SE_pi', 0.073, ow['A']['pi_se'],      tol=0.001)
chk('TC1','A F',     2.9,   ow['A']['f_stat'],     tol=0.1)
chk('TC1','A beta',  0.007, ow['A']['beta_emp'],   tol=0.001)
chk('TC1','A OWE',   0.054, ow['A']['owe'],        tol=0.002)
chk('TC1','A OWE SE',0.276, ow['A']['owe_se'],     tol=0.002)
chk('TC1','B pi',   -0.065, ow['B']['pi_wage'],    tol=0.001)
chk('TC1','B F',     0.6,   ow['B']['f_stat'],     tol=0.05)
chk('TC1','B beta',  0.120, ow['B']['beta_emp'],   tol=0.001)
chk('TC1','C pi',    0.142, ow['C']['pi_wage'],    tol=0.001)
chk('TC1','C F',     2.4,   ow['C']['f_stat'],     tol=0.05)
chk('TC1','C OWE',   1.576, ow['C']['owe'],        tol=0.002)
chk('TC1','Pooled OWE', 0.114, ow['pooled']['owe'], tol=0.001)
chk('TC1','Pooled SE',  0.267, ow['pooled']['owe_se'], tol=0.002)

# ── TABLE A1 panel ────────────────────────────────────────────────────────────
pr = panel['results']
chk('TA1','N treat',    1171, pr['n_treatment'])
chk('TA1','N ctrl',     3903, pr['n_control'])
chk('TA1','T re-int %', 24.4, pr['treatment']['n_reinterviewed']/pr['treatment']['n_2021']*100, tol=0.2)
chk('TA1','C re-int %', 23.6, pr['control']['n_reinterviewed']/pr['control']['n_2021']*100,    tol=0.2)
chk('TA1','T attr %',   75.6, pr['treatment']['attrition_rate']*100, tol=0.1)
chk('TA1','C attr %',   76.4, pr['control']['attrition_rate']*100,   tol=0.1)
chk('TA1','T emp ret',  63.9, pr['treatment']['emp_ret_reint_wtd']*100, tol=0.1)
chk('TA1','C emp ret',  60.1, pr['control']['emp_ret_reint_wtd']*100,   tol=0.1)
chk('TA1','DiD emp',     3.8, pr['did']['did_emp_ret_reint_wtd']*100,  tol=0.1)
chk('TA1','T form ret', 46.8, pr['treatment']['formal_dep_ret_reint_wtd']*100, tol=0.1)
chk('TA1','C form ret', 49.9, pr['control']['formal_dep_ret_reint_wtd']*100,   tol=0.1)
chk('TA1','DiD form',   -3.1, pr['did']['did_formal_dep_ret_reint_wtd']*100,   tol=0.1)
chk('TA1','T dlogw',   0.307, pr['treatment']['dlogw_stayers_wtd'],  tol=0.001)
chk('TA1','C dlogw',   0.154, pr['control']['dlogw_stayers_wtd'],    tol=0.001)
chk('TA1','DiD dlogw', 0.153, pr['did']['did_dlogw_stayers_wtd'],    tol=0.002)
chk('TA1','T hrs',      0.7,  pr['treatment']['dhours_stayers_wtd'], tol=0.1)
chk('TA1','C hrs',     -1.3,  pr['control']['dhours_stayers_wtd'],   tol=0.1)
chk('TA1','DiD hrs',    2.0,  pr['did']['did_dhours_stayers_wtd'],   tol=0.1)

# ── EPE Lima Table E1 ─────────────────────────────────────────────────────────
ep = epe_lima['results']
chk('TE1','A ratio',  1.031, ep['Event_A']['ratio'],  tol=0.002)
chk('TE1','A CI lo',  0.696, ep['Event_A']['ci_lo'],  tol=0.002)
chk('TE1','A CI hi',  1.630, ep['Event_A']['ci_hi'],  tol=0.002)
chk('TE1','B ratio',  0.733, ep['Event_B']['ratio'],  tol=0.002)
chk('TE1','B CI lo',  0.601, ep['Event_B']['ci_lo'],  tol=0.002)
chk('TE1','B CI hi',  1.028, ep['Event_B']['ci_hi'],  tol=0.002)
chk('TE1','C ratio',  0.885, ep['Event_C']['ratio'],  tol=0.002)
chk('TE1','C CI lo',  0.696, ep['Event_C']['ci_lo'],  tol=0.002)
chk('TE1','C CI hi',  1.131, ep['Event_C']['ci_hi'],  tol=0.002)

# ── EPEN DEP Kaitz ────────────────────────────────────────────────────────────
ed = hours['epen_dep_bunching']
chk('epen_dep','pre 2022 kaitz', 0.47, ed['pre_2022']['kaitz'], tol=0.01)

# ── MDE values ────────────────────────────────────────────────────────────────
mde = power['did_mde']
chk('MDE','A pp',  0.76,  mde['A']['mde_1sd_kaitz_dept_pp'], tol=0.02)
chk('MDE','B pp',  0.986, mde['B']['mde_1sd_kaitz_dept_pp'], tol=0.01)
mow = mde['C']['mde_1sd_kaitz_dept_pp']
chk('MDE','C pp',  1.748, mow, tol=0.01)

# ── Compression t-test p-value ────────────────────────────────────────────────
ct = sanity['check4_compression_controls']['t_test']
chk('Sec8','comp t-test p', 1.96e-6, ct['p'], tol=1e-7)

# ── Audit JSON structure ──────────────────────────────────────────────────────
print('\n=== AUDIT JSON top keys ===')
print(list(audit.keys())[:15])
if 'sample_counts' in audit:
    sc = audit['sample_counts']
    if isinstance(sc, dict):
        print('sample_counts years:', list(sc.keys())[:10])
        if '2015' in sc:
            print('  2015:', sc['2015'])
    else:
        print('sample_counts type:', type(sc).__name__, '/ value preview:', str(sc)[:200])

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('FINAL VERIFICATION SUMMARY')
print('='*80)
print('OK: {}  |  MISMATCH: {}  |  HARDCODED (no JSON): {}'.format(ok, fail, hardcoded))
print()
print('{:<8} {:<25} {:>10} {:>10} {:>10}  Note'.format('Table','Cell','Paper','JSON','Status'))
print('-'*80)
for table, cell, pv, jv, status, note in rows:
    flag = ' <-- FLAG' if status in ('MISMATCH','MISSING') else ''
    print('{:<8} {:<25} {:>10} {:>10} {:>10}{}'.format(table, cell, pv, jv, status, flag))
