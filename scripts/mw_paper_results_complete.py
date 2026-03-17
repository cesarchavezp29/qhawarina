"""
Final aggregation: Compile all results into mw_paper_results_complete.json
Run after all estimation scripts complete.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json, os
from datetime import datetime

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

margins   = load('D:/Nexus/nexus/exports/data/mw_complete_margins.json')
heter     = load('D:/Nexus/nexus/exports/data/mw_heterogeneity.json')
panel     = load('D:/Nexus/nexus/exports/data/mw_panel_decomposition.json')
owe       = load('D:/Nexus/nexus/exports/data/mw_iv_owe.json')
hours_ep  = load('D:/Nexus/nexus/exports/data/mw_hours_epen_dep.json')
epen_ciu  = load('D:/Nexus/nexus/exports/data/mw_epen_ciu_annual_bunching.json')
epen_dep  = load('D:/Nexus/nexus/exports/data/mw_epen_results.json')
annual    = load('D:/Nexus/nexus/exports/data/mw_annual_robustness_results.json')
audit     = load('D:/Nexus/nexus/exports/data/mw_data_audit_complete.json')

def fmt(v, d=3):
    if v is None: return None
    if isinstance(v, (int, float)):
        return round(float(v), d)
    return v

# TABLE 2: Bunching
br = margins.get('bunching_revised', {})
table2 = {}
for ev in ['A', 'B', 'C']:
    ev_data = br.get(ev, {})
    table2[ev] = {
        sg: {
            'missing_pp': fmt(ev_data[sg].get('missing_mass_pp'), 2),
            'excess_pp':  fmt(ev_data[sg].get('excess_mass_pp'), 2),
            'ratio':      fmt(ev_data[sg].get('ratio'), 3),
            'emp_chg_pct':fmt(ev_data[sg].get('employment_change_pct'), 2),
        }
        for sg in ['all', 'formal_dep', 'informal', 'dependent']
        if sg in ev_data and isinstance(ev_data[sg], dict)
    }

# TABLE 3: Panel decomposition
table3 = panel.get('results', {'feasible': False, 'note': 'Panel 978 infeasible'})

# TABLE 4: Heterogeneity
table4 = {}
for sg in heter.get('subgroups', []):
    if not isinstance(sg, dict): continue
    label = sg.get('label', sg.get('name'))
    ratios = {}
    for ev in ['A', 'B', 'C']:
        ev_key = f'event_{ev}'
        ev_data = sg.get(ev_key, {})
        ratios[ev] = {
            'ratio': fmt(ev_data.get('bunching_ratio'), 3),
            'unreliable': ev_data.get('ratio_unreliable', False),
        }
    table4[label] = {
        'ratios': ratios,
        'avg_ratio': fmt(sg.get('avg_ratio'), 3),
        'n_pre_2015': sg.get('n_pre_2015'),
    }

# TABLE 5: OWE
table5 = owe.get('owe', {})

# FIGURE 1: Event study
figure1 = owe.get('event_study', {})

# TABLE 6: Hours
table6 = hours_ep.get('hours_intensive_margin', {})

# TABLE A2: EPEN DEP bunching
tableA2 = hours_ep.get('epen_dep_bunching', {})

# TABLE A1: EPEN CIU annual (corroboration)
tableA1 = {
    'all':   epen_ciu.get('results', {}).get('all', {}),
    'lima':  epen_ciu.get('results', {}).get('lima_hi', {}),
    'other': epen_ciu.get('results', {}).get('other_cities', {}),
    'method': 'Lee-Saez single-period (no 2022 annual available)',
}

# Informal sector bunching (lighthouse)
lighthouse = {ev: margins.get('informal_decomp', {}).get(ev, {}) for ev in ['A', 'B', 'C']}

# Wage compression
wage_comp = {ev: v for ev, v in margins.get('wage_compression', {}).items() if ev != 'series'}

output = {
    'metadata': {
        'generated': datetime.now().isoformat(),
        'description': 'Complete MW paper results — all tables and figures',
        'sources': {
            'primary': 'ENAHO Module 500 CS (2015-2023)',
            'secondary': ['ENAHO Panel 978', 'EPEN DEP annual', 'EPEN CIU annual 2023'],
            'events': {
                'A': 'S/750->850, May 2016, pre=2015, post=2017',
                'B': 'S/850->930, Apr 2018, pre=2017, post=2019',
                'C': 'S/930->1025, May 2022, pre=2021, post=2023',
            }
        },
        'variables': {
            'wage': 'p524a1 (monthly cash wage), fallback i524a1/12',
            'employment': 'ocu500==1',
            'dependent': 'cat07p500a1==2 (or p507 in {3,4,6})',
            'formality': 'ocupinf==2 (social security OR contract)',
            'hours': 'p513t (total weekly hours, primary job)',
            'weight': 'factor07i500a (cross-section) / facpanel2123 (panel)',
            'dept': 'first 2 digits of ubigeo (24 departments)',
        }
    },
    'table2_bunching_enaho_cs': table2,
    'table3_panel_decomposition': table3,
    'table4_heterogeneity': table4,
    'table5_owe_elasticity': table5,
    'table6_hours_did': table6,
    'figure1_event_study': figure1,
    'tableA1_epen_ciu_corroboration': tableA1,
    'tableA2_epen_dep_bunching': tableA2,
    'supplemental_lighthouse': lighthouse,
    'supplemental_wage_compression': wage_comp,
    'supplemental_annual_robustness': {
        ev: {
            'employed': ev_data.get('employed', {}),
            'formal':   ev_data.get('formal', {}),
            'log_wage': ev_data.get('log_wage', {}),
        }
        for ev, ev_data in annual.get('events', {}).items()
    },
}

out_path = 'D:/Nexus/nexus/exports/data/mw_paper_results_complete.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Saved: {out_path}")
file_size = os.path.getsize(out_path) / 1024
print(f"Size: {file_size:.0f} KB")

# Print summary
print("\n=== PAPER RESULTS SUMMARY ===\n")
print("TABLE 2: Bunching (ENAHO CS)")
for ev in ['A','B','C']:
    fd = table2.get(ev,{}).get('formal_dep',{})
    inf = table2.get(ev,{}).get('informal',{})
    print(f"  Event {ev}: formal ratio={fd.get('ratio')}  informal ratio={inf.get('ratio')}")

print("\nTABLE 3: Panel Decomposition")
if table3.get('feasible'):
    print(f"  Feasible. N_treat={table3.get('n_treatment')}")
else:
    print(f"  INFEASIBLE: {table3.get('n_treatment')} treatment workers ({table3.get('note','')})")

print("\nTABLE 4: Heterogeneity (sample)")
for label, data in list(table4.items())[:6]:
    r = data.get('ratios',{})
    ra = r.get('A',{}).get('ratio','—'); rb = r.get('B',{}).get('ratio','—'); rc = r.get('C',{}).get('ratio','—')
    print(f"  {label:<28} A={ra}  B={rb}  C={rc}  avg={data.get('avg_ratio')}")

print("\nTABLE 5: OWE Elasticity")
for ev in ['A','B','C','pooled']:
    r = table5.get(ev,{})
    print(f"  Event {ev}: OWE={r.get('owe')}  SE={r.get('owe_se')}  F={r.get('f_stat')}")

print("\nTABLE 6: Hours DiD")
for ev, r in table6.items():
    if r: print(f"  Event {ev}: DiD={r.get('did')} h/week")

print("\nTABLE A2: EPEN DEP Bunching (Event C)")
for label, r in tableA2.items():
    if r: print(f"  {label}: ratio={r.get('bunching_ratio')}  missing={r.get('missing_mass_pp')}pp")

EVENTS_DICT = {'A':{'pre':2015},'B':{'pre':2017},'C':{'pre':2021}}
print("\nFIGURE 1: Event Study (check log)")
for ev, ev_data in figure1.items():
    if not isinstance(ev_data, dict): continue
    for outcome, coefs in ev_data.items():
        if not isinstance(coefs, dict): continue
        pre_vals = {yr: d for yr, d in coefs.items() if str(yr).isdigit() and int(yr) < EVENTS_DICT.get(ev,{}).get('pre',9999)}
        print(f"  {ev} {outcome}: pre-coefs={list(pre_vals.keys())}")
        break
