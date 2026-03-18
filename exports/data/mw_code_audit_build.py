import json, datetime

now = datetime.datetime.utcnow().isoformat() + 'Z'

filter_n = {
    2015: {'n_final': 9952, 'wage_var': 'p524a1', 'median_wage': 1400},
    2017: {'n_final': 10789, 'wage_var': 'p524a1', 'median_wage': 1500},
    2019: {'n_final': 10745, 'wage_var': 'p524a1', 'median_wage': 1700},
    2021: {'n_final': 8855, 'wage_var': 'p524a1', 'median_wage': 1800},
    2023: {'n_final': 9650, 'wage_var': 'p524a1', 'median_wage': 1870}
}

issues = [
    {
        'id': 'ISSUE_1',
        'script': 'mw_heterogeneity_bunching.py',
        'severity': 'MEDIUM',
        'description': (
            'Header docstring (line 11) and JSON metadata key (line 571) say p507 in {3,4} '
            'but actual code (line 156) uses dep.isin([3,4,6]) matching mw_complete_margins.py. '
            'Documentation inconsistency.'
        ),
        'fix': 'Update header docstring and JSON sample string from p507 in {3,4} to p507 in {3,4,6}'
    },
    {
        'id': 'ISSUE_2',
        'script': 'mw_heterogeneity_bunching.py',
        'severity': 'MEDIUM',
        'description': (
            'Header docstring (line 12) says wage = i524a1/12 but code (lines 178-190) '
            'prefers p524a1 with i524a1/12 as fallback. Matches mw_complete_margins.py but contradicts docstring.'
        ),
        'fix': 'Update header docstring from "monthly wage (i524a1/12)" to "monthly wage (p524a1 preferred, fallback i524a1/12)"'
    },
    {
        'id': 'ISSUE_3',
        'script': 'mw_heterogeneity_bunching.py, mw_iv_owe.py, mw_hours_epen_dep_bunching.py',
        'severity': 'LOW',
        'description': (
            'Scripts list factor07i500a as primary weight (before fac500a) but it does not exist '
            'in ENAHO data (Audit 1B: only fac500a found in 2015 DTA). '
            'Fallback to fac500a works correctly. Weight inconsistently labeled.'
        ),
        'fix': 'No functional fix needed; document that actual weight used is fac500a throughout all years.'
    },
    {
        'id': 'ISSUE_4',
        'script': 'mw_iv_owe.py',
        'severity': 'HIGH',
        'description': (
            'dept_kaitz() docstring says "Kaitz = MW_old / median_formal_wage by dept" '
            'but implementation uses np.average (weighted mean) at line 154: '
            'np.average(g["wage"], weights=g["wt"]). '
            'The instrument is MW/weighted-mean, NOT MW/median. Paper description must reflect this or code must be fixed.'
        ),
        'fix': 'Replace np.average with a weighted median in dept_kaitz(), OR update all paper references from "median" to "weighted mean". This affects instrument validity claims.'
    },
    {
        'id': 'ISSUE_5',
        'script': 'mw_hours_epen_dep_bunching.py',
        'severity': 'MEDIUM',
        'description': (
            'Table 6 script uses wage < 15000 (line 55) while mw_complete_margins.py '
            'and mw_heterogeneity_bunching.py use WAGE_MAX=6000. '
            'Table 6 sample includes workers earning up to S/15000; Tables 2-4 cap at S/6000. '
            'Samples not directly comparable.'
        ),
        'fix': 'Set WAGE_MAX=6000 in mw_hours_epen_dep_bunching.py to match other scripts, or document the intentional difference.'
    },
    {
        'id': 'ISSUE_6',
        'script': 'mw_complete_margins.py (Table 1 generation)',
        'severity': 'HIGH',
        'description': (
            'Table 1 N values (2015=10134, 2017=11026, 2019=10991, 2021=9123, 2023=9996) '
            'do NOT match bunching filter chain Ns (2015=9952, 2017=10789, 2019=10745, 2021=8855, 2023=9650). '
            'Discrepancies: 2015=+182, 2017=+237, 2019=+246, 2021=+268, 2023=+346. '
            'Bunching script caps at S/6000; Table 1 likely uses no upper cap or different cap.'
        ),
        'fix': (
            'Run filter chain without S/6000 upper bound to reproduce Table 1 Ns. '
            'Add footnote distinguishing full formal_dep sample (Table 1) from bunching sample (wage < S/6000, Tables 2-4).'
        )
    },
    {
        'id': 'ISSUE_7',
        'script': 'mw_complete_margins.py',
        'severity': 'MEDIUM',
        'description': (
            'Main bunching counterfactual (run_bunching_event, line 298) uses weighted np.average '
            'for background shift calculation. '
            'Informal sector decomposition (Part 5, line 1196) uses unweighted delta[clean].mean(). '
            'Inconsistent counterfactual method within same script.'
        ),
        'fix': 'Replace bg = delta[clean].mean() in Part 5 (line 1196) with weighted np.average using reciprocal-magnitude weights, matching the main bunching function.'
    },
    {
        'id': 'ISSUE_8',
        'script': 'mw_iv_owe.py',
        'severity': 'LOW',
        'description': (
            'EVENT_STUDY_YEARS comment for Event A says "omit 2015 -> base; or 2016 -> base", '
            'indicating ambiguity. Code uses base_year = pre_yr = 2015 for Event A. '
            'The alternative 2016 base is never used but creates reader confusion.'
        ),
        'fix': 'Remove "or 2016 -> base" from EVENT_STUDY_YEARS comment for Event A since code always uses pre_yr (2015) as base.'
    }
]

report = {
    'generated_at': now,
    'checks': {
        'script1_margins': {
            'dependent_set': 'p507 in {3.0, 4.0, 6.0}',
            'wage_var': 'p524a1 (preferred), fallback i524a1/12',
            'weight_var': 'fac500a (factor07i500a not present in data)',
            'wage_max': 6000,
            'bin_width': 25,
            'counterfactual_method': 'weighted_mean_background: np.average(delta[bc > 2*mw_new], weights=1/(|delta|+eps))',
            'compression_zone': '[MW_new, 1.5*MW_new)',
            'high_wage_zone': '[2.0*MW_new, 3.0*MW_new)',
            'placebo_thresholds': ['1100->1200', '1400->1500'],
            'placebo_source': 'mw_sanity_checks.json check6_placebo',
            'fac500a_is_factor07i500a': False,
            'filter_chain_n': {str(yr): r['n_final'] for yr, r in filter_n.items()},
            'filter_chain_detail': {
                str(yr): {
                    'n_final': r['n_final'],
                    'wage_var': r['wage_var'],
                    'median_wage': r['median_wage']
                } for yr, r in filter_n.items()
            }
        },
        'script2_heterogeneity': {
            'dependent_set_header_WRONG': 'p507 in {3,4} (docstring says this — WRONG)',
            'dependent_set_code_CORRECT': 'p507 in {3,4,6} (actual code — matches mw_complete_margins)',
            'wage_var_header_WRONG': 'i524a1/12 only (docstring says this — WRONG)',
            'wage_var_code_CORRECT': 'p524a1 preferred, fallback i524a1/12 (actual code)',
            'weight_var_header': 'fac500a (stated in header)',
            'weight_var_code': 'factor07i500a first (absent), then fac500a (actual used)',
            'wage_max': 6000,
            'bin_width': 25,
            'total_row_matches_table2': True,
            'total_ratios': {'A': 0.7136, 'B': 0.847, 'C': 0.839},
            'table2_ratios': {'A': 0.696, 'B': 0.829, 'C': 0.830},
            'total_ratio_diffs': {'A': 0.0176, 'B': 0.018, 'C': 0.009},
            'sector_coverage': 'CIIU 01-99 via ciiu_sector(); 8 named sectors + residual other. Coverage exhaustive for formal workers.',
            'firm_size_variable': 'p512b (exact count, primary), p512a (categories, fallback)'
        },
        'script3_iv_owe': {
            'rf_sample': 'all_working_age (ages 14-65, all workers)',
            'fs_sample': 'formal_dep employed (wage>0, wage<15000)',
            'kaitz_weighted': True,
            'kaitz_actual': 'MW_old / weighted_mean_formal_wage_by_dept using np.average',
            'kaitz_docstring_WRONG': 'Says median_formal_wage but code uses weighted mean',
            'clustering': 'cluster-robust SE by dept (cov_type=cluster, groups=dept); 25 departments',
            'omitted_years': {'A': 2015, 'B': 2017, 'C': 2021},
            'n_wage_obs': {'A': 21032, 'B': 21922, 'C': 18939},
            'n_emp_obs': {'A': 158108, 'B': 159735, 'C': 148990}
        },
        'script4_panel': {
            'treat_lo': 790.5,
            'ctrl_lo': 1230.0,
            'ctrl_hi': 2562.5,
            'formula_treat_lo': '0.85 * MW_OLD (930)',
            'formula_ctrl_lo': '1.2 * MW_NEW (1025)',
            'formula_ctrl_hi': '2.5 * MW_NEW (1025)',
            'attrition_correct': True,
            'attrition_definition': 'matched_23 = facpanel2123 > 0 AND notna(). Not-observed = panel attrition, NOT unemployment. Code correctly avoids misclassifying attrition as job loss.'
        },
        'script5_hours': {
            'hours_var': 'p513t (usual weekly hours, ENAHO)',
            'pre_treatment_band': '[0.85*mw_old, mw_new)',
            'pre_control_band': '[1.5*mw_new, 3.0*mw_new]',
            'post_treatment_band': '[mw_new, 1.3*mw_new] (bunched-up zone proxy)',
            'post_control_band': '[1.5*mw_new, 3.0*mw_new]',
            'wage_max_used': 15000,
            'wage_max_inconsistency': 'INCONSISTENT with other scripts (6000). See ISSUE_5.'
        }
    },
    'cross_script_consistency': {
        'mw_values_consistent': True,
        'mw_values_note': 'All 5 scripts use: A(750->850), B(850->930), C(930->1025)',
        'year_mapping_consistent': True,
        'year_mapping_note': 'All scripts use pre/post: A(2015/2017), B(2017/2019), C(2021/2023)',
        'formal_dep_consistent': True,
        'formal_dep_note': 'All scripts: ocu500==1 AND p507 in {3,4,6} AND ocupinf==2',
        'wage_var_consistent': True,
        'wage_var_note': 'All scripts prefer p524a1, fallback i524a1/12. mw_heterogeneity_bunching docstring incorrectly says i524a1/12 only.',
        'weight_var_consistent': True,
        'weight_var_note': 'mw_complete_margins uses fac500a exclusively. Other scripts try factor07i500a first (absent in data), fall back to fac500a. All end up using fac500a.',
        'wage_max_inconsistent': True,
        'wage_max_note': 'mw_hours_epen_dep_bunching.py uses 15000; all other scripts use 6000.',
        'n_2015_across_scripts': {
            'margins_bunching': 9952,
            'heterogeneity_bunching': 9952,
            'hours_epen': 'not stored in JSON output',
            'iv_owe_wage_sample': 21032,
            'iv_owe_emp_sample': 158108
        }
    },
    'table1_vs_bunching_n': {
        '2015': {'table1': 10134, 'bunching': 9952, 'diff': 182, 'match': False, 'note': 'Bunching caps at S/6000; Table 1 likely uses no upper cap'},
        '2017': {'table1': 11026, 'bunching': 10789, 'diff': 237, 'match': False, 'note': 'Bunching caps at S/6000'},
        '2019': {'table1': 10991, 'bunching': 10745, 'diff': 246, 'match': False, 'note': 'Bunching caps at S/6000'},
        '2021': {'table1': 9123,  'bunching': 8855,  'diff': 268, 'match': False, 'note': 'Bunching caps at S/6000'},
        '2023': {'table1': 9996,  'bunching': 9650,  'diff': 346, 'match': False, 'note': 'Bunching caps at S/6000'}
    },
    'issues_found': issues,
    'verdict': 'FAIL',
    'verdict_note': 'Two HIGH severity issues: (1) Kaitz instrument uses weighted mean not median, contradicting docstring/paper; (2) Table 1 N values do not match bunching sample Ns across all 5 years. Plus 3 MEDIUM and 3 LOW issues.'
}

with open('D:/Nexus/nexus/exports/data/mw_code_audit.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print('JSON written successfully')
print(f'Issues: {len(issues)}')
print(f'Verdict: FAIL')
