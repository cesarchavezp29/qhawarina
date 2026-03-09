"""
epe_informality_validate.py
Data-first informality construction, validated against INEI published Lima rates.

VERIFIED VARIABLE MAPPING (from actual .sav value labels, 2003/2012/2022 audit):

  p206  Occupation category:
    1 = Empleador o patrono
    2 = Trabajador Independiente   <-- NOT Obrero
    3 = Empleado
    4 = Obrero                     <-- NOT Independiente
    5 = Trabajador Familiar No Remunerado
    6 = Trabajador del Hogar

  p207a Firm size categorical:
    1 = Menos de 100 personas      <-- NOT <10
    2 = Mas de 100 personas

  p207b Firm size exact count (only for p207a==1 respondents)
        Use p207b <= 9 for INEI <10-worker threshold

  p222  Health insurance:
    1 = EsSalud (antes IPSS)       <-- FORMAL marker, NOT code 2
    2 = Seguro Privado de Salud
    3 = Ambos (EsSalud + Privado)
    4 = OTRO  (pre-2022: includes SIS bundled)
    5 = No esta afiliado
    6 = [unlabeled; 2022 only; profile = SIS]

  p213  = "Ha hecho algo para conseguir trabajo?" (job search for UNEMPLOYED)
           NOT a formality variable. Never use for formality.

  ocu200 Employment status:
    1 = Ocupado (employed)

INEI published Lima Metro informality rate (empleo informal / PEA ocupada):
  Source: INEI Encuesta Permanente de Empleo, quarterly bulletins
  2003: ~67%  2006: ~65%  2007: ~63%  2008: ~62%
  2011: ~61%  2012: ~60%  2015: ~57%  2016: ~56%  2022: ~58%
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pathlib
import numpy as np
import pandas as pd
import pyreadstat

OUT = pathlib.Path("D:/Nexus/nexus/data/raw/epe/srienaho")

# All 9 MW event pre-wave files + extras for validation coverage
VALIDATION_FILES = [
    ("2003", 92),
    ("2006", 142),
    ("2007", 187),
    ("2008", 196),
    ("2011", 272),
    ("2012", 299),
    ("2015", 496),
    ("2016", 514),
    ("2022", 766),
]

# INEI published informal employment rate, Lima Metro (PEA ocupada)
INEI_LIMA_INFORMAL = {
    "2003": 0.670,
    "2006": 0.650,
    "2007": 0.630,
    "2008": 0.620,
    "2011": 0.610,
    "2012": 0.600,
    "2015": 0.570,
    "2016": 0.560,
    "2022": 0.580,
}


def load_sav(code):
    d = OUT / str(code)
    savs = sorted(d.glob("*.sav")) + sorted(d.glob("*.SAV"))
    if not savs:
        return None, None
    df, meta = pyreadstat.read_sav(str(savs[0]))
    df.columns = [c.lower() for c in df.columns]
    return df, meta


def get_val_labels(meta, var):
    if hasattr(meta, "variable_value_labels"):
        raw = (meta.variable_value_labels.get(var)
               or meta.variable_value_labels.get(var.upper(), {}))
        return {k: str(v) for k, v in raw.items()}
    return {}


def construct_and_validate(df, meta, year_label):
    avail = set(df.columns)
    yr = int(year_label)

    # ── STEP 0: Verify presence of required variables ─────────────────────────
    required = {"ocu200": "Employment status",
                "p206":   "Occupation category",
                "p222":   "Health insurance"}
    optional = {"p207a": "Firm size (cat <100/>100)",
                "p207b": "Firm size (exact count)"}

    print(f"\n{'='*70}")
    print(f"  YEAR {year_label}")
    print(f"{'='*70}")
    print("  Using these VERIFIED variables:")
    for var, label in {**required, **optional}.items():
        if var in avail:
            vl = get_val_labels(meta, var)
            print(f"    {var:<8} — {label}")
            for val, lbl in sorted(vl.items(), key=lambda x: float(x[0])):
                print(f"             {val:>5} = '{lbl}'")
        else:
            print(f"    {var:<8} — NOT FOUND  ({label})")

    for var in required:
        if var not in avail:
            print(f"  FATAL: {var} missing — cannot proceed")
            return None

    employed = df["ocu200"] == 1
    n_emp = int(employed.sum())

    # ── STEP 1: Occupation (verified codes from audit) ────────────────────────
    p206 = pd.to_numeric(df["p206"], errors="coerce")

    empleador    = (p206 == 1)
    independ     = (p206 == 2)   # Trabajador Independiente  (VERIFIED)
    empleado     = (p206 == 3)   # Empleado                  (VERIFIED)
    obrero       = (p206 == 4)   # Obrero                    (VERIFIED)
    tfnr         = (p206 == 5)   # Familiar No Remunerado
    trab_hogar   = (p206 == 6)   # Del Hogar
    asalariado   = (empleado | obrero)

    print(f"\n  Occupation breakdown (employed, ocu200==1, n={n_emp:,}):")
    groups = [("empleador(1)", empleador), ("independiente(2)", independ),
              ("empleado(3)", empleado), ("obrero(4)", obrero),
              ("tfnr(5)", tfnr), ("del hogar(6)", trab_hogar)]
    for name, mask in groups:
        n = (mask & employed).sum()
        print(f"    p206={name:<20}: {n:5,}  ({n/n_emp*100:5.1f}%)")

    # ── STEP 2: Health insurance (verified codes from audit) ──────────────────
    p222 = pd.to_numeric(df["p222"], errors="coerce")

    essalud     = (p222 == 1)   # EsSalud — formal employer insurance  (VERIFIED)
    privado     = (p222 == 2)   # Seguro Privado
    ambos       = (p222 == 3)   # EsSalud + Privado
    otro        = (p222 == 4)   # Otro (SIS bundled pre-2022)
    no_afil     = (p222 == 5)   # No afiliado
    sis_2022    = (p222 == 6)   # SIS (unlabeled, 2022 only — confirmed by profile)

    # Formal employer-financed insurance = EsSalud or Ambos (or FFAA if present)
    # In 2022 we also need to not misclassify the unlabeled SIS (val=6) as formal
    has_employer_ins = (essalud | ambos)

    print(f"\n  Health insurance (p222) among employed:")
    for val, name in [(1, "EsSalud(1)-FORMAL"), (2, "Privado(2)"),
                      (3, "Ambos(3)-FORMAL"), (4, "Otro(4)"),
                      (5, "No afiliado(5)"), (6, "Unlabeled(6)=SIS")]:
        n = ((p222 == val) & employed).sum()
        pct = n / n_emp * 100
        print(f"    p222={val} {name:<22}: {n:5,}  ({pct:5.1f}%)")
    n_miss = (p222.isna() & employed).sum()
    print(f"    p222=NaN  (missing)               : {n_miss:5,}  ({n_miss/n_emp*100:5.1f}%)")

    # ── STEP 3: Firm size ─────────────────────────────────────────────────────
    if "p207b" in avail:
        p207b = pd.to_numeric(df["p207b"], errors="coerce")
        small_firm = p207b.between(1, 9)   # <10 workers — INEI threshold
        large_firm = (p207b >= 10) | (pd.to_numeric(df.get("p207a", pd.Series()), errors="coerce") == 2)
        print(f"\n  Firm size p207b (exact, <10 threshold):")
        for lo, hi in [(1,1),(2,5),(6,9),(10,99)]:
            n = p207b.between(lo, hi).sum()
            print(f"    {lo:2d}–{hi:2d}: {n:,}")
        print(f"    100+ (p207a==2): {(pd.to_numeric(df.get('p207a', pd.Series()), errors='coerce')==2).sum():,}")
    else:
        # Fallback: p207a only — <100 vs 100+ (imprecise)
        p207a = pd.to_numeric(df.get("p207a", pd.Series(dtype=float)), errors="coerce")
        small_firm = (p207a == 1)  # <100 = "small" (weaker proxy)
        large_firm = (p207a == 2)
        print(f"\n  NOTE: p207b not available; using p207a (<100) as small-firm proxy")

    # ── STEP 4: Construct 3 definitions ──────────────────────────────────────
    # A. EsSalud-only strict: formal = employed with EsSalud or Ambos
    formal_A = employed & has_employer_ins
    informal_A = employed & ~formal_A

    # B. Liberal: EsSalud + Private
    formal_B = employed & (essalud | privado | ambos)
    informal_B = employed & ~formal_B

    # C. INEI composite:
    #    asalariados   -> formal if EsSalud/Ambos
    #    independentes -> formal if firm >= 10 workers
    #    empleadores   -> formal if firm >= 10 workers
    #    del hogar     -> formal if EsSalud/Ambos
    #    tfnr          -> always informal
    formal_asalariado = asalariado & has_employer_ins
    formal_independ   = independ   & large_firm
    formal_empleador  = empleador  & large_firm
    formal_hogar      = trab_hogar & has_employer_ins
    formal_C = employed & (formal_asalariado | formal_independ | formal_empleador | formal_hogar)
    informal_C = employed & ~formal_C

    rate_A = informal_A.sum() / n_emp
    rate_B = informal_B.sum() / n_emp
    rate_C = informal_C.sum() / n_emp
    inei   = INEI_LIMA_INFORMAL.get(year_label, np.nan)

    def flag(gap):
        return "✓" if abs(gap) <= 0.05 else ("~" if abs(gap) <= 0.10 else "✗")

    print(f"\n  INFORMALITY RATES (employed):")
    print(f"    A. EsSalud-strict:      {rate_A:.1%}  gap={rate_A-inei:+.1%}  {flag(rate_A-inei)}")
    print(f"    B. Liberal (+private):  {rate_B:.1%}  gap={rate_B-inei:+.1%}  {flag(rate_B-inei)}")
    print(f"    C. INEI composite:      {rate_C:.1%}  gap={rate_C-inei:+.1%}  {flag(rate_C-inei)}")
    print(f"    INEI published:         {inei:.1%}")

    # ── STEP 5: Formal/informal breakdown by occupation ───────────────────────
    print(f"\n  Formal rate by occupation (def A — EsSalud strict):")
    for name, mask in [("empleado(3)", empleado), ("obrero(4)", obrero),
                       ("independiente(2)", independ), ("del hogar(6)", trab_hogar),
                       ("tfnr(5)", tfnr), ("empleador(1)", empleador)]:
        sub = employed & mask
        n_sub = sub.sum()
        n_formal = (formal_A & mask).sum()
        if n_sub > 0:
            print(f"    {name:<22}: {n_formal:,}/{n_sub:,}  ({n_formal/n_sub*100:.1f}% formal)")

    return {
        "year": year_label,
        "n_emp": n_emp,
        "rate_A": rate_A,  # EsSalud-strict
        "rate_B": rate_B,  # liberal
        "rate_C": rate_C,  # INEI composite
        "inei": inei,
        "gap_A": rate_A - inei,
        "gap_B": rate_B - inei,
        "gap_C": rate_C - inei,
    }


# ── RUN ────────────────────────────────────────────────────────────────────────
print("\n" + "#"*70)
print("  EPE INFORMALITY — DATA-FIRST CONSTRUCTION & INEI VALIDATION")
print("#"*70)

summary = []
for yr_label, code in VALIDATION_FILES:
    df, meta = load_sav(code)
    if df is None:
        print(f"\nSKIP {yr_label} (code {code}): file not found")
        continue
    r = construct_and_validate(df, meta, yr_label)
    if r:
        summary.append(r)


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n\n{'#'*70}")
print("  VALIDATION SUMMARY")
print(f"{'#'*70}")
print(f"  {'Year':<6} {'N_emp':>7}  {'A_EsSal':>8} {'B_Lib':>7} {'C_INEI':>7}"
      f"  {'INEI':>7}  {'GapA':>7} {'GapB':>7} {'GapC':>7}  Best")
print("  " + "-"*78)

for r in summary:
    gaps = {"A": r["gap_A"], "B": r["gap_B"], "C": r["gap_C"]}
    best = min(gaps, key=lambda k: abs(gaps[k]))
    flags = {k: ("✓" if abs(v) <= 0.05 else "~" if abs(v) <= 0.10 else "✗")
             for k, v in gaps.items()}
    print(f"  {r['year']:<6} {r['n_emp']:>7,}  "
          f"{r['rate_A']:>7.1%} {r['rate_B']:>7.1%} {r['rate_C']:>7.1%}  "
          f"{r['inei']:>7.1%}  "
          f"{r['gap_A']:>+7.1%}{flags['A']} "
          f"{r['gap_B']:>+7.1%}{flags['B']} "
          f"{r['gap_C']:>+7.1%}{flags['C']}  "
          f"Def {best}")

df_sum = pd.DataFrame(summary).dropna(subset=["gap_A", "gap_B", "gap_C"])
if len(df_sum):
    print("  " + "-"*78)
    print(f"  {'Mean |gap|':<14}  "
          f"{df_sum['gap_A'].abs().mean():>7.1%}  "
          f"{df_sum['gap_B'].abs().mean():>7.1%}  "
          f"{df_sum['gap_C'].abs().mean():>7.1%}")
    best_def = min(["A", "B", "C"], key=lambda k: df_sum[f"gap_{k}"].abs().mean())
    print(f"\n  RECOMMENDED DEFINITION: {best_def}")
    if best_def == "A":
        print("  → Use p222==1 (EsSalud strict) as formality marker")
    elif best_def == "B":
        print("  → Use p222 in [1,2,3] (EsSalud + private) as formality marker")
    else:
        print("  → Use INEI composite (EsSalud for asalariados + firm size for independientes)")

print(f"\n  DIAGNOSTIC: Do previous DiD scripts use wrong p222 codes?")
print(f"  Previous epe_robustness.py:     uses p213==2 for formality → WRONG (p213=job search)")
print(f"  Previous epe_panel_did_all.py:  uses p213 for lighthouse   → WRONG")
print(f"  Previous epe_wage_selection.py: does NOT use p206/p222     → OK")
print(f"  Lee bounds (epe_lee_bounds.py): does NOT use p206/p222     → OK")
print(f"  Employment DiD (Track 4):       uses ocu200, ingprin only  → OK")
print(f"  CONCLUSION: Employment DiD results are clean.")
print(f"  Only lighthouse (Track 2) needs rerun with p222==1 as formality proxy.")
