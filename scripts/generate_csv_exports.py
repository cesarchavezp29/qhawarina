#!/usr/bin/env python3
"""
generate_csv_exports.py

Reads JSON files from exports/data/ and writes CSV equivalents to exports/data/csv/.
CSVs are UTF-8 with BOM so Excel opens them correctly with Spanish characters.

Outputs:
  precios_diarios.csv       — daily_price_index.json  → series[]
  riesgo_politico.csv       — political_index_daily.json → daily_series[] (IRP columns)
  riesgo_economico.csv      — political_index_daily.json → daily_series[] (IRE columns)
  indices_riesgo_mensual.csv — risk_index_monthly_peaks.json → months[]
  pbi_nowcast.csv           — gdp_nowcast.json → quarterly_series[]
  pobreza_nacional.csv      — poverty_nowcast.json → historical_series[]
  pobreza_departamental.csv — poverty_nowcast.json → departments[]
  tipo_cambio.csv           — fx_interventions.json → daily_series[]
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

EXPORTS_DIR = Path(__file__).parent.parent / "exports" / "data"
CSV_DIR = EXPORTS_DIR / "csv"
HEADER = f"# Fuente: Qhawarina (qhawarina.pe) | CC BY 4.0 | Generado: {date.today()}"


def load_json(filename: str) -> dict:
    path = EXPORTS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(filename: str, rows: list, fieldnames: list | None = None) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  SKIP {filename} — no rows")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path = CSV_DIR / filename
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(HEADER + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  OK  {filename} — {len(rows)} rows")


def export_precios_diarios() -> None:
    data = load_json("daily_price_index.json")
    series = data.get("series", [])
    if not series:
        print("  SKIP precios_diarios.csv — daily_price_index.json missing or empty")
        return
    # Keep key columns; all others (per-category indices) are also included
    write_csv("precios_diarios.csv", series)


def export_riesgo_politico() -> None:
    data = load_json("political_index_daily.json")
    series = data.get("daily_series", [])
    if not series:
        print("  SKIP riesgo_politico.csv — political_index_daily.json missing or empty")
        return
    cols = ["date", "political_7d", "political_raw", "n_articles"]
    rows = [{k: row.get(k, "") for k in cols} for row in series]
    write_csv("riesgo_politico.csv", rows, cols)


def export_riesgo_economico() -> None:
    data = load_json("political_index_daily.json")
    series = data.get("daily_series", [])
    if not series:
        print("  SKIP riesgo_economico.csv — political_index_daily.json missing or empty")
        return
    cols = ["date", "economic_7d", "economic_raw", "n_articles"]
    rows = [{k: row.get(k, "") for k in cols} for row in series if row.get("economic_7d") is not None]
    if not rows:
        print("  SKIP riesgo_economico.csv — no economic_7d data")
        return
    write_csv("riesgo_economico.csv", rows, cols)


def export_indices_riesgo_mensual() -> None:
    data = load_json("risk_index_monthly_peaks.json")
    months = data.get("months", [])
    if not months:
        print("  SKIP indices_riesgo_mensual.csv — risk_index_monthly_peaks.json missing or empty")
        return
    cols = [
        "month",
        "irp_7d_peak", "irp_peak_date", "irp_event",
        "ire_7d_peak", "ire_peak_date", "ire_event",
    ]
    rows = [{k: m.get(k, "") for k in cols} for m in months]
    write_csv("indices_riesgo_mensual.csv", rows, cols)


def export_pbi_nowcast() -> None:
    data = load_json("gdp_nowcast.json")
    series = data.get("quarterly_series", [])
    if not series:
        print("  SKIP pbi_nowcast.csv — gdp_nowcast.json missing or empty")
        return
    write_csv("pbi_nowcast.csv", series)


def export_pobreza_nacional() -> None:
    data = load_json("poverty_nowcast.json")
    series = data.get("historical_series", [])
    if not series:
        print("  SKIP pobreza_nacional.csv — poverty_nowcast.json missing or empty")
        return
    write_csv("pobreza_nacional.csv", series)


def export_pobreza_departamental() -> None:
    data = load_json("poverty_nowcast.json")
    departments = data.get("departments", [])
    if not departments:
        print("  SKIP pobreza_departamental.csv — poverty_nowcast.json departments missing")
        return
    write_csv("pobreza_departamental.csv", departments)


def export_tipo_cambio() -> None:
    data = load_json("fx_interventions.json")
    series = data.get("daily_series", [])
    if not series:
        print("  SKIP tipo_cambio.csv — fx_interventions.json missing or empty")
        return
    # Select readable columns; skip monthly_series
    cols = [
        "date", "fx", "spot_net_purchases", "swaps_net",
        "total_intervention", "reference_rate",
        "bond_sol_10y", "bond_usd_10y", "bvl",
    ]
    # Use only columns that exist in the data
    available = list(series[0].keys())
    cols = [c for c in cols if c in available] or available
    rows = [{k: row.get(k, "") for k in cols} for row in series]
    write_csv("tipo_cambio.csv", rows, cols)


def main() -> None:
    print(f"\nGenerating CSV exports → {CSV_DIR}\n")
    export_precios_diarios()
    export_riesgo_politico()
    export_riesgo_economico()
    export_indices_riesgo_mensual()
    export_pbi_nowcast()
    export_pobreza_nacional()
    export_pobreza_departamental()
    export_tipo_cambio()
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
