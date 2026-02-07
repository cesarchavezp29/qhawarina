"""Discover and download expanded BCRP series (regional, inflation components, sectoral).

This script probes the BCRP API for series codes in known ranges, verifies them,
creates YAML catalogs, and downloads all data.

Priorities covered:
  - Priority 2: Regional/departmental series (exports, imports by department)
  - Priority 3: Inflation component series (IPC by group, by classification)
  - Priority 4: Sectoral GDP monthly indices (manufacturing, fishing, etc.)

Usage:
    python scripts/download_expanded_bcrp.py                    # full discovery + download
    python scripts/download_expanded_bcrp.py --discover-only    # just probe, don't download
    python scripts/download_expanded_bcrp.py --download-only    # use existing catalogs
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import pandas as pd
import yaml

from src.ingestion.bcrp import BCRPClient
from src.utils.dates import parse_bcrp_period
from src.utils.io import save_parquet, ensure_dir
from config.settings import RAW_BCRP_DIR

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "download_expanded.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.expanded_bcrp")

# ── BCRP API probing ────────────────────────────────────────────────────────

BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def probe_code(code: str, start: str = "2023-1", end: str = "2023-6",
               fallback_start: str | None = None, fallback_end: str | None = None) -> dict | None:
    """Probe a single BCRP code. Returns {code, name, sample_count} or None."""
    for s, e in [(start, end)] + ([(fallback_start, fallback_end)] if fallback_start else []):
        url = f"{BASE_URL}/{code}/json/{s}/{e}/esp"
        try:
            with httpx.Client(timeout=20, headers=HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
                text = resp.text.strip()
                if not text.startswith("{"):
                    continue
                data = json.loads(text)
                # Extract series name from config
                series_list = data.get("config", {}).get("series", [])
                if not series_list:
                    continue
                name = series_list[0].get("name", "")
                periods = data.get("periods", [])
                # Count non-null values
                n_values = 0
                for p in periods:
                    for v in p.get("values", []):
                        if v and str(v).strip() not in ("", "n.d.", "-"):
                            n_values += 1
                if n_values == 0:
                    continue
                return {"code": code, "name": name, "sample_count": n_values}
        except Exception:
            continue
    return None


def probe_range(prefix: str, start: int, end: int, suffix: str,
                delay: float = 1.0) -> list[dict]:
    """Probe a range of BCRP codes and return valid ones."""
    results = []
    total = end - start + 1
    for i, num in enumerate(range(start, end + 1)):
        code = f"{prefix}{num:05d}{suffix}"
        result = probe_code(code)
        if result:
            results.append(result)
            logger.info("  [%d/%d] FOUND: %s — %s", i + 1, total, code, result["name"])
        else:
            logger.debug("  [%d/%d] not found: %s", i + 1, total, code)
        time.sleep(delay)
    return results


def probe_codes(codes: list[str], delay: float = 1.0,
                fallback_start: str | None = None,
                fallback_end: str | None = None) -> list[dict]:
    """Probe specific BCRP codes and return valid ones."""
    results = []
    for i, code in enumerate(codes):
        result = probe_code(code, fallback_start=fallback_start,
                            fallback_end=fallback_end)
        if result:
            results.append(result)
            logger.info("  [%d/%d] FOUND: %s — %s",
                        i + 1, len(codes), code, result["name"])
        else:
            logger.debug("  [%d/%d] not found: %s", i + 1, len(codes), code)
        time.sleep(delay)
    return results


# ── Series code definitions ─────────────────────────────────────────────────

# Priority 2: Regional series
REGIONAL_EXPORT_CODES = [f"RD{38085 + i}BM" for i in range(27)]  # RD38085BM-RD38111BM
REGIONAL_IMPORT_CODES = [f"RD{38112 + i}BM" for i in range(25)]  # RD38112BM-RD38136BM

# Priority 3: Inflation component codes (discovered via API probing)
IPC_COMPONENT_CODES = (
    # IPC Lima var% mensual by group (Alimentos, Vestido, Vivienda, etc.)
    [f"PN{n:05d}PM" for n in range(1260, 1294)]
    + [f"PN{n:05d}PM" for n in range(1295, 1300)]
    # IPC sectorial var% mensual
    + [f"PN{n:05d}PM" for n in range(1300, 1318)]
    # IPC transables/no transables
    + [f"PN{n:05d}PM" for n in range(1335, 1339)]
    # IPC var% 12 meses by group
    + [f"PN{n:05d}PM" for n in range(1364, 1388)]
    # IPC newer indices (base Dic.2021=100)
    + [f"PN{n:05d}PM" for n in range(38705, 38712)]
    # Food price indices
    + [f"PN{n:05d}PM" for n in range(39444, 39448)]
)
# Remove codes already in the national catalog
EXISTING_IPC_CODES = {"PN01271PM", "PN01273PM", "PN38706PM", "PN39445PM", "PN01383PM"}
IPC_COMPONENT_CODES = [c for c in IPC_COMPONENT_CODES if c not in EXISTING_IPC_CODES]

# Priority 4: Sectoral production indices (monthly, index 2007=100)
SECTORAL_CODES = [
    # Manufacturing production indices
    "PN02079AM",  # Manufacturing total (index 2007=100)
    "PN02028AM",  # Manufacturing no primaria
    "PN02068AM",  # Cement production index
    "PN02024AM",  # Fishmeal & fish oil
    "PN01958AM",  # Manufacturing services var%
    # Capacity utilization
    "PN37618AM",  # Capacity utilization - primary resource processors
    # Business expectations by sector
    "PD39768AM",  # Business expectations - manufacturing
    # Fishing production
    "PN38083AM",  # Fishing continental frozen
    # Mining production indices
    "PN01840AM",  # Mining production copper
    "PN01841AM",  # Mining production gold
    "PN01842AM",  # Mining production silver
    "PN01843AM",  # Mining production zinc
    "PN01844AM",  # Mining production lead
    "PN01845AM",  # Mining production iron
    "PN01846AM",  # Mining production tin
    "PN01847AM",  # Mining production molybdenum
    # Additional production indices
    "PD37965AM",  # Electricity var% 12m
    "PD37968AM",  # Cement production var%
    # Additional fishing
    "PN38064AM",  # Fishing total
    "PN38065AM",  # Fishing maritime
    "PN38066AM",  # Fishing continental
]

# Additional regional probing ranges
REGIONAL_CREDIT_PROBE = [f"RD{n:05d}MM" for n in range(38137, 38170)]
REGIONAL_DEPOSIT_PROBE = [f"RD{n:05d}MM" for n in range(38170, 38200)]
# IPC by city probes
IPC_CITY_PROBE = [f"PN{n:05d}PM" for n in range(1388, 1420)]


# ── Discovery phase ─────────────────────────────────────────────────────────

def run_discovery() -> dict:
    """Discover all valid BCRP codes across priorities 2-4."""
    all_discovered = {
        "regional_exports": [],
        "regional_imports": [],
        "ipc_components": [],
        "sectoral_production": [],
        "regional_credit": [],
        "regional_deposits": [],
        "ipc_cities": [],
    }

    # Priority 2a: Regional exports (data may not extend to 2023, use 2020 fallback)
    logger.info("=" * 60)
    logger.info("PRIORITY 2: Probing regional export series (%d codes)...",
                len(REGIONAL_EXPORT_CODES))
    all_discovered["regional_exports"] = probe_codes(
        REGIONAL_EXPORT_CODES, delay=1.0,
        fallback_start="2020-1", fallback_end="2020-6",
    )
    logger.info("  Found %d valid export series",
                len(all_discovered["regional_exports"]))

    # Priority 2b: Regional imports (same fallback)
    logger.info("Probing regional import series (%d codes)...",
                len(REGIONAL_IMPORT_CODES))
    all_discovered["regional_imports"] = probe_codes(
        REGIONAL_IMPORT_CODES, delay=1.0,
        fallback_start="2020-1", fallback_end="2020-6",
    )
    logger.info("  Found %d valid import series",
                len(all_discovered["regional_imports"]))

    # Priority 3: IPC components
    logger.info("=" * 60)
    logger.info("PRIORITY 3: Probing IPC component series (%d codes)...",
                len(IPC_COMPONENT_CODES))
    all_discovered["ipc_components"] = probe_codes(IPC_COMPONENT_CODES, delay=1.0)
    logger.info("  Found %d valid IPC component series",
                len(all_discovered["ipc_components"]))

    # Priority 4: Sectoral production
    logger.info("=" * 60)
    logger.info("PRIORITY 4: Probing sectoral production series (%d codes)...",
                len(SECTORAL_CODES))
    all_discovered["sectoral_production"] = probe_codes(SECTORAL_CODES, delay=1.0)
    logger.info("  Found %d valid sectoral series",
                len(all_discovered["sectoral_production"]))

    # Additional probing: regional credit (use fallback dates)
    logger.info("Probing regional credit series (%d codes)...",
                len(REGIONAL_CREDIT_PROBE))
    all_discovered["regional_credit"] = probe_codes(
        REGIONAL_CREDIT_PROBE, delay=1.0,
        fallback_start="2020-1", fallback_end="2020-6",
    )
    logger.info("  Found %d regional credit series",
                len(all_discovered["regional_credit"]))

    # Additional probing: regional deposits (use fallback dates)
    logger.info("Probing regional deposit series (%d codes)...",
                len(REGIONAL_DEPOSIT_PROBE))
    all_discovered["regional_deposits"] = probe_codes(
        REGIONAL_DEPOSIT_PROBE, delay=1.0,
        fallback_start="2020-1", fallback_end="2020-6",
    )
    logger.info("  Found %d regional deposit series",
                len(all_discovered["regional_deposits"]))

    # Additional probing: IPC by city
    logger.info("Probing IPC by city series (%d codes)...",
                len(IPC_CITY_PROBE))
    all_discovered["ipc_cities"] = probe_codes(IPC_CITY_PROBE, delay=1.0)
    logger.info("  Found %d IPC city series",
                len(all_discovered["ipc_cities"]))

    # Save discovery results
    discovery_path = PROJECT_ROOT / "config" / "discovered_series.json"
    with open(discovery_path, "w", encoding="utf-8") as f:
        json.dump(all_discovered, f, indent=2, ensure_ascii=False)
    logger.info("Discovery results saved to %s", discovery_path)

    return all_discovered


# ── Catalog generation ───────────────────────────────────────────────────────

# BCRP department codes → UBIGEO mapping
DEPT_NAMES_ORDER = [
    "Amazonas", "Ancash", "Apurímac", "Arequipa", "Ayacucho",
    "Cajamarca", "Callao", "Cusco", "Huancavelica", "Huánuco",
    "Ica", "Junín", "La Libertad", "Lambayeque", "Lima",
    "Loreto", "Madre de Dios", "Moquegua", "Pasco", "Piura",
    "Puno", "San Martín", "Tacna", "Tumbes", "Ucayali",
]
DEPT_UBIGEO = {
    "Amazonas": "01", "Ancash": "02", "Apurímac": "03", "Arequipa": "04",
    "Ayacucho": "05", "Cajamarca": "06", "Callao": "07", "Cusco": "08",
    "Huancavelica": "09", "Huánuco": "10", "Ica": "11", "Junín": "12",
    "La Libertad": "13", "Lambayeque": "14", "Lima": "15", "Loreto": "16",
    "Madre de Dios": "17", "Moquegua": "18", "Pasco": "19", "Piura": "20",
    "Puno": "21", "San Martín": "22", "Tacna": "23", "Tumbes": "24",
    "Ucayali": "25",
}


def guess_department(series_name: str) -> str | None:
    """Try to extract department name from series name."""
    for dept in DEPT_NAMES_ORDER:
        if dept.lower() in series_name.lower():
            return dept
    # Handle accent variants
    if "apurimac" in series_name.lower():
        return "Apurímac"
    if "huanuco" in series_name.lower():
        return "Huánuco"
    if "san martin" in series_name.lower():
        return "San Martín"
    return None


def generate_regional_catalog(discovered: dict) -> dict:
    """Generate regional_series_catalog.yaml content."""
    catalog = {
        "regional": {
            "exports_by_department": {
                "description": "Exportaciones FOB por departamento (millones USD)",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
            "imports_by_customs": {
                "description": "Importaciones CIF por aduana (millones USD)",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
        }
    }

    # Exports
    for s in discovered.get("regional_exports", []):
        dept = guess_department(s["name"])
        entry = {
            "code": s["code"],
            "name": s["name"],
            "status": "verified",
        }
        if dept:
            entry["department"] = dept
            entry["ubigeo"] = DEPT_UBIGEO.get(dept, "")
        catalog["regional"]["exports_by_department"]["series"].append(entry)

    # Imports
    for s in discovered.get("regional_imports", []):
        entry = {
            "code": s["code"],
            "name": s["name"],
            "status": "verified",
        }
        catalog["regional"]["imports_by_customs"]["series"].append(entry)

    # Regional credit (if discovered)
    if discovered.get("regional_credit"):
        catalog["regional"]["credit_by_department"] = {
            "description": "Crédito por departamento",
            "frequency": "monthly",
            "source": "BCRP",
            "series": [],
        }
        for s in discovered["regional_credit"]:
            dept = guess_department(s["name"])
            entry = {
                "code": s["code"],
                "name": s["name"],
                "status": "verified",
            }
            if dept:
                entry["department"] = dept
                entry["ubigeo"] = DEPT_UBIGEO.get(dept, "")
            catalog["regional"]["credit_by_department"]["series"].append(entry)

    # Regional deposits (if discovered)
    if discovered.get("regional_deposits"):
        catalog["regional"]["deposits_by_department"] = {
            "description": "Depósitos por departamento",
            "frequency": "monthly",
            "source": "BCRP",
            "series": [],
        }
        for s in discovered["regional_deposits"]:
            dept = guess_department(s["name"])
            entry = {
                "code": s["code"],
                "name": s["name"],
                "status": "verified",
            }
            if dept:
                entry["department"] = dept
                entry["ubigeo"] = DEPT_UBIGEO.get(dept, "")
            catalog["regional"]["deposits_by_department"]["series"].append(entry)

    return catalog


def generate_inflation_catalog(discovered: dict) -> dict:
    """Generate inflation_components_catalog.yaml content."""
    catalog = {
        "inflation_components": {
            "ipc_groups_var_mensual": {
                "description": "IPC Lima por grupo — variación porcentual mensual",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
            "ipc_groups_var_12m": {
                "description": "IPC Lima por grupo — variación porcentual 12 meses",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
            "ipc_classification": {
                "description": "IPC Lima clasificaciones (subyacente, transables, etc.)",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
            "ipc_indices": {
                "description": "IPC índices de nivel (base 2009=100 o Dic.2021=100)",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            },
        }
    }

    # IPC by city (if discovered)
    if discovered.get("ipc_cities"):
        catalog["inflation_components"]["ipc_by_city"] = {
            "description": "IPC por ciudad",
            "frequency": "monthly",
            "source": "BCRP",
            "series": [],
        }

    for s in discovered.get("ipc_components", []):
        name_lower = s["name"].lower()
        entry = {
            "code": s["code"],
            "name": s["name"],
            "status": "verified",
        }

        # Classify by type
        if "índice" in name_lower or "indice" in name_lower or "index" in name_lower:
            catalog["inflation_components"]["ipc_indices"]["series"].append(entry)
        elif "12 meses" in name_lower or "12m" in name_lower or "interanual" in name_lower:
            catalog["inflation_components"]["ipc_groups_var_12m"]["series"].append(entry)
        elif ("subyacente" in name_lower or "transable" in name_lower
              or "no transable" in name_lower or "importad" in name_lower):
            catalog["inflation_components"]["ipc_classification"]["series"].append(entry)
        else:
            catalog["inflation_components"]["ipc_groups_var_mensual"]["series"].append(entry)

    for s in discovered.get("ipc_cities", []):
        entry = {
            "code": s["code"],
            "name": s["name"],
            "status": "verified",
        }
        catalog["inflation_components"]["ipc_by_city"]["series"].append(entry)

    # Remove empty categories
    catalog["inflation_components"] = {
        k: v for k, v in catalog["inflation_components"].items()
        if v.get("series")
    }

    return catalog


def generate_sectoral_catalog(discovered: dict) -> dict:
    """Generate sectoral additions to main catalog."""
    catalog = {
        "sectoral": {
            "production_indices": {
                "description": "Producción sectorial — índices y volúmenes mensuales",
                "frequency": "monthly",
                "source": "BCRP",
                "series": [],
            }
        }
    }

    for s in discovered.get("sectoral_production", []):
        entry = {
            "code": s["code"],
            "name": s["name"],
            "status": "verified",
        }
        catalog["sectoral"]["production_indices"]["series"].append(entry)

    return catalog


# ── Download phase ───────────────────────────────────────────────────────────

def download_from_catalog(catalog: dict, section_key: str,
                          output_path: Path, label: str) -> pd.DataFrame:
    """Download all series from a catalog section using BCRPClient."""
    client = BCRPClient(request_delay=1.5)

    all_codes = []
    code_meta = {}

    section = catalog.get(section_key, {})
    for cat_key, cat_data in section.items():
        if not isinstance(cat_data, dict):
            continue
        for entry in cat_data.get("series", []):
            code = entry["code"]
            all_codes.append(code)
            code_meta[code] = {
                "category": cat_key,
                "name": entry.get("name", ""),
            }

    if not all_codes:
        logger.warning("No codes found in %s section", section_key)
        return pd.DataFrame()

    logger.info("Downloading %d %s series from BCRP...", len(all_codes), label)

    df = client.fetch_series(
        all_codes,
        start_year=2004, start_month=1,
        end_year=2026, end_month=2,
    )

    if df.empty:
        logger.warning("No data downloaded for %s", label)
        return df

    # Enrich with metadata
    df["category"] = df["series_code"].map(
        lambda c: code_meta.get(c, {}).get("category", "unknown")
    )
    df["source"] = "BCRP"
    df["frequency_original"] = "M"

    # Save
    ensure_dir(output_path.parent)
    save_parquet(df, output_path)
    logger.info("Saved %s data to %s (%d rows, %d series)",
                label, output_path, len(df), df["series_code"].nunique())

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover and download expanded BCRP series"
    )
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover codes, don't download data")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download using existing catalogs")
    args = parser.parse_args()

    config_dir = PROJECT_ROOT / "config"
    regional_catalog_path = config_dir / "regional_series_catalog.yaml"
    inflation_catalog_path = config_dir / "inflation_components_catalog.yaml"
    sectoral_catalog_path = config_dir / "sectoral_production_catalog.yaml"
    discovery_path = config_dir / "discovered_series.json"

    t0 = time.time()

    # ── DISCOVERY ──
    if not args.download_only:
        logger.info("=" * 60)
        logger.info("NEXUS Expanded BCRP Series Discovery")
        logger.info("=" * 60)

        discovered = run_discovery()

        # Generate catalogs
        regional_cat = generate_regional_catalog(discovered)
        inflation_cat = generate_inflation_catalog(discovered)
        sectoral_cat = generate_sectoral_catalog(discovered)

        # Save catalogs
        with open(regional_catalog_path, "w", encoding="utf-8") as f:
            yaml.dump(regional_cat, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
        logger.info("Saved regional catalog: %s", regional_catalog_path)

        with open(inflation_catalog_path, "w", encoding="utf-8") as f:
            yaml.dump(inflation_cat, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
        logger.info("Saved inflation catalog: %s", inflation_catalog_path)

        with open(sectoral_catalog_path, "w", encoding="utf-8") as f:
            yaml.dump(sectoral_cat, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
        logger.info("Saved sectoral catalog: %s", sectoral_catalog_path)

        # Summary
        total_found = sum(len(v) for v in discovered.values())
        logger.info("")
        logger.info("DISCOVERY SUMMARY:")
        for key, series_list in discovered.items():
            logger.info("  %s: %d series", key, len(series_list))
        logger.info("  TOTAL: %d new series discovered", total_found)

    if args.discover_only:
        elapsed = time.time() - t0
        logger.info("Discovery completed in %.1fs", elapsed)
        return 0

    # ── DOWNLOAD ──
    if args.download_only:
        # Load existing catalogs
        if not regional_catalog_path.exists():
            logger.error("Regional catalog not found. Run discovery first.")
            return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOADING EXPANDED SERIES DATA")
    logger.info("=" * 60)

    # Load catalogs
    with open(regional_catalog_path, "r", encoding="utf-8") as f:
        regional_cat = yaml.safe_load(f) or {}
    with open(inflation_catalog_path, "r", encoding="utf-8") as f:
        inflation_cat = yaml.safe_load(f) or {}
    with open(sectoral_catalog_path, "r", encoding="utf-8") as f:
        sectoral_cat = yaml.safe_load(f) or {}

    # Download regional
    regional_df = download_from_catalog(
        regional_cat, "regional",
        RAW_BCRP_DIR / "bcrp_regional.parquet",
        "regional",
    )

    # Download inflation components
    inflation_df = download_from_catalog(
        inflation_cat, "inflation_components",
        RAW_BCRP_DIR / "inflation_components.parquet",
        "inflation components",
    )

    # Download sectoral production
    sectoral_df = download_from_catalog(
        sectoral_cat,
        "sectoral",
        RAW_BCRP_DIR / "sectoral_production.parquet",
        "sectoral production",
    )

    # ── FINAL SUMMARY ──
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPANDED BCRP DOWNLOAD COMPLETE in %.1fs", elapsed)
    logger.info("  Regional: %d rows, %d series",
                len(regional_df) if not regional_df.empty else 0,
                regional_df["series_code"].nunique() if not regional_df.empty else 0)
    logger.info("  Inflation: %d rows, %d series",
                len(inflation_df) if not inflation_df.empty else 0,
                inflation_df["series_code"].nunique() if not inflation_df.empty else 0)
    logger.info("  Sectoral: %d rows, %d series",
                len(sectoral_df) if not sectoral_df.empty else 0,
                sectoral_df["series_code"].nunique() if not sectoral_df.empty else 0)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
