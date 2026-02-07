"""Explore BCRP API to discover all regional/departmental series codes.

Uses the BCRP API with proper headers and rate limiting to systematically
probe known and hypothesized series code ranges.
"""

import json
import time
import httpx

BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# Peru departments in standard order
DEPARTMENTS = [
    "Amazonas", "Ancash", "Apurímac", "Arequipa", "Ayacucho",
    "Cajamarca", "Callao", "Cusco", "Huancavelica", "Huánuco",
    "Ica", "Junín", "La Libertad", "Lambayeque", "Lima",
    "Loreto", "Madre de Dios", "Moquegua", "Pasco", "Piura",
    "Puno", "San Martín", "Tacna", "Tumbes", "Ucayali"
]

def test_series(code: str, start: str = "2023-1", end: str = "2023-6") -> dict:
    """Test a single series code against the BCRP API."""
    url = f"{BASE_URL}/{code}/json/{start}/{end}/esp"
    try:
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return {"code": code, "valid": False, "error": f"HTTP {resp.status_code}"}
            text = resp.text.strip()
            if not text.startswith("{"):
                return {"code": code, "valid": False, "error": "HTML response"}
            data = json.loads(text)
            config = data.get("config", {})
            series_meta = config.get("series", [])
            periods = data.get("periods", [])
            if not series_meta:
                return {"code": code, "valid": False, "error": "No series metadata"}
            name = series_meta[0].get("name", "unknown")
            n_periods = len(periods)
            # Get a sample value
            sample = None
            for p in periods:
                vals = p.get("values", [])
                if vals and vals[0] not in ("n.d.", "", None):
                    sample = vals[0]
                    break
            return {
                "code": code, "valid": True, "name": name,
                "n_periods": n_periods, "sample": sample, "error": None
            }
    except Exception as e:
        return {"code": code, "valid": False, "error": str(e)}


def probe_range(prefix: str, start_num: int, end_num: int, suffix: str, delay: float = 1.5):
    """Probe a range of series codes."""
    results = []
    for num in range(start_num, end_num + 1):
        code = f"{prefix}{num:05d}{suffix}" if len(str(start_num)) <= 5 else f"{prefix}{num}{suffix}"
        # Auto-format based on existing pattern
        if start_num >= 10000:
            code = f"{prefix}{num}{suffix}"
        else:
            code = f"{prefix}{num:05d}{suffix}"

        result = test_series(code)
        if result["valid"]:
            print(f"  FOUND: {code} -> {result['name']} ({result['n_periods']} periods, sample={result['sample']})")
        results.append(result)
        time.sleep(delay)
    return results


def probe_specific_codes(codes: list, delay: float = 1.5):
    """Probe specific series codes."""
    results = []
    for code in codes:
        result = test_series(code)
        if result["valid"]:
            print(f"  FOUND: {code} -> {result['name']} ({result['n_periods']} periods, sample={result['sample']})")
        else:
            print(f"  MISS:  {code} -> {result['error']}")
        results.append(result)
        time.sleep(delay)
    return results


def save_results(results, output_path="logs/explore_regional_results.json"):
    """Save probe results to JSON for later catalog building."""
    import json
    valid = [r for r in results if r["valid"]]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(valid)} valid series to {output_path}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Explore BCRP regional series")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated probe sections to run (e.g., 'mining,tourism')",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save valid results to JSON",
    )
    args = parser.parse_args()

    only_sections = set()
    if args.only:
        only_sections = {s.strip().lower() for s in args.only.split(",")}

    def should_run(section_name: str) -> bool:
        if not only_sections:
            return True
        return any(s in section_name.lower() for s in only_sections)

    all_found = []

    # =====================================================================
    # 1. ELECTRICITY PRODUCTION BY DEPARTMENT (RD13043DM - RD13067DM)
    # =====================================================================
    if should_run("electricity_production"):
        print("\n=== ELECTRICITY PRODUCTION BY DEPARTMENT (RD13043DM-RD13092DM) ===")
        results = probe_range("RD", 13043, 13092, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} electricity production series")

    # =====================================================================
    # 2. ELECTRICITY SALES BY DEPARTMENT (RD13093DM - RD13142DM)
    # =====================================================================
    if should_run("electricity_sales"):
        print("\n=== ELECTRICITY SALES BY DEPARTMENT (RD13093DM-RD13142DM) ===")
        results = probe_range("RD", 13093, 13142, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} electricity sales series")

    # =====================================================================
    # 3. MINING — TIN BY DEPARTMENT (RD12968DM - RD12977DM)
    # Between copper (RD12951-RD12967) and gold (RD12978-RD12995)
    # =====================================================================
    if should_run("mining_tin"):
        print("\n=== MINING TIN BY DEPARTMENT (RD12968DM-RD12977DM) ===")
        results = probe_range("RD", 12968, 12977, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} tin mining series")

    # =====================================================================
    # 4. MINING — SILVER BY DEPARTMENT (RD12996DM - RD13012DM)
    # After gold (RD12978-RD12995)
    # =====================================================================
    if should_run("mining_silver"):
        print("\n=== MINING SILVER BY DEPARTMENT (RD12996DM-RD13015DM) ===")
        results = probe_range("RD", 12996, 13015, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} silver mining series")

    # =====================================================================
    # 5. MINING — ZINC BY DEPARTMENT (RD13013DM - RD13029DM)
    # After silver
    # =====================================================================
    if should_run("mining_zinc"):
        print("\n=== MINING ZINC BY DEPARTMENT (RD13013DM-RD13032DM) ===")
        results = probe_range("RD", 13013, 13032, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} zinc mining series")

    # =====================================================================
    # 6. MINING — LEAD BY DEPARTMENT (RD13030DM - RD13042DM)
    # Before electricity
    # =====================================================================
    if should_run("mining_lead"):
        print("\n=== MINING LEAD BY DEPARTMENT (RD13030DM-RD13042DM) ===")
        results = probe_range("RD", 13030, 13042, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} lead mining series")

    # =====================================================================
    # 7. EXPORTS BY DEPARTMENT (RD38085BM - RD38111BM)
    # =====================================================================
    if should_run("exports"):
        print("\n=== EXPORTS BY DEPARTMENT (RD38085BM-RD38111BM) ===")
        results = probe_range("RD", 38085, 38111, "BM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} export series")

    # =====================================================================
    # 8. IMPORTS BY CUSTOMS (RD38112BM - RD38136BM)
    # =====================================================================
    if should_run("imports"):
        print("\n=== IMPORTS BY CUSTOMS (RD38112BM-RD38136BM) ===")
        results = probe_range("RD", 38112, 38136, "BM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} import series")

    # =====================================================================
    # 9. TOURISM — ARRIVALS BY DEPARTMENT (RD13335DM - RD13409DM)
    # =====================================================================
    if should_run("tourism_arrivals"):
        print("\n=== TOURISM ARRIVALS (RD13335DM-RD13409DM) ===")
        results = probe_range("RD", 13335, 13409, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} tourism arrival series")

    # =====================================================================
    # 10. TOURISM — OVERNIGHT STAYS (RD13410DM - RD13484DM)
    # Hypothesized range after arrivals
    # =====================================================================
    if should_run("tourism_nights"):
        print("\n=== TOURISM OVERNIGHT STAYS (RD13410DM-RD13484DM) ===")
        results = probe_range("RD", 13410, 13484, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} tourism overnight series")

    # =====================================================================
    # 11. GOV CAPEX LOCAL (RD14012DM - RD14037DM)
    # Between local spending (RD13986-RD14011) and regional spending (RD14038-RD14064)
    # =====================================================================
    if should_run("gov_capex_local"):
        print("\n=== GOV CAPEX LOCAL (RD14012DM-RD14037DM) ===")
        results = probe_range("RD", 14012, 14037, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} gov capex local series")

    # =====================================================================
    # 12. GOV CAPEX REGIONAL (RD14065DM - RD14090DM)
    # After regional spending (RD14038-RD14064)
    # =====================================================================
    if should_run("gov_capex_regional"):
        print("\n=== GOV CAPEX REGIONAL (RD14065DM-RD14090DM) ===")
        results = probe_range("RD", 14065, 14090, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} gov capex regional series")

    # =====================================================================
    # 13. INCOME TAX DETAIL (RD13801DM - RD13850DM)
    # After tax revenue total (RD13774-RD13800)
    # =====================================================================
    if should_run("income_tax"):
        print("\n=== INCOME TAX DETAIL (RD13801DM-RD13850DM) ===")
        results = probe_range("RD", 13801, 13850, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} income tax detail series")

    # =====================================================================
    # 14. ANCHOVETA BY PORT (RD13143DM - RD13200DM)
    # After energy sales range
    # =====================================================================
    if should_run("anchoveta"):
        print("\n=== ANCHOVETA BY PORT (RD13143DM-RD13200DM) ===")
        results = probe_range("RD", 13143, 13200, "DM", delay=1.5)
        found = [r for r in results if r["valid"]]
        all_found.extend(found)
        print(f"Found {len(found)} anchoveta series")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL FOUND REGIONAL SERIES")
    print("=" * 80)
    for r in all_found:
        print(f"{r['code']:15s} | {r['name']}")
    print(f"\nTotal found: {len(all_found)}")

    if args.save:
        save_results(all_found)
