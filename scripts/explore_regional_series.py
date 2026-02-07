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


if __name__ == "__main__":
    all_found = []

    # =====================================================================
    # 1. ELECTRICITY PRODUCTION BY DEPARTMENT (RD13043DM - RD13067DM)
    # Known from the production page: 25 departments
    # =====================================================================
    print("\n=== ELECTRICITY PRODUCTION BY DEPARTMENT (RD13043DM-RD13092DM) ===")
    # Test a broader range to catch variation series too
    results = probe_range("RD", 13043, 13092, "DM", delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)
    print(f"Found {len(found)} electricity series")

    # =====================================================================
    # 2. ELECTRICITY SALES BY DEPARTMENT (RD13093DM - RD13142DM)
    # =====================================================================
    print("\n=== ELECTRICITY SALES BY DEPARTMENT (RD13093DM-RD13142DM) ===")
    results = probe_range("RD", 13093, 13142, "DM", delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)
    print(f"Found {len(found)} electricity sales series")

    # =====================================================================
    # 3. EXPORTS BY DEPARTMENT (RD38085BM - RD38111BM)
    # Known: 27 series (25 depts + "No registrado" + Total)
    # =====================================================================
    print("\n=== EXPORTS BY DEPARTMENT (RD38085BM-RD38111BM) ===")
    results = probe_range("RD", 38085, 38111, "BM", delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)
    print(f"Found {len(found)} export series")

    # =====================================================================
    # 4. IMPORTS BY DEPARTMENT/CUSTOMS (RD38112BM - RD38136BM)
    # =====================================================================
    print("\n=== IMPORTS BY CUSTOMS (RD38112BM-RD38136BM) ===")
    results = probe_range("RD", 38112, 38136, "BM", delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)
    print(f"Found {len(found)} import series")

    # =====================================================================
    # 5. TOURISM - ARRIVALS BY DEPARTMENT (RD13335DM - RD13409DM)
    # =====================================================================
    print("\n=== TOURISM ARRIVALS (RD13335DM-RD13409DM) ===")
    results = probe_range("RD", 13335, 13409, "DM", delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)
    print(f"Found {len(found)} tourism arrival series")

    # =====================================================================
    # 6. CREDIT BY DEPARTMENT - Probe around P001257BRM pattern
    # The known code P001257BRM suggests a different prefix pattern
    # =====================================================================
    print("\n=== CREDIT BY DEPARTMENT (probing P00125xBRM pattern) ===")
    credit_codes = [f"P00125{i}BRM" for i in range(0, 10)]
    credit_codes += [f"P0012{i:02d}BRM" for i in range(50, 85)]
    results = probe_specific_codes(credit_codes, delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)

    # Also try the PN pattern for credit by department
    print("\n=== CREDIT BY DEPARTMENT (probing PN pattern) ===")
    # Try common departmental credit codes
    credit_pn_codes = [f"PN005{i:02d}MM" for i in range(16, 50)]
    results = probe_specific_codes(credit_pn_codes, delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)

    # =====================================================================
    # 7. IPC BY CITY - Try various patterns
    # IPC Lima is PN01271PM. Try nearby codes for other cities.
    # =====================================================================
    print("\n=== IPC BY CITY (probing around PN0127xPM) ===")
    ipc_codes = [f"PN0127{i}PM" for i in range(0, 10)]
    ipc_codes += [f"PN0128{i}PM" for i in range(0, 10)]
    ipc_codes += [f"PN0129{i}PM" for i in range(0, 10)]
    ipc_codes += [f"PN0130{i}PM" for i in range(0, 10)]
    results = probe_specific_codes(ipc_codes, delay=1.5)
    found = [r for r in results if r["valid"]]
    all_found.extend(found)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL FOUND REGIONAL SERIES")
    print("=" * 80)
    for r in all_found:
        print(f"{r['code']:15s} | {r['name']}")
    print(f"\nTotal found: {len(all_found)}")
