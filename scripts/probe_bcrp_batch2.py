"""Batch 2: Electricity sales, tourism arrivals, IPC by city, and credit/deposits by dept."""

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

def test_series(code, start="2023-1", end="2023-6"):
    url = f"{BASE_URL}/{code}/json/{start}/{end}/esp"
    try:
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return {"code": code, "valid": False, "name": None, "error": f"HTTP {resp.status_code}"}
            text = resp.text.strip()
            if not text.startswith("{"):
                return {"code": code, "valid": False, "name": None, "error": "HTML"}
            data = json.loads(text)
            series_meta = data.get("config", {}).get("series", [])
            periods = data.get("periods", [])
            if not series_meta:
                return {"code": code, "valid": False, "name": None, "error": "No meta"}
            name = series_meta[0].get("name", "?")
            return {"code": code, "valid": True, "name": name, "n": len(periods), "error": None}
    except Exception as e:
        return {"code": code, "valid": False, "name": None, "error": str(e)[:80]}

def probe(codes, label, delay=1.6):
    print(f"\n=== {label} ===")
    found = []
    for i, code in enumerate(codes):
        r = test_series(code)
        if r["valid"]:
            print(f"  OK  {code:15s} -> {r['name'][:90]} ({r['n']} per)")
            found.append(r)
        else:
            print(f"  --  {code:15s} -> {r['error']}")
        if i < len(codes) - 1:
            time.sleep(delay)
    return found

if __name__ == "__main__":
    all_found = []

    # Electricity sales by dept: RD13093DM to RD13142DM (50 = 25 GWh + 25 var%)
    codes = [f"RD{n}DM" for n in range(13093, 13143)]
    all_found += probe(codes, "ELECTRICITY SALES BY DEPT")

    # Tourism arrivals by dept: RD13335DM to RD13409DM (75 = 25*3: total/national/foreign)
    # Sample first 5 to understand structure, then sample strategically
    sample_tourism = [f"RD{n}DM" for n in range(13335, 13345)]
    all_found += probe(sample_tourism, "TOURISM ARRIVALS (sample first 10)")

    # Then jump to get remaining samples
    sample_tourism2 = [f"RD{n}DM" for n in range(13345, 13360)]
    all_found += probe(sample_tourism2, "TOURISM ARRIVALS (sample 10-25)")

    # Jump to see what's after department tourism for total/national/foreign split
    sample_tourism3 = [f"RD{n}DM" for n in range(13360, 13410)]
    all_found += probe(sample_tourism3, "TOURISM ARRIVALS (dept split cont'd)")

    # Try IPC by city - INEI publishes IPC for 26 cities
    # Try RD prefix for regional IPC
    ipc_city_codes = [f"RD{n}PM" for n in range(12900, 12920)]
    all_found += probe(ipc_city_codes, "IPC BY CITY (RD12900-12919 PM)")

    # Also try with DM suffix
    ipc_city_dm = [f"RD{n}DM" for n in range(12900, 12917)]
    all_found += probe(ipc_city_dm, "IPC BY CITY (RD12900-12916 DM)")

    # Credit by department - try PN codes that might have departmental credit
    # BCRP publishes "Credito al sector privado por departamento"
    # Known pattern from web: P001257BRM
    # Try variations of this pattern
    credit_dept = [f"P00125{i}BRM" for i in range(0, 10)]
    credit_dept += [f"P00126{i}BRM" for i in range(0, 10)]
    credit_dept += [f"P00127{i}BRM" for i in range(0, 10)]
    credit_dept += [f"P00128{i}BRM" for i in range(0, 10)]
    credit_dept += [f"P00129{i}BRM" for i in range(0, 10)]
    all_found += probe(credit_dept, "CREDIT BY DEPT (P001250-P001299 BRM)")

    # Deposits by department - try similar pattern
    deposit_dept = [f"P00130{i}BRM" for i in range(0, 10)]
    deposit_dept += [f"P00131{i}BRM" for i in range(0, 10)]
    deposit_dept += [f"P00132{i}BRM" for i in range(0, 10)]
    all_found += probe(deposit_dept, "DEPOSITS BY DEPT (P001300-P001329 BRM)")

    print("\n\n" + "="*80)
    print("BATCH 2 SUMMARY")
    print("="*80)
    for r in all_found:
        print(f"  {r['code']:15s} | {r['name'][:90]}")
    print(f"\nTotal: {len(all_found)} valid series")
