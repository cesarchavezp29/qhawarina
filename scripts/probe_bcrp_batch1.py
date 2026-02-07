"""Batch 1: Probe electricity production by department + sample exports + IPC cities."""

import json
import time
import sys
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
            print(f"  OK  {code:15s} -> {r['name']} ({r['n']} periods)")
            found.append(r)
        else:
            print(f"  --  {code:15s} -> {r['error']}")
        if i < len(codes) - 1:
            time.sleep(delay)
    return found

if __name__ == "__main__":
    all_found = []

    # Electricity production by dept: RD13043DM to RD13067DM (25 depts)
    codes = [f"RD{n}DM" for n in range(13043, 13068)]
    all_found += probe(codes, "ELECTRICITY PRODUCTION BY DEPT")

    # Electricity production variations: RD13068DM to RD13092DM
    codes = [f"RD{n}DM" for n in range(13068, 13093)]
    all_found += probe(codes, "ELECTRICITY PRODUCTION VAR%")

    # Exports by department: RD38085BM to RD38111BM (27 series)
    codes = [f"RD{n}BM" for n in range(38085, 38112)]
    all_found += probe(codes, "EXPORTS BY DEPARTMENT")

    # IPC by city - probe around known IPC codes
    # Lima IPC is PN01271PM, PN01273PM, PN38706PM
    # Try broader range
    ipc_codes = [f"PN{n:05d}PM" for n in range(1270, 1310)]
    all_found += probe(ipc_codes, "IPC SERIES (PN01270PM-PN01309PM)")

    # IPC index codes (PN38xxx range)
    ipc_idx = [f"PN{n}PM" for n in range(38700, 38730)]
    all_found += probe(ipc_idx, "IPC INDEX SERIES (PN38700PM-PN38729PM)")

    print("\n\n" + "="*80)
    print("BATCH 1 SUMMARY")
    print("="*80)
    for r in all_found:
        print(f"  {r['code']:15s} | {r['name']}")
    print(f"\nTotal: {len(all_found)} valid series")
