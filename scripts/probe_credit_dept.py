"""Probe BCRP API for departmental credit series codes."""

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

    # === BATCH 1: User-specified codes with PN prefix, FM suffix ===
    codes = [f"PN0{n}FM" for n in range(3498, 3530)]
    all_found += probe(codes, "PN03498FM-PN03529FM (FM suffix)")

    # === BATCH 2: RD prefix, FM suffix ===
    codes = [f"RD0{n}FM" for n in range(3498, 3530)]
    all_found += probe(codes, "RD03498FM-RD03529FM (regional FM)")

    # === BATCH 3: PN prefix, MM suffix (monetary/credit) ===
    codes = [f"PN0{n}MM" for n in range(3498, 3530)]
    all_found += probe(codes, "PN03498MM-PN03529MM (MM suffix)")

    # === BATCH 4: RD prefix, MM suffix ===
    codes = [f"RD0{n}MM" for n in range(3498, 3530)]
    all_found += probe(codes, "RD03498MM-RD03529MM (regional MM)")

    # === BATCH 5: Try around known credit code PN00518MM ===
    # National credit is PN00518MM. Regional might be nearby.
    codes = [f"PN00{n}MM" for n in range(518, 560)]
    all_found += probe(codes, "PN00518MM-PN00559MM (near national credit)")

    # === BATCH 6: RD prefix near credit code ===
    codes = [f"RD00{n}MM" for n in range(518, 560)]
    all_found += probe(codes, "RD00518MM-RD00559MM (RD near national credit)")

    print("\n\n" + "=" * 80)
    print("SUMMARY: ALL FOUND SERIES")
    print("=" * 80)
    for r in all_found:
        print(f"  {r['code']:15s} | {r['name']}")
    print(f"\nTotal: {len(all_found)} valid series")
