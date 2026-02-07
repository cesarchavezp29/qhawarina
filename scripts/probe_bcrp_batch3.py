"""Batch 3: Mineral production by dept, overnight stays, IPC by city search,
credit/deposit by dept with different code patterns."""

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
            print(f"  OK  {code:15s} -> {r['name'][:100]} ({r['n']} per)")
            found.append(r)
        else:
            print(f"  --  {code:15s} -> {r['error']}")
        if i < len(codes) - 1:
            time.sleep(delay)
    return found

if __name__ == "__main__":
    all_found = []

    # =====================================================================
    # Mineral production: copper by dept RD12951DM onwards
    # From the web scrape: RD12951DM = Copper Total, up to RD13042DM
    # Let's sample the first 30 to understand the structure
    # =====================================================================
    print(">>> MINERAL PRODUCTION (RD12951-RD12980)")
    codes = [f"RD{n}DM" for n in range(12951, 12981)]
    all_found += probe(codes, "MINERAL PRODUCTION (copper+gold)")

    # Continue to get zinc, silver, lead, tin, molybdenum, iron
    print("\n>>> MINERAL PRODUCTION (RD12981-RD13042)")
    codes = [f"RD{n}DM" for n in range(12981, 13043)]
    all_found += probe(codes, "MINERAL PRODUCTION (zinc+others)")

    # =====================================================================
    # Overnight stays by department: RD13410DM-RD13484DM (75 series)
    # Structure: 25 depts * 3 (total/national/foreign)
    # =====================================================================
    print("\n>>> OVERNIGHT STAYS (RD13410-RD13484)")
    codes = [f"RD{n}DM" for n in range(13410, 13485)]
    all_found += probe(codes, "OVERNIGHT STAYS BY DEPT")

    # =====================================================================
    # Average stay duration: RD13485DM-RD13559DM (75 series)
    # =====================================================================
    print("\n>>> AVERAGE STAY (RD13485-RD13559)")
    codes = [f"RD{n}DM" for n in range(13485, 13560)]
    all_found += probe(codes, "AVERAGE STAY DURATION BY DEPT")

    # =====================================================================
    # IPC by city: try PN013xxPM range more broadly
    # IPC for other cities might be PN01310-PN01400
    # =====================================================================
    print("\n>>> IPC EXTENDED (PN01310-PN01400 PM)")
    codes = [f"PN{n:05d}PM" for n in range(1310, 1401)]
    all_found += probe(codes, "IPC EXTENDED RANGE")

    print("\n\n" + "="*80)
    print("BATCH 3 SUMMARY")
    print("="*80)
    for r in all_found:
        print(f"  {r['code']:15s} | {r['name'][:100]}")
    print(f"\nTotal: {len(all_found)} valid series")
