"""Verify new BCRP series codes against the live API."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.bcrp import BCRPClient

NEW_CODES = [
    # Commodity prices
    ("PN01652XM", "Precio del cobre"),
    ("PN01654XM", "Precio del oro"),
    ("PN01660XM", "Precio del petroleo WTI"),
    ("PN01655XM", "Precio de la plata"),
    ("PN01657XM", "Precio del zinc"),
    # Food prices
    ("PN39445PM", "IPC alimentos y bebidas (var% 12 meses)"),
    ("PN01383PM", "IPC alimentos y bebidas (indice)"),
    # Trade
    ("PN38923BM", "Terminos de intercambio"),
    # Monetary
    ("PN01013MM", "Liquidez total sistema financiero"),
    ("PN00178MM", "Base monetaria"),
    # Employment
    ("PN38063GM", "Empleo formal privado"),
    ("PN31879GM", "Empleo Lima Metropolitana"),
    # Confidence
    ("PD37981AM", "Confianza empresarial"),
    ("PD12912AM", "Confianza del consumidor"),
]


def main():
    client = BCRPClient(request_delay=1.5)

    verified = []
    failed = []

    for code, label in NEW_CODES:
        print(f"Verifying {code} ({label})...", end=" ", flush=True)
        result = client.verify_series(code)
        if result["valid"]:
            print(f"OK - {result['name']} ({result['sample_count']} values)")
            verified.append((code, result["name"]))
        else:
            print(f"FAILED - {result['error']}")
            failed.append((code, label, result["error"]))

    print(f"\n{'='*60}")
    print(f"Verified: {len(verified)}/{len(NEW_CODES)}")
    print(f"Failed:   {len(failed)}/{len(NEW_CODES)}")

    if failed:
        print(f"\nFailed codes:")
        for code, label, error in failed:
            print(f"  {code} ({label}): {error}")


if __name__ == "__main__":
    main()
