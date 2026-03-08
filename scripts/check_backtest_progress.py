"""Quick progress checker for running backtests."""

import re
from pathlib import Path

TASK_DIR = Path("D:/temp/claude/D--Nexus-nexus/tasks")

def check_progress():
    """Check progress of inflation and GDP backtests."""

    # Inflation backtest
    inf_file = TASK_DIR / "b157126.output"
    if inf_file.exists():
        with open(inf_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Find last progress line
        progress_lines = [l for l in lines if "Backtest progress:" in l]
        if progress_lines:
            match = re.search(r'(\d+)/(\d+)', progress_lines[-1])
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                pct = (current / total) * 100
                print(f"[INFLATION] {current}/{total} ({pct:.1f}%)")

        # Check if finished
        if any("Results saved" in l for l in lines[-50:]):
            print("[INFLATION] COMPLETED")

    # GDP backtest
    gdp_file = TASK_DIR / "bee24e8.output"
    if gdp_file.exists():
        with open(gdp_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Find last progress line
        progress_lines = [l for l in lines if "Backtest progress:" in l]
        if progress_lines:
            match = re.search(r'(\d+)/(\d+)', progress_lines[-1])
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                pct = (current / total) * 100
                print(f"[GDP] {current}/{total} ({pct:.1f}%)")

        # Check if finished
        if any("Results saved" in l for l in lines[-50:]):
            print("[GDP] COMPLETED")

    print("\nRe-run this script to check again, or use:")
    print("   python scripts/check_backtest_progress.py")

if __name__ == "__main__":
    check_progress()
