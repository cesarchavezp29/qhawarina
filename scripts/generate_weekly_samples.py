"""Generate 3 sample weekly PDF reports for design comparison.

Usage:
    python scripts/generate_weekly_samples.py [--no-narrative] [--date YYYY-MM-DD]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "daily_instability" / "daily_index_v2.parquet"
CLASSIFIED_PATH = PROJECT_ROOT / "data" / "raw" / "rss" / "articles_classified.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "daily_instability" / "reports" / "samples"


def main():
    parser = argparse.ArgumentParser(description="Generate weekly report samples")
    parser.add_argument("--date", default=None,
                        help="End date for the week (default: latest in data)")
    parser.add_argument("--no-narrative", action="store_true",
                        help="Skip Claude API narrative generation")
    args = parser.parse_args()

    import pandas as pd
    from src.reporting.weekly_report_v2 import generate_all_samples

    # Determine end date
    if args.date:
        end_date = args.date
    else:
        idx = pd.read_parquet(INDEX_PATH)
        end_date = pd.to_datetime(idx["date"]).max().strftime("%Y-%m-%d")
        # Go back to last Sunday
        end_dt = pd.Timestamp(end_date)
        # Use the most recent Saturday as end of reporting week
        days_since_sat = (end_dt.dayofweek + 2) % 7
        if days_since_sat == 0:
            report_end = end_dt
        else:
            report_end = end_dt - pd.Timedelta(days=days_since_sat)
        end_date = report_end.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  QHAWARINA Weekly Report Sample Generator")
    print(f"  Week ending: {end_date}")
    print(f"  Narrative: {'YES (Claude API)' if not args.no_narrative else 'DISABLED'}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    paths = generate_all_samples(
        end_date=end_date,
        index_path=str(INDEX_PATH),
        classified_path=str(CLASSIFIED_PATH),
        output_dir=str(OUTPUT_DIR),
        generate_narrative=not args.no_narrative,
    )

    print(f"\n{'='*60}")
    print(f"  Generated {len(paths)} sample reports:")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:40s} ({size_kb:.0f} KB)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
