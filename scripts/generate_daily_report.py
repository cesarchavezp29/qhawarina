"""Generate daily instability report.

Usage:
    python scripts/generate_daily_report.py                     # today's report
    python scripts/generate_daily_report.py --date 2026-01-21   # specific date
    python scripts/generate_daily_report.py --narrative          # include Claude summary
    python scripts/generate_daily_report.py --last 7             # last 7 days
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DAILY_DIR, RAW_RSS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nexus.daily_report")


def main():
    parser = argparse.ArgumentParser(description="Generate Daily Instability Report")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to report on (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--last", type=int, default=None,
        help="Generate reports for the last N days",
    )
    parser.add_argument(
        "--narrative", action="store_true",
        help="Include Claude-generated narrative summary",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save reports (default: data/processed/daily_instability/reports/)",
    )
    args = parser.parse_args()

    index_path = PROCESSED_DAILY_DIR / "daily_index.parquet"
    classified_path = RAW_RSS_DIR / "articles_classified.parquet"

    if not index_path.exists():
        logger.error("Daily index not found: %s", index_path)
        logger.error("Run 'python scripts/build_daily_index.py' first.")
        return 1

    if not classified_path.exists():
        logger.error("Classified articles not found: %s", classified_path)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DAILY_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.reporting.daily_report import generate_report, generate_narrative

    # Determine dates
    if args.last:
        today = datetime.now().date()
        dates = [(today - timedelta(days=i)).isoformat() for i in range(args.last)]
        dates.reverse()
    elif args.date:
        dates = [args.date]
    else:
        dates = [datetime.now().date().isoformat()]

    for date in dates:
        logger.info("Generating report for %s...", date)

        narrative = None
        if args.narrative:
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if has_key:
                try:
                    narrative = generate_narrative(date, index_path, classified_path)
                    logger.info("  Narrative generated")
                except Exception as e:
                    logger.warning("  Narrative generation failed: %s", e)
            else:
                logger.warning("  ANTHROPIC_API_KEY not set — skipping narrative")

        report = generate_report(date, index_path, classified_path, narrative=narrative)

        report_path = output_dir / f"report_{date}.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("  Saved: %s", report_path)

        # Also print to stdout
        print(report)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
