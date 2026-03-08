"""Generate Qhawarina PDF instability reports.

Usage:
    python scripts/generate_report.py --type daily --date 2026-01-21
    python scripts/generate_report.py --type weekly --date 2026-01-21
    python scripts/generate_report.py --type monthly --date 2026-01
    python scripts/generate_report.py --type daily --last 7
    python scripts/generate_report.py --type daily --date 2026-01-21 --narrative
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
logger = logging.getLogger("nexus.generate_report")

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"


def main():
    parser = argparse.ArgumentParser(description="Generate Qhawarina PDF Reports")
    parser.add_argument(
        "--type", type=str, required=True, choices=["daily", "weekly", "monthly"],
        help="Report type: daily, weekly, or monthly",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD for daily/weekly, YYYY-MM for monthly)",
    )
    parser.add_argument(
        "--last", type=int, default=None,
        help="Generate reports for the last N periods",
    )
    parser.add_argument(
        "--narrative", action="store_true",
        help="Include Claude-generated narrative summary",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
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

    from src.reporting.pdf_report import (
        generate_daily_pdf,
        generate_weekly_pdf,
        generate_monthly_pdf,
    )

    base_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DAILY_DIR / "reports"

    # Build list of targets
    today = datetime.now().date()

    if args.type == "daily":
        output_dir = base_dir / "daily"
        if args.last:
            dates = [(today - timedelta(days=i)).isoformat() for i in range(args.last)]
            dates.reverse()
        elif args.date:
            dates = [args.date]
        else:
            dates = [today.isoformat()]

        for date in dates:
            logger.info("Generating daily report for %s...", date)
            narrative = _get_narrative(date, args.narrative, index_path, classified_path)
            out = output_dir / f"report_{date}.pdf"
            generate_daily_pdf(date, index_path, classified_path, out, LOGO_PATH, narrative)
            logger.info("  Saved: %s", out)

    elif args.type == "weekly":
        output_dir = base_dir / "weekly"
        if args.last:
            targets = []
            for i in range(args.last):
                d = today - timedelta(weeks=i)
                yr, wk, _ = d.isocalendar()
                targets.append((yr, wk))
            targets.reverse()
        elif args.date:
            d = datetime.strptime(args.date[:10], "%Y-%m-%d").date()
            yr, wk, _ = d.isocalendar()
            targets = [(yr, wk)]
        else:
            yr, wk, _ = today.isocalendar()
            targets = [(yr, wk)]

        for year, week in targets:
            logger.info("Generating weekly report for %d-W%02d...", year, week)
            narrative = _get_weekly_narrative(
                year, week, args.narrative, index_path, classified_path,
            )
            out = output_dir / f"report_{year}-W{week:02d}.pdf"
            generate_weekly_pdf(year, week, index_path, classified_path, out, LOGO_PATH, narrative)
            logger.info("  Saved: %s", out)

    elif args.type == "monthly":
        output_dir = base_dir / "monthly"
        if args.last:
            targets = []
            for i in range(args.last):
                d = today.replace(day=1) - timedelta(days=i * 28)
                targets.append((d.year, d.month))
            # Dedup and sort
            targets = sorted(set(targets))
        elif args.date:
            parts = args.date.split("-")
            targets = [(int(parts[0]), int(parts[1]))]
        else:
            targets = [(today.year, today.month)]

        for year, month in targets:
            logger.info("Generating monthly report for %d-%02d...", year, month)
            narrative = _get_monthly_narrative(
                year, month, args.narrative, index_path, classified_path,
            )
            out = output_dir / f"report_{year}-{month:02d}.pdf"
            generate_monthly_pdf(year, month, index_path, classified_path, out, LOGO_PATH, narrative)
            logger.info("  Saved: %s", out)

    return 0


def _get_narrative(date, enabled, index_path, classified_path):
    """Get Claude narrative for a daily report."""
    if not enabled:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — skipping narrative")
        return None
    try:
        from src.reporting.daily_report import generate_narrative
        return generate_narrative(date, index_path, classified_path)
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return None


def _get_weekly_narrative(year, week, enabled, index_path, classified_path):
    """Get Claude narrative for a weekly report."""
    if not enabled:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — skipping narrative")
        return None
    try:
        from src.reporting.daily_report import generate_period_narrative
        start = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u").date()
        end = start + timedelta(days=6)
        return generate_period_narrative(
            str(start), str(end), index_path, classified_path, "semanal",
        )
    except Exception as e:
        logger.warning("Weekly narrative failed: %s", e)
        return None


def _get_monthly_narrative(year, month, enabled, index_path, classified_path):
    """Get Claude narrative for a monthly report."""
    if not enabled:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — skipping narrative")
        return None
    try:
        import calendar
        from src.reporting.daily_report import generate_period_narrative
        _, last_day = calendar.monthrange(year, month)
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day:02d}"
        return generate_period_narrative(
            start, end, index_path, classified_path, "mensual",
        )
    except Exception as e:
        logger.warning("Monthly narrative failed: %s", e)
        return None


if __name__ == "__main__":
    sys.exit(main())
