"""Pipeline validation gates + Gmail alerting for Qhawarina.

Validates data quality after each scrape step and sends a daily
summary email (success or failure) at the end of the pipeline run.

Usage:
    python scripts/validate_pipeline.py --check supermarket
    python scripts/validate_pipeline.py --check rss
    python scripts/validate_pipeline.py --alert email

Exit codes:
    0 = validation passed (or alert sent)
    1 = validation failed
"""

import argparse
import json
import logging
import os
import smtplib
import ssl
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pipeline_validation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.validate")

STATUS_PATH = PROJECT_ROOT / "data" / "pipeline_status.json"
RAW_SUPERMARKET_DIR = PROJECT_ROOT / "data" / "raw" / "supermarket"
RAW_RSS_DIR = PROJECT_ROOT / "data" / "raw" / "rss"
WEBSITE_DATA = Path(os.environ.get("QHAWARINA_WEB_DATA", "D:/qhawarina/public/assets/data"))


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _load_status() -> dict:
    """Load existing pipeline_status.json or return empty dict."""
    if STATUS_PATH.exists():
        try:
            with open(STATUS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_status(status: dict) -> None:
    """Write pipeline_status.json."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    logger.info("  Status saved: %s", STATUS_PATH)


# ---------------------------------------------------------------------------
# Validation: supermarket scrape
# ---------------------------------------------------------------------------

def validate_supermarket_scrape() -> tuple[bool, list[str]]:
    """Validate today's supermarket snapshot quality.

    Checks:
        1. Snapshot file for today exists
        2. Product count ≥70% of yesterday's count
        3. At least 2 out of 3 stores scraped successfully
        4. No negative prices
        5. Median price change overnight <10%

    Returns:
        (passed: bool, errors: list[str])
    """
    errors: list[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    snap_dir = RAW_SUPERMARKET_DIR / "snapshots"
    today_snap = snap_dir / f"{today}.parquet"
    yesterday_snap = snap_dir / f"{yesterday}.parquet"

    # 1. Snapshot for today must exist
    if not today_snap.exists():
        errors.append(
            f"No snapshot for today ({today}) — scraper may have failed or not yet run"
        )
        return False, errors  # cannot check further without today's data

    try:
        import pandas as pd

        df_today = pd.read_parquet(today_snap)
        n_today = len(df_today)
        logger.info("  Today's snapshot: %d products", n_today)

        # 2. Product count ≥70% of yesterday
        if yesterday_snap.exists():
            df_yest = pd.read_parquet(yesterday_snap)
            n_yest = len(df_yest)
            if n_yest > 0 and n_today < n_yest * 0.70:
                errors.append(
                    f"Product count dropped >30%: {n_today:,} today vs {n_yest:,} yesterday "
                    f"({n_today / n_yest:.0%})"
                )
        else:
            logger.info("  No yesterday snapshot to compare — skipping count check")

        # 3. At least 2/3 stores active
        store_col = next((c for c in ("store", "source", "tienda") if c in df_today.columns), None)
        if store_col:
            active_stores = df_today[store_col].nunique()
            if active_stores < 2:
                errors.append(
                    f"Only {active_stores}/3 stores scraped today (need ≥2). "
                    f"Active: {sorted(df_today[store_col].unique())}"
                )
        else:
            logger.warning("  No store/source column found — skipping store count check")

        # 4. No negative prices
        price_col = next((c for c in ("price", "precio", "value") if c in df_today.columns), None)
        if price_col:
            n_neg = (df_today[price_col] < 0).sum()
            if n_neg > 0:
                errors.append(f"{n_neg} negative prices in today's snapshot")

            # 5. Median price change <10% overnight
            if yesterday_snap.exists() and price_col in df_yest.columns:
                med_today = df_today[price_col].median()
                med_yest = df_yest[price_col].median()
                if med_yest > 0:
                    change = abs(med_today / med_yest - 1)
                    if change > 0.10:
                        errors.append(
                            f"Median price changed {change:.1%} overnight "
                            f"(S/{med_yest:.2f} → S/{med_today:.2f}). "
                            f"Threshold: 10% — possible scraper error"
                        )

    except Exception as exc:
        errors.append(f"Could not read snapshot parquet: {exc}")

    passed = len(errors) == 0
    return passed, errors


# ---------------------------------------------------------------------------
# Validation: RSS / political index scrape
# ---------------------------------------------------------------------------

def validate_rss_scrape() -> tuple[bool, list[str]]:
    """Validate RSS articles cache quality.

    Checks:
        1. articles_cache.parquet exists
        2. ≥10 articles published today
        3. ≥3 different news sources active today
        4. ≥5 articles in the last 48 hours

    Returns:
        (passed: bool, errors: list[str])
    """
    errors: list[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    cutoff_48h = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%d")

    cache_path = RAW_RSS_DIR / "articles_cache.parquet"

    if not cache_path.exists():
        errors.append(
            f"articles_cache.parquet not found at {cache_path} — RSS fetch may have failed"
        )
        return False, errors

    try:
        import pandas as pd

        df = pd.read_parquet(cache_path)

        if "published" not in df.columns:
            errors.append("articles_cache.parquet missing 'published' column")
            return False, errors

        # Normalize published date to YYYY-MM-DD string
        df["_date"] = pd.to_datetime(df["published"], errors="coerce").dt.strftime("%Y-%m-%d")

        # 1. ≥10 articles today
        n_today = (df["_date"] == today).sum()
        logger.info("  Articles today: %d", n_today)
        if n_today < 10:
            errors.append(
                f"Only {n_today} articles published today (need ≥10) — RSS feed may be stale"
            )

        # 2. ≥3 sources active today
        source_col = next((c for c in ("source", "feed_name") if c in df.columns), None)
        if source_col:
            df_today = df[df["_date"] == today]
            active_sources = df_today[source_col].nunique() if not df_today.empty else 0
            logger.info("  Active sources today: %d", active_sources)
            if active_sources < 3:
                errors.append(
                    f"Only {active_sources} sources active today (need ≥3). "
                    f"Active: {sorted(df_today[source_col].unique()) if not df_today.empty else []}"
                )
        else:
            logger.warning("  No 'source' column found in articles cache")

        # 3. ≥5 articles in last 48h
        n_48h = (df["_date"] >= cutoff_48h).sum()
        if n_48h < 5:
            errors.append(
                f"Only {n_48h} articles in last 48h (need ≥5) — feeds may be broken"
            )

    except Exception as exc:
        errors.append(f"Could not read articles cache: {exc}")

    passed = len(errors) == 0
    return passed, errors


# ---------------------------------------------------------------------------
# Email alerting
# ---------------------------------------------------------------------------

def send_email_alert(status: dict) -> None:
    """Send a Gmail summary of the pipeline run.

    Requires env vars:
        GMAIL_ADDRESS      — sender Gmail address
        GMAIL_APP_PASSWORD — app-specific password (Gmail 2FA)
        ALERT_EMAIL        — recipient (defaults to GMAIL_ADDRESS)
    """
    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    recipient = os.environ.get("ALERT_EMAIL", gmail_address)

    if not gmail_address or not gmail_password:
        logger.warning(
            "GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set — skipping email alert"
        )
        return

    today = datetime.now().strftime("%Y-%m-%d")
    run_time = datetime.now().strftime("%H:%M:%S")

    # Collect all errors across checks
    all_errors: list[str] = []
    for check_key, check_val in status.items():
        if isinstance(check_val, dict) and not check_val.get("passed", True):
            for err in check_val.get("errors", []):
                all_errors.append(f"[{check_key}] {err}")

    n_errors = len(all_errors)

    if n_errors == 0:
        subject = f"Qhawarina {today} — Pipeline OK"
        body_header = "PIPELINE OK — No errors detected."
    elif n_errors == 1:
        subject = f"Qhawarina {today} — 1 error en pipeline"
        body_header = "PIPELINE ERROR — 1 issue detected."
    else:
        subject = f"Qhawarina {today} — {n_errors} errores en pipeline"
        body_header = f"PIPELINE ERRORS — {n_errors} issues detected."

    lines = [
        body_header,
        "",
        f"Date      : {today}",
        f"Run time  : {run_time} (PET)",
        "",
        "CHECK RESULTS " + "─" * 40,
    ]

    for check_key, check_val in status.items():
        if isinstance(check_val, dict):
            passed = check_val.get("passed", True)
            icon = "OK  " if passed else "FAIL"
            lines.append(f"  [{icon}] {check_key}")
            for err in check_val.get("errors", []):
                lines.append(f"        • {err}")
        elif check_key not in ("date", "run_time"):
            lines.append(f"  {check_key}: {check_val}")

    if all_errors:
        lines += [
            "",
            "ERROR DETAILS " + "─" * 39,
        ]
        for err in all_errors:
            lines.append(f"  • {err}")

    lines += [
        "",
        "─" * 53,
        "Qhawarina automated pipeline monitor",
        "qhawarina.pe",
    ]

    body = "\n".join(lines)
    raw_message = (
        f"Subject: {subject}\r\n"
        f"From: {gmail_address}\r\n"
        f"To: {recipient}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{body}"
    )

    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, [recipient], raw_message.encode("utf-8"))
        logger.info("  Email sent to %s — %s", recipient, subject)
    except Exception as exc:
        logger.error("  Email send failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qhawarina pipeline validation + email alerting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check",
        choices=["supermarket", "rss"],
        help="Run a specific validation check and update pipeline_status.json",
    )
    parser.add_argument(
        "--alert",
        choices=["email"],
        help="Send alert based on pipeline_status.json",
    )
    args = parser.parse_args()

    if not args.check and not args.alert:
        parser.print_help()
        return 1

    today = datetime.now().strftime("%Y-%m-%d")
    status = _load_status()

    # Reset status on a new day
    if status.get("date") != today:
        status = {"date": today}

    if args.check == "supermarket":
        logger.info("=" * 60)
        logger.info("SUPERMARKET VALIDATION")
        logger.info("=" * 60)
        passed, errors = validate_supermarket_scrape()
        status["supermarket"] = {"passed": passed, "errors": errors}
        status["run_time"] = datetime.now().isoformat()
        _save_status(status)

        if passed:
            logger.info("  PASSED — supermarket scrape looks good")
        else:
            logger.warning("  FAILED — %d issue(s):", len(errors))
            for err in errors:
                logger.warning("    • %s", err)

        return 0 if passed else 1

    elif args.check == "rss":
        logger.info("=" * 60)
        logger.info("RSS VALIDATION")
        logger.info("=" * 60)
        passed, errors = validate_rss_scrape()
        status["rss"] = {"passed": passed, "errors": errors}
        status["run_time"] = datetime.now().isoformat()
        _save_status(status)

        if passed:
            logger.info("  PASSED — RSS articles look good")
        else:
            logger.warning("  FAILED — %d issue(s):", len(errors))
            for err in errors:
                logger.warning("    • %s", err)

        return 0 if passed else 1

    elif args.alert == "email":
        logger.info("=" * 60)
        logger.info("EMAIL ALERT")
        logger.info("=" * 60)
        status["run_time"] = datetime.now().isoformat()
        _save_status(status)

        # Copy pipeline_status.json to website
        if WEBSITE_DATA.exists():
            import shutil
            dest = WEBSITE_DATA / "pipeline_status.json"
            shutil.copy2(STATUS_PATH, dest)
            logger.info("  Copied pipeline_status.json → %s", dest)
        else:
            logger.warning("  Website data dir not found: %s", WEBSITE_DATA)

        send_email_alert(status)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
