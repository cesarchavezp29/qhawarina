"""
=============================================================================
QHAWARINA DAILY PIPELINE — run_daily_pipeline.py
=============================================================================

FULL DOCUMENTATION: D:/nexus/nexus/PIPELINE.md  ← READ THIS FIRST

Quick reference:
  - Runs daily at 21:00 Lima (PET = UTC-5) via Windows Task Scheduler
  - Total runtime: ~30 minutes
  - Log: logs/daily_pipeline_YYYY-MM-DD.log

STEPS (in order):
  1. Supermarket scrape  — Plaza Vea, Metro, Wong (~42k products)
  2. Political index     — RSS feeds + Claude Haiku → IRP + IRE indices
  3. Export web data     — All JSON/CSV for website (export_web_data.py)
  4. Sync to web         — copies exports/ → D:/qhawarina/public/assets/data/
  5. Monthly poverty     — runs on 15th only (GBR nowcast model)
  6. Git push            — disabled by default, use --push flag

KEY OUTPUT: data/processed/daily_instability/daily_index.parquet
  ↑ This is the IRP/IRE source of truth. Updated daily.
  ↑ Do NOT use data/processed/daily/daily_index.parquet (legacy, stuck 2026-03-13)

IRP/IRE SCALE:
  0-50   = quiet   |  50-100 = moderate  |  100-180 = elevated
  180-250 = high   |  250+   = very high (major political shock)

USAGE:
    python scripts/run_daily_pipeline.py                   # standard (no push)
    python scripts/run_daily_pipeline.py --push            # + git commit/push
    python scripts/run_daily_pipeline.py --skip-scrape     # skip supermarket
    python scripts/run_daily_pipeline.py --skip-political  # skip RSS/Haiku
    python scripts/run_daily_pipeline.py --no-push         # (alias for default)

KNOWN BUGS (fixed):
  - .iloc[-1] IRP/IRE partial-day UTC bleed → fixed in export_web_data.py
    (now uses last row with n_articles_total >= 100)

=============================================================================
"""

import argparse
import logging
import os
import smtplib
import ssl
import subprocess
import sys
from datetime import date, datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"daily_pipeline_{date.today().isoformat()}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
    encoding="utf-8",
)
logger = logging.getLogger("nexus.daily")

PYTHON = sys.executable
SCRIPTS = PROJECT_ROOT / "scripts"
WEB_DIR = Path("D:/qhawarina")


def run(label: str, cmd: list, cwd=PROJECT_ROOT, timeout: int = None) -> bool:
    logger.info("-- %s ----------------------------------", label)
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT after %ds: %s — skipping, pipeline continues", timeout, label)
        return False
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
    if result.returncode != 0:
        logger.error("FAILED (%d): %s", result.returncode, label)
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-20:]:
                logger.error("  %s", line)
        return False
    return True


def _load_env() -> dict:
    """Load .env from project root into a dict."""
    env = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                v = v.split(" #")[0].strip().strip('"').strip("'")
                env[k.strip()] = v
    return env


def send_success_email(today: date, results: dict) -> None:
    """Send daily summary email with IRP/IRE values and chart images attached."""
    env = _load_env()
    addr = env.get("GMAIL_ADDRESS") or os.environ.get("GMAIL_ADDRESS", "")
    pwd  = (env.get("GMAIL_APP_PASSWORD") or os.environ.get("GMAIL_APP_PASSWORD", "")).replace(" ", "")
    if not addr or not pwd:
        logger.warning("GMAIL credentials not found — skipping success email")
        return

    # Load IRP/IRE values from political_index_daily.json
    irp_val = ire_val = irp_level = ire_level = "?"
    top_drivers = []
    try:
        import json
        data_path = PROJECT_ROOT / "exports" / "data" / "political_index_daily.json"
        if data_path.exists():
            data = json.loads(data_path.read_text(encoding="utf-8"))
            cur = data.get("current", {})
            irp_val   = f"{cur.get('political_raw', cur.get('prr', '?')):.1f}"
            ire_val   = f"{cur.get('economic_raw',  cur.get('ere', '?')):.1f}"
            irp_level = cur.get("political_raw_level", cur.get("level", "?"))
            ire_level = cur.get("economic_raw_level",  cur.get("economic_level", "?"))
            top_drivers = cur.get("top_political_drivers", [])[:5]
    except Exception as e:
        logger.warning("Could not read index data for email: %s", e)

    steps_summary = "\n".join(
        f"  {'✅' if ok else '❌'} {step}" for step, ok in results.items()
    )
    drivers_text = ""
    if top_drivers:
        drivers_text = "\nTop drivers IRP:\n" + "\n".join(
            f"  [{d.get('score','?')}] {d.get('title','')[:80]}"
            for d in top_drivers if isinstance(d, dict)
        )

    subject = f"[Qhawarina] Pipeline OK — {today.isoformat()} | IRP={irp_val} {irp_level} | IRE={ire_val} {ire_level}"
    body = f"""Pipeline diario completado exitosamente.

Fecha   : {today.isoformat()}
IRP     : {irp_val} — {irp_level}
IRE     : {ire_val} — {ire_level}
{drivers_text}

Pasos:
{steps_summary}

— Qhawarina automated monitor
"""

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = addr
    msg["To"]      = addr
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Attach IRP and IRE chart images
    images_dir = PROJECT_ROOT / "output" / "twitter_images"
    for name, fname in [("IRP", f"irp_{today}.png"), ("IRE", f"ire_{today}.png")]:
        img_path = images_dir / fname
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    img = MIMEImage(f.read(), name=fname)
                    img.add_header("Content-Disposition", "attachment", filename=fname)
                    msg.attach(img)
                logger.info("Attached %s chart to email", name)
            except Exception as e:
                logger.warning("Could not attach %s: %s", fname, e)

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(addr, pwd)
            server.sendmail(addr, [addr], msg.as_bytes())
        logger.info("Success email sent to %s", addr)
    except Exception as e:
        logger.error("Failed to send success email: %s", e)


def step_scrape_supermarket() -> bool:
    """Step 1: Scrape supermarket prices and save snapshot.

    Skips if snapshot already exists for today (idempotent).
    Retries up to 3 times with 5-minute delays on failure.
    """
    import time
    from pathlib import Path as _Path
    snap_path = PROJECT_ROOT / "data" / "raw" / "supermarket" / "snapshots" / f"{date.today().isoformat()}.parquet"
    if snap_path.exists():
        logger.info("Snapshot for %s already exists (%s) — skipping scrape", date.today(), snap_path.name)
        return True

    script = f"""
import sys, pandas as pd
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.ingestion.supermarket import SupermarketScraper
from datetime import date
today = date.today()
scraper = SupermarketScraper()
df = scraper.scrape_all_stores(today)
if not df.empty:
    scraper.save_daily_snapshot(df, today)
    by_store = df.groupby('store')['product_name'].count()
    print(f'Saved {{len(df)}} products for {{today}}')
    print(by_store.to_string())
else:
    print('WARNING: scraper returned empty dataframe')
    sys.exit(1)
"""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        logger.info("Scrape attempt %d/%d", attempt, max_attempts)
        ok = run(
            "Supermarket scrape",
            [PYTHON, "-c", script],
            timeout=1800,  # 30 min max — never block the rest of the pipeline
        )
        if ok:
            return True
        if attempt < max_attempts:
            logger.warning("Scrape failed — waiting 5 minutes before retry...")
            time.sleep(300)

    # All attempts failed — send alert
    logger.error("Supermarket scrape failed after %d attempts", max_attempts)
    run(
        "Send failure alert",
        [PYTHON, str(SCRIPTS / "send_pipeline_alert.py"),
         "run_daily_pipeline",
         f"Supermarket scrape failed after {max_attempts} attempts on {date.today().isoformat()}"],
    )
    return False


def step_political_index() -> bool:
    """Step 2: Rebuild political instability index."""
    return run(
        "Political index",
        [PYTHON, str(SCRIPTS / "build_daily_index.py")],
    )


def step_export() -> bool:
    """Step 3: Export all web data (includes price index rebuild)."""
    return run(
        "Export web data",
        [PYTHON, str(SCRIPTS / "export_web_data.py")],
    )


def step_sync() -> bool:
    """Step 4: Sync exports to qhawarina web project."""
    return run(
        "Sync to web",
        [PYTHON, str(SCRIPTS / "sync_web_data.py")],
    )


def step_monthly_poverty() -> bool:
    """
    Monthly poverty nowcast — runs on the 15th of each month.

    Sequence:
      1. regional_download — fetch latest departmental BCRP series
      2. regional          — rebuild panel_departmental_monthly.parquet
      3. build_poverty_nowcast.py  — rebuild vintage_panel + monthly_indicators
      4. generate_poverty_nowcast.py — produce poverty_nowcast.json + copy to web

    Returns True even if some sub-steps warn (non-fatal), False only on hard failure.
    """
    today_dt = datetime.now()
    if today_dt.day != 15:
        logger.info("Monthly poverty: skipping (today is the %dth, runs on the 15th)", today_dt.day)
        return True  # not an error — just not time yet

    logger.info("=" * 60)
    logger.info("MONTHLY POVERTY NOWCAST — %s", today_dt.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    # Step 1: Fetch latest regional BCRP data
    ok1 = run(
        "Regional BCRP download",
        [PYTHON, str(SCRIPTS / "update_nexus.py"), "--only", "regional_download"],
    )
    if not ok1:
        logger.warning("Regional download failed — poverty nowcast will use existing data")

    # Step 2: Rebuild departmental monthly panel
    ok2 = run(
        "Regional panel rebuild",
        [PYTHON, str(SCRIPTS / "update_nexus.py"), "--only", "regional"],
    )
    if not ok2:
        logger.warning("Regional panel rebuild failed — poverty nowcast may be stale")

    # Step 3: Rebuild vintage panel + monthly indicators
    ok3 = run(
        "Build poverty vintage panel",
        [PYTHON, str(SCRIPTS / "build_poverty_nowcast.py")],
    )
    if not ok3:
        logger.error("build_poverty_nowcast.py FAILED — aborting nowcast generation")
        return False

    # Step 4: Generate nowcast + export JSON to web
    ok4 = run(
        "Generate poverty nowcast",
        [PYTHON, str(SCRIPTS / "generate_poverty_nowcast.py")],
    )
    if not ok4:
        logger.error("generate_poverty_nowcast.py FAILED")
        return False

    logger.info("Monthly poverty nowcast complete.")
    return True


def step_git_push(today: date, include_poverty: bool = False) -> bool:
    """Step 5: Git commit + push in qhawarina repo."""
    if include_poverty:
        msg = f"Daily update {today.isoformat()}: prices + political index + poverty nowcast"
    else:
        msg = f"Daily update {today.isoformat()}: prices + political index"

    # Pull before staging so we don't diverge from remote
    subprocess.run(["git", "pull", "--rebase", "origin", "master"],
                   cwd=WEB_DIR, capture_output=True)

    ok = run("Git add", ["git", "add", "public/assets/data/"], cwd=WEB_DIR)
    if not ok:
        return False

    # Check if there's anything to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        cwd=WEB_DIR, capture_output=True, text=True
    )
    if not result.stdout.strip():
        logger.info("  No changes to commit (data unchanged)")
        return True

    ok = run(
        "Git commit",
        ["git", "commit", "-m", msg],
        cwd=WEB_DIR,
    )
    if not ok:
        return False

    # Push with one retry: if rejected, pull --rebase and try again
    result = subprocess.run(["git", "push"], cwd=WEB_DIR, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    logger.warning("  Git push rejected — pulling --rebase and retrying")
    subprocess.run(["git", "pull", "--rebase", "origin", "master"],
                   cwd=WEB_DIR, capture_output=True)
    return run("Git push (retry)", ["git", "push"], cwd=WEB_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-political", action="store_true")
    parser.add_argument("--no-push", action="store_true")  # kept for backward compat
    parser.add_argument("--push", action="store_true", help="Push to GitHub after export (off by default)")
    args = parser.parse_args()

    today = date.today()
    logger.info("=" * 60)
    logger.info("QHAWARINA DAILY PIPELINE — %s", today.isoformat())
    logger.info("=" * 60)

    results = {}

    if not args.skip_scrape:
        results["scrape"] = step_scrape_supermarket()
    else:
        logger.info("Skipping supermarket scrape (--skip-scrape)")

    if not args.skip_political:
        results["political"] = step_political_index()
    else:
        logger.info("Skipping political index (--skip-political)")

    results["export"] = step_export()
    results["sync"] = step_sync()

    poverty_ran = False
    poverty_result = step_monthly_poverty()  # has its own day-15 guard; returns True on non-15th
    if datetime.now().day == 15:
        results["poverty"] = poverty_result
        poverty_ran = poverty_result

    if args.push:
        results["push"] = step_git_push(today, include_poverty=poverty_ran)
    else:
        logger.info("Skipping git push (auto-push disabled — run with --push to deploy)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %s", datetime.now().strftime("%H:%M:%S"))
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %-12s %s", step, status)
    logger.info("Log: %s", log_file)
    logger.info("=" * 60)

    failed = [k for k, v in results.items() if not v]
    if failed:
        run(
            "Send failure alert",
            [PYTHON, str(SCRIPTS / "send_pipeline_alert.py"),
             "run_daily_pipeline",
             f"Steps failed on {today.isoformat()}: {', '.join(failed)}"],
        )
    else:
        send_success_email(today, results)

    # ── Publicar tweets IRP e IRE después del git push ─────────────────────
    import subprocess, time as _time
    logger.info("[PIPELINE] Publicando tweet IRP...")
    subprocess.run([sys.executable, str(SCRIPTS / "tweet_irp.py"), "--no-delay"])
    _time.sleep(120)
    logger.info("[PIPELINE] Publicando tweet IRE...")
    subprocess.run([sys.executable, str(SCRIPTS / "tweet_ire.py"), "--no-delay"])
    logger.info("[PIPELINE] Tweets publicados.")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
