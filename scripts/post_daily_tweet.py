"""Post daily political risk + financial stress tweet via Twitter/X API.

Reads latest processed data and posts a summary tweet to @qhawarinape.
Designed to run daily after the main pipeline (e.g. 08:30 PET).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import tweepy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("post_daily_tweet")

# ── Paths ──────────────────────────────────────────────────────────────────────
POLITICAL_PATH  = project_root / "exports/data/political_index_daily.json"
FX_PATH         = project_root / "exports/data/fx_interventions.json"

# ── Twitter credentials (loaded from .env) ────────────────────────────────────
from dotenv import load_dotenv
import os
load_dotenv(project_root / ".env")

API_KEY             = os.environ["TWITTER_API_KEY"]
API_SECRET          = os.environ["TWITTER_API_SECRET"]
ACCESS_TOKEN        = os.environ["TWITTER_ACCESS_TOKEN"]
ACCESS_TOKEN_SECRET = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]


def load_political() -> dict:
    with open(POLITICAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_fx() -> dict:
    with open(FX_PATH, encoding="utf-8") as f:
        return json.load(f)


def severity_emoji(severity: str) -> str:
    return {
        "critical": "🔴",
        "high":     "🟠",
        "medium":   "🟡",
        "low":      "🟢",
        "minimal":  "⚪",
    }.get(severity.lower(), "⚪")


def build_tweet(political: dict, fx: dict) -> str:
    # Political index — latest daily entry
    daily = political.get("daily_index", [])
    latest = daily[-1] if daily else {}
    z_score   = latest.get("z_score", 0.0)
    severity  = latest.get("severity", "low")
    date_str  = latest.get("date", datetime.today().strftime("%Y-%m-%d"))

    # Format date nicely
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%-d %b %Y") if sys.platform != "win32" else dt.strftime("%d %b %Y").lstrip("0")
    except Exception:
        date_label = date_str

    emoji = severity_emoji(severity)

    # Financial stress — from political_index composite or fx
    fin_stress = latest.get("financial_stress", None)
    fin_line = f"💹 Estrés financiero: {fin_stress:+.2f}" if fin_stress is not None else ""

    # Exchange rate — latest daily from fx
    fx_daily = fx.get("daily", [])
    tc = None
    if fx_daily:
        for row in reversed(fx_daily):
            if row.get("fx") is not None:
                tc = row["fx"]
                break
    tc_line = f"💵 TC: S/ {tc:.3f}" if tc else ""

    # Severity label
    severity_label = {
        "critical": "Crítico",
        "high":     "Alto",
        "medium":   "Moderado",
        "low":      "Bajo",
        "minimal":  "Mínimo",
    }.get(severity.lower(), severity.capitalize())

    lines = [
        f"📊 Riesgo Político Perú | {date_label}",
        "",
        f"{emoji} Índice: {z_score:+.2f} ({severity_label})",
    ]
    if fin_line:
        lines.append(fin_line)
    if tc_line:
        lines.append(tc_line)
    lines += [
        "",
        "Actualización diaria | qhawarina.pe",
        "#EconomíaPerú #RiesgoPolitico",
    ]

    return "\n".join(lines)


def post_tweet(text: str) -> str:
    client = tweepy.Client(
        consumer_key=API_KEY,
        consumer_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET,
    )
    response = client.create_tweet(text=text)
    tweet_id = response.data["id"]
    logger.info("Tweet posted: https://x.com/qhawarinape/status/%s", tweet_id)
    return tweet_id


def main() -> int:
    logger.info("Loading political index data...")
    try:
        political = load_political()
    except FileNotFoundError:
        logger.error("political_index.json not found at %s", POLITICAL_PATH)
        return 1

    logger.info("Loading FX data...")
    try:
        fx = load_fx()
    except FileNotFoundError:
        logger.warning("fx_interventions.json not found — TC will be omitted")
        fx = {"daily": []}

    tweet = build_tweet(political, fx)
    logger.info("Tweet text:\n%s", tweet)

    if len(tweet) > 280:
        logger.error("Tweet too long (%d chars) — truncate or shorten", len(tweet))
        return 1

    try:
        post_tweet(tweet)
    except tweepy.TweepyException as e:
        logger.error("Twitter API error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
