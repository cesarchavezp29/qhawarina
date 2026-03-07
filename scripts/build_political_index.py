"""Build the Political Instability Index — full pipeline.

Steps:
1. Fetch Wikipedia pages via MediaWiki API
2. Extract dated political events
3. Deduplicate events
4. Build cabinet timeline (PCM premiers)
5. Match scraped events against ground truth
6. Classify events with Claude API (severity 1-3)
7. Compute cabinet stability component
8. Build monthly + weekly composite index
9. Save all outputs

Usage:
    python scripts/build_political_index.py                    # full pipeline
    python scripts/build_political_index.py --skip-fetch       # use cached pages
    python scripts/build_political_index.py --skip-claude      # skip Claude classification
    python scripts/build_political_index.py --validate-only    # just metrics
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    PROCESSED_NATIONAL_DIR,
    PROCESSED_POLITICAL_DIR,
    RAW_POLITICAL_DIR,
)
from src.ingestion.political import (
    deduplicate_events,
    extract_all_events,
    extract_cabinet_timeline,
    fetch_all_pages,
    load_ground_truth,
    match_all_events,
)
from src.processing.cabinet_stability import (
    compute_cabinet_instability,
)
from src.processing.financial_component import (
    build_confidence_score,
    build_financial_score,
)
from src.processing.political_index import (
    build_monthly_index,
    build_weekly_index,
    prepare_events_monthly,
    prepare_events_weekly,
)
from src.utils.io import save_parquet

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "political_index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nexus.political_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Build Political Instability Index")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Use cached Wikipedia pages from previous run")
    parser.add_argument("--skip-claude", action="store_true",
                        help="Skip Claude API classification (use GT fallback)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Just compute validation metrics from existing data")
    args = parser.parse_args()

    t0 = time.time()
    PROCESSED_POLITICAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_POLITICAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ground truth ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ground truth")
    gt = load_ground_truth()
    logger.info("  %d events, %d high-impact, %d presidents",
                len(gt["events"]), len(gt["high_impact"]), len(gt["presidents"]))

    # ── Step 2: Fetch Wikipedia pages ────────────────────────────────────
    cached_pages_path = RAW_POLITICAL_DIR / "wikipedia_pages_cache.parquet"

    if args.skip_fetch and cached_pages_path.exists():
        logger.info("STEP 2: Loading cached Wikipedia pages")
        pages_df = pd.read_parquet(cached_pages_path)
        pages = pages_df.to_dict("records")
    else:
        logger.info("=" * 60)
        logger.info("STEP 2: Fetching %d Wikipedia pages via API", 35)
        pages = fetch_all_pages(delay=0.5)
        # Cache for future runs
        pages_df = pd.DataFrame(pages)
        save_parquet(pages_df, cached_pages_path)
        logger.info("  Cached to %s", cached_pages_path.name)

    # ── Step 3: Extract events ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting political events from text")
    raw_events = extract_all_events(pages)
    logger.info("  Raw events: %d", len(raw_events))

    if raw_events.empty:
        logger.error("No events extracted! Check Wikipedia pages.")
        return 1

    # Save raw
    save_parquet(raw_events, RAW_POLITICAL_DIR / "scraped_events_raw.parquet")

    # ── Step 4: Deduplicate ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Deduplicating events")
    events = deduplicate_events(raw_events)
    logger.info("  After dedup: %d events", len(events))

    # ── Step 5: Build cabinet timeline ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Building cabinet timeline")
    # Find the PCM page text
    pcm_text = ""
    for page in pages:
        if "Presidente_del_Consejo_de_Ministros" in page.get("title", ""):
            pcm_text = page.get("text", "")
            break

    cabinet_timeline = extract_cabinet_timeline(pcm_text)
    save_parquet(cabinet_timeline, RAW_POLITICAL_DIR / "cabinet_timeline.parquet")
    logger.info("  %d premiers in timeline", len(cabinet_timeline))

    # ── Step 6: Match against ground truth ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Matching events against ground truth")
    events = match_all_events(events, gt["events"])

    n_matched = events["ground_truth_event_id"].dropna().nunique()
    recall = n_matched / len(gt["events"]) * 100
    logger.info("  Matched: %d/%d unique GT events (recall=%.1f%%)",
                n_matched, len(gt["events"]), recall)

    # ── Step 7: Claude API Classification ─────────────────────────────────
    if not args.skip_claude and not args.validate_only:
        logger.info("=" * 60)
        logger.info("STEP 7: Classifying events with Claude API")
        try:
            from src.nlp.classifier import classify_events_batch
            events = classify_events_batch(events)
        except ImportError:
            logger.warning("  anthropic not installed — falling back to GT severity")
            args.skip_claude = True

    if args.skip_claude or args.validate_only:
        logger.info("STEP 7: Skipping classification (using GT fallback)")
        if "severity_claude" not in events.columns:
            events["severity_claude"] = 3  # default medium
            events["severity_claude_bin3"] = 2
            events["severity_claude_confidence"] = 0.0
            events["severity_claude_label"] = ""
            # Override with GT where available
            mask = events["severity_gt"].notna()
            gt_to_score5 = {1: 1, 2: 3, 3: 5}
            for idx in events[mask].index:
                gt_val = int(events.loc[idx, "severity_gt"])
                events.loc[idx, "severity_claude"] = gt_to_score5.get(gt_val, 3)
                events.loc[idx, "severity_claude_bin3"] = gt_val

    # ── Step 8: Assign final scores ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Computing final severity scores")

    events["severity_final"] = events["severity_claude"]
    events["severity_final_bin3"] = events["severity_claude_bin3"]

    # Event weight in index: severity / 5
    events["weight_in_index"] = events["severity_final"] / 5.0

    # Assign event_id
    events["event_id"] = range(1, len(events) + 1)
    events["event_type"] = ""  # will be enriched from GT
    events["market_hours"] = pd.NA
    events["constitutional_basis"] = pd.NA

    # Enrich matched events with GT metadata
    for idx, row in events[events["ground_truth_event_id"].notna()].iterrows():
        gt_id = int(row["ground_truth_event_id"])
        gt_row = gt["events"][gt["events"]["event_id"] == gt_id]
        if not gt_row.empty:
            gt_row = gt_row.iloc[0]
            events.loc[idx, "event_type"] = gt_row.get("event_type", "")
            events.loc[idx, "market_hours"] = gt_row.get("market_hours", pd.NA)
            events.loc[idx, "constitutional_basis"] = gt_row.get("constitutional_basis", pd.NA)

    # ── Step 8b: Validation metrics (if GT available) ─────────────────────
    gt_matched = events[events["severity_gt"].notna()]
    if len(gt_matched) > 0:
        correct = (gt_matched["severity_claude_bin3"].astype(int) == gt_matched["severity_gt"].astype(int)).sum()
        accuracy = correct / len(gt_matched) * 100
        diff = abs(gt_matched["severity_claude_bin3"].astype(int) - gt_matched["severity_gt"].astype(int))
        severe = (diff >= 2).sum()
        logger.info("  Claude vs GT: %d/%d correct (%.1f%%), %d severe errors",
                    correct, len(gt_matched), accuracy, severe)

    # Save events
    save_parquet(events, PROCESSED_POLITICAL_DIR / "events.parquet")
    logger.info("  Saved %d events to events.parquet", len(events))

    # ── Step 10: Compute cabinet stability ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10: Computing cabinet stability")

    cabinet_monthly = compute_cabinet_instability(cabinet_timeline, freq="M")
    cabinet_weekly = compute_cabinet_instability(cabinet_timeline, freq="W-FRI")

    save_parquet(cabinet_monthly, PROCESSED_POLITICAL_DIR / "cabinet_stability_monthly.parquet")
    save_parquet(cabinet_weekly, PROCESSED_POLITICAL_DIR / "cabinet_stability_weekly.parquet")

    # ── Step 11: Prepare event scores at monthly/weekly ──────────────────
    logger.info("=" * 60)
    logger.info("STEP 11: Aggregating event scores to monthly/weekly")

    events_monthly = prepare_events_monthly(events)
    events_weekly = prepare_events_weekly(events)

    # ── Step 12: Financial component ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 12: Building financial stress component")

    panel_path = PROCESSED_NATIONAL_DIR / "panel_national_monthly.parquet"
    if panel_path.exists():
        panel_df = pd.read_parquet(panel_path)
        financial_monthly = build_financial_score(panel_df, freq="M")
        logger.info("  Financial score: %d months", len(financial_monthly))
    else:
        logger.warning("  Panel not found at %s — using neutral financial scores", panel_path)
        financial_monthly = pd.DataFrame({
            "date": events_monthly["date"],
            "financial_score": 0.0,
        })

    # Align financial to weekly dates (forward-fill monthly values)
    if not financial_monthly.empty:
        fin_ts = financial_monthly.set_index("date")["financial_score"]
        fin_weekly = fin_ts.reindex(events_weekly["date"], method="ffill").fillna(0.0)
        financial_weekly = pd.DataFrame({
            "date": events_weekly["date"],
            "financial_score": fin_weekly.values,
        })
    else:
        financial_weekly = pd.DataFrame({
            "date": events_weekly["date"],
            "financial_score": 0.0,
        })

    # ── Step 13: Business confidence component ───────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 13: Building business confidence component")

    if panel_path.exists():
        confidence_monthly = build_confidence_score(panel_df, series_id="PD37981AM")
        logger.info("  Confidence score: %d months", len(confidence_monthly))
    else:
        logger.warning("  Panel not found — using neutral confidence scores")
        confidence_monthly = pd.DataFrame({
            "date": events_monthly["date"],
            "confidence_score": 50.0,
        })

    # ── Step 14: Build composite indices ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 14: Building composite indices")

    monthly_index = build_monthly_index(
        events_monthly, financial_monthly, cabinet_monthly, confidence_monthly,
    )
    weekly_index = build_weekly_index(
        events_weekly, financial_weekly, cabinet_weekly,
    )

    save_parquet(monthly_index, PROCESSED_POLITICAL_DIR / "political_index_monthly.parquet")
    save_parquet(weekly_index, PROCESSED_POLITICAL_DIR / "political_index_weekly.parquet")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("POLITICAL INDEX PIPELINE COMPLETE (%.1f min)", elapsed / 60)
    logger.info("=" * 60)
    logger.info("  Events extracted:      %d", len(events))
    logger.info("  GT matched:            %d/%d (recall %.1f%%)",
                n_matched, len(gt["events"]), recall)
    logger.info("  Cabinet premiers:      %d", len(cabinet_timeline))
    logger.info("  Monthly index:         %d months", len(monthly_index))
    logger.info("  Weekly index:          %d weeks", len(weekly_index))
    logger.info("  Output dir:            %s", PROCESSED_POLITICAL_DIR)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
