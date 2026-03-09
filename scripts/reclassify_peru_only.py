"""
Reclassify only non-GDELT (Peru-source) articles with dual political+economic scores.
Saves every 500 articles. Resume-safe: skips articles where pol != 0 OR eco != 0.

IMPORTANT: GDELT articles (zeroed to 0.0 by Level 2 fix) are intentionally excluded.
Only elcomercio, gestion, larepublica, andina, rpp, correo and archive feeds get scored.
"""
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import pandas as pd
from src.utils.io import save_parquet
from src.nlp.classifier import classify_articles_dual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "reclassify_peru.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("reclassify_peru")

PARQUET = PROJECT_ROOT / "data" / "raw" / "rss" / "articles_classified.parquet"
CHUNK_SIZE = 500
BATCH_SIZE = 20
DELAY = 0.3


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. ABORTING.")
        return 1

    df = pd.read_parquet(PARQUET)
    log.info("Loaded %d articles total", len(df))

    # Only process non-GDELT articles
    is_gdelt = df["feed_name"].str.contains("gdelt", case=False, na=False)
    log.info("GDELT (skip): %d | Non-GDELT (to score): %d", is_gdelt.sum(), (~is_gdelt).sum())

    # Among non-GDELT: find articles that need scoring
    # An article needs scoring if BOTH scores are 0.0 (may be falsely zeroed)
    # Articles with any score > 0 are already correctly scored — skip them.
    non_gdelt_idx = df.index[~is_gdelt]
    non_gdelt = df.loc[non_gdelt_idx]

    needs_mask = (non_gdelt["political_score"] == 0.0) & (non_gdelt["economic_score"] == 0.0)
    already_nonzero = (~needs_mask).sum()
    to_classify_idx = non_gdelt_idx[needs_mask].tolist()
    total = len(to_classify_idx)

    log.info("Non-GDELT already nonzero (skip): %d | Need scoring: %d", already_nonzero, total)

    if total == 0:
        log.info("All non-GDELT articles already scored. Done.")
        return 0

    classified_count = 0
    t0 = time.time()

    for chunk_start in range(0, total, CHUNK_SIZE):
        chunk_idx = to_classify_idx[chunk_start: chunk_start + CHUNK_SIZE]
        chunk_df = df.loc[chunk_idx].copy()

        log.info("=== Chunk %d-%d / %d ===", chunk_start + 1, chunk_start + len(chunk_idx), total)

        try:
            classified = classify_articles_dual(
                chunk_df,
                batch_size=BATCH_SIZE,
                model="claude-haiku-4-5-20251001",
                delay=DELAY,
            )
        except RuntimeError as e:
            log.error("Classification FAILED: %s", e)
            log.error("Parquet NOT updated for this chunk. Stopping.")
            return 1

        # Sanity check
        both_zero = (
            (classified["political_score"].fillna(0) == 0) &
            (classified["economic_score"].fillna(0) == 0)
        ).mean()
        if both_zero > 0.98 and len(classified) > 100:
            log.error("ABORT: %.0f%% scored 0 on both. Likely API issue.", both_zero * 100)
            return 1

        # Merge back
        df.loc[chunk_idx, "political_score"] = pd.to_numeric(
            classified["political_score"], errors="coerce"
        ).values
        df.loc[chunk_idx, "economic_score"] = pd.to_numeric(
            classified["economic_score"], errors="coerce"
        ).values
        classified_count += len(chunk_idx)

        # Save after every chunk
        save_parquet(df, PARQUET)
        elapsed = time.time() - t0
        rate = classified_count / elapsed * 60
        remaining_min = (total - classified_count) / (rate / 60) if rate > 0 else 0
        log.info(
            "  Saved. Progress: %d/%d (%.1f%%) | %.0f/min | ETA: %.1f min",
            classified_count, total,
            100 * classified_count / total,
            rate,
            remaining_min / 60,
        )

    # Final summary
    non_gdelt_final = df.loc[non_gdelt_idx]
    log.info("=== RECLASSIFICATION COMPLETE ===")
    log.info("Total classified: %d", classified_count)
    log.info("political_score mean (non-GDELT): %.1f", non_gdelt_final["political_score"].mean())
    log.info("political nonzero (non-GDELT): %d", (non_gdelt_final["political_score"] > 0).sum())
    log.info("economic nonzero (non-GDELT): %d", (non_gdelt_final["economic_score"] > 0).sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
