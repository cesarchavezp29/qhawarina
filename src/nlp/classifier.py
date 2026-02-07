"""Claude API classifier for political event severity.

Replaces XLM-RoBERTa zero-shot (38.6% accuracy) with Claude API
for direct severity classification on the ordinal 1-3 scale matching
ground truth.

Uses batched prompts to classify events efficiently.
"""

import json
import logging
import time

import pandas as pd

logger = logging.getLogger("nexus.nlp.classifier")

# Binning: NLP 1-5 → ordinal 1-3 (kept for backward compatibility)
SCORE_TO_BIN3 = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

# Labels (kept for reference/compatibility)
CANDIDATE_LABELS = [
    "rutina politica",            # Score 1
    "tension politica moderada",  # Score 2
    "conflicto institucional",    # Score 3
    "crisis constitucional",      # Score 4
    "quiebre institucional",      # Score 5
]

LABEL_TO_SCORE = {
    "rutina politica": 1,
    "tension politica moderada": 2,
    "conflicto institucional": 3,
    "crisis constitucional": 4,
    "quiebre institucional": 5,
}

SEVERITY_PROMPT = """You are an expert political analyst specializing in Peru. Classify the severity of each political event on a 1-3 ordinal scale:

SCALE:
1 = LOW: Routine cabinet change, protocol inauguration, minor political tension, interpellation, minor protest
2 = MEDIUM: Vote of confidence, approved censure, significant protest, corruption scandal, arrest of former official, judicial ruling on political case, state of emergency
3 = HIGH: President removed/resigned, Congress dissolved, self-coup, massacre/mass killings, president arrested, major constitutional crisis

IMPORTANT: Consider the broader context of each event's month. An event that might seem routine in isolation (e.g., cabinet reshuffle, investigation opened) should be rated HIGHER if it occurs during a month with massacres, mass protests, or constitutional crises.
{context_block}
EVENTS TO CLASSIFY:
{events_block}

Respond with ONLY a JSON array. Each element: {{"id": <event_number>, "score": <1-3>, "label": "<one of: low/medium/high>"}}
No explanation, no markdown, just the JSON array."""

SINGLE_EVENT_PROMPT = """You are an expert political analyst specializing in Peru. Classify the severity of this political event on a 1-3 ordinal scale:

SCALE:
1 = LOW: Routine cabinet change, protocol inauguration, minor political tension
2 = MEDIUM: Vote of confidence, censure, significant protest, corruption scandal, arrest/trial of politician, state of emergency
3 = HIGH: President removed/resigned, Congress dissolved, self-coup, massacre, major constitutional crisis

EVENT:
Date: {date}
Description: {description}
President: {president}

Respond with ONLY a JSON object: {{"score": <1-3>, "label": "<low/medium/high>"}}"""

# Module-level client (lazy)
_client = None


def _get_client():
    """Lazy-load the Anthropic client."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
        logger.info("Anthropic client initialized")
    return _client


def classify_event(
    text: str,
    date: str = "",
    president: str = "",
    client=None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Classify a single event with Claude API.

    Returns dict with: severity_nlp (1-5 mapped), severity_nlp_confidence,
    severity_nlp_label, severity_nlp_bin3 (1-3).
    """
    if client is None:
        client = _get_client()

    prompt = SINGLE_EVENT_PROMPT.format(
        date=date,
        description=text,
        president=president,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        score3 = max(1, min(3, int(result["score"])))
        # Map bin3 back to 1-5 scale for column compatibility
        bin3_to_5 = {1: 1, 2: 3, 3: 5}
        score5 = bin3_to_5[score3]
        label = result.get("label", "")

        return {
            "severity_nlp": score5,
            "severity_nlp_confidence": 1.0,
            "severity_nlp_label": label,
            "severity_nlp_bin3": score3,
        }
    except Exception as e:
        logger.warning("Classification failed: %s", e)
        return {
            "severity_nlp": 3,
            "severity_nlp_confidence": 0.0,
            "severity_nlp_label": f"ERROR: {e}",
            "severity_nlp_bin3": 2,
        }


def _build_month_contexts(
    all_events: list[dict],
    min_events: int = 3,
    top_n: int = 3,
) -> dict[str, str]:
    """Build context summaries for months with many events.

    Groups events by YYYY-MM, and for months with >= min_events,
    picks the top_n longest descriptions as representative context.
    Returns dict mapping YYYY-MM -> context string.
    """
    from collections import defaultdict
    by_month: dict[str, list[dict]] = defaultdict(list)
    for ev in all_events:
        ym = ev["date"][:7]  # YYYY-MM
        by_month[ym].append(ev)

    contexts = {}
    for ym, month_events in by_month.items():
        if len(month_events) < min_events:
            continue
        # Pick top_n longest descriptions as most informative
        sorted_evs = sorted(month_events, key=lambda e: len(e["description"]), reverse=True)
        snippets = [e["description"][:80].strip() for e in sorted_evs[:top_n]]
        contexts[ym] = (
            f"{ym}: {len(month_events)} events this month, including: "
            + "; ".join(snippets)
        )
    return contexts


def _classify_batch_chunk(
    events: list[dict],
    client,
    model: str = "claude-haiku-4-5-20251001",
    month_contexts: dict[str, str] | None = None,
) -> list[dict]:
    """Classify a batch of events in a single API call.

    events: list of dicts with id, date, description, president.
    month_contexts: optional dict mapping YYYY-MM -> context string.
    Returns list of dicts with id, score (1-3), label.
    """
    # Build context block from relevant months
    context_block = ""
    if month_contexts:
        relevant_months = set()
        for e in events:
            ym = e["date"][:7]
            if ym in month_contexts:
                relevant_months.add(ym)
        if relevant_months:
            ctx_lines = [month_contexts[ym] for ym in sorted(relevant_months)]
            context_block = "\nMONTH CONTEXT (for reference):\n" + "\n".join(ctx_lines) + "\n"

    # Build events block
    lines = []
    for e in events:
        lines.append(f"Event {e['id']}: [{e['date']}] ({e['president']}) {e['description'][:300]}")
    events_block = "\n".join(lines)

    prompt = SEVERITY_PROMPT.format(events_block=events_block, context_block=context_block)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=len(events) * 50 + 100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        results = json.loads(raw)
        return results
    except Exception as e:
        logger.warning("Batch classification failed: %s", e)
        # Return defaults
        return [{"id": ev["id"], "score": 2, "label": "medium"} for ev in events]


def classify_events_batch(
    events_df: pd.DataFrame,
    text_column: str = "event_description",
    batch_size: int = 20,
    client=None,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.3,
) -> pd.DataFrame:
    """Classify all events in a DataFrame using Claude API.

    Sends events in batches of batch_size per API call for efficiency.
    Adds columns: severity_nlp, severity_nlp_confidence, severity_nlp_label,
    severity_nlp_bin3.
    """
    if client is None:
        client = _get_client()

    df = events_df.copy()
    n = len(df)

    logger.info("Classifying %d events with Claude API (batch_size=%d)...", n, batch_size)

    # Prepare event dicts
    all_events = []
    for i, (idx, row) in enumerate(df.iterrows()):
        date_str = row["date"]
        if hasattr(date_str, "strftime"):
            date_str = date_str.strftime("%Y-%m-%d")
        all_events.append({
            "id": i + 1,
            "idx": idx,
            "date": str(date_str),
            "description": str(row[text_column]),
            "president": str(row.get("president_affected", "")),
        })

    # Initialize result columns
    df["severity_nlp"] = 3
    df["severity_nlp_confidence"] = 0.0
    df["severity_nlp_label"] = ""
    df["severity_nlp_bin3"] = 2

    # Build month contexts for enriched classification
    month_contexts = _build_month_contexts(all_events)
    if month_contexts:
        logger.info("  Month contexts built for %d months with 3+ events", len(month_contexts))

    # Process in batches
    bin3_to_5 = {1: 1, 2: 3, 3: 5}
    processed = 0

    for batch_start in range(0, n, batch_size):
        batch = all_events[batch_start:batch_start + batch_size]
        results = _classify_batch_chunk(
            batch, client=client, model=model, month_contexts=month_contexts,
        )

        # Map results back by id
        result_map = {}
        for r in results:
            rid = int(r["id"])
            result_map[rid] = r

        for ev in batch:
            eid = ev["id"]
            idx = ev["idx"]
            r = result_map.get(eid, {"score": 2, "label": "medium"})
            score3 = max(1, min(3, int(r["score"])))
            label = r.get("label", "")

            df.loc[idx, "severity_nlp_bin3"] = score3
            df.loc[idx, "severity_nlp"] = bin3_to_5[score3]
            df.loc[idx, "severity_nlp_confidence"] = 1.0
            df.loc[idx, "severity_nlp_label"] = label

        processed += len(batch)
        if processed % 100 == 0 or processed == n:
            logger.info("  Classified %d/%d events", processed, n)

        time.sleep(delay)

    # Summary
    dist3 = df["severity_nlp_bin3"].value_counts().sort_index()
    logger.info("Bin3 distribution: %s", dict(dist3))
    dist5 = df["severity_nlp"].value_counts().sort_index()
    logger.info("Score distribution (mapped 1-5): %s", dict(dist5))

    return df


def compute_monthly_event_score(
    events_df: pd.DataFrame,
    score_column: str = "severity_nlp",
) -> pd.Series:
    """Aggregate event severity to monthly frequency.

    Uses max severity in each month as the monthly score.
    Months without events get score 0.
    """
    df = events_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[score_column].max()
    return monthly


def compute_weekly_event_score(
    events_df: pd.DataFrame,
    score_column: str = "severity_nlp",
) -> pd.Series:
    """Aggregate event severity to weekly frequency (Friday).

    Uses max severity in each week.
    """
    df = events_df.copy()
    df = df.set_index("date")
    weekly = df[score_column].resample("W-FRI").max()
    weekly = weekly.fillna(0)
    return weekly
