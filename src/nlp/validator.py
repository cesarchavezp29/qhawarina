"""Claude Sonnet validation for political event severity classification.

Validates a subset of ~200 events against Claude Sonnet's judgment,
cross-checking the Haiku classifier. All comparison with ground truth
is done on the 3-bin ordinal scale.
"""

import json
import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.nlp.validator")

SCORE_TO_BIN3 = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

RUBRIC_PROMPT = """Eres un analista politico experto en Peru. Clasifica la severidad del
siguiente evento politico peruano en una escala de 1 a 5:

ESCALA:
1 = Rutina politica: cambio ministerial rutinario, inauguracion protocolar
2 = Tension politica moderada: amenaza de censura, interpelacion, protesta menor
3 = Conflicto institucional: voto de confianza, censura aprobada, protesta significativa
4 = Crisis constitucional: vacancia iniciada, disolucion amenazada, estado de emergencia
5 = Quiebre institucional: presidente destituido/renunciado, Congreso disuelto, autogolpe

EVENTO:
Fecha: {date}
Descripcion: {description}
Presidente en ejercicio: {president}

Responde SOLO con un JSON:
{{"score": <1-5>, "reasoning": "<explicacion en 1-2 oraciones>"}}"""


def validate_single_event(
    event: dict,
    client=None,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """Validate a single event with Claude API.

    Parameters
    ----------
    event : dict with date, event_description, president_affected
    client : anthropic.Anthropic instance
    model : Claude model ID

    Returns
    -------
    dict with severity_sonnet, severity_sonnet_bin3, severity_sonnet_reasoning
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    date_str = event.get("date", "")
    if hasattr(date_str, "strftime"):
        date_str = date_str.strftime("%Y-%m-%d")

    prompt = RUBRIC_PROMPT.format(
        date=date_str,
        description=event.get("event_description", ""),
        president=event.get("president_affected", ""),
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Parse JSON (handle potential markdown wrapping)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        score = int(result["score"])
        return {
            "severity_sonnet": score,
            "severity_sonnet_bin3": SCORE_TO_BIN3.get(score, 2),
            "severity_sonnet_reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        logger.warning("Claude validation failed for event: %s", e)
        return {
            "severity_sonnet": pd.NA,
            "severity_sonnet_bin3": pd.NA,
            "severity_sonnet_reasoning": f"ERROR: {e}",
        }


def validate_batch(
    events_df: pd.DataFrame,
    sample_size: int = 200,
    delay: float = 0.5,
    client=None,
    model: str = "claude-sonnet-4-5-20250929",
) -> pd.DataFrame:
    """Validate a stratified sample of events with Claude API.

    Selects: all GT-matched events + random sample to reach sample_size.
    Adds columns: severity_sonnet, severity_sonnet_bin3, severity_sonnet_reasoning.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    df = events_df.copy()

    # Select sample: all GT-matched + random fill
    gt_matched = df[df["severity_gt"].notna()].index.tolist()
    remaining = df[df["severity_gt"].isna()].index.tolist()
    n_remaining = max(0, sample_size - len(gt_matched))

    if n_remaining > 0 and remaining:
        rng = np.random.default_rng(42)
        extra = rng.choice(remaining, size=min(n_remaining, len(remaining)), replace=False)
        sample_idx = gt_matched + extra.tolist()
    else:
        sample_idx = gt_matched

    logger.info("Claude validation: %d events (%d GT-matched + %d random)",
                len(sample_idx), len(gt_matched), len(sample_idx) - len(gt_matched))

    # Initialize columns
    df["severity_sonnet"] = pd.NA
    df["severity_sonnet_bin3"] = pd.NA
    df["severity_sonnet_reasoning"] = ""

    for i, idx in enumerate(sample_idx):
        row = df.loc[idx]
        result = validate_single_event(row.to_dict(), client=client, model=model)
        df.loc[idx, "severity_sonnet"] = result["severity_sonnet"]
        df.loc[idx, "severity_sonnet_bin3"] = result["severity_sonnet_bin3"]
        df.loc[idx, "severity_sonnet_reasoning"] = result["severity_sonnet_reasoning"]

        if (i + 1) % 20 == 0:
            logger.info("  Validated %d/%d events", i + 1, len(sample_idx))
        time.sleep(delay)

    logger.info("Claude validation complete: %d events processed",
                df["severity_sonnet"].notna().sum())
    return df


# ── Validation metrics ──────────────────────────────────────────────────────

def compute_validation_metrics(
    events_df: pd.DataFrame,
) -> dict:
    """Compute validation metrics between Claude Haiku, Sonnet, and GT (all on 1-3 scale).

    Returns dict with accuracy, kappa, confusion matrices, severe errors.
    """
    metrics = {}

    # Claude Haiku (main classifier) vs GT
    gt_matched = events_df[events_df["severity_gt"].notna()].copy()
    if len(gt_matched) > 0:
        gt_matched["severity_gt"] = gt_matched["severity_gt"].astype(int)
        gt_matched["severity_claude_bin3"] = gt_matched["severity_claude_bin3"].astype(int)

        correct = (gt_matched["severity_claude_bin3"] == gt_matched["severity_gt"]).sum()
        metrics["claude_vs_gt_accuracy"] = round(correct / len(gt_matched), 3)

        # Severe errors: GT=3 classified as bin3=1, or GT=1 classified as bin3=3
        severe = (
            ((gt_matched["severity_gt"] == 3) & (gt_matched["severity_claude_bin3"] == 1)) |
            ((gt_matched["severity_gt"] == 1) & (gt_matched["severity_claude_bin3"] == 3))
        ).sum()
        metrics["claude_vs_gt_severe_errors"] = int(severe)

        from sklearn.metrics import confusion_matrix, cohen_kappa_score
        cm = confusion_matrix(
            gt_matched["severity_gt"],
            gt_matched["severity_claude_bin3"],
            labels=[1, 2, 3],
        )
        metrics["claude_vs_gt_confusion"] = cm.tolist()
        metrics["claude_vs_gt_kappa"] = round(
            cohen_kappa_score(gt_matched["severity_gt"], gt_matched["severity_claude_bin3"]),
            3,
        )

    # Sonnet (cross-validation) vs GT
    sonnet_gt = events_df[
        events_df["severity_gt"].notna() & events_df["severity_sonnet_bin3"].notna()
    ].copy()
    if len(sonnet_gt) > 0:
        sonnet_gt["severity_gt"] = sonnet_gt["severity_gt"].astype(int)
        sonnet_gt["severity_sonnet_bin3"] = sonnet_gt["severity_sonnet_bin3"].astype(int)

        correct = (sonnet_gt["severity_sonnet_bin3"] == sonnet_gt["severity_gt"]).sum()
        metrics["sonnet_vs_gt_accuracy"] = round(correct / len(sonnet_gt), 3)

        from sklearn.metrics import confusion_matrix, cohen_kappa_score
        cm = confusion_matrix(
            sonnet_gt["severity_gt"],
            sonnet_gt["severity_sonnet_bin3"],
            labels=[1, 2, 3],
        )
        metrics["sonnet_vs_gt_confusion"] = cm.tolist()
        metrics["sonnet_vs_gt_kappa"] = round(
            cohen_kappa_score(sonnet_gt["severity_gt"], sonnet_gt["severity_sonnet_bin3"]),
            3,
        )

    # Claude Haiku vs Sonnet (inter-rater)
    both = events_df[
        events_df["severity_claude_bin3"].notna() & events_df["severity_sonnet_bin3"].notna()
    ].copy()
    if len(both) > 0:
        both["severity_claude_bin3"] = both["severity_claude_bin3"].astype(int)
        both["severity_sonnet_bin3"] = both["severity_sonnet_bin3"].astype(int)

        from sklearn.metrics import cohen_kappa_score
        metrics["claude_vs_sonnet_kappa"] = round(
            cohen_kappa_score(both["severity_claude_bin3"], both["severity_sonnet_bin3"]),
            3,
        )

    return metrics
