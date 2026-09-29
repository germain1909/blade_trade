#!/usr/bin/env python3
"""
Pure, deterministic output-quality validation for Phase 2 V2's structured
LLM response. No LLM calls here. Does exactly one semantics-preserving
normalization (the "null"-string fix) and otherwise only ANNOTATES —
never silently changes a substantive field (decision, direction,
tradeability, confidence) and never re-asks the model for a different
answer within this pass.

Flags found here are diagnostic/methodology signal (e.g. "did the v2
prompt/schema changes actually reduce empty-array output"), not something
fed back into the prompt or used to alter the frozen prediction.
"""

from __future__ import annotations

_NULLABLE_STRING_FIELDS = ("setup", "invalidation_concept")
_FACTOR_ARRAY_FIELDS = (
    "directional_bull_case", "directional_bear_case", "directional_decisive_factors",
    "tradeability_supporting_factors", "tradeability_opposing_factors",
)
_MIN_SUMMARY_WORDS_FOR_EMPTY_ARRAY_FLAG = 15
_MIN_WORDS_FOR_TRADEABILITY_REASON = 5
_LOW_CONFIDENCE_TRADE_THRESHOLD = 0.55  # symmetric near-coin-flip cutoff, not fit to any outcome data


def validate_and_normalize_v2(parsed: dict) -> tuple[dict, tuple[str, ...]]:
    """Returns (normalized_dict, quality_flags). `parsed` is the raw dict
    already returned by json.loads on the model's structured response."""
    normalized = dict(parsed)
    flags: list[str] = []

    # 1. "null"-string -> None normalization (semantics-preserving: both are
    # schema-valid values for a nullable string field, so this doesn't
    # change what the model "meant," only how it's represented downstream).
    for field in _NULLABLE_STRING_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str) and value.strip().lower() == "null":
            normalized[field] = None
            flags.append(f"NULL_STRING_NORMALIZED:{field}")

    # 2. Empty factor array despite a substantive summary.
    summary = str(normalized.get("analysis_summary", "") or "")
    summary_word_count = len(summary.split())
    if summary_word_count > _MIN_SUMMARY_WORDS_FOR_EMPTY_ARRAY_FLAG:
        for field in _FACTOR_ARRAY_FIELDS:
            arr = normalized.get(field)
            if isinstance(arr, list) and len(arr) == 0:
                flags.append(f"EMPTY_FACTOR_ARRAY_WITH_SUBSTANTIVE_SUMMARY:{field}")

    # 3. Missing/trivial tradeability_reason.
    reason = str(normalized.get("tradeability_reason", "") or "")
    if len(reason.split()) < _MIN_WORDS_FOR_TRADEABILITY_REASON:
        flags.append("TRADEABILITY_REASON_MISSING_OR_TRIVIAL")

    # 4. Directional/tradeability inconsistency: TRADE endorsed on a
    # near-coin-flip directional read with no directional factors given —
    # the shape of the V1 pilot's one internally-incoherent episode.
    try:
        directional_confidence = float(normalized.get("directional_confidence", 0.0))
    except (TypeError, ValueError):
        directional_confidence = 0.0
    tradeability = str(normalized.get("tradeability", "")).strip().upper()
    bull = normalized.get("directional_bull_case") or []
    bear = normalized.get("directional_bear_case") or []
    if tradeability == "TRADE" and directional_confidence < _LOW_CONFIDENCE_TRADE_THRESHOLD and not (bull or bear):
        flags.append("DIRECTIONAL_TRADEABILITY_INCONSISTENCY")

    return normalized, tuple(flags)


# --- Phase 3 additions -----------------------------------------------------
# Extends this module rather than forking it (see approved Phase 3 plan §1)
# -- output_quality.py has no persisted-row dependency on staying
# byte-identical the way a prompt/schema module does.

_ANCHORING_ACTIONS = ("MAINTAIN", "STRENGTHEN")


def check_anchoring_suspected(review_action: str, confidence: float,
                               opposite_thesis_evidence_strength: float) -> tuple[str, ...]:
    """Anti-anchoring audit flag (approved Phase 3 plan §6 item 2): the
    model is asked, every single review, how strong the OPPOSITE
    direction's evidence currently is -- even while maintaining its own
    thesis. If that self-reported opposite-strength is at or above the
    thesis's own confidence while the model still chose to MAINTAIN or
    STRENGTHEN, that's a visible, after-the-fact inconsistency worth
    flagging. Diagnostic only -- never overrides review_action itself
    (the forced-invalidation circuit breaker in active_thesis.py is the
    only thing authorized to override review_action)."""
    if review_action in _ANCHORING_ACTIONS and opposite_thesis_evidence_strength >= confidence:
        return ("ANCHORING_SUSPECTED",)
    return ()


def check_thesis_output_quality(parsed: dict) -> tuple[str, ...]:
    """Lightweight structural sanity checks on a ThesisReview response,
    mirroring validate_and_normalize_v2's philosophy: annotate, never
    silently fix. Does NOT normalize "null" strings itself -- ThesisReview's
    schema has no nullable-string fields prone to that wrinkle except
    terminal_reason, which is legitimately null most of the time."""
    flags: list[str] = []
    action = str(parsed.get("review_action", "")).strip().upper()
    what_changed = str(parsed.get("what_changed_since_previous", "") or "")
    if len(what_changed.split()) < 3:
        flags.append("WHAT_CHANGED_MISSING_OR_TRIVIAL")
    if action in ("INVALIDATE", "REVERSE") and not (parsed.get("terminal_reason") or "").strip():
        flags.append("TERMINAL_REASON_MISSING")
    if action == "REVERSE" and str(parsed.get("reversal_direction") or "").strip().upper() not in ("LONG", "SHORT"):
        flags.append("REVERSAL_DIRECTION_MISSING")
    return tuple(flags)
