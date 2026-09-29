#!/usr/bin/env python3
"""
LLM-calling wrapper for PostEntryReview -- pairs with
prompts/post_entry_review_v1.py. Deliberately has no ability to touch
price/level fields (the schema doesn't have any) -- see that module's
docstring. Its output can only ever cause an EARLIER exit (THESIS_EXIT,
lowest precedence among post-entry exits in trade_plan_replay.py), never
move or cancel the hard invalidation/target boundaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from llm_provider import LLMProvider
from market_analysis import MarketAnalysis
from prompts import post_entry_review_v1


@dataclass(frozen=True)
class PostEntryReviewResult:
    thesis_status: str  # "THESIS_STILL_VALID" | "THESIS_WEAKENING" | "THESIS_INVALIDATED"
    evidence_strengthened: tuple[str, ...]
    evidence_weakened: tuple[str, ...]
    evidence_disappeared: tuple[str, ...]
    new_evidence_against: tuple[str, ...]
    original_assumptions_still_hold: bool
    opposite_thesis_evidence_strength: float
    reasoning: str
    raw_response: str
    latency_s: float


_VALID_STATUSES = {"THESIS_STILL_VALID", "THESIS_WEAKENING", "THESIS_INVALIDATED"}


def run_post_entry_review(
    provider: LLMProvider, thesis_statement: str, direction: str, setup_type: str, ma: MarketAnalysis,
) -> PostEntryReviewResult:
    prompt = post_entry_review_v1.build_prompt(thesis_statement, direction, setup_type, ma)
    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=post_entry_review_v1.SYSTEM_PROMPT, user=prompt,
        schema=post_entry_review_v1.JSON_SCHEMA, schema_name="post_entry_review_v1",
    )
    latency = time.time() - t0

    status = str(parsed.get("thesis_status", "THESIS_STILL_VALID")).strip().upper()
    if status not in _VALID_STATUSES:
        status = "THESIS_STILL_VALID"  # fail closed to "still valid" -- never fabricate an early exit

    try:
        opposite_strength = max(0.0, min(1.0, float(parsed.get("opposite_thesis_evidence_strength", 0.5))))
    except (TypeError, ValueError):
        opposite_strength = 0.5

    return PostEntryReviewResult(
        thesis_status=status,
        evidence_strengthened=tuple(parsed.get("evidence_strengthened", []) or []),
        evidence_weakened=tuple(parsed.get("evidence_weakened", []) or []),
        evidence_disappeared=tuple(parsed.get("evidence_disappeared", []) or []),
        new_evidence_against=tuple(parsed.get("new_evidence_against", []) or []),
        original_assumptions_still_hold=bool(parsed.get("original_assumptions_still_hold", True)),
        opposite_thesis_evidence_strength=opposite_strength,
        reasoning=str(parsed.get("reasoning", "") or ""),
        raw_response=raw, latency_s=latency,
    )
