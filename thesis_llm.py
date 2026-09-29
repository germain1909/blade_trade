#!/usr/bin/env python3
"""
LLM-calling wrappers for ThesisFormation and ThesisReview -- pairs with
prompts/thesis_formation_v1.py / prompts/thesis_review_v1.py the same way
phase2_analysis.py pairs with prompts/phase2_decision_v1.py. active_thesis.py
itself stays pure (dataclass + state machine, no LLM calls); this module is
where the LLM's output becomes a new ActiveThesis row.

The forced-invalidation circuit breaker (active_thesis.check_forced_invalidation)
is applied AFTER the model's own review, not instead of it -- the full model
response is still generated and persisted (so the audit trail shows what the
model actually thought), but the EFFECTIVE review_action/state/terminal_reason
are overridden to INVALIDATED if the thesis's own declared conditions have
deterministically fired, regardless of what the model said.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

from active_thesis import ActiveThesis, check_forced_invalidation, next_state
from llm_provider import LLMProvider
from market_analysis import MarketAnalysis
from output_quality import check_anchoring_suspected, check_thesis_output_quality
from prompts import thesis_formation_v1, thesis_review_v1
from thesis_config import ThesisConfig

_VALID_DIRECTIONS = {"LONG", "SHORT"}


@dataclass(frozen=True)
class ThesisFormationResult:
    thesis_status: str  # "CANDIDATE" | "NO_THESIS"
    no_thesis_reason: Optional[str]
    thesis: Optional[ActiveThesis]
    candidate_structures: tuple[dict, ...]
    raw_response: str
    latency_s: float


def run_thesis_formation(
    provider: LLMProvider, ma: MarketAnalysis, cfg: ThesisConfig,
) -> ThesisFormationResult:
    prompt, candidates = thesis_formation_v1.build_prompt(ma)
    candidate_names = tuple(c["name"] for c in candidates)
    schema = thesis_formation_v1.build_schema(candidate_names)

    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=thesis_formation_v1.SYSTEM_PROMPT, user=prompt,
        schema=schema, schema_name="thesis_formation_v1",
    )
    latency = time.time() - t0

    status = str(parsed.get("thesis_status", "NO_THESIS")).strip().upper()
    if status not in ("CANDIDATE", "NO_THESIS"):
        status = "NO_THESIS"

    if status == "NO_THESIS":
        return ThesisFormationResult(
            thesis_status="NO_THESIS", no_thesis_reason=parsed.get("no_thesis_reason"),
            thesis=None, candidate_structures=candidates, raw_response=raw, latency_s=latency,
        )

    direction = str(parsed.get("direction", "")).strip().upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"thesis_formation returned CANDIDATE with an invalid direction: {parsed!r}")

    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    thesis_id = str(uuid.uuid4())
    thesis = ActiveThesis(
        thesis_id=thesis_id, version=0, parent_thesis_id=None,
        created_at=ma.timestamp, last_reviewed_at=ma.timestamp,
        symbol=ma.symbol, direction=direction, state="CANDIDATE",
        thesis_statement=str(parsed.get("thesis_statement", "") or ""),
        intended_horizon_bars=cfg.intended_horizon_bars, confidence=confidence,
        supporting_evidence=tuple(parsed.get("supporting_evidence", []) or []),
        opposing_evidence=tuple(parsed.get("opposing_evidence", []) or []),
        structural_basis=tuple(parsed.get("structural_basis", []) or []),
        important_levels=tuple(parsed.get("important_levels", []) or []),
        expected_market_behavior=str(parsed.get("expected_market_behavior", "") or ""),
        confirmation_still_needed=tuple(parsed.get("confirmation_still_needed", []) or []),
        invalidation_conditions=tuple(parsed.get("invalidation_conditions", []) or []),
        structural_objectives=tuple(parsed.get("structural_objectives", []) or []),
        risks=tuple(parsed.get("risks", []) or []),
        evidence_strengthened=(), evidence_weakened=(), evidence_disappeared=(),
        new_evidence_against=(), original_assumptions_still_hold=True,
        opposite_thesis_evidence_strength=1.0 - confidence,
        what_changed_since_previous="thesis formed",
        review_action="MAINTAIN", terminal_reason=None,
        market_analysis_id=ma.market_analysis_id, prior_thesis_version_id=None,
        model_name=provider.config.model, provider="OpenAICompatProvider",
        prompt_version=thesis_formation_v1.PROMPT_VERSION,
        inference_settings_json={"temperature": provider.config.temperature, "base_url": provider.config.base_url},
        raw_response=raw, latency_s=latency,
    )
    return ThesisFormationResult(
        thesis_status="CANDIDATE", no_thesis_reason=None, thesis=thesis,
        candidate_structures=candidates, raw_response=raw, latency_s=latency,
    )


@dataclass(frozen=True)
class ThesisReviewResult:
    new_version: ActiveThesis           # the new row under the SAME thesis_id (state may be terminal)
    reversed_new_thesis_id: Optional[str]  # if REVERSE, the caller must separately call
                                            # run_thesis_formation-equivalent logic for the new opposite
                                            # thesis; this field just flags that a reversal happened and
                                            # carries the reversal_direction for the caller to act on
    reversal_direction: Optional[str]
    forced_invalidation: bool           # True if the circuit breaker overrode the model's own action
    quality_flags: tuple[str, ...]
    raw_response: str
    latency_s: float


def run_thesis_review(
    provider: LLMProvider, thesis: ActiveThesis, ma: MarketAnalysis,
    prior_missed_or_abandoned: tuple[dict, ...] = (),
) -> ThesisReviewResult:
    prompt = thesis_review_v1.build_prompt(thesis, ma, prior_missed_or_abandoned)
    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=thesis_review_v1.SYSTEM_PROMPT, user=prompt,
        schema=thesis_review_v1.JSON_SCHEMA, schema_name="thesis_review_v1",
    )
    latency = time.time() - t0

    action = str(parsed.get("review_action", "MAINTAIN")).strip().upper()
    if action not in ("MAINTAIN", "STRENGTHEN", "WEAKEN", "INVALIDATE", "REVERSE"):
        action = "MAINTAIN"

    forced = False
    forced_reason = check_forced_invalidation(thesis, ma.snapshot)
    terminal_reason = parsed.get("terminal_reason")
    reversal_direction = parsed.get("reversal_direction")
    if forced_reason is not None and action != "INVALIDATE":
        action = "INVALIDATE"
        terminal_reason = f"[FORCED] {forced_reason}"
        forced = True
        reversal_direction = None
    elif forced_reason is not None:
        terminal_reason = f"[FORCED] {forced_reason}"
        forced = True

    new_state = next_state(thesis.state, action)

    try:
        updated_confidence = max(0.0, min(1.0, float(parsed.get("updated_confidence", thesis.confidence))))
    except (TypeError, ValueError):
        updated_confidence = thesis.confidence
    try:
        opposite_strength = max(0.0, min(1.0, float(parsed.get("opposite_thesis_evidence_strength", 0.5))))
    except (TypeError, ValueError):
        opposite_strength = 0.5

    new_version = ActiveThesis(
        thesis_id=thesis.thesis_id, version=thesis.version + 1, parent_thesis_id=thesis.parent_thesis_id,
        created_at=thesis.created_at, last_reviewed_at=ma.timestamp,
        symbol=thesis.symbol, direction=thesis.direction, state=new_state,
        thesis_statement=thesis.thesis_statement, intended_horizon_bars=thesis.intended_horizon_bars,
        confidence=updated_confidence,
        supporting_evidence=tuple(parsed.get("updated_supporting_evidence", []) or []),
        opposing_evidence=tuple(parsed.get("updated_opposing_evidence", []) or []),
        structural_basis=thesis.structural_basis,
        important_levels=thesis.important_levels,
        expected_market_behavior=thesis.expected_market_behavior,
        confirmation_still_needed=tuple(parsed.get("updated_confirmation_still_needed", []) or []),
        invalidation_conditions=thesis.invalidation_conditions,
        structural_objectives=thesis.structural_objectives,
        risks=thesis.risks,
        evidence_strengthened=tuple(parsed.get("evidence_strengthened", []) or []),
        evidence_weakened=tuple(parsed.get("evidence_weakened", []) or []),
        evidence_disappeared=tuple(parsed.get("evidence_disappeared", []) or []),
        new_evidence_against=tuple(parsed.get("new_evidence_against", []) or []),
        original_assumptions_still_hold=bool(parsed.get("original_assumptions_still_hold", True)),
        opposite_thesis_evidence_strength=opposite_strength,
        what_changed_since_previous=str(parsed.get("what_changed_since_previous", "") or ""),
        review_action=action, terminal_reason=terminal_reason,
        market_analysis_id=ma.market_analysis_id, prior_thesis_version_id=f"{thesis.thesis_id}:{thesis.version}",
        model_name=provider.config.model, provider="OpenAICompatProvider",
        prompt_version=thesis_review_v1.PROMPT_VERSION,
        inference_settings_json={"temperature": provider.config.temperature, "base_url": provider.config.base_url},
        raw_response=raw, latency_s=latency,
    )

    quality_flags = check_thesis_output_quality(parsed)
    quality_flags += check_anchoring_suspected(action, updated_confidence, opposite_strength)

    return ThesisReviewResult(
        new_version=new_version,
        reversed_new_thesis_id=str(uuid.uuid4()) if action == "REVERSE" else None,
        reversal_direction=reversal_direction if action == "REVERSE" else None,
        forced_invalidation=forced, quality_flags=quality_flags, raw_response=raw, latency_s=latency,
    )
