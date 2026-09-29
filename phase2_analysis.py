#!/usr/bin/env python3
"""
Final Phase 2 synthesis call — LONG / SHORT / NO_TRADE with structured
reasoning, constrained to prompts/phase2_decision_v1.py's schema. See that
module's docstring for why NO_TRADE is structurally first-class and why
"confidence" means evidentiary strength, not win probability.

This is the only call in the pipeline that produces a decision. Everything
upstream (OpportunitySnapshot, TrendAnalysis, VisionAnalysis) is either
pure deterministic computation or a description of chart structure — no
direction is ever assumed before this point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from llm_provider import LLMProvider
from opportunity_snapshot import OpportunitySnapshot
from prompts import phase2_decision_v1
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis


@dataclass(frozen=True)
class AnalysisResult:
    decision: str  # "LONG" | "SHORT" | "NO_TRADE"
    confidence: float
    setup: Optional[str]
    thesis: Optional[str]
    reason: Optional[str]
    supporting_factors: tuple[str, ...]
    opposing_factors: tuple[str, ...]
    key_levels: tuple[dict, ...]
    invalidation_concept: Optional[str]
    analysis_summary: str
    raw_response: str
    latency_s: float


_VALID_DECISIONS = {"LONG", "SHORT", "NO_TRADE"}


def run_phase2_analysis(
    provider: LLMProvider,
    snapshot: OpportunitySnapshot,
    trend: Optional[TrendAnalysis],
    vision: VisionAnalysis,
) -> AnalysisResult:
    user = phase2_decision_v1.build_prompt(snapshot, trend, vision)
    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=phase2_decision_v1.SYSTEM_PROMPT,
        user=user,
        schema=phase2_decision_v1.JSON_SCHEMA,
        schema_name="phase2_decision_v1",
    )
    latency = time.time() - t0

    decision = str(parsed.get("decision", "NO_TRADE")).strip().upper()
    if decision not in _VALID_DECISIONS:
        # Schema enum constraint should make this unreachable, but fail
        # closed to NO_TRADE rather than propagate an unrecognized value —
        # NO_TRADE must never require the grammar constraint alone to hold.
        decision = "NO_TRADE"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    key_levels = parsed.get("key_levels", []) or []

    return AnalysisResult(
        decision=decision,
        confidence=confidence,
        setup=parsed.get("setup"),
        thesis=parsed.get("thesis"),
        reason=parsed.get("reason"),
        supporting_factors=tuple(str(f) for f in parsed.get("supporting_factors", []) or []),
        opposing_factors=tuple(str(f) for f in parsed.get("opposing_factors", []) or []),
        key_levels=tuple(key_levels),
        invalidation_concept=parsed.get("invalidation_concept"),
        analysis_summary=str(parsed.get("analysis_summary", "")),
        raw_response=raw,
        latency_s=latency,
    )
