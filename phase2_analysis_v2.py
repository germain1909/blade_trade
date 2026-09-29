#!/usr/bin/env python3
"""
Phase 2 V2 synthesis call — directional_view + tradeability as two
independent, always-answered outputs from one call, constrained to
prompts/phase2_decision_v2.py's schema. See that module and RESEARCH_NOTES
for why this replaces V1's single 3-way decision plus a separate
forced-direction diagnostic call.

Does not replace phase2_analysis.py/AnalysisResult (V1, frozen, matches
the already-persisted 12-episode pilot).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from llm_provider import LLMProvider
from opportunity_snapshot import OpportunitySnapshot
from output_quality import validate_and_normalize_v2
from prompts import phase2_decision_v2
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis


@dataclass(frozen=True)
class AnalysisResultV2:
    directional_view: str  # "LONG" | "SHORT" — always forced
    directional_confidence: float
    directional_bull_case: tuple[str, ...]
    directional_bear_case: tuple[str, ...]
    directional_decisive_factors: tuple[str, ...]

    tradeability: str  # "TRADE" | "NO_TRADE"
    tradeability_confidence: float
    tradeability_reason: str
    tradeability_supporting_factors: tuple[str, ...]
    tradeability_opposing_factors: tuple[str, ...]

    setup: Optional[str]
    key_levels: tuple[dict, ...]
    invalidation_concept: Optional[str]
    analysis_summary: str

    raw_response: str
    latency_s: float
    quality_flags: tuple[str, ...]


_VALID_DIRECTIONS = {"LONG", "SHORT"}
_VALID_TRADEABILITY = {"TRADE", "NO_TRADE"}


def run_phase2_analysis_v2(
    provider: LLMProvider,
    snapshot: OpportunitySnapshot,
    trend: Optional[TrendAnalysis],
    vision: VisionAnalysis,
) -> tuple[AnalysisResultV2, str, tuple[str, ...]]:
    """Returns (result, prompt_text, levels_shown_to_model) — the latter two
    for persistence/reproducibility."""
    user, levels_shown = phase2_decision_v2.build_prompt(snapshot, trend, vision)
    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=phase2_decision_v2.SYSTEM_PROMPT,
        user=user,
        schema=phase2_decision_v2.JSON_SCHEMA,
        schema_name="phase2_decision_v2",
    )
    latency = time.time() - t0

    normalized, quality_flags = validate_and_normalize_v2(parsed)

    direction = str(normalized.get("directional_view", "")).strip().upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"v2 analysis returned an invalid directional_view: {normalized!r}")
    tradeability = str(normalized.get("tradeability", "")).strip().upper()
    if tradeability not in _VALID_TRADEABILITY:
        raise ValueError(f"v2 analysis returned an invalid tradeability: {normalized!r}")

    def _conf(key: str) -> float:
        try:
            v = float(normalized.get(key, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))

    result = AnalysisResultV2(
        directional_view=direction,
        directional_confidence=_conf("directional_confidence"),
        directional_bull_case=tuple(str(f) for f in normalized.get("directional_bull_case", []) or []),
        directional_bear_case=tuple(str(f) for f in normalized.get("directional_bear_case", []) or []),
        directional_decisive_factors=tuple(str(f) for f in normalized.get("directional_decisive_factors", []) or []),
        tradeability=tradeability,
        tradeability_confidence=_conf("tradeability_confidence"),
        tradeability_reason=str(normalized.get("tradeability_reason", "") or ""),
        tradeability_supporting_factors=tuple(
            str(f) for f in normalized.get("tradeability_supporting_factors", []) or []),
        tradeability_opposing_factors=tuple(
            str(f) for f in normalized.get("tradeability_opposing_factors", []) or []),
        setup=normalized.get("setup"),
        key_levels=tuple(normalized.get("key_levels", []) or []),
        invalidation_concept=normalized.get("invalidation_concept"),
        analysis_summary=str(normalized.get("analysis_summary", "") or ""),
        raw_response=raw,
        latency_s=latency,
        quality_flags=quality_flags,
    )
    return result, user, levels_shown
