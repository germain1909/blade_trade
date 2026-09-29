#!/usr/bin/env python3
"""
DIAGNOSTIC-ONLY forced-direction call. See prompts/forced_direction_v1.py's
docstring — this exists to separate "the model sees a lean but abstains"
from "the model genuinely sees balanced evidence." Never feeds into the
real Phase 2 decision pipeline, never persisted to the `analyses` table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from llm_provider import LLMProvider
from prompts import forced_direction_v1


@dataclass(frozen=True)
class ForcedDirectionResult:
    forced_direction: str  # "LONG" | "SHORT"
    directional_confidence: float
    bull_case: tuple[str, ...]
    bear_case: tuple[str, ...]
    decisive_factors: tuple[str, ...]
    raw_response: str
    latency_s: float


_VALID_DIRECTIONS = {"LONG", "SHORT"}


def run_forced_direction_diagnostic(
    provider: LLMProvider,
    snapshot: dict,
    trend: dict | None,
    vision: dict,
) -> ForcedDirectionResult:
    user = forced_direction_v1.build_prompt(snapshot, trend, vision)
    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=forced_direction_v1.SYSTEM_PROMPT,
        user=user,
        schema=forced_direction_v1.JSON_SCHEMA,
        schema_name="forced_direction_diagnostic_v1",
    )
    latency = time.time() - t0

    direction = str(parsed.get("forced_direction", "")).strip().upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"diagnostic call returned an invalid forced_direction: {parsed!r}")

    try:
        confidence = float(parsed.get("directional_confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return ForcedDirectionResult(
        forced_direction=direction,
        directional_confidence=confidence,
        bull_case=tuple(str(f) for f in parsed.get("bull_case", []) or []),
        bear_case=tuple(str(f) for f in parsed.get("bear_case", []) or []),
        decisive_factors=tuple(str(f) for f in parsed.get("decisive_factors", []) or []),
        raw_response=raw,
        latency_s=latency,
    )
