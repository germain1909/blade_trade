#!/usr/bin/env python3
"""
Calls the local vision model on a point-in-time chart, constrained to
prompts/vision_v1.py's JSON schema. See that module for why vision is asked
about pattern/structure only, never asked to re-derive numbers Python
already computed exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from llm_provider import LLMProvider
from opportunity_snapshot import OpportunitySnapshot
from prompts import vision_v1
from trend_analysis import TrendAnalysis


@dataclass(frozen=True)
class VisionAnalysis:
    trend_structure: str
    pattern: Optional[str]
    compression: bool
    breakout_structure: Optional[str]
    rejection_structure: Optional[str]
    visual_supporting_factors: tuple[str, ...]
    visual_risk_factors: tuple[str, ...]
    raw_response: str
    latency_s: float


def run_vision_analysis(
    provider: LLMProvider,
    snapshot: OpportunitySnapshot,
    trend: Optional[TrendAnalysis],
    image_b64: str,
) -> VisionAnalysis:
    user_text = vision_v1.build_prompt(snapshot, trend)
    t0 = time.time()
    parsed, raw = provider.complete_vision_structured(
        system=vision_v1.SYSTEM_PROMPT,
        user_text=user_text,
        image_b64=image_b64,
        schema=vision_v1.JSON_SCHEMA,
        schema_name="vision_analysis_v1",
    )
    latency = time.time() - t0

    return VisionAnalysis(
        trend_structure=str(parsed.get("trend_structure", "")),
        pattern=parsed.get("pattern"),
        compression=bool(parsed.get("compression", False)),
        breakout_structure=parsed.get("breakout_structure"),
        rejection_structure=parsed.get("rejection_structure"),
        visual_supporting_factors=tuple(str(f) for f in parsed.get("visual_supporting_factors", []) or []),
        visual_risk_factors=tuple(str(f) for f in parsed.get("visual_risk_factors", []) or []),
        raw_response=raw,
        latency_s=latency,
    )
