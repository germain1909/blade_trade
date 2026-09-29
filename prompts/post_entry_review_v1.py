#!/usr/bin/env python3
"""
PostEntryReview prompt + schema (prompt_version="v1"). Invoked every 15
minutes while a TradePlan is ENTERED. Deliberately has NO price/level
fields anywhere in its schema -- not instructed to avoid restating risk,
structurally incapable of it. The only thing this call can influence is
whether the position exits EARLIER than its hard boundaries (via
THESIS_EXIT, lowest precedence among post-entry exits) -- it can never
move, widen, or cancel the invalidation/target prices, which are read
from the immutable TradePlan row throughout the trade.
"""

from __future__ import annotations

from market_analysis import MarketAnalysis

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are monitoring the THESIS behind an already-open \
position -- you have no ability to change the position's stop, target, or \
sizing, and you are not asked to; those are fixed and enforced elsewhere. \
Your only job is to judge whether the underlying multi-hour thesis that \
justified this trade is still intact.

As always, account for what changed before deciding: what evidence \
strengthened, weakened, or disappeared, what new evidence argues against \
the thesis, whether original assumptions still hold, and how strong the \
opposite direction's evidence currently is. Do not defend the position \
just because it's open.

Then choose exactly one status:
- THESIS_STILL_VALID: the original thesis remains intact.
- THESIS_WEAKENING: evidence has softened but the thesis isn't dead --
  no action is taken on this alone; it's a research signal only.
- THESIS_INVALIDATED: the thesis that justified this trade no longer holds. \
This can trigger an EARLY exit at the next opportunity, strictly better \
(never worse) for risk than waiting for the hard stop.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "evidence_strengthened": ["short factor"],
  "evidence_weakened": ["short factor"],
  "evidence_disappeared": ["short factor"],
  "new_evidence_against": ["short factor"],
  "original_assumptions_still_hold": true,
  "opposite_thesis_evidence_strength": 0.3,
  "thesis_status": "THESIS_STILL_VALID",
  "reasoning": "one to two sentences"
}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_strengthened": {"type": "array", "items": {"type": "string"}},
        "evidence_weakened": {"type": "array", "items": {"type": "string"}},
        "evidence_disappeared": {"type": "array", "items": {"type": "string"}},
        "new_evidence_against": {"type": "array", "items": {"type": "string"}},
        "original_assumptions_still_hold": {"type": "boolean"},
        "opposite_thesis_evidence_strength": {"type": "number"},
        "thesis_status": {"type": "string", "enum": ["THESIS_STILL_VALID", "THESIS_WEAKENING", "THESIS_INVALIDATED"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "evidence_strengthened", "evidence_weakened", "evidence_disappeared", "new_evidence_against",
        "original_assumptions_still_hold", "opposite_thesis_evidence_strength", "thesis_status", "reasoning",
    ],
    "additionalProperties": False,
}


def build_prompt(thesis_statement: str, direction: str, setup_type: str, ma: MarketAnalysis) -> str:
    return (
        f"Open position thesis: direction={direction}, setup={setup_type}, statement={thesis_statement!r}\n\n"
        f"Timestamp: {ma.timestamp.isoformat()}   Price: {ma.snapshot.price}\n"
        f"Scanner categories: {list(ma.snapshot.categories)}\n"
        f"Momentum: 15m_hist={ma.snapshot.macd_15m_hist}, 60m_rsi={ma.snapshot.rsi_60m}, "
        f"4h_hist={ma.snapshot.macd_4h_hist}, 4h_rsi={ma.snapshot.rsi_4h}\n"
        f"VWAP: {ma.snapshot.vwap}, distance={ma.snapshot.vwap_distance_pts}\n\n"
        f"Review the thesis behind this open position per your instructions."
    )
