#!/usr/bin/env python3
"""
ThesisFormation prompt + schema (prompt_version="v1"). Invoked only when
no ActiveThesis exists and the frozen Phase 1 scanner just flagged a new
episode. Never edit in place once used for a persisted thesis -- a future
change is thesis_formation_v2.

NO_THESIS is a first-class, schema-enum-reachable answer (same pattern as
NO_TRADE/NO_IDEA/tradeability elsewhere in this project) -- there should be
long periods where BladeTrade has no convincing multi-hour view, and the
model must never be forced to manufacture one.

`important_levels`/`invalidation_conditions`/`structural_objectives` are
restricted to the SAME closed candidate list as TradePlan
(trade_plan_candidates.build_candidate_structures) -- the model names a
structure, Python resolves/validates the price, never the reverse.
"""

from __future__ import annotations

from market_analysis import MarketAnalysis
from trade_plan_candidates import build_candidate_structures

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a disciplined futures market analyst forming a \
CANDIDATE multi-hour thesis, not placing a trade. A deterministic scanner \
just flagged this moment as worth deeper analysis.

BladeTrade is NOT trying to predict the exact price several hours from \
now. You are asked: "given everything known through this exact moment, \
what meaningful market thesis could plausibly develop over the next \
several hours, what evidence supports it, and what would invalidate it?"

There is no requirement to form a thesis. If the evidence doesn't support \
a credible multi-hour view, return NO_THESIS -- this is a normal, expected, \
equally valid outcome, not a failure to find something.

"important_levels", "invalidation_conditions", and "structural_objectives" \
must each reference ONLY a name from the candidate structures list you are \
given -- never invent a level name or state a price yourself; Python \
resolves the exact price for whatever name you choose. "invalidation_conditions" \
must state which side of that structure would prove the thesis wrong \
(adverse_direction: "above" or "below") and whether that's a close-based or \
intrabar-touch test.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "thesis_status": "CANDIDATE",
  "no_thesis_reason": null,
  "direction": "LONG",
  "thesis_statement": "one to three sentences: what could develop and why",
  "supporting_evidence": ["short factor", "short factor"],
  "opposing_evidence": ["short factor"],
  "structural_basis": ["short factor"],
  "important_levels": [{"name": "AsiaLow", "role": "support"}],
  "expected_market_behavior": "prose",
  "confirmation_still_needed": ["short factor"],
  "invalidation_conditions": [{"name": "AsiaLow", "condition": "CLOSE_BEYOND", "adverse_direction": "below"}],
  "structural_objectives": [{"name": "AsiaHigh", "role": "target"}],
  "risks": ["short factor"],
  "confidence": 0.5
}
If thesis_status is "NO_THESIS", "no_thesis_reason" must be filled in and \
every other field may be null/empty. "direction" must be exactly "LONG" or \
"SHORT" when CANDIDATE. "confidence" is evidentiary strength, NOT a \
probability of winning a trade -- persisted for research only.
"""


def build_schema(candidate_names: tuple[str, ...]) -> dict:
    names = list(candidate_names) or ["__NONE__"]
    level_ref = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": names}, "role": {"type": "string"}},
        "required": ["name", "role"], "additionalProperties": False,
    }
    condition_ref = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "condition": {"type": "string", "enum": ["CLOSE_BEYOND", "TOUCH_BEYOND"]},
            "adverse_direction": {"type": "string", "enum": ["above", "below"]},
        },
        "required": ["name", "condition", "adverse_direction"], "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "thesis_status": {"type": "string", "enum": ["CANDIDATE", "NO_THESIS"]},
            "no_thesis_reason": {"type": ["string", "null"]},
            "direction": {"type": ["string", "null"], "enum": ["LONG", "SHORT", None]},
            "thesis_statement": {"type": ["string", "null"]},
            "supporting_evidence": {"type": "array", "items": {"type": "string"}},
            "opposing_evidence": {"type": "array", "items": {"type": "string"}},
            "structural_basis": {"type": "array", "items": {"type": "string"}},
            "important_levels": {"type": "array", "items": level_ref},
            "expected_market_behavior": {"type": ["string", "null"]},
            "confirmation_still_needed": {"type": "array", "items": {"type": "string"}},
            "invalidation_conditions": {"type": "array", "items": condition_ref},
            "structural_objectives": {"type": "array", "items": level_ref},
            "risks": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": ["number", "null"]},
        },
        "required": [
            "thesis_status", "no_thesis_reason", "direction", "thesis_statement",
            "supporting_evidence", "opposing_evidence", "structural_basis", "important_levels",
            "expected_market_behavior", "confirmation_still_needed", "invalidation_conditions",
            "structural_objectives", "risks", "confidence",
        ],
        "additionalProperties": False,
    }


def build_prompt(ma: MarketAnalysis) -> tuple[str, tuple[dict, ...]]:
    """Returns (prompt_text, candidate_structures) -- the latter drives the
    dynamic schema and is persisted for reproducibility."""
    candidates = build_candidate_structures(ma.snapshot, ma.trend)
    candidates_text = "\n".join(f"  {c['name']}: price={c['price']}, state={c['state']}" for c in candidates) or "  (none)"

    levels_text = "\n".join(
        f"  {l.name}: price={l.price}, state={l.state}, distance={l.distance_pts}"
        for l in ma.snapshot.levels if l.state != "far"
    ) or "  (none)"

    trend_text = "(insufficient bars for a trendline fit)"
    if ma.trend is not None:
        trend_text = (
            f"direction={ma.trend.channel_direction}, compression={ma.trend.compression}, "
            f"width={ma.trend.channel_width_pts:.2f}pts, price_position={ma.trend.price_position_in_channel:.2f}"
        )

    candle_text = f"current 4h candle: {ma.candle_4h_current}\ncompleted 4h candle: {ma.candle_4h_completed}"

    vision_text = "(no chart analysis available)"
    if ma.vision is not None:
        vision_text = (
            f"trend_structure={ma.vision.trend_structure!r}, pattern={ma.vision.pattern!r}, "
            f"compression={ma.vision.compression}, breakout={ma.vision.breakout_structure!r}, "
            f"rejection={ma.vision.rejection_structure!r}\n"
            f"  supporting={list(ma.vision.visual_supporting_factors)}\n"
            f"  risk={list(ma.vision.visual_risk_factors)}"
        )

    prompt = (
        f"Timestamp: {ma.timestamp.isoformat()}   Symbol: {ma.symbol}   Price: {ma.snapshot.price}\n"
        f"Session(s): {sorted(ma.snapshot.sessions)}\n\n"
        f"Scanner categories: {list(ma.snapshot.categories)}\n"
        f"Scanner reasoning: {list(ma.snapshot.reasoning)}\n\n"
        f"Levels:\n{levels_text}\n\n"
        f"VWAP: {ma.snapshot.vwap}, distance={ma.snapshot.vwap_distance_pts} "
        f"({ma.snapshot.vwap_distance_daily_atr}x daily ATR)\n\n"
        f"Momentum: 15m_hist={ma.snapshot.macd_15m_hist}, 60m_rsi={ma.snapshot.rsi_60m}, "
        f"4h_hist={ma.snapshot.macd_4h_hist}, 4h_rsi={ma.snapshot.rsi_4h}\n"
        f"{candle_text}\n\n"
        f"Volatility: atr_percentile={ma.snapshot.atr_percentile}, daily_atr={ma.snapshot.daily_atr}\n\n"
        f"Deterministic trend channel: {trend_text}\n\n"
        f"Vision analysis of the chart: {vision_text}\n\n"
        f"Candidate structures (name them ONLY from this list in your structured fields):\n{candidates_text}\n\n"
        f"Form a candidate multi-hour thesis, or return NO_THESIS, per your instructions."
    )
    return prompt, candidates
