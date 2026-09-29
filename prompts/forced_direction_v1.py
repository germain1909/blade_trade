#!/usr/bin/env python3
"""
DIAGNOSTIC-ONLY prompt. Not part of the Phase 2 decision pipeline and must
never be used to place, size, or influence a trade. Its only purpose is to
separate two possible explanations for a NO_TRADE-heavy pilot:

  A) the model sees a directional lean but phase2_decision_v1's prompt/
     threshold framing causes it to abstain, vs.
  B) the model genuinely sees ~balanced evidence with no lean either way.

Runs on the exact same frozen OpportunitySnapshot/TrendAnalysis/
VisionAnalysis a given episode's real Phase 2 analysis already used (reused
from the persisted `analyses` row, not recomputed) — the only variable
changed is the question asked. NO_TRADE is deliberately not offered as an
option here; "decision" enums do not include it.

prompt_version is a distinct family ("diagnostic_v1"), persisted to a
separate `diagnostic_forced_direction` table — this must never be
confused with, or overwrite, a real phase2_decision_v1 row.
"""

from __future__ import annotations

PROMPT_VERSION = "diagnostic_v1"

SYSTEM_PROMPT = """This is a DIAGNOSTIC research exercise, not a real trading \
decision. You are given the exact same point-in-time evidence used for a \
real analysis, but for this exercise NO_TRADE is not an available answer. \
You must identify which directional thesis currently has relatively \
stronger evidence, even if the overall case is weak, mixed, or you'd \
normally decline to act on it.

This is being used to measure whether a directional lean exists in the \
evidence at all, separate from whether that lean is strong enough to act \
on. Answer honestly about which side is relatively stronger, and reflect \
genuine uncertainty in "directional_confidence" (a low confidence like 0.5 \
is a fully valid answer to give if the evidence is close to balanced) \
rather than picking a side arbitrarily to sound decisive.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "forced_direction": "LONG",
  "directional_confidence": 0.5,
  "bull_case": ["short factor", "short factor"],
  "bear_case": ["short factor", "short factor"],
  "decisive_factors": ["short factor describing what tipped it"]
}
"forced_direction" must be exactly "LONG" or "SHORT". "directional_confidence" \
must be a number between 0.0 and 1.0, where 0.5 means the two sides are \
nearly balanced and you are essentially guessing which is marginally ahead.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "forced_direction": {"type": "string", "enum": ["LONG", "SHORT"]},
        "directional_confidence": {"type": "number"},
        "bull_case": {"type": "array", "items": {"type": "string"}},
        "bear_case": {"type": "array", "items": {"type": "string"}},
        "decisive_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["forced_direction", "directional_confidence", "bull_case", "bear_case", "decisive_factors"],
    "additionalProperties": False,
}


def build_prompt(snapshot: dict, trend: dict | None, vision: dict) -> str:
    levels_text = "\n".join(
        f"  {l['name']}: price={l['price']}, state={l['state']}, distance={l['distance_pts']}"
        for l in snapshot["levels"]
    )
    zones_text = "\n".join(
        f"  center={z['center']:.2f}, levels={list(z['level_names'])}"
        for z in snapshot.get("confluence_zones", [])
    ) or "  (none)"

    if trend is not None:
        trend_text = (
            f"  support: slope={trend['support_slope']:.4f}/bar, now={trend['support_price_now']:.2f}\n"
            f"  resistance: slope={trend['resistance_slope']:.4f}/bar, now={trend['resistance_price_now']:.2f}\n"
            f"  channel_width={trend['channel_width_pts']:.2f}pts ({trend['channel_width_atr_mult']:.2f}x daily ATR)\n"
            f"  direction={trend['channel_direction']}, compression={trend['compression']}\n"
            f"  price_position_in_channel={trend['price_position_in_channel']:.2f}\n"
            f"  distance_to_support={trend['distance_to_support_pts']:+.2f}, "
            f"distance_to_resistance={trend['distance_to_resistance_pts']:+.2f}\n"
        )
    else:
        trend_text = "  (insufficient bars for a trendline fit)\n"

    return (
        f"Timestamp: {snapshot['timestamp']}   Symbol: {snapshot['symbol']}   Price: {snapshot['price']}\n"
        f"Active session(s): {snapshot['sessions']}\n\n"
        f"Scanner categories: {snapshot['categories']}\n"
        f"Scanner reasoning: {snapshot['reasoning']}\n\n"
        f"Levels:\n{levels_text}\n\n"
        f"Confluence zones:\n{zones_text}\n\n"
        f"VWAP: {snapshot['vwap']}, distance={snapshot['vwap_distance_pts']}, "
        f"distance_in_daily_atr={snapshot['vwap_distance_daily_atr']}, extension_flag={snapshot['vwap_extension_flag']}\n\n"
        f"Momentum: 15m_macd_hist={snapshot['macd_15m_hist']}, "
        f"15m_bottom_reversal_recent={snapshot['macd_15m_bottom_recent']}, "
        f"15m_top_reversal_recent={snapshot['macd_15m_top_recent']}, "
        f"60m_rsi={snapshot['rsi_60m']}, 4h_macd_hist={snapshot['macd_4h_hist']}, 4h_rsi={snapshot['rsi_4h']}\n\n"
        f"Volatility: atr_60bar={snapshot['atr_60bar']}, atr_percentile={snapshot['atr_percentile']}, "
        f"daily_atr={snapshot['daily_atr']}\n\n"
        f"Deterministic trend channel (ground truth, do not re-derive):\n{trend_text}\n"
        f"Vision analysis of the chart:\n"
        f"  trend_structure={vision['trend_structure']!r}, pattern={vision['pattern']!r}, "
        f"compression={vision['compression']}\n"
        f"  breakout_structure={vision['breakout_structure']!r}, rejection_structure={vision['rejection_structure']!r}\n"
        f"  visual_supporting_factors={list(vision['visual_supporting_factors'])}\n"
        f"  visual_risk_factors={list(vision['visual_risk_factors'])}\n\n"
        f"Which directional thesis currently has stronger evidence: LONG or SHORT?"
    )
