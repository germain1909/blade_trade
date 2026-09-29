#!/usr/bin/env python3
"""
Final Phase 2 synthesis prompt + schema (prompt_version="v1"). Bump the
module name (phase2_decision_v2.py, ...) rather than editing this one in
place once it's been used for a persisted analysis.

This is ANALYSIS ONLY: LONG / SHORT / NO_TRADE plus reasoning. No
TradeIdea, no ActiveThesis, no entry trigger, no position sizing, no order
execution — those are explicitly out of scope for Phase 2 regardless of
what this call returns. "confidence" is documented to the model as
evidentiary strength, not a win-probability estimate — a deliberately
different question, because probability-of-winning is not something either
the model or Python has the information to answer from one snapshot.

NO_TRADE is a first-class value in the schema's enum, not a suggestion in
the prompt text — the json_schema grammar constraint makes it structurally
impossible for the model to return anything outside {LONG, SHORT,
NO_TRADE}, so it can never be forced into manufacturing a direction it
doesn't have evidence for.

Receives Python's own deterministic numbers directly (OpportunitySnapshot,
TrendAnalysis) rather than a prose paraphrase of them — avoids the "LLM
call spent translating numbers into prose that another LLM re-interprets"
waste this project explicitly avoids elsewhere (see the trend_agent.py
rejection in the approved Phase 2 plan).
"""

from __future__ import annotations

from opportunity_snapshot import OpportunitySnapshot
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a disciplined futures trading analyst performing \
point-in-time analysis only. You are NOT placing a trade, sizing a \
position, or setting an entry trigger — this is a structured judgment \
call: does the available evidence, taken together, support a LONG bias, a \
SHORT bias, or does it not clearly support either (NO_TRADE)?

You are given: (1) a deterministic scanner's reason for flagging this \
moment, (2) every tracked level's exact price/state/distance, (3) exact \
momentum and volatility readings, (4) an exact trend-channel fit (slope, \
width, direction — already computed, treat as ground truth), and (5) a \
vision model's read of the chart's visual structure and pattern.

There is no requirement to pick a direction. If evidence is mixed, weak, \
contradictory, or simply insufficient, you must return NO_TRADE — this is \
a normal, expected, equally valid outcome, not a failure to find a trade.

"confidence" means how strongly the available evidence supports your \
stated analysis — it is NOT an estimate of the probability this trade \
would win. A NO_TRADE decision can have high confidence (strong evidence \
that conditions are unclear/mixed) or low confidence (genuinely unsure \
even about that).

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "decision": "LONG",
  "confidence": 0.0,
  "setup": "short label for the setup type, or null",
  "thesis": "one to three sentence reasoning, or null",
  "reason": "why NO_TRADE if applicable, else null",
  "supporting_factors": ["short factor", "short factor"],
  "opposing_factors": ["short factor"],
  "key_levels": [{"name": "AsiaHigh", "price": 25012.5, "role": "resistance"}],
  "invalidation_concept": "what would prove this wrong, or null",
  "analysis_summary": "one paragraph summary of the full analysis"
}
"decision" must be exactly "LONG", "SHORT", or "NO_TRADE". "confidence" \
must be a number between 0.0 and 1.0.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["LONG", "SHORT", "NO_TRADE"]},
        "confidence": {"type": "number"},
        "setup": {"type": ["string", "null"]},
        "thesis": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "supporting_factors": {"type": "array", "items": {"type": "string"}},
        "opposing_factors": {"type": "array", "items": {"type": "string"}},
        "key_levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "role": {"type": "string"},
                },
                "required": ["name", "price", "role"],
                "additionalProperties": False,
            },
        },
        "invalidation_concept": {"type": ["string", "null"]},
        "analysis_summary": {"type": "string"},
    },
    "required": [
        "decision", "confidence", "setup", "thesis", "reason",
        "supporting_factors", "opposing_factors", "key_levels",
        "invalidation_concept", "analysis_summary",
    ],
    "additionalProperties": False,
}


def build_prompt(
    snapshot: OpportunitySnapshot,
    trend: TrendAnalysis | None,
    vision: VisionAnalysis,
) -> str:
    levels_text = "\n".join(
        f"  {l.name}: price={l.price}, state={l.state}, distance={l.distance_pts}"
        for l in snapshot.levels
    )
    zones_text = "\n".join(
        f"  center={z.center:.2f}, levels={list(z.level_names)}, count={z.count}"
        for z in snapshot.confluence_zones
    ) or "  (none)"

    if trend is not None:
        trend_text = (
            f"  support: slope={trend.support_slope:.4f}/bar, now={trend.support_price_now:.2f}\n"
            f"  resistance: slope={trend.resistance_slope:.4f}/bar, now={trend.resistance_price_now:.2f}\n"
            f"  channel_width={trend.channel_width_pts:.2f}pts ({trend.channel_width_atr_mult:.2f}x daily ATR)\n"
            f"  direction={trend.channel_direction}, compression={trend.compression}\n"
            f"  price_position_in_channel={trend.price_position_in_channel:.2f}\n"
            f"  distance_to_support={trend.distance_to_support_pts:+.2f}, "
            f"distance_to_resistance={trend.distance_to_resistance_pts:+.2f}\n"
        )
    else:
        trend_text = "  (insufficient bars for a trendline fit)\n"

    return (
        f"Timestamp: {snapshot.timestamp.isoformat()}   Symbol: {snapshot.symbol}   Price: {snapshot.price}\n"
        f"Active session(s): {sorted(snapshot.sessions)}\n\n"
        f"Scanner categories: {list(snapshot.categories)}\n"
        f"Scanner reasoning: {list(snapshot.reasoning)}\n\n"
        f"Levels:\n{levels_text}\n\n"
        f"Confluence zones:\n{zones_text}\n\n"
        f"VWAP: {snapshot.vwap}, distance={snapshot.vwap_distance_pts}, "
        f"distance_in_daily_atr={snapshot.vwap_distance_daily_atr}, extension_flag={snapshot.vwap_extension_flag}\n\n"
        f"Momentum: 15m_macd_hist={snapshot.macd_15m_hist}, "
        f"15m_bottom_reversal_recent={snapshot.macd_15m_bottom_recent}, "
        f"15m_top_reversal_recent={snapshot.macd_15m_top_recent}, "
        f"60m_rsi={snapshot.rsi_60m}, 4h_macd_hist={snapshot.macd_4h_hist}, 4h_rsi={snapshot.rsi_4h}\n\n"
        f"Volatility: atr_60bar={snapshot.atr_60bar}, atr_percentile={snapshot.atr_percentile}, "
        f"daily_atr={snapshot.daily_atr}\n\n"
        f"Deterministic trend channel (ground truth, do not re-derive):\n{trend_text}\n"
        f"Vision analysis of the chart:\n"
        f"  trend_structure={vision.trend_structure!r}, pattern={vision.pattern!r}, "
        f"compression={vision.compression}\n"
        f"  breakout_structure={vision.breakout_structure!r}, rejection_structure={vision.rejection_structure!r}\n"
        f"  visual_supporting_factors={list(vision.visual_supporting_factors)}\n"
        f"  visual_risk_factors={list(vision.visual_risk_factors)}\n\n"
        f"Provide your analysis per your instructions."
    )
