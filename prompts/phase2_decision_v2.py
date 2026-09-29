#!/usr/bin/env python3
"""
Phase 2 V2 synthesis prompt + schema (prompt_version="v2"). Never edit this
in place once used for a persisted analysis — a further change is a new
_v3 module. prompts/phase2_decision_v1.py stays exactly as-is, matching
the already-completed 12-episode pilot.

Replaces V1's single 3-way decision (LONG/SHORT/NO_TRADE) with two
independent, always-answered questions:
  1. directional_view (LONG|SHORT, always forced — no neutral/abstain)
  2. tradeability (TRADE|NO_TRADE) — a separate judgment about whether
     NOW is a good time to act on that view.
This lets the model say "SHORT is the stronger thesis, but NO_TRADE
because price is extended/poorly located/unconfirmed" in one frozen
record, replacing the need for V1's separate forced-direction diagnostic
call entirely.

Field order in JSON_SCHEMA is deliberate: each factor array sits
immediately before its confidence number, and analysis_summary (the only
free-text field) comes last — aimed at the V1 pilot's biggest output-
quality issue (structured factor arrays left empty despite a substantive
prose summary in 7/12 episodes) by front-loading structured content.
"""

from __future__ import annotations

from opportunity_snapshot import OpportunitySnapshot
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis
from level_relevance import relevant_levels, relevant_reasoning

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are a disciplined futures trading analyst performing \
point-in-time analysis only. You are NOT placing a trade, sizing a \
position, or setting an entry trigger.

You must answer two SEPARATE questions:

1. DIRECTIONAL VIEW: Which directional thesis does the evidence more \
strongly support, LONG or SHORT? You must always answer this — there is \
no neutral or abstain option for this question, even if the evidence is \
mixed. "directional_confidence" means how strongly the evidence supports \
that direction, NOT the probability the trade would win.

2. TRADEABILITY: Independently of your directional view, is NOW a good \
time to act on it (TRADE), or not (NO_TRADE)? A high-confidence \
directional view can still be NO_TRADE if price is poorly located, \
over-extended, or lacks confirmation. A low-confidence directional view \
can still be TRADE if location and risk/reward are favorable despite \
mixed evidence. These are two different questions — do not let one \
silently determine the other. "tradeability_reason" must always be filled \
in with real, specific content, whether the answer is TRADE or NO_TRADE.

Example shape 1 (illustrative only, not a real setup):
{
  "directional_view": "LONG",
  "directional_bull_case": ["momentum turning up on the 15m timeframe", "price holding above a broken resistance level"],
  "directional_bear_case": ["higher-timeframe momentum still negative"],
  "directional_decisive_factors": ["15m reversal confirmed with expanding histogram"],
  "directional_confidence": 0.7,
  "tradeability": "NO_TRADE",
  "tradeability_supporting_factors": ["directional evidence is reasonably clear"],
  "tradeability_opposing_factors": ["price already extended well above VWAP", "no nearby level to define risk against"],
  "tradeability_reason": "The long thesis is reasonably supported, but price is already extended with no nearby level to anchor risk, so this is not a good moment to act on it.",
  "tradeability_confidence": 0.65,
  "setup": null,
  "key_levels": [],
  "invalidation_concept": null,
  "analysis_summary": "..."
}

Example shape 2 (illustrative only, not a real setup):
{
  "directional_view": "SHORT",
  "directional_bull_case": ["oversold on shorter-timeframe RSI"],
  "directional_bear_case": ["price rejected cleanly from resistance", "falling trend channel with negative slope"],
  "directional_decisive_factors": ["rejection at resistance with a falling channel"],
  "directional_confidence": 0.6,
  "tradeability": "TRADE",
  "tradeability_supporting_factors": ["price at a well-defined resistance level with room to a support target"],
  "tradeability_opposing_factors": ["directional evidence is only moderately strong"],
  "tradeability_reason": "Even though the directional case is only moderate, the location (at resistance, with a clear level to risk against and room to a target) makes this a reasonable moment to act.",
  "tradeability_confidence": 0.6,
  "setup": "resistance rejection",
  "key_levels": [{"name": "ExampleResistance", "price": 100.0, "role": "resistance"}],
  "invalidation_concept": "a close back above the resistance level",
  "analysis_summary": "..."
}

Respond with ONLY a single JSON object in the exact shape shown (no \
markdown fences, no extra keys). Every factor you rely on in \
analysis_summary must also appear in the appropriate factors array above \
it — do not leave a factors array empty if your summary discusses \
relevant evidence. "directional_view" must be exactly "LONG" or "SHORT". \
"tradeability" must be exactly "TRADE" or "NO_TRADE". Both confidence \
fields must be numbers between 0.0 and 1.0.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "directional_view": {"type": "string", "enum": ["LONG", "SHORT"]},
        "directional_bull_case": {"type": "array", "items": {"type": "string"}},
        "directional_bear_case": {"type": "array", "items": {"type": "string"}},
        "directional_decisive_factors": {"type": "array", "items": {"type": "string"}},
        "directional_confidence": {"type": "number"},

        "tradeability": {"type": "string", "enum": ["TRADE", "NO_TRADE"]},
        "tradeability_supporting_factors": {"type": "array", "items": {"type": "string"}},
        "tradeability_opposing_factors": {"type": "array", "items": {"type": "string"}},
        "tradeability_reason": {"type": "string"},
        "tradeability_confidence": {"type": "number"},

        "setup": {"type": ["string", "null"]},
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
        "directional_view", "directional_bull_case", "directional_bear_case",
        "directional_decisive_factors", "directional_confidence",
        "tradeability", "tradeability_supporting_factors", "tradeability_opposing_factors",
        "tradeability_reason", "tradeability_confidence",
        "setup", "key_levels", "invalidation_concept", "analysis_summary",
    ],
    "additionalProperties": False,
}


def build_prompt(
    snapshot: OpportunitySnapshot,
    trend: TrendAnalysis | None,
    vision: VisionAnalysis,
) -> tuple[str, tuple[str, ...]]:
    """Returns (prompt_text, levels_shown_to_model) — the second value is
    persisted separately for reproducibility of exactly what the model saw
    after the relevance filter (§3 of the approved V2 plan)."""
    shown_levels = relevant_levels(snapshot.levels, snapshot.daily_atr)
    excluded_names = {l.name for l in snapshot.levels} - {l.name for l in shown_levels}
    shown_reasoning = relevant_reasoning(snapshot.reasoning, excluded_names)

    levels_text = "\n".join(
        f"  {l.name}: price={l.price}, state={l.state}, distance={l.distance_pts}"
        for l in shown_levels
    ) or "  (none within relevance range)"
    zones_text = "\n".join(
        f"  center={z.center:.2f}, levels={list(z.level_names)}"
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

    prompt = (
        f"Timestamp: {snapshot.timestamp.isoformat()}   Symbol: {snapshot.symbol}   Price: {snapshot.price}\n"
        f"Active session(s): {sorted(snapshot.sessions)}\n\n"
        f"Scanner categories: {list(snapshot.categories)}\n"
        f"Scanner reasoning: {list(shown_reasoning)}\n\n"
        f"Levels (only those within a plausible one-day move of current price are shown):\n{levels_text}\n\n"
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
    return prompt, tuple(l.name for l in shown_levels)
