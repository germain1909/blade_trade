#!/usr/bin/env python3
"""
Vision-analysis prompt + schema (prompt_version="v1"). Bump the module name
(vision_v2.py, ...) rather than editing this one in place once it's been
used for a persisted analysis, so past rows stay reproducible.

The model is shown the chart PLUS the already-computed TrendAnalysis
numbers as text (slope, channel width, direction) — it is asked about
things Python cannot compute from the raw bars alone (pattern geometry,
how candles behave near the drawn lines), never asked to re-derive the
slope/direction/channel-width Python already has exactly. Re-deriving a
number the caller already computed exactly, from a picture, would only
add noise on top of ground truth.
"""

from __future__ import annotations

from opportunity_snapshot import OpportunitySnapshot
from trend_analysis import TrendAnalysis

PROMPT_VERSION = "v1"

_PATTERN_VOCABULARY = (
    "double top, double bottom, triple top, triple bottom, head and shoulders, "
    "inverse head and shoulders, ascending triangle, descending triangle, "
    "symmetrical triangle, rising wedge, falling wedge, bull flag, bear flag, "
    "rounded top, rounded bottom, channel breakout, channel breakdown"
)

SYSTEM_PROMPT = f"""You are a market chart analyst. You are shown a candlestick \
chart of recent price action ending at a specific point in time, with the \
already-computed trend channel (support/resistance lines) drawn on it, plus \
the exact numeric values of that channel given to you as text.

Do NOT re-estimate the slope, channel width, or direction of the drawn \
lines — those exact numbers are already given to you as text and are \
correct by construction. Your job is to describe what those numbers alone \
don't capture: overall visual structure, whether a classical chart pattern \
is present, how price is behaving near the drawn lines (compressing into \
them, breaking through, rejecting from), and specific visual factors that \
support or work against a trade at this moment.

If you recognize a classical pattern, name it using standard terminology \
(examples: {_PATTERN_VOCABULARY}). If none clearly applies, use null.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{{
  "trend_structure": "short phrase, e.g. 'rising channel', 'range-bound', 'choppy'",
  "pattern": "named pattern or null",
  "compression": true,
  "breakout_structure": "description of a breakout in progress, or null",
  "rejection_structure": "description of a rejection in progress, or null",
  "visual_supporting_factors": ["short factor", "short factor"],
  "visual_risk_factors": ["short factor"]
}}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "trend_structure": {"type": "string"},
        "pattern": {"type": ["string", "null"]},
        "compression": {"type": "boolean"},
        "breakout_structure": {"type": ["string", "null"]},
        "rejection_structure": {"type": ["string", "null"]},
        "visual_supporting_factors": {"type": "array", "items": {"type": "string"}},
        "visual_risk_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "trend_structure", "pattern", "compression", "breakout_structure",
        "rejection_structure", "visual_supporting_factors", "visual_risk_factors",
    ],
    "additionalProperties": False,
}


def build_prompt(snapshot: OpportunitySnapshot, trend: TrendAnalysis | None) -> str:
    if trend is not None:
        trend_text = (
            f"Support line: slope={trend.support_slope:.4f} pts/bar, currently at {trend.support_price_now:.2f}\n"
            f"Resistance line: slope={trend.resistance_slope:.4f} pts/bar, currently at {trend.resistance_price_now:.2f}\n"
            f"Channel width: {trend.channel_width_pts:.2f} pts ({trend.channel_width_atr_mult:.2f}x daily ATR)\n"
            f"Channel direction: {trend.channel_direction}\n"
            f"Price position in channel: {trend.price_position_in_channel:.2f} (0.0=at support, 1.0=at resistance)\n"
            f"Distance to support: {trend.distance_to_support_pts:+.2f} pts, "
            f"to resistance: {trend.distance_to_resistance_pts:+.2f} pts\n"
        )
    else:
        trend_text = "(Not enough bars for a reliable trendline fit — judge visual structure only.)\n"

    return (
        f"Chart: {snapshot.symbol}, {len(snapshot.recent_bars)} most recent 1-minute bars, "
        f"ending {snapshot.timestamp.isoformat()}.\n\n"
        f"Already-computed trend channel (ground truth, do not re-derive):\n{trend_text}\n"
        f"Scanner context: categories={list(snapshot.categories)}\n"
        f"Describe the chart per your instructions."
    )
