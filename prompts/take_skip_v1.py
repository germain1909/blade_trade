#!/usr/bin/env python3
"""
Prompt template + response parser for the TAKE/SKIP context-filter experiment
(prompt_version="v1"). Bump the module name (take_skip_v2.py, ...) rather
than editing this one in place once an experiment has used it, so past
experiment rows stay reproducible against the exact prompt they were run with.
"""

from __future__ import annotations

import json

from decision_strategy import Decision
from market_snapshot import MarketSnapshot

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a disciplined futures trading risk analyst.

You are shown ONE candidate trade setup that a deterministic system has \
already detected. You are NOT finding the setup and you are NOT predicting \
where price will go next. Your only job is to judge whether the surrounding \
market context supports TAKING this specific setup, or whether it should be \
SKIPPED.

There is no requirement to trade every setup you are shown. When the \
context is mixed, weak, or you are unsure, prefer SKIP.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no trailing commas, no extra keys):
{
  "decision": "TAKE",
  "confidence": 0.0,
  "thesis": "one or two sentence reasoning",
  "supporting_factors": ["short factor", "short factor"],
  "opposing_factors": ["short factor"],
  "invalidation": "what would prove this wrong"
}
"decision" must be exactly "TAKE" or "SKIP". "confidence" must be a number \
between 0.0 and 1.0.
"""


def build_prompt(snapshot: MarketSnapshot) -> tuple[str, str]:
    bars_preview = "\n".join(
        f"  {b[0]}  O={b[1]:.2f} H={b[2]:.2f} L={b[3]:.2f} C={b[4]:.2f} V={b[5]:.0f}"
        for b in snapshot.recent_bars[-10:]
    )
    user = (
        f"Setup: {snapshot.setup_type.upper()} at level '{snapshot.level_name}' "
        f"({dict(snapshot.levels).get(snapshot.level_name)})\n"
        f"Proposed direction: {snapshot.setup_direction.upper()}\n"
        f"Timestamp: {snapshot.timestamp.isoformat()}   Symbol: {snapshot.symbol}   "
        f"Price: {snapshot.price}\n"
        f"Session: {snapshot.session}\n\n"
        f"Distances to known levels (price - level, in points):\n"
        + "\n".join(f"  {name}: {dist:+.2f}" for name, dist in snapshot.distance_to_levels)
        + "\n\n"
        f"Indicators as of this exact moment (nothing later):\n"
        f"  15m MACD histogram: {snapshot.macd_15m_hist}\n"
        f"    -> substantial bottom reversal seen recently: {snapshot.macd_15m_bottom_recent}\n"
        f"    -> substantial top reversal seen recently: {snapshot.macd_15m_top_recent}\n"
        f"  60m RSI: {snapshot.rsi_60m}\n"
        f"  4h MACD histogram: {snapshot.macd_4h_hist}\n"
        f"  4h RSI: {snapshot.rsi_4h}\n"
        f"  Higher-timeframe trend read: {snapshot.higher_tf_trend}\n"
        f"  Short-lookback ATR: {snapshot.atr}\n\n"
        f"Auto-tagged pattern labels active at this moment: {list(snapshot.tags)}\n\n"
        f"Most recent 1-minute bars (oldest to newest, all at or before the decision time):\n"
        f"{bars_preview}\n"
    )
    return SYSTEM_PROMPT, user


def parse_response(raw: str) -> Decision:
    """Fails closed: any parse failure or unrecognized decision value
    defaults to SKIP, never TAKE."""
    data: dict = {}
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(raw[start:end])
    except json.JSONDecodeError:
        data = {}

    action = str(data.get("decision", "SKIP")).strip().upper()
    if action not in ("TAKE", "SKIP"):
        action = "SKIP"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    supporting = data.get("supporting_factors", [])
    opposing = data.get("opposing_factors", [])

    return Decision(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        thesis=str(data.get("thesis", "")),
        supporting_factors=[str(f) for f in supporting] if isinstance(supporting, list) else [],
        opposing_factors=[str(f) for f in opposing] if isinstance(opposing, list) else [],
        invalidation=str(data.get("invalidation", "")),
        raw_response=raw,
    )
