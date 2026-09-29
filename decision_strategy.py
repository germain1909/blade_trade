#!/usr/bin/env python3
"""
Decision/Strategy interface — the swappable layer that judges a candidate
setup (already detected deterministically) as TAKE or SKIP.

`RuleBasedStrategy` reproduces backtest_manual_week.py's original
`require_macd_turn` boolean gate exactly, so an experiment run with it is
bit-for-bit identical to the pre-existing behavior — this is the baseline
(Experiment A) that `LLMStrategy` (Experiment B) is compared against.

The backtest engine (backtest_manual_week.py's run_week()) does not need to
know which concrete Strategy produced a Decision — it only calls
`strategy.decide(snapshot)` and branches on `.action`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Protocol

from market_snapshot import MarketSnapshot

Action = Literal["TAKE", "SKIP"]


@dataclass
class Decision:
    action: Action
    confidence: float = 0.0
    thesis: str = ""
    supporting_factors: list[str] = field(default_factory=list)
    opposing_factors: list[str] = field(default_factory=list)
    invalidation: str = ""
    raw_response: str = ""  # full LLM output (empty for RuleBasedStrategy), kept for the experiment log


@dataclass
class Outcome:
    """What actually (or hypothetically) happened after a Decision."""
    entry_px: float
    stop_px: float
    target_px: float
    exit_time: Optional[datetime]
    exit_px: Optional[float]
    exit_reason: Optional[str]
    pnl_usd: Optional[float]
    pnl_r: Optional[float]
    mfe: Optional[float]
    mae: Optional[float]
    outcome_type: Literal["actual", "hypothetical"]


class Strategy(Protocol):
    def decide(self, snapshot: MarketSnapshot) -> Decision: ...


class RuleBasedStrategy:
    """Reproduces the original boolean MACD-turn gate as a Decision.

    TAKE iff the 15m MACD "substantial reversal" flag matching the setup's
    direction (`macd_15m_bottom_recent` for longs, `macd_15m_top_recent` for
    shorts) was seen recently — exactly `backtest_manual_week.py`'s original
    `require_macd_turn` check, just expressed as a Decision instead of an
    inline boolean.
    """

    def __init__(self, require_macd_turn: bool = True):
        self.require_macd_turn = require_macd_turn

    def decide(self, snapshot: MarketSnapshot) -> Decision:
        if not self.require_macd_turn:
            return Decision(
                action="TAKE", confidence=1.0,
                thesis="MACD gate disabled — every cross/fail is taken unconditionally.",
            )

        is_long = snapshot.setup_direction == "long"
        macd_ok = snapshot.macd_15m_bottom_recent if is_long else snapshot.macd_15m_top_recent
        label = "bottom" if is_long else "top"

        if macd_ok:
            return Decision(
                action="TAKE", confidence=1.0,
                thesis=f"15m MACD showed a substantial {label} reversal recently.",
                supporting_factors=[f"macd_15m_{label}_recent=True"],
            )
        return Decision(
            action="SKIP", confidence=1.0,
            thesis=f"No matching 15m MACD {label} reversal seen recently.",
            opposing_factors=[f"macd_15m_{label}_recent=False"],
        )


class LLMStrategy:
    """Delegates the TAKE/SKIP call to a local LLM via an LLMProvider.

    Fails closed: any parse failure or unexpected response defaults to SKIP,
    never TAKE — a malformed response should never silently open a trade.
    """

    def __init__(self, provider, prompt_version: str = "v1"):
        self.provider = provider
        self.prompt_version = prompt_version

    def decide(self, snapshot: MarketSnapshot) -> Decision:
        from prompts.take_skip_v1 import build_prompt, parse_response

        system, user = build_prompt(snapshot)
        raw = self.provider.complete(system, user)
        return parse_response(raw)
