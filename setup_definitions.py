#!/usr/bin/env python3
"""
The six-setup controlled vocabulary and its fully deterministic candidacy/
trigger/invalidation rules (approved Phase 3 plan, section 8). Four of six
setups map directly onto session_tracker.py's already-frozen, already-
tested `_LevelWatch` state transitions -- no new pattern-detection logic
for those. `trigger_condition`/`invalidation_condition` are NEVER asked of
the model; they are looked up here from setup_type via a fixed table.

MOMENTUM_CONTINUATION is the one level-free setup: candidacy reuses the
scanner's own already-computed momentum_expansion/volatility_expansion
categories directly (opportunity_scanner.py), no new signal invented.
"""

from __future__ import annotations

from typing import Optional

from opportunity_snapshot import OpportunitySnapshot
from trade_plan_config import TradePlanConfig
from trend_analysis import TrendAnalysis

SETUP_TYPES = (
    "BREAKOUT_HOLD", "BREAKOUT_RETEST", "REJECTION", "RECLAIM",
    "PULLBACK_HOLD", "MOMENTUM_CONTINUATION",
)

# Sentinel "trigger level name" for the one setup with no anchoring level --
# TradePlan.reference_price (the CANDIDATE-formation price) is used instead.
MOMENTUM_REFERENCE = "__REFERENCE_PRICE__"

TRIGGER_CONDITION_FOR = {
    "BREAKOUT_HOLD": "CLOSE_BEYOND_LEVEL",
    "BREAKOUT_RETEST": "RETEST_TOUCH_THEN_RESUME",
    "REJECTION": "TOUCH_AND_HOLD",
    "RECLAIM": "TOUCH_AND_HOLD",
    "PULLBACK_HOLD": "TOUCH_AND_HOLD",
    "MOMENTUM_CONTINUATION": "MOMENTUM_PERSISTENCE",
}

INVALIDATION_CONDITION_FOR = {
    "BREAKOUT_HOLD": "TOUCH_BEYOND",
    "BREAKOUT_RETEST": "TOUCH_BEYOND",
    "REJECTION": "CLOSE_BEYOND",
    "RECLAIM": "CLOSE_BEYOND",
    "PULLBACK_HOLD": "CLOSE_BEYOND",
    "MOMENTUM_CONTINUATION": "MOMENTUM_REVERSAL_BEYOND",
}


def compute_candidate_setups(
    snapshot: OpportunitySnapshot,
    trend: Optional[TrendAnalysis],
    direction: str,
    cfg: TradePlanConfig,
    tick_size: float = 0.25,
) -> dict[str, tuple[str, ...]]:
    """Which setup types are structurally valid right now, and for the
    level-anchored ones, which specific level name(s) qualify. Direction
    consistency is read directly off already-computed fields (LevelState.
    distance_pts sign, TrendAnalysis.channel_direction, macd_15m_hist sign)
    -- no new state is invented."""
    candidates: dict[str, tuple[str, ...]] = {}
    want_positive = direction == "LONG"  # LONG wants price currently ABOVE the reference level

    def _names(states: set[str], want_pos: Optional[bool]) -> tuple[str, ...]:
        out = []
        for l in snapshot.levels:
            if l.state in states and l.distance_pts is not None:
                if want_pos is None or (l.distance_pts > 0) == want_pos:
                    out.append(l.name)
        return tuple(out)

    breakout_levels = _names({"broken_holding"}, want_positive)
    if breakout_levels:
        candidates["BREAKOUT_HOLD"] = breakout_levels
        candidates["BREAKOUT_RETEST"] = breakout_levels

    rejection_levels = _names({"rejecting"}, want_positive)
    if rejection_levels:
        candidates["REJECTION"] = rejection_levels

    # "reclaimed" = price recovered back above a level after breaking below it (bullish -> LONG).
    # "lost" = price fell back below a level after breaking above it (bearish -> SHORT).
    reclaim_state = "reclaimed" if direction == "LONG" else "lost"
    reclaim_levels = tuple(l.name for l in snapshot.levels if l.state == reclaim_state)
    if reclaim_levels:
        candidates["RECLAIM"] = reclaim_levels

    if trend is not None:
        aligned = (trend.channel_direction == "rising" and direction == "LONG") or \
                  (trend.channel_direction == "falling" and direction == "SHORT")
        if aligned:
            touch_dist_pts = cfg.touch_dist_ticks * tick_size
            if direction == "LONG":
                ref_name, ref_dist = "TrendlineSupport", trend.distance_to_support_pts
            else:
                ref_name, ref_dist = "TrendlineResistance", trend.distance_to_resistance_pts
            if abs(ref_dist) <= touch_dist_pts:
                candidates["PULLBACK_HOLD"] = (ref_name,)

    momentum_context = bool({"momentum_expansion", "volatility_expansion"} & set(snapshot.categories))
    if momentum_context and snapshot.macd_15m_hist is not None:
        sign_ok = (snapshot.macd_15m_hist > 0) if direction == "LONG" else (snapshot.macd_15m_hist < 0)
        if sign_ok:
            candidates["MOMENTUM_CONTINUATION"] = (MOMENTUM_REFERENCE,)

    return candidates
