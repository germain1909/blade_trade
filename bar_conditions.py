#!/usr/bin/env python3
"""
Shared, pure bar-condition primitives used by both trade_plan_replay.py
(trigger/invalidation/target checking) and active_thesis.py (the forced-
invalidation circuit breaker). Kept in one place so both modules check
"has price crossed beyond X" the same way, rather than two slightly
different implementations drifting apart.
"""

from __future__ import annotations


def crossed_beyond(direction: str, level_price: float, condition: str,
                    bar_high: float, bar_low: float, bar_close: float) -> bool:
    """direction: 'above' | 'below' -- the side price must be on for this to count.
    condition: 'CLOSE_BEYOND' (bar close test) | 'TOUCH_BEYOND' (intrabar high/low test)."""
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    if condition == "CLOSE_BEYOND":
        return bar_close > level_price if direction == "above" else bar_close < level_price
    if condition == "TOUCH_BEYOND":
        return bar_high > level_price if direction == "above" else bar_low < level_price
    raise ValueError(f"unknown condition: {condition!r}")


def touched_within(level_price: float, tolerance_pts: float, bar_high: float, bar_low: float) -> bool:
    """Did this bar's range come within tolerance_pts of level_price (a 'touch')."""
    return bar_low - tolerance_pts <= level_price <= bar_high + tolerance_pts
