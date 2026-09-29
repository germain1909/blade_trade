#!/usr/bin/env python3
"""
Deterministic 1-minute-clock TradePlan monitoring -- zero LLM calls. Every
bar, from CANDIDATE-formation onward, checks trigger/invalidation
conditions per setup_type's fixed mapping (setup_definitions.py). Same-bar
ambiguity always resolves conservatively (bar_conditions.py's primitives,
applied with a fixed precedence order here, never configurable).

Watchers are per-plan, stateful across bars (mirroring session_tracker.py's
_LevelWatch pattern) -- they keep running while a plan is CANDIDATE (for
audit) and continue unchanged when a plan later becomes ARMED; arming
itself never resets any watcher's internal counters.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from bar_conditions import crossed_beyond, touched_within
from trade_plan import TradePlan
from trade_plan_config import TradePlanConfig


class Bar(NamedTuple):
    timestamp: object
    open: float
    high: float
    low: float
    close: float


def _favorable_direction(plan_direction: str) -> str:
    """The side price must be on for 'away from the reference in the
    thesis direction' -- used to phrase distance checks uniformly."""
    return "above" if plan_direction == "LONG" else "below"


def _adverse_direction(plan_direction: str) -> str:
    return "below" if plan_direction == "LONG" else "above"


class PreEntryWatch:
    """Tracks a CANDIDATE or ARMED plan's trigger/invalidation conditions,
    bar by bar, from the bar immediately after CANDIDATE formation."""

    def __init__(self, plan: TradePlan, cfg: TradePlanConfig, tick_size: float = 0.25):
        self.plan = plan
        self.cfg = cfg
        self.confirm_distance_pts = cfg.confirm_distance_ticks * tick_size
        self.touch_dist_pts = cfg.touch_dist_ticks * tick_size
        self.hold_bars = 0
        self.retest_touched = False
        self._first_bar_checked = False
        self.adverse_hold_bars = 0  # for MOMENTUM_REVERSAL_BEYOND's sign-flip persistence

    def _reference_price(self) -> float:
        return self.plan.trigger_level_price if self.plan.trigger_level_price is not None else self.plan.reference_price

    def _favorable_distance_met(self, close_price: float) -> bool:
        ref = self._reference_price()
        if self.plan.direction == "LONG":
            return (close_price - ref) >= self.confirm_distance_pts
        return (ref - close_price) >= self.confirm_distance_pts

    def _adverse_distance_met(self, close_price: float) -> bool:
        ref = self._reference_price()
        if self.plan.direction == "LONG":
            return (ref - close_price) >= self.confirm_distance_pts
        return (close_price - ref) >= self.confirm_distance_pts

    def check_invalidation(self, bar: Bar, macd_15m_hist: Optional[float] = None) -> bool:
        """`macd_15m_hist` is only consulted for MOMENTUM_REVERSAL_BEYOND
        (MOMENTUM_CONTINUATION plans) -- ignored otherwise."""
        cond = self.plan.invalidation_condition
        direction = _adverse_direction(self.plan.direction)
        if cond in ("TOUCH_BEYOND", "CLOSE_BEYOND"):
            return crossed_beyond(direction, self.plan.invalidation_level_price, cond,
                                   bar.high, bar.low, bar.close)
        if cond == "MOMENTUM_REVERSAL_BEYOND":
            if self._adverse_distance_met(bar.close):
                return True
            # Second leg: 15m histogram sign disagreeing with the thesis direction,
            # sustained for hold_confirm_bars consecutive bars.
            if macd_15m_hist is None:
                self.adverse_hold_bars = 0
                return False
            sign_disagrees = (macd_15m_hist <= 0) if self.plan.direction == "LONG" else (macd_15m_hist >= 0)
            self.adverse_hold_bars = self.adverse_hold_bars + 1 if sign_disagrees else 0
            return self.adverse_hold_bars >= self.cfg.hold_confirm_bars
        raise ValueError(f"unknown invalidation_condition: {cond!r}")

    def check_trigger(self, bar: Bar, macd_15m_hist: Optional[float] = None) -> bool:
        cond = self.plan.trigger_condition

        if cond == "CLOSE_BEYOND_LEVEL":
            # Already true at CANDIDATE-formation time by construction (BREAKOUT_HOLD's candidacy
            # requires the level to already be broken_holding) -- fires on the first bar checked.
            fired = not self._first_bar_checked
            self._first_bar_checked = True
            return fired

        if cond == "TOUCH_AND_HOLD":
            self.hold_bars = self.hold_bars + 1 if self._favorable_distance_met(bar.close) else 0
            return self.hold_bars >= self.cfg.hold_confirm_bars

        if cond == "RETEST_TOUCH_THEN_RESUME":
            if not self.retest_touched:
                if touched_within(self.plan.trigger_level_price, self.touch_dist_pts, bar.high, bar.low):
                    self.retest_touched = True
                return False
            self.hold_bars = self.hold_bars + 1 if self._favorable_distance_met(bar.close) else 0
            return self.hold_bars >= self.cfg.hold_confirm_bars

        if cond == "MOMENTUM_PERSISTENCE":
            sign_ok = macd_15m_hist is not None and (
                (macd_15m_hist > 0) if self.plan.direction == "LONG" else (macd_15m_hist < 0)
            )
            if sign_ok and self._favorable_distance_met(bar.close):
                self.hold_bars += 1
            else:
                self.hold_bars = 0
            return self.hold_bars >= self.cfg.hold_confirm_bars

        raise ValueError(f"unknown trigger_condition: {cond!r}")


class PostEntryWatch:
    """Tracks an ENTERED plan's hard boundaries. Immutable inputs (entry
    price, resolved target price) -- nothing here, including the
    PostEntryReview LLM call, can change these. Uses the SAME
    invalidation_condition the setup was formed with (§6/§8 of the
    approved plan) -- not hardened to an intrabar touch for every setup;
    REJECTION/RECLAIM/PULLBACK_HOLD keep their CLOSE_BEYOND semantics
    post-entry too."""

    def __init__(self, plan: TradePlan, entry_price: float, cfg: TradePlanConfig, tick_size: float = 0.25):
        self.plan = plan
        self.cfg = cfg
        self.confirm_distance_pts = cfg.confirm_distance_ticks * tick_size
        self.entry_price = entry_price
        self.target_price = resolve_target_price(plan, entry_price)
        self.adverse_hold_bars = 0

    def check_invalidation(self, bar: Bar, macd_15m_hist: Optional[float] = None) -> bool:
        cond = self.plan.invalidation_condition
        direction = _adverse_direction(self.plan.direction)
        if cond in ("TOUCH_BEYOND", "CLOSE_BEYOND"):
            return crossed_beyond(direction, self.plan.invalidation_level_price, cond,
                                   bar.high, bar.low, bar.close)
        if cond == "MOMENTUM_REVERSAL_BEYOND":
            ref = self.plan.reference_price
            adverse_met = (ref - bar.close) >= self.confirm_distance_pts if self.plan.direction == "LONG" \
                else (bar.close - ref) >= self.confirm_distance_pts
            if adverse_met:
                return True
            if macd_15m_hist is None:
                self.adverse_hold_bars = 0
                return False
            sign_disagrees = (macd_15m_hist <= 0) if self.plan.direction == "LONG" else (macd_15m_hist >= 0)
            self.adverse_hold_bars = self.adverse_hold_bars + 1 if sign_disagrees else 0
            return self.adverse_hold_bars >= self.cfg.hold_confirm_bars
        raise ValueError(f"unknown invalidation_condition: {cond!r}")

    def check_target(self, bar: Bar) -> bool:
        direction = _favorable_direction(self.plan.direction)
        return crossed_beyond(direction, self.target_price, "TOUCH_BEYOND", bar.high, bar.low, bar.close)


def check_pre_entry_bar_outcome(
    watch: PreEntryWatch, bar: Bar, is_armed: bool, macd_15m_hist: Optional[float] = None,
) -> Optional[str]:
    """Returns 'ENTERED' | 'MISSED' | 'INVALIDATED_BEFORE_ENTRY' | None for
    this bar. Both checks are evaluated unconditionally (so watcher state
    advances correctly every bar); same-bar ambiguity (both trigger and
    invalidation fire) resolves conservatively -- the trigger never fires
    on a bar where invalidation also fires. A trigger firing while not
    ARMED resolves MISSED, never a retroactive ENTERED once armed later."""
    invalidated = watch.check_invalidation(bar, macd_15m_hist)
    triggered = watch.check_trigger(bar, macd_15m_hist)
    if invalidated:
        return "INVALIDATED_BEFORE_ENTRY"
    if triggered:
        return "ENTERED" if is_armed else "MISSED"
    return None


def check_post_entry_bar_outcome(
    watch: PostEntryWatch, bar: Bar, bars_held: int, max_hold_bars: int,
    thesis_invalidated_flag: bool, macd_15m_hist: Optional[float] = None,
) -> Optional[str]:
    """Returns 'INVALIDATION_HIT' | 'TARGET_HIT' | 'THESIS_EXIT' |
    'MAX_HOLD_EXIT' | None for this bar, in that fixed precedence order --
    the harder, more deterministic condition always wins a same-bar tie;
    an advisory THESIS_EXIT never overrides a harder condition that also
    fired this bar."""
    if watch.check_invalidation(bar, macd_15m_hist):
        return "INVALIDATION_HIT"
    if watch.check_target(bar):
        return "TARGET_HIT"
    if thesis_invalidated_flag:
        return "THESIS_EXIT"
    if bars_held >= max_hold_bars:
        return "MAX_HOLD_EXIT"
    return None


def resolve_target_price(plan: TradePlan, entry_price: float) -> float:
    """Python-computed, never model-stated. A named-level target resolves
    directly to its already-known price; an R-multiple target is defined
    off the actual entry/invalidation distance, resolved only once entry
    price is known (which is why this can't be computed at formation time
    for any setup whose trigger isn't CLOSE_BEYOND_LEVEL)."""
    if plan.target_choice_type == "LEVEL":
        return plan.target_price_hint
    risk_anchor = plan.invalidation_level_price if plan.invalidation_level_price is not None else plan.reference_price
    risk = abs(entry_price - risk_anchor)
    r = plan.target_r_multiple
    return entry_price + r * risk if plan.direction == "LONG" else entry_price - r * risk
