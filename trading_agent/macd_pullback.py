from __future__ import annotations

"""
MACD Leg Pullback Strategy — Version 1
=======================================
Core idea: 15M MACD cross identifies the momentum leg direction.
Wait for price to impulse, pull back to 5M 15 EMA or ATR level,
then enter on MACD re-acceleration (not on the original cross).

Timeframes:
  Signal/entry   : 5-minute bars  (15 EMA, pullback detection, entry trigger)
  Regime         : 15-minute bars (MACD leg direction, slope, histogram)
  Execution      : 1-minute bars  (stop / target management)

States:
  NEUTRAL
  → BULL_MACD_LEG_ACTIVE      (15M MACD crosses above signal)
  → BULL_IMPULSE_CONFIRMED    (MACD > signal, slope > 0, histogram expanding, price > 5M EMA)
  → BULL_PULLBACK_WAIT        (price pulls back toward 5M EMA or ATR level)
  → BULL_REACCELERATION_CONFIRMED  (MACD slope turns positive again)
  → LONG_ACTIVE               (price closes back above 5M EMA or ATR level)
  (mirror for bearish)
  → INVALIDATED → NEUTRAL     (timeout / MACD crosses back / price breaks)
"""

import csv
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, _EMA, _WilderEMA, Bar5m
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_ATR_PCTS: list[float] = [
    -1.00, -0.75, -0.50, -0.25,
     0.00,
    +0.25, +0.50, +0.75, +1.00,
]


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class MACDPBState(Enum):
    NEUTRAL                      = "NEUTRAL"
    BULL_MACD_LEG_ACTIVE         = "BULL_MACD_LEG_ACTIVE"
    BEAR_MACD_LEG_ACTIVE         = "BEAR_MACD_LEG_ACTIVE"
    BULL_IMPULSE_CONFIRMED       = "BULL_IMPULSE_CONFIRMED"
    BEAR_IMPULSE_CONFIRMED       = "BEAR_IMPULSE_CONFIRMED"
    BULL_PULLBACK_WAIT           = "BULL_PULLBACK_WAIT"
    BEAR_PULLBACK_WAIT           = "BEAR_PULLBACK_WAIT"
    BULL_REACCELERATION_CONFIRMED = "BULL_REACCELERATION_CONFIRMED"
    BEAR_REACCELERATION_CONFIRMED = "BEAR_REACCELERATION_CONFIRMED"
    LONG_ACTIVE                  = "LONG_ACTIVE"
    SHORT_ACTIVE                 = "SHORT_ACTIVE"
    INVALIDATED                  = "INVALIDATED"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MACDPBConfig:
    # 15M MACD
    macd_fast_length:           int   = 12
    macd_slow_length:           int   = 26
    macd_signal_length:         int   = 9

    # 5M EMA
    ema_length:                 int   = 15

    # Slope lookback (in completed bars of that timeframe)
    macd_slope_lookback_bars:      int   = 2
    histogram_slope_lookback_bars: int   = 2
    ema_slope_lookback_bars:       int   = 3   # bars back to measure 5M EMA slope

    # ATR level touch buffer — how many ticks from the level counts as a touch
    atr_retest_buffer_ticks:    int   = 3

    # Stop placement — 25% of daily ATR from entry
    stop_atr_pct:               float = 0.25  # stop = entry ± stop_atr_pct * daily_atr
    stop_buffer_ticks:          int   = 4     # fallback ticks if stop_atr_pct = 0

    # Entry quality — max ticks the entry bar can wick THROUGH the level in the
    # wrong direction before closing back. Filters false rejections where price
    # traded well through the level before dipping back.
    entry_clean_close_ticks:    int   = 6

    # Timeouts (in completed 15M bars)
    max_bars_after_macd_cross:  int   = 20  # max 15M bars to wait for impulse
    # Timeouts (in completed 5M bars)
    max_pullback_bars:          int   = 12  # max 5M bars in pullback before invalidation

    # Execution
    contracts:                  int   = 3
    tick_size:                  float = 0.25
    slippage_ticks:             int   = 1
    dollars_per_point:          float = 5.0

    # Daily risk
    max_trades_per_day:         int   = 3
    daily_loss_limit_usd:       float = -600.0
    daily_profit_limit_usd:     float = 1500.0

    # Session window (ET)
    trade_start_et:             dtime = dtime(8, 30)
    entry_cutoff_et:            dtime = dtime(15, 0)
    trade_end_et:               dtime = dtime(16, 0)

    # Pullback: ATR level required. EMA is a slope direction filter only.
    use_atr_level_pullback:     bool  = True
    use_ema_slope_filter:       bool  = True   # 5M EMA slope must align with direction

    # Target: always next ATR level
    use_atr_targets:            bool  = True
    fixed_r_target:             float = 2.0    # fallback if no next level exists

    # Direction filters
    allow_longs:                bool  = True
    allow_shorts:               bool  = True

    # Daily ATR period (for level calculation)
    atr_period:                 int   = 14

    # Minimum absolute ATR level percentage to consider for pullback entry.
    # e.g. 0.50 means only ±50%, ±75%, ±100% levels are valid entry points.
    # Set to 0.0 to allow all levels.
    min_atr_level_pct:          float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MACDPBTrade:
    date:                   date
    entry_time:             datetime
    direction:              str          # "long" | "short"
    entry_price:            float
    stop_price:             float
    target_price:           float
    exit_time:              datetime
    exit_price:             float
    exit_reason:            str          # "target" | "stop" | "session_end"
    pnl_usd:                float
    pnl_r:                  float
    risk_pts:               float
    mfe_usd:                float
    mae_usd:                float
    daily_open:             float
    daily_atr:              float
    session:                str
    day_of_week:            str
    hour_of_entry:          int
    # MACD context at entry
    macd15_line:            Optional[float]
    macd15_signal:          Optional[float]
    macd15_hist:            Optional[float]
    macd15_slope:           Optional[float]
    histogram_slope:        Optional[float]
    # Pullback context
    pullback_source:        str          # "ema" | "atr_level" | "both"
    pullback_extreme:       float        # low (long) or high (short) of pullback
    atr_level_used:         Optional[float]
    atr_level_pct:          Optional[float]
    # Timing
    bars_cross_to_impulse:  int          # 15M bars
    bars_impulse_to_entry:  int          # 5M bars


# ─────────────────────────────────────────────────────────────────────────────
# State-transition log
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MACDPBTransitionLog:
    timestamp:      datetime
    state_before:   str
    state_after:    str
    price:          float
    macd15_line:    Optional[float]
    macd15_signal:  Optional[float]
    macd15_hist:    Optional[float]
    macd15_slope:   Optional[float]
    atr_level:      Optional[float]
    reason:         str


# ─────────────────────────────────────────────────────────────────────────────
# Internal state machine context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _MACDPBSMState:
    state:                  MACDPBState = MACDPBState.NEUTRAL

    # MACD cross direction memory
    macd_cross_bars_15m:    int   = 0     # 15M bars since the MACD cross

    # Impulse detection
    bars_since_impulse_5m:  int   = 0     # 5M bars since impulse confirmed

    # Pullback tracking
    pullback_extreme:       Optional[float] = None   # low (long) or high (short)
    pullback_bars_5m:       int   = 0                # bars in pullback state
    pullback_source:        str   = ""               # "ema" | "atr_level"
    atr_level_used:         Optional[float] = None
    atr_level_pct:          Optional[float] = None

    # Timing metrics
    bars_cross_to_impulse:  int   = 0
    bars_impulse_start_5m:  int   = 0    # absolute 5M bar index when impulse confirmed


@dataclass
class _DailyState:
    trade_date: date
    daily_open: float
    daily_atr:  float
    levels:     list[float]
    level_pcts: list[float]


# ─────────────────────────────────────────────────────────────────────────────
# Timezone / session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


def _et_time(ts: datetime) -> dtime:
    return _et(ts).time()


def _et_date(ts: datetime) -> date:
    return _et(ts).date()


def _cme_day(ts: datetime) -> date:
    from datetime import timedelta
    et = _et(ts)
    if et.time() >= dtime(18, 0):
        return (et + timedelta(days=1)).date()
    return et.date()


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    if start < end:
        return start <= t < end
    return t >= start or t < end


def _session(ts: datetime) -> str:
    t = _et_time(ts)
    if t >= dtime(18, 0) or t < dtime(3, 0):
        return "asia"
    if dtime(3, 0) <= t < dtime(8, 0):
        return "london"
    if dtime(8, 0) <= t < dtime(16, 0):
        return "ny"
    return "other"


def _dpp(cfg: MACDPBConfig) -> float:
    return cfg.dollars_per_point * cfg.contracts


# ─────────────────────────────────────────────────────────────────────────────
# ATR level functions
# ─────────────────────────────────────────────────────────────────────────────

def calculateAtrLevels(daily_open: float, daily_atr: float) -> list[float]:
    return [round(daily_open + daily_atr * p, 4) for p in _ATR_PCTS]


def nearestAtrLevel(
    price:     float,
    levels:    list[float],
    direction: str,
    buffer:    float,
) -> Optional[tuple[float, float]]:
    """
    Find the nearest ATR level that price is within `buffer` of.
    For long pullbacks: find the highest level <= price + buffer.
    For short pullbacks: find the lowest level >= price - buffer.
    Returns (level_price, atr_pct) or None.
    """
    candidates = []
    for lvl, pct in zip(levels, _ATR_PCTS):
        if direction == "long":
            if price - buffer <= lvl <= price + buffer:
                candidates.append((lvl, pct))
        else:
            if price - buffer <= lvl <= price + buffer:
                candidates.append((lvl, pct))

    if not candidates:
        return None
    if direction == "long":
        return max(candidates, key=lambda x: x[0])
    return min(candidates, key=lambda x: x[0])


def nextAtrLevel(
    current_pct: float,
    levels:      list[float],
    direction:   str,
) -> Optional[float]:
    """Return next ATR level above (long) or below (short) the current one."""
    try:
        idx = _ATR_PCTS.index(current_pct)
    except ValueError:
        return None
    if direction == "long":
        return levels[idx + 1] if idx + 1 < len(levels) else None
    return levels[idx - 1] if idx - 1 >= 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# MACD leg detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate15mMacd(
    close:      float,
    fast_ema:   _EMA,
    slow_ema:   _EMA,
    signal_ema: _EMA,
) -> Optional[tuple[float, float, float]]:
    """Feed one 15M close; return (macd_line, signal, histogram) when warm."""
    fv = fast_ema.update(close)
    sv = slow_ema.update(close)
    if fv is None or sv is None:
        return None
    macd_line = fv - sv
    sig = signal_ema.update(macd_line)
    if sig is None:
        return None
    return macd_line, sig, macd_line - sig


def calculateMacdSlope(history: deque, lookback: int) -> Optional[float]:
    """Change in MACD line over `lookback` bars. Positive = accelerating bullish."""
    if len(history) < lookback + 1:
        return None
    return history[-1] - history[-(lookback + 1)]


def calculateHistogramSlope(history: deque, lookback: int) -> Optional[float]:
    """Change in histogram over `lookback` bars."""
    if len(history) < lookback + 1:
        return None
    return history[-1] - history[-(lookback + 1)]


def detectMacdCross(
    prev_macd:   Optional[float],
    prev_signal: Optional[float],
    cur_macd:    float,
    cur_signal:  float,
) -> Optional[str]:
    """Return 'bull' on cross above, 'bear' on cross below, None otherwise."""
    if prev_macd is None or prev_signal is None:
        return None
    if prev_macd <= prev_signal and cur_macd > cur_signal:
        return "bull"
    if prev_macd >= prev_signal and cur_macd < cur_signal:
        return "bear"
    return None


def detectImpulseConfirmed(
    direction:      str,
    macd_line:      float,
    signal_line:    float,
    macd_slope:     Optional[float],
    hist_slope:     Optional[float],
    price:          float,
    ema5m:          Optional[float],
) -> bool:
    """
    Bull: MACD > signal, MACD slope > 0, histogram expanding (slope > 0), price > EMA.
    Bear: MACD < signal, MACD slope < 0, histogram expanding (slope < 0), price < EMA.
    """
    if macd_slope is None or hist_slope is None or ema5m is None:
        return False
    if direction == "bull":
        return (
            macd_line > signal_line
            and macd_slope > 0
            and hist_slope > 0
            and price > ema5m
        )
    return (
        macd_line < signal_line
        and macd_slope < 0
        and hist_slope < 0
        and price < ema5m
    )


def detectPullbackToEma(
    direction: str,
    bar:       Bar5m,
    ema5m:     float,
    buffer:    float,
) -> bool:
    """
    Long pullback: wick touches EMA within buffer.
    Short pullback: wick touches EMA within buffer.
    """
    if direction == "bull":
        return bar.low <= ema5m + buffer
    return bar.high >= ema5m - buffer


def detectPullbackToAtrLevel(
    direction: str,
    bar:       Bar5m,
    levels:    list[float],
    buffer:    float,
) -> Optional[tuple[float, float]]:
    """Return (level_price, atr_pct) of closest ATR level touched, or None."""
    if direction == "bull":
        ref = bar.low
    else:
        ref = bar.high
    return nearestAtrLevel(ref, levels, direction, buffer)


def detectReacceleration(
    direction:   str,
    macd_line:   float,
    signal_line: float,
    macd_slope:  Optional[float],
    hist_slope:  Optional[float],
) -> bool:
    """
    Bull re-acceleration: MACD still > signal AND (slope turns positive OR histogram increases).
    Bear re-acceleration: MACD still < signal AND (slope turns negative OR histogram decreases).
    """
    if macd_slope is None or hist_slope is None:
        return False
    if direction == "bull":
        if macd_line <= signal_line:
            return False
        return macd_slope > 0 or hist_slope > 0
    else:
        if macd_line >= signal_line:
            return False
        return macd_slope < 0 or hist_slope < 0


def detectEntryConfirmation(
    direction:  str,
    bar:        Bar5m,
    ema5m:      Optional[float],
    atr_level:  Optional[float],
) -> bool:
    """
    Long: close back above retested ATR level.
    Short: close back below retested ATR level.
    """
    if direction == "bull":
        if atr_level is not None and bar.close > atr_level:
            return True
    else:
        if atr_level is not None and bar.close < atr_level:
            return True
    return False


def detectCleanClose(
    direction: str,
    bar:       Bar5m,
    atr_level: float,
    buffer:    float,
) -> bool:
    """
    Reject entries where the confirmation bar wicked significantly through
    the ATR level in the wrong direction before closing back.

    Short: bar.high must not exceed atr_level by more than buffer.
    Long:  bar.low must not be below atr_level by more than buffer.
    """
    if direction == "bull":
        return atr_level - bar.low <= buffer
    return bar.high - atr_level <= buffer


# ─────────────────────────────────────────────────────────────────────────────
# Target and stop calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculateTargetAndStop(
    direction:      str,
    entry_price:    float,
    atr_level_used: float,
    atr_level_pct:  float,
    levels:         list[float],
    stop_buffer:    float,
    daily_atr:      float,
    cfg:            MACDPBConfig,
) -> tuple[float, float]:
    """
    Stop: 25% of daily ATR from entry (or level-buffer fallback).
    Target: next ATR level in the direction of the trade.
    """
    if cfg.stop_atr_pct > 0 and daily_atr > 0:
        atr_dist = cfg.stop_atr_pct * daily_atr
        stop = entry_price - atr_dist if direction == "long" else entry_price + atr_dist
    else:
        stop = (atr_level_used - stop_buffer if direction == "long"
                else atr_level_used + stop_buffer)

    nxt = nextAtrLevel(atr_level_pct, levels, direction)
    if nxt is not None:
        target = nxt
    else:
        risk = abs(entry_price - stop)
        target = (entry_price + risk * cfg.fixed_r_target if direction == "long"
                  else entry_price - risk * cfg.fixed_r_target)

    return target, stop


# ─────────────────────────────────────────────────────────────────────────────
# Transition logger
# ─────────────────────────────────────────────────────────────────────────────

def logMACDPBTransition(
    logs:          list[MACDPBTransitionLog],
    timestamp:     datetime,
    state_before:  MACDPBState,
    state_after:   MACDPBState,
    price:         float,
    macd15_line:   Optional[float],
    macd15_signal: Optional[float],
    macd15_hist:   Optional[float],
    macd15_slope:  Optional[float],
    atr_level:     Optional[float],
    reason:        str,
) -> None:
    logs.append(MACDPBTransitionLog(
        timestamp     = timestamp,
        state_before  = state_before.value,
        state_after   = state_after.value,
        price         = price,
        macd15_line   = round(macd15_line,  6) if macd15_line  is not None else None,
        macd15_signal = round(macd15_signal,6) if macd15_signal is not None else None,
        macd15_hist   = round(macd15_hist,  6) if macd15_hist  is not None else None,
        macd15_slope  = round(macd15_slope, 6) if macd15_slope is not None else None,
        atr_level     = atr_level,
        reason        = reason,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Core state machine update (called on each completed 5M bar)
# ─────────────────────────────────────────────────────────────────────────────

def updateMacdLegStateMachine(
    bar5m:          Bar5m,
    cur_ema5m:      Optional[float],
    cur_ema_slope:  Optional[float],
    levels:         list[float],
    cfg:            MACDPBConfig,
    sm:             _MACDPBSMState,
    logs:           list[MACDPBTransitionLog],
    # 15M MACD state (updated externally when 15M bar completes)
    macd15_line:        Optional[float],
    macd15_signal:      Optional[float],
    macd15_hist:        Optional[float],
    macd15_slope:       Optional[float],
    histogram_slope:    Optional[float],
    # Cross detection from 15M stream
    macd15_cross:       Optional[str],   # "bull" | "bear" | None (this bar)
    daily_atr:          float = 0.0,
) -> Optional[dict]:
    tick       = cfg.tick_size
    atr_buf    = cfg.atr_retest_buffer_ticks  * tick
    stop_buf   = cfg.stop_buffer_ticks        * tick
    slip       = cfg.slippage_ticks           * tick

    new_entry: Optional[dict] = None
    price = bar5m.close

    def _log(before: MACDPBState, after: MACDPBState, reason: str) -> None:
        logMACDPBTransition(
            logs, bar5m.timestamp, before, after, price,
            macd15_line, macd15_signal, macd15_hist, macd15_slope,
            sm.atr_level_used, reason,
        )

    def _transition(new_state: MACDPBState, reason: str) -> None:
        _log(sm.state, new_state, reason)
        sm.state = new_state

    def _reset_to_neutral(reason: str) -> None:
        _transition(MACDPBState.INVALIDATED, reason)
        sm.pullback_extreme  = None
        sm.pullback_source   = ""
        sm.atr_level_used    = None
        sm.atr_level_pct     = None
        sm.pullback_bars_5m  = 0
        sm.bars_since_impulse_5m = 0
        _transition(MACDPBState.NEUTRAL, "reset")

    def _build_entry(direction: str) -> Optional[dict]:
        if sm.atr_level_used is None or sm.atr_level_pct is None:
            return None
        e_px = price + slip if direction == "long" else price - slip
        target, stop = calculateTargetAndStop(
            direction, e_px, sm.atr_level_used,
            sm.atr_level_pct, levels, stop_buf, daily_atr, cfg,
        )
        if direction == "long" and e_px <= stop:
            return None
        if direction == "short" and e_px >= stop:
            return None
        risk = abs(e_px - stop)
        if risk <= 0:
            return None
        return {
            "dir":               direction,
            "entry":             e_px,
            "entry_time":        bar5m.timestamp,
            "stop":              stop,
            "target":            target,
            "pullback_extreme":  sm.pullback_extreme,
            "pullback_source":   sm.pullback_source,
            "atr_level_used":    sm.atr_level_used,
            "atr_level_pct":     sm.atr_level_pct,
            "session":           _session(bar5m.timestamp),
            "macd15_line":       macd15_line,
            "macd15_signal":     macd15_signal,
            "macd15_hist":       macd15_hist,
            "macd15_slope":      macd15_slope,
            "histogram_slope":   histogram_slope,
            "bars_cross_to_impulse": sm.bars_cross_to_impulse,
            "bars_impulse_to_entry": sm.bars_since_impulse_5m,
        }

    # ── Handle any 15M MACD cross first (highest priority) ────────────────────
    if macd15_cross == "bull" and cfg.allow_longs:
        if sm.state not in (MACDPBState.LONG_ACTIVE, MACDPBState.SHORT_ACTIVE):
            prev = sm.state
            sm.macd_cross_bars_15m   = 0
            sm.bars_cross_to_impulse = 0
            sm.bars_since_impulse_5m = 0
            sm.pullback_extreme      = None
            sm.pullback_source       = ""
            sm.atr_level_used        = None
            sm.atr_level_pct         = None
            sm.pullback_bars_5m      = 0
            _log(prev, MACDPBState.BULL_MACD_LEG_ACTIVE, "macd_bull_cross")
            sm.state = MACDPBState.BULL_MACD_LEG_ACTIVE
            return None

    elif macd15_cross == "bear" and cfg.allow_shorts:
        if sm.state not in (MACDPBState.LONG_ACTIVE, MACDPBState.SHORT_ACTIVE):
            prev = sm.state
            sm.macd_cross_bars_15m   = 0
            sm.bars_cross_to_impulse = 0
            sm.bars_since_impulse_5m = 0
            sm.pullback_extreme      = None
            sm.pullback_source       = ""
            sm.atr_level_used        = None
            sm.atr_level_pct         = None
            sm.pullback_bars_5m      = 0
            _log(prev, MACDPBState.BEAR_MACD_LEG_ACTIVE, "macd_bear_cross")
            sm.state = MACDPBState.BEAR_MACD_LEG_ACTIVE
            return None

    # ── BULL_MACD_LEG_ACTIVE ──────────────────────────────────────────────────
    if sm.state == MACDPBState.BULL_MACD_LEG_ACTIVE:
        sm.macd_cross_bars_15m += 1

        # Invalidate: MACD crossed back below signal
        if macd15_line is not None and macd15_signal is not None:
            if macd15_line <= macd15_signal:
                _reset_to_neutral("macd_recrossed_below_signal")
                return None

        # Timeout
        if sm.macd_cross_bars_15m > cfg.max_bars_after_macd_cross:
            _reset_to_neutral("impulse_timeout")
            return None

        # Detect impulse — MACD directional + EMA slope aligned
        ema_slope_ok_bull = (
            not cfg.use_ema_slope_filter
            or (cur_ema_slope is not None and cur_ema_slope > 0)
        )
        if (
            macd15_line is not None and macd15_signal is not None
            and ema_slope_ok_bull
            and detectImpulseConfirmed(
                "bull", macd15_line, macd15_signal,
                macd15_slope, histogram_slope, price, cur_ema5m,
            )
        ):
            sm.bars_cross_to_impulse = sm.macd_cross_bars_15m
            sm.bars_since_impulse_5m = 0
            _transition(MACDPBState.BULL_IMPULSE_CONFIRMED, "impulse_confirmed")

    # ── BEAR_MACD_LEG_ACTIVE ──────────────────────────────────────────────────
    elif sm.state == MACDPBState.BEAR_MACD_LEG_ACTIVE:
        sm.macd_cross_bars_15m += 1

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line >= macd15_signal:
                _reset_to_neutral("macd_recrossed_above_signal")
                return None

        if sm.macd_cross_bars_15m > cfg.max_bars_after_macd_cross:
            _reset_to_neutral("impulse_timeout")
            return None

        ema_slope_ok_bear = (
            not cfg.use_ema_slope_filter
            or (cur_ema_slope is not None and cur_ema_slope < 0)
        )
        if (
            macd15_line is not None and macd15_signal is not None
            and ema_slope_ok_bear
            and detectImpulseConfirmed(
                "bear", macd15_line, macd15_signal,
                macd15_slope, histogram_slope, price, cur_ema5m,
            )
        ):
            sm.bars_cross_to_impulse = sm.macd_cross_bars_15m
            sm.bars_since_impulse_5m = 0
            _transition(MACDPBState.BEAR_IMPULSE_CONFIRMED, "impulse_confirmed")

    # ── BULL_IMPULSE_CONFIRMED ────────────────────────────────────────────────
    elif sm.state == MACDPBState.BULL_IMPULSE_CONFIRMED:
        sm.bars_since_impulse_5m += 1

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line <= macd15_signal:
                _reset_to_neutral("macd_recrossed_below_signal")
                return None

        # Pullback must reach an ATR level — EMA is a direction filter only
        pb_atr = detectPullbackToAtrLevel("bull", bar5m, levels, atr_buf)
        if pb_atr and abs(pb_atr[1]) >= cfg.min_atr_level_pct:
            sm.pullback_extreme = bar5m.low
            sm.pullback_bars_5m = 0
            sm.pullback_source  = "atr_level"
            sm.atr_level_used   = pb_atr[0]
            sm.atr_level_pct    = pb_atr[1]
            _transition(MACDPBState.BULL_PULLBACK_WAIT,
                        f"pullback_to_atr_{'+' if pb_atr[1] >= 0 else ''}{int(pb_atr[1]*100)}pct")

    # ── BEAR_IMPULSE_CONFIRMED ────────────────────────────────────────────────
    elif sm.state == MACDPBState.BEAR_IMPULSE_CONFIRMED:
        sm.bars_since_impulse_5m += 1

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line >= macd15_signal:
                _reset_to_neutral("macd_recrossed_above_signal")
                return None

        pb_atr = detectPullbackToAtrLevel("bear", bar5m, levels, atr_buf)
        if pb_atr and abs(pb_atr[1]) >= cfg.min_atr_level_pct:
            sm.pullback_extreme = bar5m.high
            sm.pullback_bars_5m = 0
            sm.pullback_source  = "atr_level"
            sm.atr_level_used   = pb_atr[0]
            sm.atr_level_pct    = pb_atr[1]
            _transition(MACDPBState.BEAR_PULLBACK_WAIT,
                        f"pullback_to_atr_{'+' if pb_atr[1] >= 0 else ''}{int(pb_atr[1]*100)}pct")

    # ── BULL_PULLBACK_WAIT ────────────────────────────────────────────────────
    elif sm.state == MACDPBState.BULL_PULLBACK_WAIT:
        sm.pullback_bars_5m  += 1
        sm.bars_since_impulse_5m += 1
        # Track worst pullback extreme
        if sm.pullback_extreme is not None:
            sm.pullback_extreme = min(sm.pullback_extreme, bar5m.low)

        # Invalidate: MACD crossed back below signal
        if macd15_line is not None and macd15_signal is not None:
            if macd15_line <= macd15_signal:
                _reset_to_neutral("macd_recrossed_below_signal")
                return None

        # Invalidate: pullback too long
        if sm.pullback_bars_5m > cfg.max_pullback_bars:
            _reset_to_neutral("pullback_timeout")
            return None

        # Detect re-acceleration
        if (
            macd15_line is not None and macd15_signal is not None
            and detectReacceleration(
                "bull", macd15_line, macd15_signal,
                macd15_slope, histogram_slope,
            )
        ):
            _transition(MACDPBState.BULL_REACCELERATION_CONFIRMED, "reacceleration")

    # ── BEAR_PULLBACK_WAIT ────────────────────────────────────────────────────
    elif sm.state == MACDPBState.BEAR_PULLBACK_WAIT:
        sm.pullback_bars_5m  += 1
        sm.bars_since_impulse_5m += 1
        if sm.pullback_extreme is not None:
            sm.pullback_extreme = max(sm.pullback_extreme, bar5m.high)

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line >= macd15_signal:
                _reset_to_neutral("macd_recrossed_above_signal")
                return None

        if sm.pullback_bars_5m > cfg.max_pullback_bars:
            _reset_to_neutral("pullback_timeout")
            return None

        if (
            macd15_line is not None and macd15_signal is not None
            and detectReacceleration(
                "bear", macd15_line, macd15_signal,
                macd15_slope, histogram_slope,
            )
        ):
            _transition(MACDPBState.BEAR_REACCELERATION_CONFIRMED, "reacceleration")

    # ── BULL_REACCELERATION_CONFIRMED ─────────────────────────────────────────
    elif sm.state == MACDPBState.BULL_REACCELERATION_CONFIRMED:
        sm.bars_since_impulse_5m += 1

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line <= macd15_signal:
                _reset_to_neutral("macd_recrossed_below_signal")
                return None

        # Entry: price closes back above the ATR level with a clean bar
        # (wick below the level must not exceed clean_close buffer)
        clean_buf = cfg.entry_clean_close_ticks * tick
        if (
            sm.atr_level_used is not None
            and detectEntryConfirmation("bull", bar5m, None, sm.atr_level_used)
            and detectCleanClose("bull", bar5m, sm.atr_level_used, clean_buf)
        ):
            entry = _build_entry("long")
            if entry is not None:
                _transition(MACDPBState.LONG_ACTIVE, "entry_triggered")
                new_entry = entry

    # ── BEAR_REACCELERATION_CONFIRMED ─────────────────────────────────────────
    elif sm.state == MACDPBState.BEAR_REACCELERATION_CONFIRMED:
        sm.bars_since_impulse_5m += 1

        if macd15_line is not None and macd15_signal is not None:
            if macd15_line >= macd15_signal:
                _reset_to_neutral("macd_recrossed_above_signal")
                return None

        # Entry: price closes back below the ATR level with a clean bar
        # (wick above the level must not exceed clean_close buffer)
        clean_buf = cfg.entry_clean_close_ticks * tick
        if (
            sm.atr_level_used is not None
            and detectEntryConfirmation("bear", bar5m, None, sm.atr_level_used)
            and detectCleanClose("bear", bar5m, sm.atr_level_used, clean_buf)
        ):
            entry = _build_entry("short")
            if entry is not None:
                _transition(MACDPBState.SHORT_ACTIVE, "entry_triggered")
                new_entry = entry

    return new_entry


# ─────────────────────────────────────────────────────────────────────────────
# Daily ATR state builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_daily_states(bars: list[Bar], atr_period: int = 14) -> list[_DailyState]:
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)
        if d not in daily:
            daily[d] = {"open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"], bar.high)
            s["low"]   = min(s["low"],  bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    wilder       = _WilderEMA(atr_period)
    prev_close: Optional[float] = None
    atr_vals: dict[date, float] = {}

    for d in sorted_dates:
        s = daily[d]
        hi, lo, cl = s["high"], s["low"], s["close"]
        tr = hi - lo if prev_close is None else max(
            hi - lo, abs(hi - prev_close), abs(lo - prev_close)
        )
        prev_close = cl
        val = wilder.update(tr)
        if val is not None:
            atr_vals[d] = val

    states = []
    for d in sorted_dates:
        if d not in atr_vals:
            continue
        s = daily[d]
        atr = atr_vals[d]
        states.append(_DailyState(
            trade_date = d,
            daily_open = s["open"],
            daily_atr  = atr,
            levels     = calculateAtrLevels(s["open"], atr),
            level_pcts = list(_ATR_PCTS),
        ))
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Position close helper
# ─────────────────────────────────────────────────────────────────────────────

def _close_position(
    p:        dict,
    exit_px:  float,
    exit_ts:  datetime,
    reason:   str,
    mfe_pts:  float,
    mae_pts:  float,
    cfg:      MACDPBConfig,
    daily_open: float,
    daily_atr:  float,
) -> MACDPBTrade:
    direction = p["dir"]
    pnl_pts   = (exit_px - p["entry"]) if direction == "long" else (p["entry"] - exit_px)
    risk_pts  = abs(p["entry"] - p["stop"])
    dpp       = _dpp(cfg)
    pnl_usd   = pnl_pts * dpp
    pnl_r     = pnl_pts / risk_pts if risk_pts > 0 else 0.0
    et_entry  = _et(p["entry_time"])

    return MACDPBTrade(
        date                   = _et_date(p["entry_time"]),
        entry_time             = p["entry_time"],
        direction              = direction,
        entry_price            = p["entry"],
        stop_price             = p["stop"],
        target_price           = p["target"],
        exit_time              = exit_ts,
        exit_price             = exit_px,
        exit_reason            = reason,
        pnl_usd                = round(pnl_usd, 2),
        pnl_r                  = round(pnl_r, 3),
        risk_pts               = round(risk_pts, 4),
        mfe_usd                = round(mfe_pts * dpp, 2),
        mae_usd                = round(mae_pts * dpp, 2),
        daily_open             = daily_open,
        daily_atr              = daily_atr,
        session                = p["session"],
        day_of_week            = et_entry.strftime("%a"),
        hour_of_entry          = et_entry.hour,
        macd15_line            = p.get("macd15_line"),
        macd15_signal          = p.get("macd15_signal"),
        macd15_hist            = p.get("macd15_hist"),
        macd15_slope           = p.get("macd15_slope"),
        histogram_slope        = p.get("histogram_slope"),
        pullback_source        = p.get("pullback_source", ""),
        pullback_extreme       = p.get("pullback_extreme", 0.0),
        atr_level_used         = p.get("atr_level_used"),
        atr_level_pct          = p.get("atr_level_pct"),
        bars_cross_to_impulse  = p.get("bars_cross_to_impulse", 0),
        bars_impulse_to_entry  = p.get("bars_impulse_to_entry", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_macd_pullback_research(
    bars: list[Bar],
    cfg:  MACDPBConfig,
) -> tuple[list[MACDPBTrade], list[MACDPBTransitionLog]]:
    """
    Walk-forward simulation of the MACD Leg Pullback strategy.
    Returns (trades, transition_logs).
    """
    daily_states  = _build_daily_states(bars, cfg.atr_period)
    state_by_date = {s.trade_date: s for s in daily_states}

    # Indicator streams — continuous, never reset between days
    acc5m  = MinuteBarAccumulator(5)
    acc15m = MinuteBarAccumulator(15)

    # 5M EMA + slope
    ema5m_ind    = _EMA(cfg.ema_length)
    cur_ema5m:   Optional[float] = None
    cur_ema_slope: Optional[float] = None
    ema5m_history: deque[float] = deque(maxlen=cfg.ema_slope_lookback_bars + 1)

    # 15M MACD
    macd15_fast   = _EMA(cfg.macd_fast_length)
    macd15_slow   = _EMA(cfg.macd_slow_length)
    macd15_signal = _EMA(cfg.macd_signal_length)

    cur_macd15_line:   Optional[float] = None
    cur_macd15_sig:    Optional[float] = None
    cur_macd15_hist:   Optional[float] = None
    prev_macd15_line:  Optional[float] = None
    prev_macd15_sig:   Optional[float] = None

    # Slope histories (15M)
    macd15_line_history: deque[float] = deque(maxlen=cfg.macd_slope_lookback_bars + 1)
    hist15_history:      deque[float] = deque(maxlen=cfg.histogram_slope_lookback_bars + 1)

    cur_macd15_slope: Optional[float] = None
    cur_hist15_slope: Optional[float] = None

    # MACD cross flag (set each 15M bar, consumed each 5M bar)
    pending_macd_cross: Optional[str] = None

    # State machine
    sm = _MACDPBSMState()

    # Open position
    position: Optional[dict] = None
    mfe_pts = mae_pts = 0.0

    # Daily counters
    cur_date:       Optional[date]  = None
    day_trade_count = 0
    day_pnl         = 0.0
    day_halted      = False

    # Output
    trades: list[MACDPBTrade]           = []
    logs:   list[MACDPBTransitionLog]   = []

    for bar in bars:
        bar_date = _cme_day(bar.timestamp)
        bar_time = _et_time(bar.timestamp)

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != cur_date:
            if position is not None:
                ds_end = state_by_date.get(cur_date) if cur_date else None
                trades.append(_close_position(
                    position, bar.open, bar.timestamp, "session_end",
                    mfe_pts, mae_pts, cfg,
                    ds_end.daily_open if ds_end else 0.0,
                    ds_end.daily_atr  if ds_end else 0.0,
                ))
                position = None
                mfe_pts = mae_pts = 0.0
            cur_date        = bar_date
            day_trade_count = 0
            day_pnl         = 0.0
            day_halted      = False
            sm              = _MACDPBSMState()
            pending_macd_cross = None

        ds = state_by_date.get(bar_date)
        if ds is None:
            continue

        # ── Feed 15M accumulator ──────────────────────────────────────────────
        completed_15m = acc15m.update(bar)
        if completed_15m is not None:
            result = calculate15mMacd(
                completed_15m.close, macd15_fast, macd15_slow, macd15_signal
            )
            if result is not None:
                prev_macd15_line = cur_macd15_line
                prev_macd15_sig  = cur_macd15_sig

                cur_macd15_line, cur_macd15_sig, cur_macd15_hist = result
                macd15_line_history.append(cur_macd15_line)
                hist15_history.append(cur_macd15_hist)

                cur_macd15_slope = calculateMacdSlope(
                    macd15_line_history, cfg.macd_slope_lookback_bars
                )
                cur_hist15_slope = calculateHistogramSlope(
                    hist15_history, cfg.histogram_slope_lookback_bars
                )

                cross = detectMacdCross(
                    prev_macd15_line, prev_macd15_sig,
                    cur_macd15_line, cur_macd15_sig,
                )
                if cross:
                    pending_macd_cross = cross

        # ── Feed 5M accumulator ───────────────────────────────────────────────
        completed_5m = acc5m.update(bar)
        if completed_5m is not None:
            ev = ema5m_ind.update(completed_5m.close)
            if ev is not None:
                cur_ema5m = ev
                ema5m_history.append(ev)
                if len(ema5m_history) >= cfg.ema_slope_lookback_bars + 1:
                    cur_ema_slope = ema5m_history[-1] - ema5m_history[0]

        # ── Manage open position on every 1M bar ──────────────────────────────
        if position is not None and bar.timestamp > position["entry_time"]:
            p = position
            if p["dir"] == "long":
                mfe_pts = max(mfe_pts, bar.high - p["entry"])
                mae_pts = max(mae_pts, p["entry"] - bar.low)
                hit_stop   = bar.low  <= p["stop"]
                hit_target = bar.high >= p["target"]
            else:
                mfe_pts = max(mfe_pts, p["entry"] - bar.low)
                mae_pts = max(mae_pts, bar.high - p["entry"])
                hit_stop   = bar.high >= p["stop"]
                hit_target = bar.low  <= p["target"]

            if hit_stop or hit_target:
                reason   = "stop" if hit_stop else "target"
                exit_px  = p["stop"] if hit_stop else p["target"]
                t = _close_position(p, exit_px, bar.timestamp, reason,
                                    mfe_pts, mae_pts, cfg,
                                    p["daily_open"], p["daily_atr"])
                trades.append(t)
                day_pnl += t.pnl_usd
                position = None
                mfe_pts = mae_pts = 0.0
                active_s = (MACDPBState.LONG_ACTIVE if p["dir"] == "long"
                            else MACDPBState.SHORT_ACTIVE)
                logMACDPBTransition(
                    logs, bar.timestamp, active_s, MACDPBState.NEUTRAL,
                    bar.close, cur_macd15_line, cur_macd15_sig,
                    cur_macd15_hist, cur_macd15_slope, None, reason,
                )
                sm.state = MACDPBState.NEUTRAL

        # ── Force-flat outside session ────────────────────────────────────────
        if position is not None and not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et):
            p = position
            t = _close_position(p, bar.close, bar.timestamp, "session_end",
                                mfe_pts, mae_pts, cfg,
                                p["daily_open"], p["daily_atr"])
            trades.append(t)
            day_pnl += t.pnl_usd
            position = None
            mfe_pts = mae_pts = 0.0
            sm.state = MACDPBState.NEUTRAL

        # ── Daily halt ────────────────────────────────────────────────────────
        if not day_halted and (
            day_pnl <= cfg.daily_loss_limit_usd
            or day_pnl >= cfg.daily_profit_limit_usd
        ):
            day_halted = True

        # ── State machine on completed 5M bars ────────────────────────────────
        if (
            completed_5m is not None
            and position is None
            and not day_halted
            and day_trade_count < cfg.max_trades_per_day
            and _in_window(bar_time, cfg.trade_start_et, cfg.entry_cutoff_et)
        ):
            cross_this_bar   = pending_macd_cross
            pending_macd_cross = None

            entry = updateMacdLegStateMachine(
                bar5m           = completed_5m,
                cur_ema5m       = cur_ema5m,
                cur_ema_slope   = cur_ema_slope,
                levels          = ds.levels,
                cfg             = cfg,
                sm              = sm,
                logs            = logs,
                macd15_line     = cur_macd15_line,
                macd15_signal   = cur_macd15_sig,
                macd15_hist     = cur_macd15_hist,
                macd15_slope    = cur_macd15_slope,
                histogram_slope = cur_hist15_slope,
                macd15_cross    = cross_this_bar,
                daily_atr       = ds.daily_atr,
            )

            if entry is not None:
                entry["daily_open"] = ds.daily_open
                entry["daily_atr"]  = ds.daily_atr
                position        = entry
                mfe_pts = mae_pts = 0.0
                day_trade_count += 1

    # Force-close any position still open at end of data
    if position is not None:
        last = bars[-1]
        ds_last = state_by_date.get(_cme_day(last.timestamp))
        t = _close_position(
            position, last.close, last.timestamp, "session_end",
            mfe_pts, mae_pts, cfg,
            ds_last.daily_open if ds_last else 0.0,
            ds_last.daily_atr  if ds_last else 0.0,
        )
        trades.append(t)

    return trades, logs


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_macd_pb_trades_csv(trades: list[MACDPBTrade], path: Path) -> None:
    if not trades:
        return
    fields = list(MACDPBTrade.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {f: getattr(t, f) for f in fields}
            row["date"]       = str(row["date"])
            row["entry_time"] = str(row["entry_time"])
            row["exit_time"]  = str(row["exit_time"])
            w.writerow(row)


def export_macd_pb_logs_csv(logs: list[MACDPBTransitionLog], path: Path) -> None:
    if not logs:
        return
    fields = list(MACDPBTransitionLog.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for lg in logs:
            row = {f: getattr(lg, f) for f in fields}
            row["timestamp"] = str(row["timestamp"])
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _stats_block(label: str, group: list[MACDPBTrade], col: int = 26) -> str:
    if not group:
        return f"  {label:<{col}}  (no trades)"
    wins   = [t for t in group if t.pnl_usd > 0]
    losses = [t for t in group if t.pnl_usd <= 0]
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pnl    = sum(t.pnl_usd for t in group)
    wr     = len(wins) / len(group) * 100
    avg_w  = gw / len(wins)   if wins   else 0.0
    avg_l  = gl / len(losses) if losses else 0.0
    avg_r  = statistics.mean(t.pnl_r for t in group)
    pf     = gw / gl          if gl     else float("inf")
    pf_s   = f"{pf:.2f}"     if pf != float("inf") else "∞"
    return (
        f"  {label:<{col}}  N={len(group):>4}  W={len(wins):>3}  L={len(losses):>3}  "
        f"WR={wr:>5.1f}%  PnL=${pnl:>10,.0f}  AvgR={avg_r:>+5.2f}  PF={pf_s:>5}  "
        f"AvgW=${avg_w:>6,.0f}  AvgL=${avg_l:>6,.0f}"
    )


def print_macd_pb_summary(
    trades: list[MACDPBTrade],
    logs:   list[MACDPBTransitionLog],
    cfg:    MACDPBConfig,
) -> None:
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pnl    = sum(t.pnl_usd for t in trades)
    wr     = len(wins) / len(trades) * 100
    avg_w  = gw / len(wins)   if wins   else 0.0
    avg_l  = gl / len(losses) if losses else 0.0
    avg_r  = statistics.mean(t.pnl_r for t in trades)
    pf     = gw / gl          if gl     else float("inf")
    pf_s   = f"{pf:.2f}"     if pf != float("inf") else "∞"

    equity = peak = max_dd = 0.0
    daily_pnl: dict[date, float] = defaultdict(float)
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity   += t.pnl_usd
        peak      = max(peak, equity)
        max_dd    = max(max_dd, peak - equity)
        daily_pnl[t.date] += t.pnl_usd

    sharpe = 0.0
    dpnl   = list(daily_pnl.values())
    if len(dpnl) >= 2:
        mu  = statistics.mean(dpnl)
        std = statistics.stdev(dpnl)
        sharpe = mu / std * math.sqrt(252) if std > 0 else 0.0

    dates = sorted({t.date for t in trades})

    print(f"\n{'='*90}")
    print(f"  MACD LEG PULLBACK STRATEGY  |  "
          f"15M MACD({cfg.macd_fast_length}/{cfg.macd_slow_length}/{cfg.macd_signal_length})  |  "
          f"5M EMA({cfg.ema_length})  |  {cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  {len(dates)} trading days")
    print(f"{'='*90}")

    print(f"\n── OVERALL ───────────────────────────────────────────────────────────────────────────")
    print(f"  Total trades     : {len(trades)}")
    print(f"  Wins / Losses    : {len(wins)} / {len(losses)}")
    print(f"  Win rate         : {wr:.1f}%")
    print(f"  Total PnL        : ${pnl:>10,.2f}")
    print(f"  Profit factor    : {pf_s}")
    print(f"  Avg win          : ${avg_w:>8,.2f}")
    print(f"  Avg loss         : ${avg_l:>8,.2f}")
    print(f"  Avg R            : {avg_r:>+7.3f}")
    print(f"  Max drawdown     : ${max_dd:>8,.2f}")
    print(f"  Sharpe (ann.)    : {sharpe:.2f}")

    print(f"\n── LONG / SHORT ──────────────────────────────────────────────────────────────────────")
    print(_stats_block("Long",  [t for t in trades if t.direction == "long"]))
    print(_stats_block("Short", [t for t in trades if t.direction == "short"]))

    print(f"\n── BY SESSION ────────────────────────────────────────────────────────────────────────")
    for ses in ("asia", "london", "ny", "other"):
        grp = [t for t in trades if t.session == ses]
        if grp:
            print(_stats_block(ses, grp))

    print(f"\n── BY DAY OF WEEK ────────────────────────────────────────────────────────────────────")
    for dow in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        grp = [t for t in trades if t.day_of_week == dow]
        if grp:
            print(_stats_block(dow, grp))

    print(f"\n── BY PULLBACK SOURCE ────────────────────────────────────────────────────────────────")
    for src in ("ema", "atr_level", "both"):
        grp = [t for t in trades if t.pullback_source == src]
        if grp:
            print(_stats_block(f"Source: {src}", grp))

    print(f"\n── BY ATR LEVEL USED ─────────────────────────────────────────────────────────────────")
    lvl_groups: dict[Optional[float], list[MACDPBTrade]] = defaultdict(list)
    for t in trades:
        lvl_groups[t.atr_level_pct].append(t)
    for pct in sorted(k for k in lvl_groups if k is not None):
        lbl = f"ATR {'+' if pct >= 0 else ''}{int(pct * 100)}%"
        print(_stats_block(lbl, lvl_groups[pct]))
    if None in lvl_groups:
        print(_stats_block("ATR level: none (EMA only)", lvl_groups[None]))

    print(f"\n── BY EXIT REASON ────────────────────────────────────────────────────────────────────")
    er_groups: dict[str, list[MACDPBTrade]] = defaultdict(list)
    for t in trades:
        er_groups[t.exit_reason].append(t)
    for reason in sorted(er_groups.keys()):
        grp = er_groups[reason]
        print(f"  {reason:<16}  N={len(grp):>4}  PnL=${sum(t.pnl_usd for t in grp):>10,.0f}")

    print(f"\n── TIMING METRICS ────────────────────────────────────────────────────────────────────")
    print(f"  Avg 15M bars cross→impulse  : {statistics.mean(t.bars_cross_to_impulse for t in trades):.1f}")
    print(f"  Avg 5M bars impulse→entry   : {statistics.mean(t.bars_impulse_to_entry for t in trades):.1f}")

    # MACD slope analysis at entry
    slope_trades = [t for t in trades if t.macd15_slope is not None]
    if slope_trades:
        print(f"\n── 15M MACD SLOPE AT ENTRY ───────────────────────────────────────────────────────────")
        aligned = [t for t in slope_trades if
                   (t.direction == "long" and t.macd15_slope > 0) or
                   (t.direction == "short" and t.macd15_slope < 0)]
        opposed = [t for t in slope_trades if t not in aligned]

        def _ms(label: str, grp: list) -> str:
            if not grp:
                return f"  {label:<26}  (none)"
            w   = [t for t in grp if t.pnl_usd > 0]
            gw_ = sum(t.pnl_usd for t in w)
            gl_ = abs(sum(t.pnl_usd for t in grp if t.pnl_usd <= 0))
            pf_ = f"{gw_/gl_:.2f}" if gl_ else "∞"
            wr_ = len(w) / len(grp) * 100
            avg = sum(t.pnl_usd for t in grp) / len(grp)
            return (f"  {label:<26}  N={len(grp):>4}  WR={wr_:>5.1f}%  "
                    f"AvgPnL=${avg:>8.2f}  PF={pf_}")

        print(_ms("MACD slope aligned with trade", aligned))
        print(_ms("MACD slope opposed to trade",   opposed))

    # State machine stats
    cross_count      = sum(1 for lg in logs if "macd_bull_cross" in lg.reason or "macd_bear_cross" in lg.reason)
    impulse_count    = sum(1 for lg in logs if lg.reason == "impulse_confirmed")
    pullback_count   = sum(1 for lg in logs if lg.reason == "pullback_detected")
    reaccel_count    = sum(1 for lg in logs if lg.reason == "reacceleration")
    entry_count      = sum(1 for lg in logs if lg.reason == "entry_triggered")
    invalid_count    = sum(1 for lg in logs if lg.state_after == "INVALIDATED")

    print(f"\n── STATE MACHINE STATISTICS ──────────────────────────────────────────────────────────")
    print(f"  MACD crosses detected       : {cross_count}")
    print(f"  Impulses confirmed          : {impulse_count}")
    print(f"  Pullbacks detected          : {pullback_count}")
    print(f"  Re-accelerations confirmed  : {reaccel_count}")
    print(f"  Entry triggers              : {entry_count}")
    print(f"  Invalidations               : {invalid_count}")

    if cross_count > 0:
        print(f"  Cross→impulse rate          : {impulse_count/cross_count*100:.1f}%")
    if impulse_count > 0:
        print(f"  Impulse→pullback rate       : {pullback_count/impulse_count*100:.1f}%")
    if pullback_count > 0:
        print(f"  Pullback→reaccel rate       : {reaccel_count/pullback_count*100:.1f}%")
    if reaccel_count > 0:
        print(f"  Reaccel→entry rate          : {entry_count/reaccel_count*100:.1f}%")

    # Invalidation breakdown
    inv_reasons: dict[str, int] = defaultdict(int)
    for lg in logs:
        if lg.state_after == "INVALIDATED":
            inv_reasons[lg.reason] += 1
    if inv_reasons:
        print(f"\n  Invalidation reasons:")
        for reason, count in sorted(inv_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<40}  {count}")

    print()
