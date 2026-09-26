from __future__ import annotations

"""
ATR Level State Machine Strategy — Version 1
=============================================
Entry logic: Break → Retest → Hold → Entry Trigger

Structural decisions : 5-minute bars  (15 EMA slope + state machine)
Momentum             : 15-minute bars (RSI, RSI MA, RSI MA slope)
Execution            : 1-minute bars  (stop / target management)

States:
  NEUTRAL → BULL_READY / BEAR_READY
  BULL_READY → BULL_BREAK_CONFIRMED  (5M close above ATR level)
  BULL_BREAK_CONFIRMED → BULL_RETESTING  (price pulls back to level)
  BULL_RETESTING → LONG_ACTIVE  (5M close back above level)
  Any setup state → INVALIDATED → NEUTRAL  (timeout / bias loss / close through level)
  (mirror for bearish side)
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

from .indicators import MinuteBarAccumulator, _EMA, _WilderEMA, RSI, Bar5m
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_ATR_PCTS: list[float] = [
    -1.50, -1.25, -1.00, -0.75, -0.50, -0.25,
     0.00, +0.25, +0.50, +0.75, +1.00, +1.25, +1.50,
]

_TRADE_START  = dtime(8, 30)   # ET — no new entries before
_ENTRY_CUTOFF = dtime(15, 0)   # ET — no new entries at/after
_TRADE_END    = dtime(16, 0)   # ET — force-flat any open position


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    NEUTRAL              = "NEUTRAL"
    BULL_READY           = "BULL_READY"
    BEAR_READY           = "BEAR_READY"
    BULL_BREAK_CONFIRMED = "BULL_BREAK_CONFIRMED"
    BEAR_BREAK_CONFIRMED = "BEAR_BREAK_CONFIRMED"
    BULL_RETESTING       = "BULL_RETESTING"
    BEAR_RETESTING       = "BEAR_RETESTING"
    LONG_ACTIVE          = "LONG_ACTIVE"
    SHORT_ACTIVE         = "SHORT_ACTIVE"
    INVALIDATED          = "INVALIDATED"


class MarketBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATRSMConfig:
    # Daily ATR
    atr_period:                      int   = 14

    # Indicator settings
    ema_length:                      int   = 15    # period of 5M EMA
    rsi_length:                      int   = 14    # period of 15M RSI
    rsi_ma_length:                   int   = 14    # EMA period applied to RSI values
    slope_lookback_bars:             int   = 3     # bars back for slope comparison

    # Setup buffers (in ticks)
    retest_buffer_ticks:             int   = 1     # wick must reach within this many ticks of level
    min_retest_depth:                float = 0.0   # min pts price must penetrate the level (0 = off)
    max_retest_depth:                float = 0.0   # max pts before level is considered broken (0 = off)
    invalidation_buffer_ticks:       int   = 4     # max close past level before invalidation
    stop_buffer_ticks:               int   = 4     # ticks beyond retest extreme for stop (unused when stop_atr_pct set)
    stop_atr_pct:                    float = 0.25  # stop distance as fraction of daily ATR from entry (0 = use retest extreme)

    # Timeouts (in completed 5M bars)
    max_bars_after_break_for_retest: int   = 20    # ~100 minutes
    max_bars_after_retest_for_entry: int   = 10    # ~50 minutes

    # Execution
    contracts:                       int   = 3
    tick_size:                       float = 0.25
    slippage_ticks:                  int   = 1
    dollars_per_point:               float = 5.0

    # Daily risk management
    max_trades_per_day:              int   = 3
    daily_loss_limit_usd:            float = -600.0
    daily_profit_limit_usd:          float = 1500.0

    # Session window (ET).  Default = NY session only.
    # For full CME day use: trade_start_et=dtime(18,0), entry_cutoff_et=dtime(16,30), trade_end_et=dtime(17,0)
    trade_start_et:                  dtime = dtime(8, 30)
    entry_cutoff_et:                 dtime = dtime(15, 0)
    trade_end_et:                    dtime = dtime(16, 0)

    # 5-minute MACD
    macd_fast_length:                int   = 12
    macd_slow_length:                int   = 26
    macd_signal_length:              int   = 9
    min_macd_separation:             float = 0.0   # 0 = log only, no filter
    min_macd_slope:                  float = 0.0   # 0 = log only, no filter
    filter_macd_slope_direction:     bool  = False # long only if slope>0, short only if slope<0
    filter_rsi_direction:            bool  = False # long only if RSI<rsi_long_max, short only if RSI>rsi_short_min
    rsi_long_max:                    float = 50.0  # block longs above this RSI
    rsi_short_min:                   float = 50.0  # block shorts below this RSI

    # Toggles
    allow_longs:                     bool  = True
    allow_shorts:                    bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATRSMTrade:
    date:                date
    entry_time:          datetime
    direction:           str            # "long" | "short"
    break_level_pct:     float          # ATR pct of the broken level (e.g. 0.25)
    break_level_price:   float
    retest_extreme:      float          # retest low (long) or high (short)
    entry_price:         float
    stop_price:          float
    target_price:        float
    exit_time:           datetime
    exit_price:          float
    exit_reason:         str            # "target" | "stop" | "session_end"
    pnl_usd:             float
    pnl_r:               float          # R multiple  (pnl_pts / risk_pts)
    risk_pts:            float
    mfe_usd:             float
    mae_usd:             float
    daily_open:          float
    daily_atr:           float
    session:             str            # "asia" | "london" | "ny" | "other"
    day_of_week:         str
    hour_of_entry:       int
    rsi_at_entry:        Optional[float]
    rsi_ma_at_entry:     Optional[float]
    ema_slope_at_entry:  Optional[float]
    bars_to_retest:      int            # 5M bars: break → first retest bar
    bars_to_entry:       int            # 5M bars: first retest bar → entry bar
    macd_line:           Optional[float]
    macd_signal:         Optional[float]
    macd_hist:           Optional[float]
    macd_separation:     Optional[float]
    macd_slope:          Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# State-transition log entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateTransitionLog:
    timestamp:    datetime
    state_before: str
    state_after:  str
    price:        float
    atr_level:    Optional[float]
    rsi:          Optional[float]
    rsi_ma:       Optional[float]
    rsi_ma_slope: Optional[float]
    ema_slope:    Optional[float]
    reason:       str


# ─────────────────────────────────────────────────────────────────────────────
# Internal mutable state machine context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SMState:
    state:                      State         = State.NEUTRAL
    break_level_price:          Optional[float] = None
    break_level_pct:            Optional[float] = None
    bars_since_break:           int           = 0
    retest_extreme:             Optional[float] = None
    bars_since_retest:          int           = 0
    bars_to_retest_at_detection: int          = 0


@dataclass
class _DailyState:
    trade_date: date
    daily_open: float
    daily_atr:  float
    levels:     list[float]


# ─────────────────────────────────────────────────────────────────────────────
# Private timezone / session helpers
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
    """
    CME futures session day: bars at or after 18:00 ET belong to the
    NEXT calendar day's session (matching TradingView daily bars for MES).
    Session for day D runs from D-1 18:00 ET → D 17:00 ET.
    """
    from datetime import timedelta
    et = _et(ts)
    if et.time() >= dtime(18, 0):
        return (et + timedelta(days=1)).date()
    return et.date()


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    """True if t is within [start, end). Handles overnight windows where start > end."""
    if start < end:
        return start <= t < end
    return t >= start or t < end   # e.g. 18:00 → 17:00 spans midnight


def _session(ts: datetime) -> str:
    t = _et_time(ts)
    if t >= dtime(18, 0) or t < dtime(3, 0):
        return "asia"
    if dtime(3, 0) <= t < dtime(8, 0):
        return "london"
    if dtime(8, 0) <= t < dtime(16, 0):
        return "ny"
    return "other"


def _dpp(cfg: ATRSMConfig) -> float:
    return cfg.dollars_per_point * cfg.contracts


def _pts_to_usd(pts: float, cfg: ATRSMConfig) -> float:
    return pts * _dpp(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Required strategy functions
# ─────────────────────────────────────────────────────────────────────────────

def calculateAtrLevels(daily_open: float, daily_atr: float) -> list[float]:
    """Return the 13 ATR level prices in ascending order."""
    return [round(daily_open + daily_atr * p, 4) for p in _ATR_PCTS]


def calculateEmaSlope(history: deque, lookback: int) -> Optional[float]:
    """Raw change in EMA value over `lookback` completed bars. None if not enough history."""
    if len(history) < lookback + 1:
        return None
    return history[-1] - history[-(lookback + 1)]


def calculateRsiMaSlope(history: deque, lookback: int) -> Optional[float]:
    """Raw change in RSI MA over `lookback` completed 15M bars. None if not enough history."""
    if len(history) < lookback + 1:
        return None
    return history[-1] - history[-(lookback + 1)]


def getBullBearState(
    rsi:          Optional[float],
    rsi_ma:       Optional[float],
    rsi_ma_slope: Optional[float],
    ema_slope:    Optional[float],
) -> MarketBias:
    """
    BULLISH : RSI > RSI_MA  AND  RSI_MA slope > 0  AND  5M EMA slope > 0
    BEARISH : RSI < RSI_MA  AND  RSI_MA slope < 0  AND  5M EMA slope < 0
    NEUTRAL : everything else (indicators not warm, or mixed signals)
    """
    if any(v is None for v in (rsi, rsi_ma, rsi_ma_slope, ema_slope)):
        return MarketBias.NEUTRAL
    if rsi > rsi_ma and rsi_ma_slope > 0 and ema_slope > 0:  # type: ignore[operator]
        return MarketBias.BULLISH
    if rsi < rsi_ma and rsi_ma_slope < 0 and ema_slope < 0:  # type: ignore[operator]
        return MarketBias.BEARISH
    return MarketBias.NEUTRAL


def findBrokenAtrLevel(
    prev_close: float,
    curr_close: float,
    levels:     list[float],
    direction:  str,
) -> Optional[tuple[float, float]]:
    """
    Detect a fresh ATR level cross on a completed 5M bar.

    Long  : prev_close was AT-OR-BELOW the level, curr_close is ABOVE.
            Returns (level_price, atr_pct) of the HIGHEST level crossed.
    Short : prev_close was AT-OR-ABOVE the level, curr_close is BELOW.
            Returns (level_price, atr_pct) of the LOWEST level crossed.
    None if no level was freshly crossed.
    """
    if direction == "long":
        crossed = [
            (lvl, pct)
            for lvl, pct in zip(levels, _ATR_PCTS)
            if prev_close <= lvl < curr_close
        ]
        return max(crossed, key=lambda x: x[0]) if crossed else None
    else:
        crossed = [
            (lvl, pct)
            for lvl, pct in zip(levels, _ATR_PCTS)
            if curr_close < lvl <= prev_close
        ]
        return min(crossed, key=lambda x: x[0]) if crossed else None


def detectRetest(
    bar:        Bar5m,
    level:      float,
    direction:  str,
    buffer:     float,
) -> bool:
    """
    Long retest  : wick touches level within buffer AND bar closes back above level.
    Short retest : wick touches level within buffer AND bar closes back below level.
    Both conditions required — a wick that closes through the level is not a valid retest.
    """
    if direction == "long":
        return bar.low <= level + buffer and bar.close > level
    return bar.high >= level - buffer and bar.close < level


def _retest_wick_touched(bar: Bar5m, level: float, direction: str, buffer: float) -> bool:
    """True when price reached the level but the close requirement failed."""
    if direction == "long":
        return bar.low <= level + buffer and bar.close <= level
    return bar.high >= level - buffer and bar.close >= level


def detectEntryTrigger(bar: Bar5m, level: float, direction: str) -> bool:
    """
    Long  : 5M bar closes above the ATR level after retest held.
    Short : 5M bar closes below the ATR level after retest held.
    """
    if direction == "long":
        return bar.close > level
    return bar.close < level


def calculateTargetAndStop(
    direction:         str,
    entry_price:       float,
    break_level_price: float,
    break_level_pct:   float,
    retest_extreme:    float,
    levels:            list[float],
    stop_buffer:       float,
) -> Optional[tuple[float, float]]:
    """
    Target = next ATR level in the direction of trade.
    Stop   = beyond retest extreme (or break level, whichever is further back)
             by `stop_buffer`.

    Returns (target_price, stop_price) or None if no target level exists.
    """
    try:
        idx = _ATR_PCTS.index(break_level_pct)
    except ValueError:
        return None

    if direction == "long":
        if idx + 1 >= len(levels):
            return None
        target = levels[idx + 1]
        stop   = min(retest_extreme, break_level_price) - stop_buffer
    else:
        if idx - 1 < 0:
            return None
        target = levels[idx - 1]
        stop   = max(retest_extreme, break_level_price) + stop_buffer

    return target, stop


def logStateTransition(
    logs:         list[StateTransitionLog],
    timestamp:    datetime,
    state_before: State,
    state_after:  State,
    price:        float,
    atr_level:    Optional[float],
    rsi:          Optional[float],
    rsi_ma:       Optional[float],
    rsi_ma_slope: Optional[float],
    ema_slope:    Optional[float],
    reason:       str,
) -> None:
    logs.append(StateTransitionLog(
        timestamp    = timestamp,
        state_before = state_before.value,
        state_after  = state_after.value,
        price        = price,
        atr_level    = atr_level,
        rsi          = round(rsi,         2) if rsi          is not None else None,
        rsi_ma       = round(rsi_ma,      2) if rsi_ma       is not None else None,
        rsi_ma_slope = round(rsi_ma_slope,4) if rsi_ma_slope is not None else None,
        ema_slope    = round(ema_slope,   4) if ema_slope     is not None else None,
        reason       = reason,
    ))


def updateStateMachine(
    bar5m:         Bar5m,
    bias:          MarketBias,
    levels:        list[float],
    prev_5m_close: Optional[float],
    cfg:           ATRSMConfig,
    sm:            _SMState,
    logs:          list[StateTransitionLog],
    cur_rsi:       Optional[float],
    cur_rsi_ma:    Optional[float],
    cur_rsi_ma_slope: Optional[float],
    cur_ema_slope: Optional[float],
    daily_atr:          float = 0.0,
    cur_macd_line:      Optional[float] = None,
    cur_macd_signal:    Optional[float] = None,
    cur_macd_hist:      Optional[float] = None,
    cur_macd_separation: Optional[float] = None,
    cur_macd_slope:     Optional[float] = None,
) -> Optional[dict]:
    """
    Advance the state machine by one completed 5M bar.
    Mutates `sm` in place.
    Returns a position dict if an entry was triggered, else None.
    """
    tick     = cfg.tick_size
    retest_buf = cfg.retest_buffer_ticks       * tick
    inv_buf    = cfg.invalidation_buffer_ticks * tick
    stop_buf   = cfg.stop_buffer_ticks         * tick
    slip       = cfg.slippage_ticks            * tick

    new_entry: Optional[dict] = None

    def _log(before: State, after: State, reason: str) -> None:
        logStateTransition(
            logs, bar5m.timestamp, before, after, bar5m.close,
            sm.break_level_price, cur_rsi, cur_rsi_ma,
            cur_rsi_ma_slope, cur_ema_slope, reason,
        )

    def _transition(new_state: State, reason: str) -> None:
        _log(sm.state, new_state, reason)
        sm.state = new_state

    def _build(direction: str) -> Optional[dict]:
        nonlocal new_entry
        ts_info = calculateTargetAndStop(
            direction, bar5m.close, sm.break_level_price,
            sm.break_level_pct, sm.retest_extreme, levels, stop_buf,
        )
        if ts_info is None:
            return None
        target, retest_stop = ts_info
        e_px = bar5m.close + slip if direction == "long" else bar5m.close - slip
        # ATR-based stop overrides retest-extreme stop when configured
        if cfg.stop_atr_pct > 0 and daily_atr > 0:
            atr_dist = cfg.stop_atr_pct * daily_atr
            stop = (e_px - atr_dist) if direction == "long" else (e_px + atr_dist)
        else:
            stop = retest_stop
        if direction == "long" and e_px <= stop:
            return None
        if direction == "short" and e_px >= stop:
            return None
        # Retest depth filter
        depth = (
            (sm.break_level_price - sm.retest_extreme) if direction == "long"
            else (sm.retest_extreme - sm.break_level_price)
        )
        if cfg.min_retest_depth > 0 and depth < cfg.min_retest_depth:
            return None
        if cfg.max_retest_depth > 0 and depth > cfg.max_retest_depth:
            return None

        # MACD filters
        if cfg.min_macd_separation > 0 and cur_macd_separation is not None:
            if cur_macd_separation < cfg.min_macd_separation:
                return None
        if cfg.min_macd_slope > 0 and cur_macd_slope is not None:
            if abs(cur_macd_slope) < cfg.min_macd_slope:
                return None
        if cfg.filter_macd_slope_direction and cur_macd_slope is not None:
            if direction == "long"  and cur_macd_slope <= 0:
                return None
            if direction == "short" and cur_macd_slope >= 0:
                return None
        if cfg.filter_rsi_direction and cur_rsi is not None:
            if direction == "long"  and cur_rsi > cfg.rsi_long_max:
                return None
            if direction == "short" and cur_rsi < cfg.rsi_short_min:
                return None
        return {
            "dir":               direction,
            "entry":             e_px,
            "entry_time":        bar5m.timestamp,
            "stop":              stop,
            "target":            target,
            "break_level_pct":   sm.break_level_pct,
            "break_level_price": sm.break_level_price,
            "retest_extreme":    sm.retest_extreme,
            "session":           _session(bar5m.timestamp),
            "rsi_at_entry":      cur_rsi,
            "rsi_ma_at_entry":   cur_rsi_ma,
            "ema_slope_at_entry": cur_ema_slope,
            "bars_to_retest":    sm.bars_to_retest_at_detection,
            "bars_to_entry":     sm.bars_since_retest,
            "macd_line":         cur_macd_line,
            "macd_signal":       cur_macd_signal,
            "macd_hist":         cur_macd_hist,
            "macd_separation":   cur_macd_separation,
            "macd_slope":        cur_macd_slope,
        }

    def _invalidate(reason: str) -> None:
        _transition(State.INVALIDATED, reason)
        sm.break_level_price = None
        sm.break_level_pct   = None
        sm.retest_extreme    = None
        # Reset to NEUTRAL first, then immediately re-evaluate
        _transition(State.NEUTRAL, "post_invalidation_reset")
        if bias == MarketBias.BULLISH and cfg.allow_longs:
            _transition(State.BULL_READY, "bullish_bias")
        elif bias == MarketBias.BEARISH and cfg.allow_shorts:
            _transition(State.BEAR_READY, "bearish_bias")

    # ── NEUTRAL ───────────────────────────────────────────────────────────────
    if sm.state == State.NEUTRAL:
        if bias == MarketBias.BULLISH and cfg.allow_longs:
            _transition(State.BULL_READY, "bullish_bias")
        elif bias == MarketBias.BEARISH and cfg.allow_shorts:
            _transition(State.BEAR_READY, "bearish_bias")

    # ── BULL_READY ────────────────────────────────────────────────────────────
    if sm.state == State.BULL_READY:
        if bias != MarketBias.BULLISH:
            _transition(State.NEUTRAL, "bias_lost")
        elif prev_5m_close is not None:
            broken = findBrokenAtrLevel(prev_5m_close, bar5m.close, levels, "long")
            if broken:
                sm.break_level_price, sm.break_level_pct = broken
                sm.bars_since_break = 0
                sm.retest_extreme   = None
                sm.bars_since_retest = 0
                _transition(State.BULL_BREAK_CONFIRMED,
                            f"break_above_{sm.break_level_pct:.2f}")

    # ── BEAR_READY ────────────────────────────────────────────────────────────
    elif sm.state == State.BEAR_READY:
        if bias != MarketBias.BEARISH:
            _transition(State.NEUTRAL, "bias_lost")
        elif prev_5m_close is not None:
            broken = findBrokenAtrLevel(prev_5m_close, bar5m.close, levels, "short")
            if broken:
                sm.break_level_price, sm.break_level_pct = broken
                sm.bars_since_break = 0
                sm.retest_extreme   = None
                sm.bars_since_retest = 0
                _transition(State.BEAR_BREAK_CONFIRMED,
                            f"break_below_{sm.break_level_pct:.2f}")

    # ── BULL_BREAK_CONFIRMED ──────────────────────────────────────────────────
    elif sm.state == State.BULL_BREAK_CONFIRMED:
        sm.bars_since_break += 1

        # If a HIGHER level was broken, update to that level and reset timer
        if prev_5m_close is not None:
            new_brk = findBrokenAtrLevel(prev_5m_close, bar5m.close, levels, "long")
            if new_brk and new_brk[0] > sm.break_level_price:
                _log(sm.state, sm.state, f"higher_break_{new_brk[1]:.2f}")
                sm.break_level_price = new_brk[0]
                sm.break_level_pct   = new_brk[1]
                sm.bars_since_break  = 0

        # Log wick-to-level that fails the close-back requirement
        if _retest_wick_touched(bar5m, sm.break_level_price, "long", retest_buf):
            _log(sm.state, sm.state, f"retest_wick_rejected_close={bar5m.close:.2f}")

        if bias != MarketBias.BULLISH:
            _invalidate("bias_lost")
        elif sm.bars_since_break >= cfg.max_bars_after_break_for_retest:
            _invalidate("retest_timeout")
        elif bar5m.close < sm.break_level_price - inv_buf:
            _invalidate("close_below_level")
        elif detectRetest(bar5m, sm.break_level_price, "long", retest_buf):
            sm.retest_extreme              = bar5m.low
            sm.bars_since_retest           = 0
            sm.bars_to_retest_at_detection = sm.bars_since_break
            _transition(State.BULL_RETESTING, "retest_detected")
            # close > level is guaranteed by new detectRetest — entry always fires here
            new_entry = _build("long")
            if new_entry:
                _transition(State.LONG_ACTIVE, "entry_trigger")

    # ── BEAR_BREAK_CONFIRMED ──────────────────────────────────────────────────
    elif sm.state == State.BEAR_BREAK_CONFIRMED:
        sm.bars_since_break += 1

        if prev_5m_close is not None:
            new_brk = findBrokenAtrLevel(prev_5m_close, bar5m.close, levels, "short")
            if new_brk and new_brk[0] < sm.break_level_price:
                _log(sm.state, sm.state, f"lower_break_{new_brk[1]:.2f}")
                sm.break_level_price = new_brk[0]
                sm.break_level_pct   = new_brk[1]
                sm.bars_since_break  = 0

        # Log wick-to-level that fails the close-back requirement
        if _retest_wick_touched(bar5m, sm.break_level_price, "short", retest_buf):
            _log(sm.state, sm.state, f"retest_wick_rejected_close={bar5m.close:.2f}")

        if bias != MarketBias.BEARISH:
            _invalidate("bias_lost")
        elif sm.bars_since_break >= cfg.max_bars_after_break_for_retest:
            _invalidate("retest_timeout")
        elif bar5m.close > sm.break_level_price + inv_buf:
            _invalidate("close_above_level")
        elif detectRetest(bar5m, sm.break_level_price, "short", retest_buf):
            sm.retest_extreme              = bar5m.high
            sm.bars_since_retest           = 0
            sm.bars_to_retest_at_detection = sm.bars_since_break
            _transition(State.BEAR_RETESTING, "retest_detected")
            # close < level is guaranteed by new detectRetest — entry always fires here
            new_entry = _build("short")
            if new_entry:
                _transition(State.SHORT_ACTIVE, "entry_trigger")

    # ── BULL_RETESTING ────────────────────────────────────────────────────────
    elif sm.state == State.BULL_RETESTING:
        sm.bars_since_retest += 1
        sm.retest_extreme = min(sm.retest_extreme, bar5m.low)  # type: ignore[arg-type]

        if bias != MarketBias.BULLISH:
            _invalidate("bias_lost")
        elif bar5m.close < sm.break_level_price - inv_buf:  # type: ignore[operator]
            _invalidate("close_below_level")
        elif sm.bars_since_retest >= cfg.max_bars_after_retest_for_entry:
            _invalidate("entry_timeout")
        elif detectEntryTrigger(bar5m, sm.break_level_price, "long"):  # type: ignore[arg-type]
            new_entry = _build("long")
            if new_entry:
                _transition(State.LONG_ACTIVE, "entry_trigger")

    # ── BEAR_RETESTING ────────────────────────────────────────────────────────
    elif sm.state == State.BEAR_RETESTING:
        sm.bars_since_retest += 1
        sm.retest_extreme = max(sm.retest_extreme, bar5m.high)  # type: ignore[arg-type]

        if bias != MarketBias.BEARISH:
            _invalidate("bias_lost")
        elif bar5m.close > sm.break_level_price + inv_buf:  # type: ignore[operator]
            _invalidate("close_above_level")
        elif sm.bars_since_retest >= cfg.max_bars_after_retest_for_entry:
            _invalidate("entry_timeout")
        elif detectEntryTrigger(bar5m, sm.break_level_price, "short"):  # type: ignore[arg-type]
            new_entry = _build("short")
            if new_entry:
                _transition(State.SHORT_ACTIVE, "entry_trigger")

    return new_entry


# ─────────────────────────────────────────────────────────────────────────────
# Daily ATR state builder  (same approach as atr_levels.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_daily_states(bars: list[Bar]) -> list[_DailyState]:
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)   # CME session: 18:00 ET D-1 → 17:00 ET D
        if d not in daily:
            daily[d] = {"open": bar.open, "high": bar.high,
                        "low": bar.low,  "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"],  bar.high)
            s["low"]   = min(s["low"],   bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    wilder       = _WilderEMA(14)
    prev_close: Optional[float] = None
    atr_vals: dict[date, float] = {}

    for d in sorted_dates:
        s  = daily[d]
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
    cfg:      ATRSMConfig,
) -> ATRSMTrade:
    direction = p["dir"]
    pnl_pts   = (exit_px - p["entry"]) if direction == "long" else (p["entry"] - exit_px)
    risk_pts  = abs(p["entry"] - p["stop"])
    dpp       = _dpp(cfg)
    pnl_usd   = pnl_pts * dpp
    pnl_r     = pnl_pts / risk_pts if risk_pts > 0 else 0.0
    et_entry  = _et(p["entry_time"])

    return ATRSMTrade(
        date                = _et_date(p["entry_time"]),
        entry_time          = p["entry_time"],
        direction           = direction,
        break_level_pct     = p["break_level_pct"],
        break_level_price   = p["break_level_price"],
        retest_extreme      = p["retest_extreme"],
        entry_price         = p["entry"],
        stop_price          = p["stop"],
        target_price        = p["target"],
        exit_time           = exit_ts,
        exit_price          = exit_px,
        exit_reason         = reason,
        pnl_usd             = round(pnl_usd,  2),
        pnl_r               = round(pnl_r,    3),
        risk_pts            = round(risk_pts,  4),
        mfe_usd             = round(mfe_pts * dpp, 2),
        mae_usd             = round(mae_pts * dpp, 2),
        daily_open          = p["daily_open"],
        daily_atr           = p["daily_atr"],
        session             = p["session"],
        day_of_week         = et_entry.strftime("%a"),
        hour_of_entry       = et_entry.hour,
        rsi_at_entry        = p.get("rsi_at_entry"),
        rsi_ma_at_entry     = p.get("rsi_ma_at_entry"),
        ema_slope_at_entry  = p.get("ema_slope_at_entry"),
        bars_to_retest      = p["bars_to_retest"],
        bars_to_entry       = p["bars_to_entry"],
        macd_line           = p.get("macd_line"),
        macd_signal         = p.get("macd_signal"),
        macd_hist           = p.get("macd_hist"),
        macd_separation     = p.get("macd_separation"),
        macd_slope          = p.get("macd_slope"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_atr_sm_research(
    bars: list[Bar],
    cfg:  ATRSMConfig,
) -> tuple[list[ATRSMTrade], list[StateTransitionLog]]:
    """
    Walk-forward simulation of the ATR Level State Machine strategy.
    Returns (trades, transition_logs).
    """
    daily_states  = _build_daily_states(bars)
    state_by_date = {s.trade_date: s for s in daily_states}

    # Continuous indicator streams (never reset between days)
    acc5m        = MinuteBarAccumulator(5)
    acc15m       = MinuteBarAccumulator(15)
    ema5         = _EMA(cfg.ema_length)
    rsi15        = RSI(cfg.rsi_length)
    rsi_ma       = _EMA(cfg.rsi_ma_length)
    macd_fast    = _EMA(cfg.macd_fast_length)
    macd_slow    = _EMA(cfg.macd_slow_length)
    macd_sig_ema = _EMA(cfg.macd_signal_length)

    # Rolling slope history (oldest → newest)
    ema_history:    deque[float] = deque(maxlen=cfg.slope_lookback_bars + 1)
    rsi_ma_history: deque[float] = deque(maxlen=cfg.slope_lookback_bars + 1)

    # Latest warm indicator values
    cur_rsi:          Optional[float] = None
    cur_rsi_ma:       Optional[float] = None
    cur_rsi_ma_slope: Optional[float] = None
    cur_ema_slope:       Optional[float] = None
    cur_macd_line:       Optional[float] = None
    cur_macd_signal:     Optional[float] = None
    cur_macd_hist:       Optional[float] = None
    cur_macd_separation: Optional[float] = None
    cur_macd_slope:      Optional[float] = None
    prev_macd_line:      Optional[float] = None

    # State machine (reset each day)
    sm = _SMState()

    # Open position  (None or dict)
    position: Optional[dict] = None
    mfe_pts  = 0.0
    mae_pts  = 0.0

    # Daily counters
    cur_date:       Optional[date]  = None
    day_trade_count = 0
    day_pnl         = 0.0
    day_halted      = False
    prev_5m_close:  Optional[float] = None

    # Output
    trades: list[ATRSMTrade]         = []
    logs:   list[StateTransitionLog] = []

    for bar in bars:
        bar_date = _cme_day(bar.timestamp)   # CME session day
        bar_time = _et_time(bar.timestamp)

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != cur_date:
            if position is not None:
                trades.append(_close_position(
                    position, bar.open, bar.timestamp, "session_end",
                    mfe_pts, mae_pts, cfg,
                ))
                position = None
                mfe_pts = mae_pts = 0.0
            cur_date        = bar_date
            day_trade_count = 0
            day_pnl         = 0.0
            day_halted      = False
            sm              = _SMState()
            prev_5m_close   = None

        ds = state_by_date.get(bar_date)
        if ds is None:
            continue   # skip ATR warmup days

        # ── Feed 15M accumulator ──────────────────────────────────────────────
        completed_15m = acc15m.update(bar)
        if completed_15m is not None:
            r = rsi15.update(completed_15m.close)
            if r is not None:
                cur_rsi = r
                rma = rsi_ma.update(r)
                if rma is not None:
                    cur_rsi_ma = rma
                    rsi_ma_history.append(rma)
                    cur_rsi_ma_slope = calculateRsiMaSlope(
                        rsi_ma_history, cfg.slope_lookback_bars
                    )

        # ── Feed 5M accumulator ───────────────────────────────────────────────
        completed_5m = acc5m.update(bar)
        if completed_5m is not None:
            ev = ema5.update(completed_5m.close)
            if ev is not None:
                ema_history.append(ev)
                cur_ema_slope = calculateEmaSlope(ema_history, cfg.slope_lookback_bars)

            # MACD on 5M closes
            fast_v = macd_fast.update(completed_5m.close)
            slow_v = macd_slow.update(completed_5m.close)
            if fast_v is not None and slow_v is not None:
                prev_macd_line = cur_macd_line
                cur_macd_line  = fast_v - slow_v
                sig_v = macd_sig_ema.update(cur_macd_line)
                if sig_v is not None:
                    cur_macd_signal     = sig_v
                    cur_macd_hist       = cur_macd_line - sig_v
                    cur_macd_separation = abs(cur_macd_line - sig_v)
                if prev_macd_line is not None:
                    cur_macd_slope = cur_macd_line - prev_macd_line


        # ── Manage open position on 1M bars ───────────────────────────────────
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
                if hit_stop:
                    reason, exit_px = "stop", p["stop"]
                else:
                    reason, exit_px = "target", p["target"]

                t = _close_position(p, exit_px, bar.timestamp, reason,
                                    mfe_pts, mae_pts, cfg)
                trades.append(t)
                day_pnl += t.pnl_usd
                position = None
                mfe_pts = mae_pts = 0.0
                active_s = State.LONG_ACTIVE if p["dir"] == "long" else State.SHORT_ACTIVE
                logStateTransition(
                    logs, bar.timestamp, active_s, State.NEUTRAL,
                    bar.close, None, cur_rsi, cur_rsi_ma,
                    cur_rsi_ma_slope, cur_ema_slope, reason,
                )
                sm.state = State.NEUTRAL

        # ── Force-flat at session end ─────────────────────────────────────────
        if position is not None and not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et):
            p = position
            t = _close_position(p, bar.close, bar.timestamp, "session_end",
                                mfe_pts, mae_pts, cfg)
            trades.append(t)
            day_pnl += t.pnl_usd
            position = None
            mfe_pts = mae_pts = 0.0
            sm.state = State.NEUTRAL

        # ── Daily halt check ──────────────────────────────────────────────────
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
            bias = getBullBearState(
                cur_rsi, cur_rsi_ma, cur_rsi_ma_slope, cur_ema_slope
            )
            entry = updateStateMachine(
                bar5m         = completed_5m,
                bias          = bias,
                levels        = ds.levels,
                prev_5m_close = prev_5m_close,
                cfg           = cfg,
                sm            = sm,
                logs          = logs,
                cur_rsi       = cur_rsi,
                cur_rsi_ma    = cur_rsi_ma,
                cur_rsi_ma_slope = cur_rsi_ma_slope,
                cur_ema_slope        = cur_ema_slope,
                daily_atr            = ds.daily_atr,
                cur_macd_line        = cur_macd_line,
                cur_macd_signal      = cur_macd_signal,
                cur_macd_hist        = cur_macd_hist,
                cur_macd_separation  = cur_macd_separation,
                cur_macd_slope       = cur_macd_slope,
            )
            if entry is not None:
                entry["daily_open"] = ds.daily_open
                entry["daily_atr"]  = ds.daily_atr
                position        = entry
                mfe_pts = mae_pts = 0.0
                day_trade_count += 1

            prev_5m_close = completed_5m.close

    # Force-close any position still open at end of data
    if position is not None:
        last = bars[-1]
        t = _close_position(position, last.close, last.timestamp, "session_end",
                            mfe_pts, mae_pts, cfg)
        trades.append(t)

    return trades, logs


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_atr_sm_trades_csv(trades: list[ATRSMTrade], path: Path) -> None:
    if not trades:
        return
    fields = list(ATRSMTrade.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {f: getattr(t, f) for f in fields}
            row["date"]       = str(row["date"])
            row["entry_time"] = str(row["entry_time"])
            row["exit_time"]  = str(row["exit_time"])
            w.writerow(row)


def export_state_logs_csv(logs: list[StateTransitionLog], path: Path) -> None:
    if not logs:
        return
    fields = list(StateTransitionLog.__dataclass_fields__)
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

def _stats_block(label: str, group: list[ATRSMTrade], col: int = 22) -> str:
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
    pf     = gw / gl           if gl    else float("inf")
    pf_s   = f"{pf:.2f}"       if pf != float("inf") else "∞"
    return (
        f"  {label:<{col}}  N={len(group):>4}  W={len(wins):>3}  L={len(losses):>3}  "
        f"WR={wr:>5.1f}%  PnL=${pnl:>10,.0f}  AvgR={avg_r:>+5.2f}  PF={pf_s:>5}  "
        f"AvgW=${avg_w:>6,.0f}  AvgL=${avg_l:>6,.0f}"
    )


def print_atr_sm_summary(
    trades: list[ATRSMTrade],
    logs:   list[StateTransitionLog],
    cfg:    ATRSMConfig,
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
    pf     = gw / gl           if gl    else float("inf")
    pf_s   = f"{pf:.2f}"       if pf != float("inf") else "∞"

    # Drawdown
    equity = peak = max_dd = 0.0
    daily_pnl: dict[date, float] = defaultdict(float)
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity    += t.pnl_usd
        peak       = max(peak, equity)
        max_dd     = max(max_dd, peak - equity)
        daily_pnl[t.date] += t.pnl_usd

    sharpe = 0.0
    dpnl = list(daily_pnl.values())
    if len(dpnl) >= 2:
        mu  = statistics.mean(dpnl)
        std = statistics.stdev(dpnl)
        sharpe = mu / std * math.sqrt(252) if std > 0 else 0.0

    dates = sorted({t.date for t in trades})

    print(f"\n{'='*80}")
    print(f"  ATR LEVEL STATE MACHINE  |  5M EMA({cfg.ema_length})  |  "
          f"15M RSI({cfg.rsi_length}) MA({cfg.rsi_ma_length})  |  "
          f"{cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  {len(dates)} trading days")
    print(f"{'='*80}")

    print(f"\n── OVERALL ───────────────────────────────────────────────────────────────────")
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

    print(f"\n── LONG / SHORT ──────────────────────────────────────────────────────────────")
    print(_stats_block("Long",  [t for t in trades if t.direction == "long"]))
    print(_stats_block("Short", [t for t in trades if t.direction == "short"]))

    print(f"\n── BY SESSION ────────────────────────────────────────────────────────────────")
    for ses in ("asia", "london", "ny", "other"):
        grp = [t for t in trades if t.session == ses]
        if grp:
            print(_stats_block(ses, grp))

    print(f"\n── BY DAY OF WEEK ────────────────────────────────────────────────────────────")
    for dow in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        grp = [t for t in trades if t.day_of_week == dow]
        if grp:
            print(_stats_block(dow, grp))

    print(f"\n── BY ATR LEVEL (break level) ────────────────────────────────────────────────")
    lvl_groups: dict[float, list[ATRSMTrade]] = defaultdict(list)
    for t in trades:
        lvl_groups[t.break_level_pct].append(t)
    for pct in sorted(lvl_groups.keys()):
        lbl = f"{'+' if pct >= 0 else ''}{int(pct * 100)}%"
        print(_stats_block(f"Level {lbl}", lvl_groups[pct]))

    print(f"\n── BY EXIT REASON ────────────────────────────────────────────────────────────")
    er_groups: dict[str, list[ATRSMTrade]] = defaultdict(list)
    for t in trades:
        er_groups[t.exit_reason].append(t)
    for reason in sorted(er_groups.keys()):
        grp = er_groups[reason]
        print(f"  {reason:<16}  N={len(grp):>4}  PnL=${sum(t.pnl_usd for t in grp):>10,.0f}")

    print(f"\n── RETEST QUALITY ────────────────────────────────────────────────────────────")
    print(f"  Avg bars break→retest  : {statistics.mean(t.bars_to_retest for t in trades):.1f}")
    print(f"  Avg bars retest→entry  : {statistics.mean(t.bars_to_entry  for t in trades):.1f}")

    same_bar  = [t for t in trades if t.bars_to_retest <= 1]
    delayed   = [t for t in trades if t.bars_to_retest > 1]
    rej_count = sum(1 for lg in logs if "retest_wick_rejected" in lg.reason)
    print(f"  Same-bar retests       : {len(same_bar)} ({len(same_bar)/len(trades)*100:.0f}%)")
    print(f"  Delayed retests        : {len(delayed)} ({len(delayed)/len(trades)*100:.0f}%)")
    print(f"  Rejected retest wicks  : {rej_count}")
    print(_stats_block("Same-bar retest",  same_bar,  col=22))
    print(_stats_block("Delayed retest",   delayed,   col=22))

    # Retest depth: how far price penetrated the level
    def _retest_depth(t: ATRSMTrade) -> float:
        if t.direction == "long":
            return t.break_level_price - t.retest_extreme   # pts below level
        return t.retest_extreme - t.break_level_price       # pts above level

    depths = sorted(_retest_depth(t) for t in trades)
    n = len(depths)
    if n >= 4:
        d1, d2, d3 = depths[n//4], depths[n//2], depths[3*n//4]
        print(f"\n  Retest depth quartiles: Q1={d1:.2f}  Q2={d2:.2f}  Q3={d3:.2f} pts")
        dbuckets = [
            (f"Depth Q1 (<={d1:.2f})",  [t for t in trades if _retest_depth(t) <= d1]),
            (f"Depth Q2 (<={d2:.2f})",  [t for t in trades if d1 < _retest_depth(t) <= d2]),
            (f"Depth Q3 (<={d3:.2f})",  [t for t in trades if d2 < _retest_depth(t) <= d3]),
            (f"Depth Q4 (>{d3:.2f})",   [t for t in trades if _retest_depth(t) > d3]),
        ]
        for lbl, grp in dbuckets:
            print(_stats_block(lbl, grp, col=22))

    # MACD analysis
    macd_trades = [t for t in trades if t.macd_slope is not None and t.macd_separation is not None]
    if macd_trades:
        print(f"\n── MACD AT ENTRY (5M, {cfg.macd_fast_length}/{cfg.macd_slow_length}/{cfg.macd_signal_length}) ────────────────────────────────────")

        # Slope direction aligned with trade
        aligned   = [t for t in macd_trades if
                     (t.direction == "long"  and t.macd_slope > 0) or
                     (t.direction == "short" and t.macd_slope < 0)]
        opposed   = [t for t in macd_trades if t not in aligned]

        def _macd_stats(label: str, grp: list) -> str:
            if not grp:
                return f"  {label:<22}  (none)"
            w   = [t for t in grp if t.pnl_usd > 0]
            gw  = sum(t.pnl_usd for t in w)
            gl  = abs(sum(t.pnl_usd for t in grp if t.pnl_usd <= 0))
            pf  = f"{gw/gl:.2f}" if gl else "∞"
            wr  = len(w) / len(grp) * 100
            avg = sum(t.pnl_usd for t in grp) / len(grp)
            return (f"  {label:<22}  N={len(grp):>4}  WR={wr:>5.1f}%  "
                    f"AvgPnL=${avg:>8.2f}  PF={pf}")

        print(_macd_stats("Slope aligned",  aligned))
        print(_macd_stats("Slope opposed",  opposed))

        # Separation quartiles
        seps = sorted(t.macd_separation for t in macd_trades)
        n    = len(seps)
        q1, q2, q3 = seps[n//4], seps[n//2], seps[3*n//4]
        print(f"\n  Separation quartile cutoffs: Q1={q1:.4f}  Q2={q2:.4f}  Q3={q3:.4f}")
        buckets = [
            (f"Sep Q1 (<={q1:.4f})",  [t for t in macd_trades if t.macd_separation <= q1]),
            (f"Sep Q2 (<={q2:.4f})",  [t for t in macd_trades if q1 < t.macd_separation <= q2]),
            (f"Sep Q3 (<={q3:.4f})",  [t for t in macd_trades if q2 < t.macd_separation <= q3]),
            (f"Sep Q4 (>{q3:.4f})",   [t for t in macd_trades if t.macd_separation > q3]),
        ]
        for lbl, grp in buckets:
            print(_macd_stats(lbl, grp))

        # MACD slope magnitude quartiles
        slopes = sorted(abs(t.macd_slope) for t in macd_trades)
        s1, s2, s3 = slopes[n//4], slopes[n//2], slopes[3*n//4]
        print(f"\n  |Slope| quartile cutoffs:    Q1={s1:.4f}  Q2={s2:.4f}  Q3={s3:.4f}")
        sbuckets = [
            (f"|Slope| Q1 (<={s1:.4f})", [t for t in macd_trades if abs(t.macd_slope) <= s1]),
            (f"|Slope| Q2 (<={s2:.4f})", [t for t in macd_trades if s1 < abs(t.macd_slope) <= s2]),
            (f"|Slope| Q3 (<={s3:.4f})", [t for t in macd_trades if s2 < abs(t.macd_slope) <= s3]),
            (f"|Slope| Q4 (>{s3:.4f})",  [t for t in macd_trades if abs(t.macd_slope) > s3]),
        ]
        for lbl, grp in sbuckets:
            print(_macd_stats(lbl, grp))

    # State machine stats from logs
    inv_count    = sum(1 for lg in logs if lg.state_after == "INVALIDATED")
    break_count  = sum(1 for lg in logs if "break_above" in lg.reason or "break_below" in lg.reason)
    retest_count = sum(1 for lg in logs if lg.reason == "retest_detected")
    entry_count  = sum(1 for lg in logs if lg.reason == "entry_trigger")
    print(f"\n── STATE MACHINE STATISTICS ─────────────────────────────────────────────────")
    print(f"  Total ATR breaks detected  : {break_count}")
    print(f"  Retests detected           : {retest_count}")
    print(f"  Entry triggers             : {entry_count}")
    print(f"  Invalidations              : {inv_count}")
    if retest_count > 0:
        print(f"  Retest → entry conv. rate  : {entry_count/retest_count*100:.1f}%")
    if break_count > 0:
        print(f"  Break → retest conv. rate  : {retest_count/break_count*100:.1f}%")

    print()
