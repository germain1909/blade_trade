from __future__ import annotations

"""
ORBreakoutStrategy
==================
Opening Range Breakout / Failed Breakout strategy for MES / MGC futures.

Per-window flow:
  1. Build opening range (OR) during the first `or_minutes` of each window.
  2. After OR is complete, detect breakouts and failed breakouts.
  3. Apply VWAP / 5m MACD / 1m RSI filters.
  4. One trade per 4-hour window; once taken no further entries that window.

Four cases:
  Case 1 — Bullish breakout continuation  (long)
  Case 2 — Bearish breakout continuation  (short)
  Case 3 — Bull trap  / failed high break (short)
  Case 4 — Bear trap  / failed low  break (long)

VWAP resets at midnight ET (calendar day boundary).
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from .indicators import MACD, MACDResult, RSI, VWAP, FiveMinAccumulator
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_WIN_NAMES = ("london", "ny_open", "ny_pm")


# ---------------------------------------------------------------------------
# Public enums / abstract base
# ---------------------------------------------------------------------------

class Signal(Enum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal: ...

    def last_stop_price(self) -> float | None:
        return None

    def last_target_price(self) -> float | None:
        return None

    def last_trade_metadata(self) -> dict:
        return {}

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ORBreakoutConfig:
    # ── Windows (ET start hour, ET start minute; each spans window_hours) ───
    windows:      tuple[tuple[int, int], ...] = ((4, 0), (8, 0), (12, 0))
    window_hours: int   = 4
    or_minutes:   int   = 30    # opening range build duration; 15 for faster OR

    # ── Case toggles ──────────────────────────────────────────────────────────
    enable_case1: bool = True   # bullish breakout continuation
    enable_case2: bool = True   # bearish breakout continuation
    enable_case3: bool = True   # bull trap  → short
    enable_case4: bool = True   # bear trap  → long

    # ── 5-minute MACD filter ──────────────────────────────────────────────────
    require_macd: bool = True
    macd_fast:    int  = 16
    macd_slow:    int  = 40
    macd_signal:  int  = 9

    # ── 1-minute RSI pullback (Cases 1 and 2 only) ────────────────────────────
    rsi_len:              int   = 14
    require_rsi_pullback: bool  = True
    long_pullback_rsi:    float = 40.0   # RSI dips below this then turns up
    short_pullback_rsi:   float = 60.0   # RSI rises above this then turns down

    # ── VWAP filter ───────────────────────────────────────────────────────────
    require_vwap: bool = True

    # ── Exits ─────────────────────────────────────────────────────────────────
    stop_mode:      str   = "fixed"   # "fixed" | "swing"
    stop_ticks:     int   = 24
    tick_size:      float = 0.25      # MES: 0.25 pts / tick
    target_rr:      float = 2.0
    swing_lookback: int   = 10


# ---------------------------------------------------------------------------
# Internal per-window state
# ---------------------------------------------------------------------------

class _WinState:
    __slots__ = (
        "idx", "name", "win_start_et", "or_end_et", "win_end_et",
        "or_high", "or_low", "or_complete", "or_mid",
        "broke_above", "broke_below", "break_high", "break_low",
        "failed_above", "failed_below",
        "rsi_pulled", "trade_taken", "active_case",
    )

    def __init__(self, idx: int, name: str,
                 win_start_et: dtime, or_end_et: dtime, win_end_et: dtime) -> None:
        self.idx           = idx
        self.name          = name
        self.win_start_et  = win_start_et
        self.or_end_et     = or_end_et
        self.win_end_et    = win_end_et

        self.or_high       = float("-inf")
        self.or_low        = float("inf")
        self.or_complete   = False
        self.or_mid        = 0.0

        self.broke_above   = False
        self.broke_below   = False
        self.break_high    = float("-inf")
        self.break_low     = float("inf")

        self.failed_above  = False
        self.failed_below  = False

        self.rsi_pulled    = False
        self.trade_taken   = False
        self.active_case   = ""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ORBreakoutStrategy(Strategy):
    """
    Opening Range Breakout strategy.  One trade per 4-hour window.
    """

    def __init__(self, config: ORBreakoutConfig | None = None) -> None:
        self._cfg = config or ORBreakoutConfig()
        self._init_state()

    # ── Public interface ──────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Signal:
        self._last_stop     = None
        self._last_target   = None
        self._last_metadata = {}

        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        et      = ts.astimezone(_ET)
        t_et    = et.time()
        date_et = et.date()

        # VWAP resets at midnight ET each calendar day
        if date_et != self._vwap_date:
            self._vwap.reset()
            self._vwap_date = date_et

        # 5m MACD — no lookahead
        bar5 = self._acc.update(bar)
        if bar5 is not None:
            if self._macd.result is not None:
                self._prev_macd = self._macd.result
            self._macd.update(bar5.close)

        # 1m RSI
        rsi = self._rsi.update(bar.close)

        # VWAP
        self._vwap.update(bar)

        # Swing stop windows
        self._swing_lows.append(bar.low)
        self._swing_highs.append(bar.high)

        # Window detection and transition
        win_idx = self._window_idx(t_et)
        if win_idx != self._active_win_idx:
            self._active_win_idx = win_idx
            self._active_win = (
                self._make_window(win_idx) if win_idx is not None else None
            )

        w = self._active_win
        signal = Signal.FLAT

        if w is not None and not w.trade_taken:
            # OR building phase
            if not w.or_complete:
                if t_et < w.or_end_et:
                    if w.or_high == float("-inf"):
                        w.or_high = bar.high
                        w.or_low  = bar.low
                    else:
                        w.or_high = max(w.or_high, bar.high)
                        w.or_low  = min(w.or_low,  bar.low)
                else:
                    if w.or_high != float("-inf"):
                        w.or_complete = True
                        w.or_mid = (w.or_high + w.or_low) / 2.0

            # Setup evaluation — only after OR is locked and RSI is warm
            if w.or_complete and rsi is not None:
                signal = self._evaluate(bar, w, rsi)

        self._prev_close = bar.close
        if rsi is not None:
            self._prev_rsi = rsi
        return signal

    def last_stop_price(self) -> float | None:
        return self._last_stop

    def last_target_price(self) -> float | None:
        return self._last_target

    def last_trade_metadata(self) -> dict:
        return self._last_metadata

    def reset(self) -> None:
        self._init_state()

    # ── Core evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, bar: Bar, w: _WinState, rsi: float) -> Signal:
        c = bar.close
        pc = self._prev_close
        if pc is None:
            return Signal.FLAT

        # Detect fresh breakouts (only in one direction at a time)
        if not w.broke_above and not w.failed_above:
            if c > w.or_high and pc <= w.or_high:
                w.broke_above = True
                w.break_high  = bar.high
                w.rsi_pulled  = False

        if not w.broke_below and not w.failed_below:
            if c < w.or_low and pc >= w.or_low:
                w.broke_below = True
                w.break_low   = bar.low
                w.rsi_pulled  = False

        # Detect failed breakouts
        if w.broke_above and not w.failed_above:
            if c <= w.or_high:
                w.failed_above = True
                w.broke_above  = False

        if w.broke_below and not w.failed_below:
            if c >= w.or_low:
                w.failed_below = True
                w.broke_below  = False

        # RSI pullback tracking (Cases 1 / 2)
        if w.broke_above and rsi < self._cfg.long_pullback_rsi:
            w.rsi_pulled = True
        if w.broke_below and rsi > self._cfg.short_pullback_rsi:
            w.rsi_pulled = True

        prev_rsi = self._prev_rsi

        # ── Case 1: Bullish breakout continuation ─────────────────────────────
        if self._cfg.enable_case1 and w.broke_above:
            if (self._rsi_long_ok(w, rsi, prev_rsi)
                    and self._vwap_long(c)
                    and self._macd_long()):
                stop = self._stop_cont(bar, Signal.LONG)
                if stop is not None and stop < c:
                    w.active_case = "Case 1"
                    return self._enter(bar, stop, Signal.LONG, w, rsi)

        # ── Case 2: Bearish breakout continuation ─────────────────────────────
        if self._cfg.enable_case2 and w.broke_below:
            if (self._rsi_short_ok(w, rsi, prev_rsi)
                    and self._vwap_short(c)
                    and self._macd_short()):
                stop = self._stop_cont(bar, Signal.SHORT)
                if stop is not None and stop > c:
                    w.active_case = "Case 2"
                    return self._enter(bar, stop, Signal.SHORT, w, rsi)

        # ── Case 3: Bull trap → short ─────────────────────────────────────────
        if self._cfg.enable_case3 and w.failed_above:
            if self._vwap_short(c) and self._macd_short():
                stop = self._stop_trap(bar, Signal.SHORT, w)
                if stop is not None and stop > c:
                    w.active_case = "Case 3"
                    return self._enter(bar, stop, Signal.SHORT, w, rsi)

        # ── Case 4: Bear trap → long ──────────────────────────────────────────
        if self._cfg.enable_case4 and w.failed_below:
            if self._vwap_long(c) and self._macd_long():
                stop = self._stop_trap(bar, Signal.LONG, w)
                if stop is not None and stop < c:
                    w.active_case = "Case 4"
                    return self._enter(bar, stop, Signal.LONG, w, rsi)

        return Signal.FLAT

    # ── Entry recording ───────────────────────────────────────────────────────

    def _enter(
        self, bar: Bar, stop: float, direction: Signal,
        w: _WinState, rsi: float,
    ) -> Signal:
        risk = abs(bar.close - stop)
        if direction is Signal.LONG:
            self._last_stop   = stop
            self._last_target = bar.close + risk * self._cfg.target_rr
        else:
            self._last_stop   = stop
            self._last_target = bar.close - risk * self._cfg.target_rr

        self._last_metadata = {
            "bias_source": w.active_case,
            "session":     w.name,
            "or_high":     w.or_high,
            "or_low":      w.or_low,
            "or_mid":      round(w.or_mid, 4),
            "vwap":        round(self._vwap.value, 4) if self._vwap.value else None,
            "rsi_1m":      round(rsi, 1),
        }
        w.trade_taken = True
        return direction

    # ── Filter helpers ────────────────────────────────────────────────────────

    def _rsi_long_ok(self, w: _WinState, rsi: float, prev_rsi: float | None) -> bool:
        if not self._cfg.require_rsi_pullback:
            return True
        return w.rsi_pulled and (prev_rsi is None or rsi > prev_rsi)

    def _rsi_short_ok(self, w: _WinState, rsi: float, prev_rsi: float | None) -> bool:
        if not self._cfg.require_rsi_pullback:
            return True
        return w.rsi_pulled and (prev_rsi is None or rsi < prev_rsi)

    def _macd_long(self) -> bool:
        if not self._cfg.require_macd:
            return True
        r = self._macd.result
        return r is None or r.macd > r.signal  # None = not warm yet → allow

    def _macd_short(self) -> bool:
        if not self._cfg.require_macd:
            return True
        r = self._macd.result
        return r is None or r.macd < r.signal

    def _vwap_long(self, close: float) -> bool:
        if not self._cfg.require_vwap:
            return True
        v = self._vwap.value
        return v is None or close > v

    def _vwap_short(self, close: float) -> bool:
        if not self._cfg.require_vwap:
            return True
        v = self._vwap.value
        return v is None or close < v

    # ── Stop helpers ──────────────────────────────────────────────────────────

    def _stop_cont(self, bar: Bar, direction: Signal) -> float | None:
        """Fixed or swing stop for breakout continuation entries."""
        if self._cfg.stop_mode == "fixed":
            dist = self._cfg.stop_ticks * self._cfg.tick_size
            return bar.close - dist if direction is Signal.LONG else bar.close + dist
        if direction is Signal.LONG:
            return min(self._swing_lows)  if self._swing_lows  else None
        return max(self._swing_highs) if self._swing_highs else None

    def _stop_trap(self, bar: Bar, direction: Signal, w: _WinState) -> float | None:
        """Stop for failed-breakout reversal entries."""
        if self._cfg.stop_mode == "fixed":
            dist = self._cfg.stop_ticks * self._cfg.tick_size
            return bar.close - dist if direction is Signal.LONG else bar.close + dist
        # Swing: above the breakout bar high (case 3) or below its low (case 4)
        if direction is Signal.SHORT:
            return w.break_high if w.break_high != float("-inf") else None
        return w.break_low if w.break_low != float("inf") else None

    # ── Window helpers ────────────────────────────────────────────────────────

    def _window_idx(self, t: dtime) -> int | None:
        for i, (wh, wm) in enumerate(self._cfg.windows):
            w_start = dtime(wh, wm)
            end_h   = wh + self._cfg.window_hours
            end_m   = wm
            if end_h >= 24:
                end_h, end_m = 23, 59
            if w_start <= t < dtime(end_h, end_m):
                return i
        return None

    def _make_window(self, idx: int) -> _WinState:
        wh, wm  = self._cfg.windows[idx]
        total_m = wm + self._cfg.or_minutes
        or_h    = wh + total_m // 60
        or_m    = total_m % 60
        end_h   = wh + self._cfg.window_hours
        end_m   = wm
        if end_h >= 24:
            end_h, end_m = 23, 59
        return _WinState(
            idx          = idx,
            name         = _WIN_NAMES[idx],
            win_start_et = dtime(wh, wm),
            or_end_et    = dtime(or_h, or_m),
            win_end_et   = dtime(end_h, end_m),
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        cfg = self._cfg
        self._vwap       = VWAP()
        self._vwap_date  = None
        self._acc        = FiveMinAccumulator()
        self._macd       = MACD(cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        self._prev_macd: MACDResult | None = None
        self._rsi        = RSI(cfg.rsi_len)
        lb = cfg.swing_lookback
        self._swing_lows:  deque[float] = deque(maxlen=lb)
        self._swing_highs: deque[float] = deque(maxlen=lb)
        self._active_win_idx: int | None  = None
        self._active_win:     _WinState | None = None
        self._prev_close: float | None = None
        self._prev_rsi:   float | None = None
        self._last_stop:     float | None = None
        self._last_target:   float | None = None
        self._last_metadata: dict         = {}
