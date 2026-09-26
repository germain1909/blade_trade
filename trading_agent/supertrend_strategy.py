from __future__ import annotations

"""
Adaptive Supertrend Strategy
============================
Ported from PineScript "Multi-Factor Regime Engine Pro [MarkitTick]".

10-feature composite confidence score dynamically adjusts Supertrend band
width.

Two modes:
  1. Flip-to-flip: enter on flip, exit on next flip.
  2. ATR levels:   enter on flip, TP/SL at nearest ATR percentage levels.
"""

import bisect
import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, Bar5m, _WilderEMA
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class SupertrendConfig:
    bar_minutes:       int   = 5
    lookback:          int   = 14
    threshold:         float = 2.0
    pred_weight:       float = 0.5
    auto_params:       bool  = True

    contracts:         int   = 1
    tick_size:         float = 0.25
    slippage_ticks:    int   = 1
    dollars_per_point: float = 5.0

    trade_start_et:    dtime = dtime(8, 30)
    trade_end_et:      dtime = dtime(16, 0)


# ── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class SupertrendTrade:
    date:                date
    entry_time:          datetime
    exit_time:           datetime
    direction:           str
    entry_price:         float
    exit_price:          float
    exit_reason:         str
    pnl_usd:             float
    confidence_at_entry: float
    bars_held:           int
    stop_price:          float = 0.0
    target_price:        float = 0.0
    daily_atr:           float = 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _auto_select(bar_minutes: int) -> tuple[int, float, float]:
    sec = bar_minutes * 60
    if sec <= 300:   return 10, 1.5, 0.70
    if sec <= 3600:  return 14, 2.0, 0.50
    if sec <= 14400: return 20, 2.5, 0.40
    return 30, 3.0, 0.30


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sma_buf(buf: deque[float]) -> float:
    return sum(buf) / len(buf) if buf else 0.0


def _stdev_buf(buf: deque[float]) -> float:
    n = len(buf)
    if n < 2:
        return 0.0
    mean = sum(buf) / n
    return (sum((x - mean) ** 2 for x in buf) / n) ** 0.5


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


# ── Core engine ──────────────────────────────────────────────────────────────

class _SupertrendEngine:
    """Processes completed N-minute bars, maintains adaptive Supertrend state."""

    def __init__(self, lb: int, ct: float, pw: float) -> None:
        self.lb = lb
        self.ct = ct
        self.pw = pw

        # ATR (Wilder smoothed)
        self._atr_fast = _WilderEMA(lb)
        self._atr_slow = _WilderEMA(lb * 3)
        self._dmi_atr  = _WilderEMA(lb)

        # Previous OHLC
        self._prev_close: Optional[float] = None
        self._prev_high:  Optional[float] = None
        self._prev_low:   Optional[float] = None

        self.atr:      Optional[float] = None
        self.atr_slow: Optional[float] = None

        # RSI (Wilder)
        self._rsi_gain = _WilderEMA(lb)
        self._rsi_loss = _WilderEMA(lb)
        self._rsi_prev: Optional[float] = None
        self.rsi: Optional[float] = None

        # DMI / ADX
        self._dmi_pdm = _WilderEMA(lb)
        self._dmi_mdm = _WilderEMA(lb)
        self._adx_ema = _WilderEMA(lb)
        self.di_plus:  Optional[float] = None
        self.di_minus: Optional[float] = None
        self.adx:      Optional[float] = None

        # Rolling close buffers for SMA fast/slow
        self._close_fast: deque[float] = deque(maxlen=lb)
        self._close_slow: deque[float] = deque(maxlen=lb * 2)

        # ROC buffers
        self._close_hist: deque[float] = deque(maxlen=lb * 2 + 1)
        self._roc_slow_hist: deque[float] = deque(maxlen=lb)

        # Bollinger Band buffer
        self._bb_buf: deque[float] = deque(maxlen=lb)

        # Volume buffer
        self._vol_buf: deque[float] = deque(maxlen=lb)

        # Bar consistency buffer (PineScript loop needs lb+2 values)
        self._bar_buf: deque[float] = deque(maxlen=lb + 2)

        # Confidence smoothing
        smooth_len = max(round(lb * (1.0 - pw) * 0.5 + 2.0), 2)
        self._conf_buf: deque[float] = deque(maxlen=smooth_len)
        self.conf_score: float = 0.5

        # Supertrend state
        self.t_dir: int = 1
        self.st_upper: Optional[float] = None
        self.st_lower: Optional[float] = None
        self.st_line:  Optional[float] = None
        self._prev_f1: Optional[float] = None

        # Flip signal: 1 = flipped bull, -1 = flipped bear, None = no flip
        self.flip: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, bar: Bar5m) -> None:
        h, l, c, v = bar.high, bar.low, bar.close, bar.volume
        hl2 = (h + l) / 2.0
        self.flip = None
        lb = self.lb

        # ── True Range ───────────────────────────────────────────────
        if self._prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))

        # ── ATR fast & slow ──────────────────────────────────────────
        v_atr = self._atr_fast.update(tr)
        v_atr_s = self._atr_slow.update(tr)
        if v_atr is not None:
            self.atr = v_atr
        if v_atr_s is not None:
            self.atr_slow = v_atr_s

        # ── RSI ──────────────────────────────────────────────────────
        if self._rsi_prev is not None:
            delta = c - self._rsi_prev
            g = self._rsi_gain.update(max(delta, 0.0))
            ls = self._rsi_loss.update(max(-delta, 0.0))
            if g is not None and ls is not None:
                self.rsi = 100.0 if ls == 0.0 else 100.0 - 100.0 / (1.0 + g / ls)
        self._rsi_prev = c

        # ── DMI / ADX ────────────────────────────────────────────────
        dmi_atr_v = self._dmi_atr.update(tr)

        if self._prev_high is not None:
            up_move = h - self._prev_high
            dn_move = self._prev_low - l
            pdm = up_move if up_move > dn_move and up_move > 0 else 0.0
            mdm = dn_move if dn_move > up_move and dn_move > 0 else 0.0
        else:
            pdm = 0.0
            mdm = 0.0

        pdm_v = self._dmi_pdm.update(pdm)
        mdm_v = self._dmi_mdm.update(mdm)

        if (pdm_v is not None and mdm_v is not None
                and dmi_atr_v is not None and dmi_atr_v > 0):
            self.di_plus = pdm_v / dmi_atr_v * 100.0
            self.di_minus = mdm_v / dmi_atr_v * 100.0
            di_sum = self.di_plus + self.di_minus
            dx = (abs(self.di_plus - self.di_minus) / di_sum * 100.0
                  if di_sum > 0 else 0.0)
            adx_v = self._adx_ema.update(dx)
            if adx_v is not None:
                self.adx = adx_v

        # ── Update previous OHLC ─────────────────────────────────────
        self._prev_close = c
        self._prev_high = h
        self._prev_low = l

        # ── Rolling buffers ──────────────────────────────────────────
        self._close_fast.append(c)
        self._close_slow.append(c)
        self._close_hist.append(c)
        self._bb_buf.append(c)
        self._vol_buf.append(float(v))
        self._bar_buf.append(c)

        # ROC slow history
        if len(self._close_hist) > lb * 2:
            old_c2 = self._close_hist[-(lb * 2 + 1)]
            if old_c2 != 0:
                self._roc_slow_hist.append((c - old_c2) / old_c2 * 100.0)

        # ── Need ATR + RSI before features ───────────────────────────
        if self.atr is None or self.rsi is None:
            return

        atr = self.atr
        atr_slow = self.atr_slow if self.atr_slow is not None else atr

        # ── f1: RSI normalised to [-1, 1] ────────────────────────────
        f1 = (self.rsi - 50.0) / 50.0

        # ── f2: ADX with directional sign ────────────────────────────
        if self.adx is not None and self.di_plus is not None:
            f2 = (min(self.adx / 50.0, 1.0)
                  * (1.0 if self.di_plus >= self.di_minus else -1.0))
        else:
            f2 = 0.0

        # ── f3: price deviation from fast SMA ────────────────────────
        if len(self._close_fast) == lb:
            sma_f = _sma_buf(self._close_fast)
            f3 = _clamp((c - sma_f) / (atr * 2.0 + 1e-9), -1.0, 1.0)
        else:
            sma_f = None
            f3 = 0.0

        # ── f4: fast/slow SMA slope ──────────────────────────────────
        if sma_f is not None and len(self._close_slow) == lb * 2:
            sma_s = _sma_buf(self._close_slow)
            f4 = _clamp((sma_f - sma_s) / (atr + 1e-9), -1.0, 1.0)
        else:
            f4 = 0.0

        # ── f5: ROC normalised by stdev of slow ROC ─────────────────
        if len(self._close_hist) > lb:
            old_c = self._close_hist[-(lb + 1)]
            roc = (c - old_c) / old_c * 100.0 if old_c != 0 else 0.0
        else:
            roc = None

        if roc is not None and len(self._roc_slow_hist) == lb:
            roc_std = _stdev_buf(self._roc_slow_hist)
            f5 = _clamp(roc / (roc_std * 2.0 + 1e-9), -1.0, 1.0)
        else:
            f5 = 0.0

        # ── f6, f7: Bollinger Band position & width ──────────────────
        if len(self._bb_buf) == lb:
            bb_mid = _sma_buf(self._bb_buf)
            bb_std = _stdev_buf(self._bb_buf)
            bb_up = bb_mid + 2.0 * bb_std
            bb_dn = bb_mid - 2.0 * bb_std
            bb_rng = bb_up - bb_dn
            f6 = (c - bb_dn) / bb_rng if bb_rng > 0 else 0.5
            f7 = min(bb_rng / (atr * 4.0 + 1e-9), 2.0) / 2.0
        else:
            f6 = 0.5
            f7 = 0.5

        # ── f8: volatility stability ─────────────────────────────────
        f8 = min(atr_slow / (atr + 1e-9) / 1.5, 1.0)

        # ── f9: volume ratio ─────────────────────────────────────────
        if len(self._vol_buf) == lb:
            vol_sma = _sma_buf(self._vol_buf)
            f9 = min(float(v) / (vol_sma * 2.0 + 1e-9), 1.0) if vol_sma > 0 else 0.5
        else:
            f9 = 0.5

        # ── f10: bull/bear bar consistency ───────────────────────────
        if len(self._bar_buf) >= lb + 2:
            bull_c = 0
            for i in range(1, lb + 1):
                if self._bar_buf[-(i + 1)] > self._bar_buf[-(i + 2)]:
                    bull_c += 1
            f10 = (bull_c - (lb - bull_c)) / lb
        else:
            f10 = 0.0

        # ── Confidence score ─────────────────────────────────────────
        dir_score = (f1 * 0.22 + f2 * 0.22 + f3 * 0.18
                     + f4 * 0.15 + f5 * 0.13 + f10 * 0.10)
        dir_norm = (dir_score + 1.0) / 2.0
        qual_score = f8 * 0.35 + f9 * 0.25 + f7 * 0.25 + abs(f2) * 0.15
        bb_extreme = abs(f6 - 0.5) * 2.0

        raw_conf = dir_norm * 0.45 + qual_score * 0.35 + bb_extreme * 0.20
        self._conf_buf.append(raw_conf)
        self.conf_score = _clamp(
            sum(self._conf_buf) / len(self._conf_buf), 0.0, 1.0)

        ml_bias = 1 if dir_score >= 0 else -1

        # ── Adaptive band ────────────────────────────────────────────
        adapt_mult = self.ct * (1.40 - self.conf_score * 0.80)
        half_bw = atr * adapt_mult
        raw_u = hl2 + half_bw
        raw_l = hl2 - half_bw

        # ── Ratcheting Supertrend state machine ──────────────────────
        prev_dir = self.t_dir
        prev_u = self.st_upper if self.st_upper is not None else raw_u
        prev_l = self.st_lower if self.st_lower is not None else raw_l

        self.st_upper = min(raw_u, prev_u) if prev_dir == -1 else raw_u
        self.st_lower = max(raw_l, prev_l) if prev_dir == 1 else raw_l

        # Regime shift detection
        atr_spiked = atr > atr_slow * 1.5
        prev_f1 = self._prev_f1 if self._prev_f1 is not None else f1
        dir_flipped = (math.copysign(1, f1) != math.copysign(1, prev_f1)
                       and abs(f1) > 0.40)
        regime_shift = atr_spiked and dir_flipped
        self._prev_f1 = f1

        flip_thresh = atr * 0.04 * (1.0 - self.conf_score * 0.5)

        if prev_dir == 1:
            if ((c < self.st_lower - flip_thresh)
                    or (regime_shift and ml_bias == -1)):
                self.t_dir = -1
                self.st_line = self.st_upper
            else:
                self.t_dir = 1
                self.st_line = self.st_lower
        else:
            if ((c > self.st_upper + flip_thresh)
                    or (regime_shift and ml_bias == 1)):
                self.t_dir = 1
                self.st_line = self.st_lower
            else:
                self.t_dir = -1
                self.st_line = self.st_upper

        if self.t_dir != prev_dir:
            self.flip = self.t_dir


# ── Walk-forward simulation ──────────────────────────────────────────────────

def run_supertrend_strategy(
    bars: list[Bar],
    cfg: SupertrendConfig,
) -> list[SupertrendTrade]:
    """Enter on Supertrend flip, exit on next flip or session end."""

    if cfg.auto_params:
        lb, ct, pw = _auto_select(cfg.bar_minutes)
    else:
        lb, ct, pw = cfg.lookback, cfg.threshold, cfg.pred_weight

    engine = _SupertrendEngine(lb, ct, pw)
    accum  = MinuteBarAccumulator(cfg.bar_minutes)

    slip = cfg.slippage_ticks * cfg.tick_size
    dpp  = cfg.dollars_per_point * cfg.contracts

    trades: list[SupertrendTrade] = []

    in_trade = False
    t_dir:  Optional[str]      = None
    t_entry: Optional[float]   = None
    t_ts:   Optional[datetime] = None
    t_conf: float = 0.0
    t_bars: int   = 0

    def _close_trade(exit_price: float, exit_time: datetime,
                     reason: str) -> None:
        nonlocal in_trade
        pts = (exit_price - t_entry if t_dir == "long"
               else t_entry - exit_price)
        trades.append(SupertrendTrade(
            date=_et(t_ts).date(),
            entry_time=t_ts,
            exit_time=exit_time,
            direction=t_dir,
            entry_price=t_entry,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_usd=round(pts * dpp, 2),
            confidence_at_entry=t_conf,
            bars_held=t_bars,
        ))
        in_trade = False

    for bar in bars:
        ts_et    = _et(bar.timestamp)
        bar_time = ts_et.time()
        in_session = _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)

        # Close at session end (checked every 1m bar)
        if in_trade and not in_session:
            _close_trade(bar.close, bar.timestamp, "session_end")

        # Accumulate N-minute bars
        completed = accum.update(bar)
        if completed is None:
            continue

        engine.update(completed)

        if in_trade:
            t_bars += 1

        if not in_session:
            continue

        # Handle flip
        if engine.flip is not None:
            if in_trade:
                _close_trade(completed.close, bar.timestamp, "flip")

            new_dir = "long" if engine.flip == 1 else "short"
            t_entry = (completed.close + slip if new_dir == "long"
                       else completed.close - slip)
            t_dir   = new_dir
            t_ts    = bar.timestamp
            t_conf  = engine.conf_score
            t_bars  = 0
            in_trade = True

    # Close any remaining position
    if in_trade and bars:
        _close_trade(bars[-1].close, bars[-1].timestamp, "data_end")

    return trades


# ── ATR percentage level helpers ─────────────────────────────────────────────

_ATR_PCTS: list[float] = [
    -1.50, -1.25, -1.00, -0.75, -0.50, -0.25,
     0.00, +0.25, +0.50, +0.75, +1.00, +1.25, +1.50,
]


def _cme_day(ts: datetime) -> date:
    """CME session day: bars at or after 18:00 ET belong to the next calendar day."""
    et = _et(ts)
    if et.time() >= dtime(18, 0):
        return (et + timedelta(days=1)).date()
    return et.date()


@dataclass
class _DayInfo:
    daily_open: float
    daily_atr:  float
    levels:     list[float]


def _build_day_info(bars: list[Bar], atr_period: int = 14) -> dict[date, _DayInfo]:
    """
    Two-pass build: daily OHLC → Wilder ATR → per-day levels.
    Uses PRIOR day's ATR for today's levels (no lookahead).
    """
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)
        if d not in daily:
            daily[d] = {"open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"],  bar.high)
            s["low"]   = min(s["low"],   bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    if not sorted_dates:
        return {}

    wilder = _WilderEMA(atr_period)
    prev_close: Optional[float] = None
    atr_vals: dict[date, float] = {}

    for d in sorted_dates:
        s = daily[d]
        hi, lo, cl = s["high"], s["low"], s["close"]
        tr = (hi - lo if prev_close is None
              else max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
        prev_close = cl
        atr = wilder.update(tr)
        if atr is not None:
            atr_vals[d] = atr

    result: dict[date, _DayInfo] = {}
    for i, d in enumerate(sorted_dates):
        prev_d = sorted_dates[i - 1] if i > 0 else None
        if prev_d is None or prev_d not in atr_vals:
            continue
        prior_atr = atr_vals[prev_d]
        day_open = daily[d]["open"]
        levels = [round(day_open + prior_atr * p, 4) for p in _ATR_PCTS]
        result[d] = _DayInfo(daily_open=day_open, daily_atr=prior_atr,
                             levels=levels)
    return result


def _nearest_levels(
    levels: list[float], entry: float, direction: str,
) -> tuple[Optional[float], Optional[float]]:
    """Return (target, stop) from ATR levels for given direction."""
    lo_idx = bisect.bisect_left(levels, entry) - 1
    hi_idx = bisect.bisect_right(levels, entry)

    level_below = levels[lo_idx] if lo_idx >= 0 else None
    level_above = levels[hi_idx] if hi_idx < len(levels) else None

    if direction == "long":
        return level_above, level_below
    return level_below, level_above


# ── Walk-forward: Supertrend + ATR level TP/SL ──────────────────────────────

def run_supertrend_atr_strategy(
    bars: list[Bar],
    cfg: SupertrendConfig,
) -> list[SupertrendTrade]:
    """
    Supertrend flip for direction, ATR percentage levels for TP/SL.
    One trade per flip — after TP/SL exit, stay flat until next flip.
    TP/SL checked on every 1m bar for precision.
    """
    if cfg.auto_params:
        lb, ct, pw = _auto_select(cfg.bar_minutes)
    else:
        lb, ct, pw = cfg.lookback, cfg.threshold, cfg.pred_weight

    day_info = _build_day_info(bars)
    engine   = _SupertrendEngine(lb, ct, pw)
    accum    = MinuteBarAccumulator(cfg.bar_minutes)

    slip = cfg.slippage_ticks * cfg.tick_size
    dpp  = cfg.dollars_per_point * cfg.contracts

    trades: list[SupertrendTrade] = []

    in_trade  = False
    t_dir:    Optional[str]      = None
    t_entry:  Optional[float]    = None
    t_stop:   Optional[float]    = None
    t_target: Optional[float]    = None
    t_ts:     Optional[datetime] = None
    t_conf:   float = 0.0
    t_bars:   int   = 0
    t_atr:    float = 0.0

    def _close(exit_price: float, exit_time: datetime, reason: str) -> None:
        nonlocal in_trade
        pts = (exit_price - t_entry if t_dir == "long"
               else t_entry - exit_price)
        trades.append(SupertrendTrade(
            date=_et(t_ts).date(),
            entry_time=t_ts,
            exit_time=exit_time,
            direction=t_dir,
            entry_price=t_entry,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_usd=round(pts * dpp, 2),
            confidence_at_entry=t_conf,
            bars_held=t_bars,
            stop_price=t_stop or 0.0,
            target_price=t_target or 0.0,
            daily_atr=t_atr,
        ))
        in_trade = False

    for bar in bars:
        ts_et    = _et(bar.timestamp)
        bar_time = ts_et.time()
        in_session = _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)

        # ── 1. Check TP/SL on every 1m bar ──────────────────────────
        if in_trade and t_stop is not None and t_target is not None:
            if t_dir == "long":
                if bar.low <= t_stop:
                    _close(t_stop, bar.timestamp, "stop")
                elif bar.high >= t_target:
                    _close(t_target, bar.timestamp, "target")
            else:
                if bar.high >= t_stop:
                    _close(t_stop, bar.timestamp, "stop")
                elif bar.low <= t_target:
                    _close(t_target, bar.timestamp, "target")

        # ── 2. Session end ───────────────────────────────────────────
        if in_trade and not in_session:
            _close(bar.close, bar.timestamp, "session_end")

        # ── 3. Accumulate N-minute bars, update engine ───────────────
        completed = accum.update(bar)
        if completed is not None:
            engine.update(completed)

        if in_trade:
            t_bars += 1

        # ── 4. Handle flips (only on completed bars, during session) ─
        if completed is not None and in_session and engine.flip is not None:
            if in_trade:
                _close(completed.close, bar.timestamp, "flip")

            new_dir = "long" if engine.flip == 1 else "short"
            entry_px = (completed.close + slip if new_dir == "long"
                        else completed.close - slip)

            bar_date = _cme_day(bar.timestamp)
            di = day_info.get(bar_date)
            if di is None:
                continue

            tp, sl = _nearest_levels(di.levels, entry_px, new_dir)
            if tp is None or sl is None:
                continue
            if new_dir == "long" and tp <= entry_px:
                continue
            if new_dir == "short" and tp >= entry_px:
                continue

            t_dir    = new_dir
            t_entry  = entry_px
            t_stop   = sl
            t_target = tp
            t_ts     = bar.timestamp
            t_conf   = engine.conf_score
            t_bars   = 0
            t_atr    = di.daily_atr
            in_trade = True

    if in_trade and bars:
        _close(bars[-1].close, bars[-1].timestamp, "data_end")

    return trades


# ── Walk-forward: Supertrend direction → hold to session end ─────────────────

def run_supertrend_hold_strategy(
    bars: list[Bar],
    cfg: SupertrendConfig,
) -> list[SupertrendTrade]:
    """
    Enter on Supertrend flip, hold until session end.
    While in a trade, subsequent flips are ignored.
    One trade at a time — pure directional bet.
    """
    if cfg.auto_params:
        lb, ct, pw = _auto_select(cfg.bar_minutes)
    else:
        lb, ct, pw = cfg.lookback, cfg.threshold, cfg.pred_weight

    engine = _SupertrendEngine(lb, ct, pw)
    accum  = MinuteBarAccumulator(cfg.bar_minutes)

    slip = cfg.slippage_ticks * cfg.tick_size
    dpp  = cfg.dollars_per_point * cfg.contracts

    trades: list[SupertrendTrade] = []

    in_trade = False
    t_dir:  Optional[str]      = None
    t_entry: Optional[float]   = None
    t_ts:   Optional[datetime] = None
    t_conf: float = 0.0
    t_bars: int   = 0

    def _close(exit_price: float, exit_time: datetime, reason: str) -> None:
        nonlocal in_trade
        pts = (exit_price - t_entry if t_dir == "long"
               else t_entry - exit_price)
        trades.append(SupertrendTrade(
            date=_et(t_ts).date(),
            entry_time=t_ts,
            exit_time=exit_time,
            direction=t_dir,
            entry_price=t_entry,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_usd=round(pts * dpp, 2),
            confidence_at_entry=t_conf,
            bars_held=t_bars,
        ))
        in_trade = False

    for bar in bars:
        ts_et    = _et(bar.timestamp)
        bar_time = ts_et.time()
        in_session = _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)

        # Close at session end only
        if in_trade and not in_session:
            _close(bar.close, bar.timestamp, "session_end")

        # Accumulate N-minute bars
        completed = accum.update(bar)
        if completed is not None:
            engine.update(completed)

        if in_trade:
            t_bars += 1

        # Enter on flip — only if flat and in session
        if (completed is not None and in_session
                and engine.flip is not None and not in_trade):
            new_dir = "long" if engine.flip == 1 else "short"
            t_entry = (completed.close + slip if new_dir == "long"
                       else completed.close - slip)
            t_dir   = new_dir
            t_ts    = bar.timestamp
            t_conf  = engine.conf_score
            t_bars  = 0
            in_trade = True

    if in_trade and bars:
        _close(bars[-1].close, bars[-1].timestamp, "data_end")

    return trades


# ── Pivot + Supertrend confluence strategy ───────────────────────────────────

@dataclass
class PivotSupertrendConfig:
    st_bar_minutes:     int   = 30
    st_lookback:        int   = 14
    st_threshold:       float = 2.0
    st_pred_weight:     float = 0.5
    st_auto_params:     bool  = True

    rej_bar_minutes:    int   = 5
    zone_buffer_pts:    float = 5.0
    min_rr:             float = 1.0
    short_tp_pct:       float = 1.0    # scale short TP distance (0.5 = half)
    allow_longs:        bool  = True
    allow_shorts:       bool  = True

    max_trades_per_day: int   = 3
    contracts:          int   = 1
    tick_size:          float = 0.25
    slippage_ticks:     int   = 1
    dollars_per_point:  float = 5.0

    trade_start_et:     dtime = dtime(8, 30)
    entry_cutoff_et:    dtime = dtime(15, 0)
    trade_end_et:       dtime = dtime(16, 0)


def _compute_traditional_pivots(
    h: float, l: float, c: float,
) -> list[tuple[str, float]]:
    p = (h + l + c) / 3.0
    levels = [
        ("S3", l - 2.0 * (h - p)),
        ("S2", p - (h - l)),
        ("S1", 2.0 * p - h),
        ("P",  p),
        ("R1", 2.0 * p - l),
        ("R2", p + (h - l)),
        ("R3", h + 2.0 * (p - l)),
    ]
    return sorted(levels, key=lambda x: x[1])


def _build_pivot_data(bars: list[Bar]) -> dict[date, list[tuple[str, float]]]:
    """Prior day OHLC → Traditional pivot levels for each trading day."""
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)
        if d not in daily:
            daily[d] = {"high": bar.high, "low": bar.low, "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"],  bar.high)
            s["low"]   = min(s["low"],   bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    result: dict[date, list[tuple[str, float]]] = {}
    for i in range(1, len(sorted_dates)):
        d = sorted_dates[i]
        prev = daily[sorted_dates[i - 1]]
        result[d] = _compute_traditional_pivots(
            prev["high"], prev["low"], prev["close"])
    return result


def _find_rejection(
    bar_5m: Bar5m,
    pivots: list[tuple[str, float]],
    st_dir: int,
    zone_buffer: float,
    used_levels: set[str],
) -> Optional[tuple[str, float]]:
    """Find a pivot rejection candle. Returns (level_name, level_price)."""
    candidates: list[tuple[str, float, float]] = []

    for name, level in pivots:
        if name in used_levels:
            continue

        if st_dir == 1:
            # Bullish: bar wicks into level from above, closes above it
            if bar_5m.low <= level + zone_buffer and bar_5m.close > level:
                candidates.append((name, level, abs(bar_5m.close - level)))
        elif st_dir == -1:
            # Bearish: bar wicks into level from below, closes below it
            if bar_5m.high >= level - zone_buffer and bar_5m.close < level:
                candidates.append((name, level, abs(bar_5m.close - level)))

    if not candidates:
        return None
    best = min(candidates, key=lambda x: x[2])
    return (best[0], best[1])


def run_pivot_supertrend_strategy(
    bars: list[Bar],
    cfg: PivotSupertrendConfig,
) -> list[SupertrendTrade]:
    """
    30m Supertrend for direction + 5m rejection at daily pivot levels.
    TP = next pivot in trade direction, SL = next pivot behind.
    """
    if cfg.st_auto_params:
        lb, ct, pw = _auto_select(cfg.st_bar_minutes)
    else:
        lb, ct, pw = cfg.st_lookback, cfg.st_threshold, cfg.st_pred_weight

    pivot_data = _build_pivot_data(bars)
    engine     = _SupertrendEngine(lb, ct, pw)
    st_accum   = MinuteBarAccumulator(cfg.st_bar_minutes)
    rej_accum  = MinuteBarAccumulator(cfg.rej_bar_minutes)

    slip = cfg.slippage_ticks * cfg.tick_size
    dpp  = cfg.dollars_per_point * cfg.contracts

    trades: list[SupertrendTrade] = []

    in_trade  = False
    t_dir:    Optional[str]      = None
    t_entry:  Optional[float]    = None
    t_stop:   Optional[float]    = None
    t_target: Optional[float]    = None
    t_ts:     Optional[datetime] = None
    t_conf:   float = 0.0
    t_bars:   int   = 0

    st_ready     = False
    cur_date:    Optional[date] = None
    day_trades   = 0
    used_levels: set[str] = set()

    def _close(exit_price: float, exit_time: datetime, reason: str) -> None:
        nonlocal in_trade
        pts = (exit_price - t_entry if t_dir == "long"
               else t_entry - exit_price)
        trades.append(SupertrendTrade(
            date=_et(t_ts).date(),
            entry_time=t_ts,
            exit_time=exit_time,
            direction=t_dir,
            entry_price=t_entry,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_usd=round(pts * dpp, 2),
            confidence_at_entry=t_conf,
            bars_held=t_bars,
            stop_price=t_stop or 0.0,
            target_price=t_target or 0.0,
        ))
        in_trade = False

    for bar in bars:
        ts_et    = _et(bar.timestamp)
        bar_time = ts_et.time()
        bar_date = _cme_day(bar.timestamp)
        in_session = _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)
        can_enter  = _in_window(bar_time, cfg.trade_start_et, cfg.entry_cutoff_et)

        # Day rollover
        if bar_date != cur_date:
            cur_date    = bar_date
            day_trades  = 0
            used_levels = set()

        # ── 1. TP/SL on every 1m bar ────────────────────────────────
        if in_trade and t_stop is not None and t_target is not None:
            if t_dir == "long":
                if bar.low <= t_stop:
                    _close(t_stop, bar.timestamp, "stop")
                elif bar.high >= t_target:
                    _close(t_target, bar.timestamp, "target")
            else:
                if bar.high >= t_stop:
                    _close(t_stop, bar.timestamp, "stop")
                elif bar.low <= t_target:
                    _close(t_target, bar.timestamp, "target")

        # ── 2. Session end ───────────────────────────────────────────
        if in_trade and not in_session:
            _close(bar.close, bar.timestamp, "session_end")

        # ── 3. Update Supertrend on 30m bars ─────────────────────────
        completed_st = st_accum.update(bar)
        if completed_st is not None:
            engine.update(completed_st)
            if engine.flip is not None:
                st_ready = True
                # Exit if Supertrend flips against position
                if in_trade:
                    if ((t_dir == "long" and engine.flip == -1)
                            or (t_dir == "short" and engine.flip == 1)):
                        _close(completed_st.close, bar.timestamp, "st_flip")

        if in_trade:
            t_bars += 1

        # ── 4. Rejection entry on 5m bars ────────────────────────────
        completed_rej = rej_accum.update(bar)
        if (completed_rej is not None and in_session and can_enter
                and not in_trade and st_ready
                and day_trades < cfg.max_trades_per_day):

            pivots = pivot_data.get(bar_date)
            if pivots is not None:
                rej = _find_rejection(
                    completed_rej, pivots, engine.t_dir,
                    cfg.zone_buffer_pts, used_levels)

                if rej is not None:
                    name, level_price = rej
                    direction = "long" if engine.t_dir == 1 else "short"

                    if direction == "long" and not cfg.allow_longs:
                        continue
                    if direction == "short" and not cfg.allow_shorts:
                        continue

                    entry_px = (completed_rej.close + slip
                                if direction == "long"
                                else completed_rej.close - slip)

                    pivot_prices = [p for _, p in pivots]
                    tp, sl = _nearest_levels(pivot_prices, entry_px, direction)
                    if tp is None or sl is None:
                        continue

                    # Scale short TP if configured
                    if direction == "short" and cfg.short_tp_pct < 1.0:
                        full_dist = abs(entry_px - tp)
                        tp = entry_px - full_dist * cfg.short_tp_pct

                    risk   = abs(entry_px - sl)
                    reward = abs(tp - entry_px)
                    if risk <= 0 or reward / risk < cfg.min_rr:
                        continue

                    t_dir    = direction
                    t_entry  = entry_px
                    t_stop   = sl
                    t_target = tp
                    t_ts     = bar.timestamp
                    t_conf   = engine.conf_score
                    t_bars   = 0
                    in_trade = True
                    used_levels.add(name)
                    day_trades += 1

    if in_trade and bars:
        _close(bars[-1].close, bars[-1].timestamp, "data_end")

    return trades


# ── Reporting ────────────────────────────────────────────────────────────────

def print_supertrend_summary(
    trades: list[SupertrendTrade],
    cfg: SupertrendConfig,
) -> None:
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
    pnl = sum(t.pnl_usd for t in trades)
    wr  = len(wins) / len(trades) * 100

    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        eq   += t.pnl_usd
        peak  = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    dates = sorted({t.date for t in trades})

    if cfg.auto_params:
        lb, ct, pw = _auto_select(cfg.bar_minutes)
    else:
        lb, ct, pw = cfg.lookback, cfg.threshold, cfg.pred_weight

    print(f"\n{'='*72}")
    print(f"  ADAPTIVE SUPERTREND  |  {cfg.bar_minutes}m bars  |  "
          f"lb={lb}  ct={ct}  pw={pw}  |  {cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  {len(dates)} trading days")
    print(f"{'='*72}")

    print(f"\n── OVERALL ──────────────────────────────────────────────────────")
    print(f"  Total trades  : {len(trades)}")
    print(f"  Wins / Losses : {len(wins)} / {len(losses)}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Total PnL     : ${pnl:>10,.2f}")
    print(f"  Profit factor : {pf_s}")
    avg_w = gw / len(wins) if wins else 0.0
    avg_l = gl / len(losses) if losses else 0.0
    print(f"  Avg win       : ${avg_w:>10,.2f}")
    print(f"  Avg loss      : ${avg_l:>10,.2f}")
    print(f"  Max drawdown  : ${max_dd:>10,.2f}")

    # Long / short breakdown
    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    print(f"\n── LONG / SHORT ─────────────────────────────────────────────────")
    for label, grp in [("Long", longs), ("Short", shorts)]:
        if not grp:
            print(f"  {label:<8}  (no trades)")
            continue
        w = [t for t in grp if t.pnl_usd > 0]
        l_ = [t for t in grp if t.pnl_usd <= 0]
        g_w = sum(t.pnl_usd for t in w)
        g_l = abs(sum(t.pnl_usd for t in l_))
        g_pf = g_w / g_l if g_l > 0 else float("inf")
        g_pf_s = f"{g_pf:.2f}" if g_pf != float("inf") else "∞"
        g_pnl = sum(t.pnl_usd for t in grp)
        g_wr  = len(w) / len(grp) * 100
        print(f"  {label:<8}  N={len(grp):>4}  {len(w)}W/{len(l_)}L  "
              f"WR={g_wr:>5.1f}%  PnL=${g_pnl:>10,.2f}  PF={g_pf_s}")

    # By exit reason
    from collections import defaultdict
    print(f"\n── BY EXIT REASON ───────────────────────────────────────────────")
    by_reason: dict[str, list[SupertrendTrade]] = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t)
    for reason in sorted(by_reason):
        grp = by_reason[reason]
        g_pnl = sum(t.pnl_usd for t in grp)
        w = sum(1 for t in grp if t.pnl_usd > 0)
        print(f"  {reason:<14}  {len(grp):>4} trades  {w}W/{len(grp)-w}L  "
              f"${g_pnl:>+10,.2f}")

    # Trade log (last 30)
    has_levels = any(t.target_price > 0 for t in trades)
    show = trades[-30:]
    if len(trades) > 30:
        print(f"\n── LAST 30 TRADES (of {len(trades)}) ─────────────────────────────────")
    else:
        print(f"\n── TRADE LOG ────────────────────────────────────────────────────")
    for t in show:
        et = t.entry_time.astimezone(_ET).strftime("%H:%M") if t.entry_time.tzinfo else "?"
        xt = t.exit_time.astimezone(_ET).strftime("%H:%M") if t.exit_time.tzinfo else "?"
        icon = "✓" if t.pnl_usd > 0 else "✗"
        lvl = (f"  sl={t.stop_price:.2f}  tp={t.target_price:.2f}"
               if has_levels else "")
        print(f"  {t.date}  {t.direction.upper():<5}  "
              f"ep={t.entry_price:.2f}{lvl}  exit={t.exit_price:.2f}  "
              f"({t.exit_reason})  conf={t.confidence_at_entry:.0%}  "
              f"{icon} ${t.pnl_usd:>+8,.2f}  {et}→{xt}")
    print()
