from __future__ import annotations

"""
HFT Scalp Strategy — Version 1
================================
1-minute automated scalp using 1M 15 EMA slope + 5M MACD slope alignment.

Entry logic:
  1. 1M 15 EMA slope and 5M MACD slope must agree on direction.
  2. Price pulls back to within pullbackBufferTicks of the 1M EMA.
  3. Price closes back through the EMA in the trend direction → entry.

Targets: fixed dollar TP / SL per contract (configurable).
Break-even requires > 67% win rate at default TP $50 / SL $100.
"""

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, _EMA
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class HFTScalpState(Enum):
    NO_TRADE               = "NO_TRADE"
    LONG_BIAS              = "LONG_BIAS"
    SHORT_BIAS             = "SHORT_BIAS"
    LONG_PULLBACK_WAIT     = "LONG_PULLBACK_WAIT"
    SHORT_PULLBACK_WAIT    = "SHORT_PULLBACK_WAIT"
    LONG_ENTRY_READY       = "LONG_ENTRY_READY"
    SHORT_ENTRY_READY      = "SHORT_ENTRY_READY"
    LONG_ACTIVE            = "LONG_ACTIVE"
    SHORT_ACTIVE           = "SHORT_ACTIVE"
    PAUSED_AFTER_LOSS      = "PAUSED_AFTER_LOSS"
    LOCKED_OUT_DAILY_LIMIT = "LOCKED_OUT_DAILY_LIMIT"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HFTScalpConfig:
    # 1M EMA
    ema_length:                  int   = 15
    ema_slope_lookback_bars:     int   = 3
    ema_slope_threshold:         float = 0.0   # min |slope| to count as trending

    # 5M MACD (computed on 5m bars)
    macd_fast_length:            int   = 12
    macd_slow_length:            int   = 26
    macd_signal_length:          int   = 9
    macd_slope_lookback_bars:    int   = 2     # lookback on MACD line history
    macd_slope_threshold:        float = 0.0   # min |slope| to count as trending

    # Pullback detection
    pullback_buffer_ticks:       int   = 3     # ticks from EMA that counts as pullback
    invalidation_buffer_ticks:   int   = 4     # ticks past EMA → cancel setup
    max_pullback_bars:           int   = 10    # 1m bars before timeout

    # Execution
    contracts:                   int   = 1
    tick_size:                   float = 0.25
    slippage_ticks:              int   = 1
    dollars_per_point:           float = 5.0

    # Fixed dollar P&L per trade
    take_profit_usd:             float = 50.0
    stop_loss_usd:               float = 100.0

    # Daily risk
    max_trades_per_day:          int   = 10
    max_consecutive_losses:      int   = 3
    daily_profit_target_usd:     float = 150.0
    daily_loss_limit_usd:        float = -300.0

    # Cooldowns (number of 1m bars)
    cooldown_after_trade_bars:   int   = 5
    cooldown_after_loss_bars:    int   = 10

    # Direction
    allow_longs:                 bool  = True
    allow_shorts:                bool  = True

    # Session (ET) — default RTH
    trade_start_et:              dtime = dtime(9, 30)
    entry_cutoff_et:             dtime = dtime(15, 30)
    trade_end_et:                dtime = dtime(16, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HFTScalpTrade:
    date:                date
    entry_time:          datetime
    exit_time:           datetime
    direction:           str      # "long" | "short"
    entry_price:         float
    stop_price:          float
    target_price:        float
    exit_price:          float
    exit_reason:         str      # "target" | "stop" | "session_end"
    pnl_usd:             float
    contracts:           int
    bars_held:           int
    ema_at_entry:        Optional[float]
    macd_line_at_entry:  Optional[float]
    macd_slope_at_entry: Optional[float]
    session:             str      # "rth" | "extended" | "overnight"


@dataclass
class HFTScalpTransitionLog:
    timestamp:    datetime
    state_before: str
    state_after:  str
    reason:       str
    price:        float
    ema:          Optional[float]
    ema_slope:    Optional[float]
    macd_line:    Optional[float]
    macd_signal:  Optional[float]
    macd_hist:    Optional[float]
    macd_slope:   Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Internal SM state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HFTSMState:
    state:               HFTScalpState = HFTScalpState.NO_TRADE
    pullback_bars:       int           = 0
    cooldown_bars:       int           = 0
    consecutive_losses:  int           = 0
    day_trades:          int           = 0
    day_pnl:             float         = 0.0
    invalidated_setups:  int           = 0
    skipped_cooldown:    int           = 0
    skipped_daily_limit: int           = 0


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def detectTrendBias(
    direction:       str,
    ema_slope:       Optional[float],
    macd_slope:      Optional[float],
    ema_threshold:   float,
    macd_threshold:  float,
) -> bool:
    if ema_slope is None or macd_slope is None:
        return False
    if direction == "bull":
        return ema_slope > ema_threshold and macd_slope > macd_threshold
    return ema_slope < -ema_threshold and macd_slope < -macd_threshold


def detectPullbackToEma(direction: str, bar: Bar, ema: float, buffer_pts: float) -> bool:
    if direction == "bull":
        return bar.low <= ema + buffer_pts
    return bar.high >= ema - buffer_pts


def detectEntryTrigger(direction: str, bar: Bar, ema: float) -> bool:
    if direction == "bull":
        return bar.close > ema
    return bar.close < ema


def detectInvalidation(
    direction:      str,
    bar:            Bar,
    ema:            float,
    inval_pts:      float,
    ema_slope:      Optional[float],
    macd_slope:     Optional[float],
    ema_threshold:  float,
    macd_threshold: float,
) -> Optional[str]:
    if direction == "bull":
        if ema_slope is not None and ema_slope <= ema_threshold:
            return "ema_slope_flat_or_negative"
        if macd_slope is not None and macd_slope <= macd_threshold:
            return "macd_slope_flat_or_negative"
        if bar.close < ema - inval_pts:
            return "price_broke_below_ema"
    else:
        if ema_slope is not None and ema_slope >= -ema_threshold:
            return "ema_slope_flat_or_positive"
        if macd_slope is not None and macd_slope >= -macd_threshold:
            return "macd_slope_flat_or_positive"
        if bar.close > ema + inval_pts:
            return "price_broke_above_ema"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Session helper
# ─────────────────────────────────────────────────────────────────────────────

def _session_label(t: dtime) -> str:
    if dtime(9, 30) <= t < dtime(16, 0):
        return "rth"
    if dtime(4, 0) <= t < dtime(9, 30):
        return "extended"
    return "overnight"


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_hft_scalp_research(
    bars: list[Bar],
    cfg:  HFTScalpConfig,
) -> tuple[list[HFTScalpTrade], list[HFTScalpTransitionLog]]:
    """Walk-forward backtest of the HFT scalp strategy on 1-minute bars."""

    tick      = cfg.tick_size
    slip      = cfg.slippage_ticks * tick
    tp_pts    = cfg.take_profit_usd  / (cfg.dollars_per_point * cfg.contracts)
    sl_pts    = cfg.stop_loss_usd    / (cfg.dollars_per_point * cfg.contracts)
    pb_buf    = cfg.pullback_buffer_ticks     * tick
    inval_buf = cfg.invalidation_buffer_ticks * tick

    # 1M EMA
    ema_ind      = _EMA(cfg.ema_length)
    ema_hist:    deque[float] = deque(maxlen=cfg.ema_slope_lookback_bars + 1)
    cur_ema:     Optional[float] = None
    cur_ema_sl:  Optional[float] = None   # slope

    # 5M MACD
    acc5m        = MinuteBarAccumulator(5)
    macd_fast    = _EMA(cfg.macd_fast_length)
    macd_slow    = _EMA(cfg.macd_slow_length)
    macd_sig_ema = _EMA(cfg.macd_signal_length)
    macd_hist:   deque[float] = deque(maxlen=cfg.macd_slope_lookback_bars + 1)
    cur_macd_ln: Optional[float] = None
    cur_macd_sg: Optional[float] = None
    cur_macd_ht: Optional[float] = None
    cur_macd_sl: Optional[float] = None   # slope of MACD line

    sm        = _HFTSMState()
    cur_day:  Optional[date] = None

    # Active trade fields
    in_trade     = False
    t_dir:       Optional[str]      = None
    t_entry:     Optional[float]    = None
    t_target:    Optional[float]    = None
    t_stop:      Optional[float]    = None
    t_ts:        Optional[datetime] = None
    t_ema:       Optional[float]    = None
    t_macd:      Optional[float]    = None
    t_macd_sl:   Optional[float]    = None
    t_bars:      int                = 0
    t_session:   str                = ""

    trades: list[HFTScalpTrade]         = []
    logs:   list[HFTScalpTransitionLog] = []

    def _log(bar: Bar, before: HFTScalpState, after: HFTScalpState, reason: str) -> None:
        logs.append(HFTScalpTransitionLog(
            timestamp    = bar.timestamp,
            state_before = before.value,
            state_after  = after.value,
            reason       = reason,
            price        = bar.close,
            ema          = cur_ema,
            ema_slope    = cur_ema_sl,
            macd_line    = cur_macd_ln,
            macd_signal  = cur_macd_sg,
            macd_hist    = cur_macd_ht,
            macd_slope   = cur_macd_sl,
        ))

    def _transition(new: HFTScalpState, bar: Bar, reason: str) -> None:
        old = sm.state
        sm.state = new
        _log(bar, old, new, reason)

    def _close_trade(bar: Bar, exit_price: float, reason: str) -> None:
        nonlocal in_trade
        pts = (exit_price - t_entry) if t_dir == "long" else (t_entry - exit_price)
        pnl = round(pts * cfg.dollars_per_point * cfg.contracts, 2)

        sm.day_pnl    += pnl
        sm.day_trades += 1

        if pnl <= 0:
            sm.consecutive_losses += 1
        else:
            sm.consecutive_losses = 0

        trades.append(HFTScalpTrade(
            date                = bar.timestamp.astimezone(_ET).date(),
            entry_time          = t_ts,
            exit_time           = bar.timestamp,
            direction           = t_dir,
            entry_price         = t_entry,
            stop_price          = t_stop,
            target_price        = t_target,
            exit_price          = exit_price,
            exit_reason         = reason,
            pnl_usd             = pnl,
            contracts           = cfg.contracts,
            bars_held           = t_bars,
            ema_at_entry        = t_ema,
            macd_line_at_entry  = t_macd,
            macd_slope_at_entry = t_macd_sl,
            session             = t_session,
        ))
        in_trade = False

        # Post-trade state
        ts_et = bar.timestamp.astimezone(_ET)
        if (sm.day_pnl <= cfg.daily_loss_limit_usd
                or sm.day_pnl >= cfg.daily_profit_target_usd
                or sm.day_trades >= cfg.max_trades_per_day
                or sm.consecutive_losses >= cfg.max_consecutive_losses):
            reason_lock = (
                "daily_loss_limit"   if sm.day_pnl <= cfg.daily_loss_limit_usd else
                "daily_profit_target" if sm.day_pnl >= cfg.daily_profit_target_usd else
                "max_trades"         if sm.day_trades >= cfg.max_trades_per_day else
                "max_consecutive_losses"
            )
            _transition(HFTScalpState.LOCKED_OUT_DAILY_LIMIT, bar, reason_lock)
        else:
            cooldown = (cfg.cooldown_after_loss_bars if pnl <= 0
                        else cfg.cooldown_after_trade_bars)
            sm.cooldown_bars = cooldown
            _transition(HFTScalpState.PAUSED_AFTER_LOSS, bar,
                        f"cooldown_{cooldown}bars_after_{'loss' if pnl <= 0 else 'trade'}")

    for bar in bars:
        ts_et    = bar.timestamp.astimezone(_ET)
        bar_date = ts_et.date()
        bar_time = ts_et.time()

        # ── Daily reset ───────────────────────────────────────────────────────
        if bar_date != cur_day:
            cur_day = bar_date
            sm.day_trades         = 0
            sm.day_pnl            = 0.0
            sm.consecutive_losses = 0
            if sm.state not in (HFTScalpState.LONG_ACTIVE, HFTScalpState.SHORT_ACTIVE):
                if sm.state != HFTScalpState.NO_TRADE:
                    _transition(HFTScalpState.NO_TRADE, bar, "new_day_reset")

        # ── Update 5M MACD ────────────────────────────────────────────────────
        completed_5m = acc5m.update(bar)
        if completed_5m is not None:
            fv = macd_fast.update(completed_5m.close)
            sv = macd_slow.update(completed_5m.close)
            if fv is not None and sv is not None:
                ml  = fv - sv
                sgv = macd_sig_ema.update(ml)
                if sgv is not None:
                    cur_macd_ln = ml
                    cur_macd_sg = sgv
                    cur_macd_ht = ml - sgv
                    macd_hist.append(ml)
                    if len(macd_hist) >= cfg.macd_slope_lookback_bars + 1:
                        cur_macd_sl = macd_hist[-1] - macd_hist[0]

        # ── Update 1M EMA ─────────────────────────────────────────────────────
        ev = ema_ind.update(bar.close)
        if ev is not None:
            cur_ema = ev
            ema_hist.append(ev)
            if len(ema_hist) >= cfg.ema_slope_lookback_bars + 1:
                cur_ema_sl = ema_hist[-1] - ema_hist[0]

        in_session = cfg.trade_start_et <= bar_time < cfg.trade_end_et
        can_enter  = cfg.trade_start_et <= bar_time < cfg.entry_cutoff_et

        # ── Active trade management ───────────────────────────────────────────
        if in_trade:
            t_bars += 1
            hit_tp = hit_sl = False

            if t_dir == "long":
                if bar.high >= t_target:
                    hit_tp = True;  exit_px = t_target
                elif bar.low <= t_stop:
                    hit_sl = True;  exit_px = t_stop - slip
            else:
                if bar.low <= t_target:
                    hit_tp = True;  exit_px = t_target
                elif bar.high >= t_stop:
                    hit_sl = True;  exit_px = t_stop + slip

            if hit_tp or hit_sl:
                _close_trade(bar, exit_px, "target" if hit_tp else "stop")
            elif not in_session:
                _close_trade(bar, bar.close, "session_end")

            if in_trade:
                continue

        # ── Skip outside session for new setups ───────────────────────────────
        if not in_session:
            if sm.state not in (HFTScalpState.NO_TRADE,
                                HFTScalpState.LOCKED_OUT_DAILY_LIMIT,
                                HFTScalpState.PAUSED_AFTER_LOSS):
                _transition(HFTScalpState.NO_TRADE, bar, "session_end_reset")
            continue

        if cur_ema is None or cur_macd_sl is None:
            continue

        # ── PAUSED_AFTER_LOSS / cooldown ──────────────────────────────────────
        if sm.state == HFTScalpState.PAUSED_AFTER_LOSS:
            sm.cooldown_bars -= 1
            if sm.cooldown_bars <= 0:
                _transition(HFTScalpState.NO_TRADE, bar, "cooldown_expired")
            continue

        # ── LOCKED_OUT_DAILY_LIMIT ────────────────────────────────────────────
        if sm.state == HFTScalpState.LOCKED_OUT_DAILY_LIMIT:
            continue

        # ── Re-check daily limits ─────────────────────────────────────────────
        if (sm.day_pnl <= cfg.daily_loss_limit_usd
                or sm.day_pnl >= cfg.daily_profit_target_usd
                or sm.day_trades >= cfg.max_trades_per_day):
            if sm.state != HFTScalpState.LOCKED_OUT_DAILY_LIMIT:
                _transition(HFTScalpState.LOCKED_OUT_DAILY_LIMIT, bar, "daily_limit_check")
            continue

        # ── NO_TRADE → look for bias ──────────────────────────────────────────
        if sm.state == HFTScalpState.NO_TRADE:
            if not can_enter:
                continue
            if cfg.allow_longs and detectTrendBias(
                    "bull", cur_ema_sl, cur_macd_sl,
                    cfg.ema_slope_threshold, cfg.macd_slope_threshold):
                _transition(HFTScalpState.LONG_BIAS, bar, "bull_bias_detected")
            elif cfg.allow_shorts and detectTrendBias(
                    "bear", cur_ema_sl, cur_macd_sl,
                    cfg.ema_slope_threshold, cfg.macd_slope_threshold):
                _transition(HFTScalpState.SHORT_BIAS, bar, "bear_bias_detected")

        # ── LONG_BIAS → wait for pullback ──────────────────────────────────────
        elif sm.state == HFTScalpState.LONG_BIAS:
            inval = detectInvalidation("bull", bar, cur_ema, inval_buf,
                                       cur_ema_sl, cur_macd_sl,
                                       cfg.ema_slope_threshold, cfg.macd_slope_threshold)
            if inval:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, f"invalidated_{inval}")
            elif not can_enter:
                _transition(HFTScalpState.NO_TRADE, bar, "past_entry_cutoff")
            elif detectPullbackToEma("bull", bar, cur_ema, pb_buf):
                sm.pullback_bars = 0
                _transition(HFTScalpState.LONG_PULLBACK_WAIT, bar, "pullback_to_ema")

        # ── SHORT_BIAS → wait for pullback ────────────────────────────────────
        elif sm.state == HFTScalpState.SHORT_BIAS:
            inval = detectInvalidation("bear", bar, cur_ema, inval_buf,
                                       cur_ema_sl, cur_macd_sl,
                                       cfg.ema_slope_threshold, cfg.macd_slope_threshold)
            if inval:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, f"invalidated_{inval}")
            elif not can_enter:
                _transition(HFTScalpState.NO_TRADE, bar, "past_entry_cutoff")
            elif detectPullbackToEma("bear", bar, cur_ema, pb_buf):
                sm.pullback_bars = 0
                _transition(HFTScalpState.SHORT_PULLBACK_WAIT, bar, "pullback_to_ema")

        # ── LONG_PULLBACK_WAIT → wait for re-cross ────────────────────────────
        elif sm.state == HFTScalpState.LONG_PULLBACK_WAIT:
            sm.pullback_bars += 1
            inval = detectInvalidation("bull", bar, cur_ema, inval_buf,
                                       cur_ema_sl, cur_macd_sl,
                                       cfg.ema_slope_threshold, cfg.macd_slope_threshold)
            if inval:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, f"invalidated_{inval}")
            elif sm.pullback_bars > cfg.max_pullback_bars:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, "pullback_timeout")
            elif not can_enter:
                _transition(HFTScalpState.NO_TRADE, bar, "past_entry_cutoff")
            elif detectEntryTrigger("bull", bar, cur_ema):
                _transition(HFTScalpState.LONG_ENTRY_READY, bar, "close_above_ema")

        # ── SHORT_PULLBACK_WAIT → wait for re-cross ───────────────────────────
        elif sm.state == HFTScalpState.SHORT_PULLBACK_WAIT:
            sm.pullback_bars += 1
            inval = detectInvalidation("bear", bar, cur_ema, inval_buf,
                                       cur_ema_sl, cur_macd_sl,
                                       cfg.ema_slope_threshold, cfg.macd_slope_threshold)
            if inval:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, f"invalidated_{inval}")
            elif sm.pullback_bars > cfg.max_pullback_bars:
                sm.invalidated_setups += 1
                _transition(HFTScalpState.NO_TRADE, bar, "pullback_timeout")
            elif not can_enter:
                _transition(HFTScalpState.NO_TRADE, bar, "past_entry_cutoff")
            elif detectEntryTrigger("bear", bar, cur_ema):
                _transition(HFTScalpState.SHORT_ENTRY_READY, bar, "close_below_ema")

        # ── ENTRY_READY → open trade ──────────────────────────────────────────
        if sm.state in (HFTScalpState.LONG_ENTRY_READY, HFTScalpState.SHORT_ENTRY_READY):
            is_long = sm.state == HFTScalpState.LONG_ENTRY_READY

            ep = bar.close + slip if is_long else bar.close - slip
            t_dir      = "long" if is_long else "short"
            t_entry    = ep
            t_target   = ep + tp_pts if is_long else ep - tp_pts
            t_stop     = ep - sl_pts if is_long else ep + sl_pts
            t_ts       = bar.timestamp
            t_ema      = cur_ema
            t_macd     = cur_macd_ln
            t_macd_sl  = cur_macd_sl
            t_bars     = 0
            t_session  = _session_label(bar_time)
            in_trade   = True

            _transition(
                HFTScalpState.LONG_ACTIVE if is_long else HFTScalpState.SHORT_ACTIVE,
                bar, "order_filled",
            )

    return trades, logs


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper
# ─────────────────────────────────────────────────────────────────────────────

def hft_scalp_summary(trades: list[HFTScalpTrade], cfg: HFTScalpConfig) -> str:
    if not trades:
        return "No trades."

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total  = sum(t.pnl_usd for t in trades)
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pf     = gw / gl if gl else float("inf")
    wr     = len(wins) / len(trades) * 100

    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    lw     = [t for t in longs  if t.pnl_usd > 0]
    sw     = [t for t in shorts if t.pnl_usd > 0]

    by_sess: dict[str, list] = {}
    for t in trades:
        by_sess.setdefault(t.session, []).append(t)

    be_wr = cfg.stop_loss_usd / (cfg.stop_loss_usd + cfg.take_profit_usd) * 100

    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    avg_bars = sum(t.bars_held for t in trades) / len(trades)

    lines = [
        f"TP ${cfg.take_profit_usd:.0f} / SL ${cfg.stop_loss_usd:.0f}"
        f"  |  Break-even WR: {be_wr:.1f}%",
        "─" * 55,
        f"  Trades     : {len(trades):>4}  ({len(wins)}W / {len(losses)}L)",
        f"  Win rate   : {wr:>6.1f}%  (need >{be_wr:.1f}%)",
        f"  Net P/L    : ${total:>10,.2f}",
        f"  Max DD     : ${max_dd:>10,.2f}",
        f"  Avg win    : ${gw/len(wins):>8,.2f}" if wins else "  Avg win    : —",
        f"  Avg loss   : ${gl/len(losses):>8,.2f}" if losses else "  Avg loss   : —",
        f"  Profit fac : {pf:.2f}",
        f"  Avg bars   : {avg_bars:.1f}",
        f"  Long  WR   : {len(lw)/len(longs)*100:.1f}%  ({len(longs)} trades)" if longs else "  Long  WR   : —",
        f"  Short WR   : {len(sw)/len(shorts)*100:.1f}%  ({len(shorts)} trades)" if shorts else "  Short WR   : —",
    ]
    if by_sess:
        lines.append("  By session :")
        for sess, ts in sorted(by_sess.items()):
            sw_ = [t for t in ts if t.pnl_usd > 0]
            lines.append(
                f"    {sess:<10}  {len(ts):>3} trades  "
                f"{len(sw_)/len(ts)*100:.1f}% WR  "
                f"${sum(t.pnl_usd for t in ts):,.2f}"
            )
    return "\n".join(lines)
