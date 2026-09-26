from __future__ import annotations

"""
POI Strategy — Pre-session level + setup confirmation
======================================================

Flow:
  1. Pre-session POI scanner identifies 3 price levels.
  2. During the session, state machines watch for price to reach each level.
  3. At the level, the strategy ANALYZES: waits for 5M MACD slope and 1M EMA
     slope to align with the POI bias before committing.
  4. Entry triggers when price closes back through the level in the bias
     direction with indicators confirmed.
  5. Stop is placed on the other side of the POI zone.  Target is the next
     POI level or a fixed R multiple.

State machine (per POI):
  WATCHING          price has not reached the zone
  ZONE_TOUCHED      first touch of the POI zone — start analyzing
  SETUP_FORMING     indicators beginning to align (MACD slope turning)
  ENTRY_CONFIRMED   full confirmation — entry placed this bar
  USED              trade taken from this level today
  EXPIRED           timed out without a valid setup

Global:
  IN_TRADE          a position is open (blocks new POI entries)
  DAILY_LOCKED      daily loss/profit limit reached
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, _EMA, Bar5m
from .projectx_client import Bar
from .poi_scanner import POI

_ET = ZoneInfo("America/New_York")


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    """True if time `t` falls in [start, end), handling sessions that cross midnight."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end   # e.g. 18:00 → 17:00 crosses midnight


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class POIZoneState(Enum):
    WATCHING          = "WATCHING"
    ZONE_TOUCHED      = "ZONE_TOUCHED"
    SETUP_FORMING     = "SETUP_FORMING"
    ENTRY_CONFIRMED   = "ENTRY_CONFIRMED"
    USED              = "USED"
    EXPIRED           = "EXPIRED"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class POIStrategyConfig:
    # Zone detection
    zone_buffer_ticks:        int   = 6     # ticks from POI price = "at the zone"
    max_zone_bars:            int   = 20    # 5m bars at zone before expiry

    # Setup confirmation filters
    require_macd_slope_align: bool  = True  # 5M MACD slope must agree with bias
    require_ema_slope_align:  bool  = True  # 1M EMA slope must agree with bias
    macd_slope_threshold:     float = 0.0
    ema_slope_threshold:      float = 0.0
    ema_slope_lookback_bars:  int   = 3

    # Entry bar quality
    entry_clean_close_ticks:  int   = 8    # max wick through level in wrong direction

    # Indicators
    ema_length:               int   = 15
    macd_fast_length:         int   = 12
    macd_slow_length:         int   = 26
    macd_signal_length:       int   = 9
    macd_slope_lookback_bars: int   = 2

    # Stop / target
    stop_buffer_ticks:        int   = 6    # ticks beyond the zone for stop
    stop_atr_pct:             float = 0.25 # alternative: % of daily ATR
    use_atr_stop:             bool  = True
    target_r_multiple:        float = 2.0  # R multiple if no next POI to target

    # Execution
    contracts:                int   = 1
    tick_size:                float = 0.25
    slippage_ticks:           int   = 1
    dollars_per_point:        float = 5.0

    # Daily risk
    max_trades_per_day:       int   = 3
    daily_loss_limit_usd:     float = -500.0
    daily_profit_limit_usd:   float = 1500.0

    # Session (ET)
    trade_start_et:           dtime = dtime(6, 0)
    entry_cutoff_et:          dtime = dtime(15, 30)
    trade_end_et:             dtime = dtime(16, 0)

    # Allow direction
    allow_longs:              bool  = True
    allow_shorts:             bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class POITrade:
    date:               date
    poi_rank:           int       # which POI (1/2/3)
    poi_price:          float     # the scanner's level
    poi_bias:           str       # "support" | "resistance" | "pivot"
    poi_label:          str       # e.g. "PDH / Asia High / ATR +75%"
    entry_time:         datetime
    exit_time:          datetime
    direction:          str       # "long" | "short"
    entry_price:        float
    stop_price:         float
    target_price:       float
    exit_price:         float
    exit_reason:        str       # "target" | "stop" | "session_end"
    pnl_usd:            float
    pnl_r:              float
    risk_pts:           float
    bars_in_zone:       int       # 5m bars spent at zone before entry
    macd_slope_entry:   Optional[float]
    ema_slope_entry:    Optional[float]


@dataclass
class POITransitionLog:
    timestamp:    datetime
    poi_rank:     int
    poi_price:    float
    state_before: str
    state_after:  str
    reason:       str
    price:        float
    macd_slope:   Optional[float]
    ema_slope:    Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Per-POI tracking state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _POIState:
    poi:          POI
    state:        POIZoneState = POIZoneState.WATCHING
    zone_bars:    int          = 0     # 5m bars spent at or near zone
    touched_time: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _at_zone(bar5m: Bar5m, poi_price: float, buffer: float) -> bool:
    """Price has touched or entered the zone around the POI."""
    return bar5m.low <= poi_price + buffer and bar5m.high >= poi_price - buffer


def _macd_aligned(bias: str, macd_slope: Optional[float], threshold: float) -> bool:
    if macd_slope is None:
        return False
    if bias == "support":
        return macd_slope > threshold
    if bias == "resistance":
        return macd_slope < -threshold
    # pivot: either direction counts
    return abs(macd_slope) > threshold


def _ema_aligned(bias: str, ema_slope: Optional[float], threshold: float) -> bool:
    if ema_slope is None:
        return False
    if bias == "support":
        return ema_slope > threshold
    if bias == "resistance":
        return ema_slope < -threshold
    return abs(ema_slope) > threshold


def _entry_direction(bias: str, bar5m: Bar5m, poi_price: float,
                     macd_slope: Optional[float]) -> Optional[str]:
    """
    Determine entry direction for a pivot POI based on which way price
    moved away from the level and where MACD is pointing.
    """
    if bias == "support":
        return "long" if bar5m.close > poi_price else None
    if bias == "resistance":
        return "short" if bar5m.close < poi_price else None
    # pivot — use MACD slope to determine direction
    if bar5m.close > poi_price and macd_slope is not None and macd_slope > 0:
        return "long"
    if bar5m.close < poi_price and macd_slope is not None and macd_slope < 0:
        return "short"
    return None


def _clean_close(direction: str, bar5m: Bar5m, poi_price: float, buffer: float) -> bool:
    """Entry bar must not wick significantly through the level in the wrong direction."""
    if direction == "long":
        return poi_price - bar5m.low <= buffer
    return bar5m.high - poi_price <= buffer


def _next_poi_target(direction: str, entry_price: float,
                     all_pois: list[_POIState]) -> Optional[float]:
    """Find the next POI price in the trade direction for use as target."""
    candidates = []
    for ps in all_pois:
        p = ps.poi.price
        if direction == "long"  and p > entry_price:
            candidates.append(p)
        if direction == "short" and p < entry_price:
            candidates.append(p)
    if not candidates:
        return None
    return min(candidates) if direction == "long" else max(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_poi_strategy(
    bars:      list[Bar],
    pois:      list[POI],
    cfg:       POIStrategyConfig,
    daily_atr: float = 0.0,
) -> tuple[list[POITrade], list[POITransitionLog]]:
    """
    Walk-forward backtest of the POI strategy.

    Parameters
    ----------
    bars      : 1-minute bars sorted oldest → newest
    pois      : exactly 3 POIs from the scanner (rank 1/2/3)
    cfg       : strategy configuration
    daily_atr : today's ATR for stop sizing (required if use_atr_stop=True)
    """
    tick  = cfg.tick_size
    slip  = cfg.slippage_ticks * tick
    zbuf  = cfg.zone_buffer_ticks     * tick
    sbuf  = cfg.stop_buffer_ticks     * tick
    cbuf  = cfg.entry_clean_close_ticks * tick

    # ── 5M bar accumulator + MACD ─────────────────────────────────────────────
    acc5m        = MinuteBarAccumulator(5)
    macd_fast    = _EMA(cfg.macd_fast_length)
    macd_slow    = _EMA(cfg.macd_slow_length)
    macd_sig_ema = _EMA(cfg.macd_signal_length)
    macd_hist:   deque[float] = deque(maxlen=cfg.macd_slope_lookback_bars + 1)
    cur_macd_sl: Optional[float] = None
    cur_macd_ln: Optional[float] = None

    # ── 1M EMA ────────────────────────────────────────────────────────────────
    ema1m     = _EMA(cfg.ema_length)
    ema_hist: deque[float] = deque(maxlen=cfg.ema_slope_lookback_bars + 1)
    cur_ema:  Optional[float] = None
    cur_ema_sl: Optional[float] = None

    # ── POI state machines ────────────────────────────────────────────────────
    poi_states = [_POIState(poi=p) for p in pois]

    # ── Active trade ──────────────────────────────────────────────────────────
    in_trade   = False
    t_dir:     Optional[str]      = None
    t_entry:   Optional[float]    = None
    t_stop:    Optional[float]    = None
    t_target:  Optional[float]    = None
    t_ts:      Optional[datetime] = None
    t_poi:     Optional[_POIState]= None
    t_risk:    float              = 0.0
    t_bars:    int                = 0
    t_macd_sl: Optional[float]    = None
    t_ema_sl:  Optional[float]    = None
    t_zone_bars: int              = 0

    # ── Daily counters ────────────────────────────────────────────────────────
    day_trades = 0
    day_pnl    = 0.0
    locked     = False

    trades: list[POITrade]         = []
    logs:   list[POITransitionLog] = []

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log_poi(ps: _POIState, bar_ts: datetime,
                 before: POIZoneState, after: POIZoneState,
                 reason: str, price: float) -> None:
        logs.append(POITransitionLog(
            timestamp    = bar_ts,
            poi_rank     = ps.poi.rank,
            poi_price    = ps.poi.price,
            state_before = before.value,
            state_after  = after.value,
            reason       = reason,
            price        = price,
            macd_slope   = cur_macd_sl,
            ema_slope    = cur_ema_sl,
        ))

    def _set_poi_state(ps: _POIState, new: POIZoneState,
                       bar_ts: datetime, reason: str, price: float) -> None:
        old = ps.state
        ps.state = new
        _log_poi(ps, bar_ts, old, new, reason, price)

    # ─────────────────────────────────────────────────────────────────────────
    # Process 1m bars one at a time
    # ─────────────────────────────────────────────────────────────────────────
    for bar in bars:
        ts_et    = bar.timestamp.astimezone(_ET)
        bar_time = ts_et.time()
        in_session = _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)
        can_enter  = _in_window(bar_time, cfg.trade_start_et, cfg.entry_cutoff_et)

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
                    macd_hist.append(ml)
                    if len(macd_hist) >= cfg.macd_slope_lookback_bars + 1:
                        cur_macd_sl = macd_hist[-1] - macd_hist[0]

            # ── Per-POI 5M analysis (runs on each completed 5m bar) ───────────
            if in_session and not locked:
                for ps in poi_states:
                    if ps.state in (POIZoneState.USED, POIZoneState.EXPIRED,
                                    POIZoneState.ENTRY_CONFIRMED):
                        continue

                    at_z = _at_zone(completed_5m, ps.poi.price, zbuf)

                    # WATCHING → ZONE_TOUCHED
                    if ps.state == POIZoneState.WATCHING and at_z:
                        ps.zone_bars    = 1
                        ps.touched_time = bar.timestamp
                        _set_poi_state(ps, POIZoneState.ZONE_TOUCHED,
                                       bar.timestamp, "zone_touched",
                                       completed_5m.close)

                    elif ps.state == POIZoneState.ZONE_TOUCHED:
                        if not at_z:
                            _set_poi_state(ps, POIZoneState.WATCHING,
                                           bar.timestamp, "left_zone_without_setup",
                                           completed_5m.close)
                            continue
                        ps.zone_bars += 1

                        # Check if setup is forming
                        macd_ok = (not cfg.require_macd_slope_align or
                                   _macd_aligned(ps.poi.bias, cur_macd_sl,
                                                 cfg.macd_slope_threshold))
                        ema_ok  = (not cfg.require_ema_slope_align or
                                   _ema_aligned(ps.poi.bias, cur_ema_sl,
                                                cfg.ema_slope_threshold))

                        if macd_ok or ema_ok:
                            _set_poi_state(ps, POIZoneState.SETUP_FORMING,
                                           bar.timestamp, "indicators_aligning",
                                           completed_5m.close)

                        if ps.zone_bars > cfg.max_zone_bars:
                            _set_poi_state(ps, POIZoneState.EXPIRED,
                                           bar.timestamp, "zone_timeout",
                                           completed_5m.close)

                    elif ps.state == POIZoneState.SETUP_FORMING:
                        if not at_z:
                            _set_poi_state(ps, POIZoneState.WATCHING,
                                           bar.timestamp, "left_zone_before_entry",
                                           completed_5m.close)
                            continue
                        ps.zone_bars += 1
                        if ps.zone_bars > cfg.max_zone_bars:
                            _set_poi_state(ps, POIZoneState.EXPIRED,
                                           bar.timestamp, "zone_timeout",
                                           completed_5m.close)

        # ── Update 1M EMA ─────────────────────────────────────────────────────
        ev = ema1m.update(bar.close)
        if ev is not None:
            cur_ema = ev
            ema_hist.append(ev)
            if len(ema_hist) >= cfg.ema_slope_lookback_bars + 1:
                cur_ema_sl = ema_hist[-1] - ema_hist[0]

        if not in_session:
            continue

        # ── Active trade management (1m resolution for TP/SL) ─────────────────
        if in_trade:
            t_bars += 1
            hit_tp = hit_sl = False
            exit_px = bar.close

            if t_dir == "long":
                if bar.high >= t_target:
                    hit_tp = True; exit_px = t_target
                elif bar.low <= t_stop:
                    hit_sl = True; exit_px = t_stop - slip
            else:
                if bar.low <= t_target:
                    hit_tp = True; exit_px = t_target
                elif bar.high >= t_stop:
                    hit_sl = True; exit_px = t_stop + slip

            force_exit = (not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et)
                          and not hit_tp and not hit_sl)

            if hit_tp or hit_sl or force_exit:
                pts   = (exit_px - t_entry) if t_dir == "long" else (t_entry - exit_px)
                pnl   = round(pts * cfg.dollars_per_point * cfg.contracts, 2)
                risk  = abs(t_entry - t_stop) * cfg.dollars_per_point * cfg.contracts
                pnl_r = round(pnl / risk, 2) if risk > 0 else 0.0
                reason = "target" if hit_tp else ("stop" if hit_sl else "session_end")

                day_pnl    += pnl
                day_trades += 1

                trades.append(POITrade(
                    date             = ts_et.date(),
                    poi_rank         = t_poi.poi.rank,
                    poi_price        = t_poi.poi.price,
                    poi_bias         = t_poi.poi.bias,
                    poi_label        = t_poi.poi.label,
                    entry_time       = t_ts,
                    exit_time        = bar.timestamp,
                    direction        = t_dir,
                    entry_price      = t_entry,
                    stop_price       = t_stop,
                    target_price     = t_target,
                    exit_price       = exit_px,
                    exit_reason      = reason,
                    pnl_usd          = pnl,
                    pnl_r            = pnl_r,
                    risk_pts         = t_risk,
                    bars_in_zone     = t_zone_bars,
                    macd_slope_entry = t_macd_sl,
                    ema_slope_entry  = t_ema_sl,
                ))

                t_poi.state = POIZoneState.USED
                in_trade = False

                if (day_pnl <= cfg.daily_loss_limit_usd
                        or day_pnl >= cfg.daily_profit_limit_usd
                        or day_trades >= cfg.max_trades_per_day):
                    locked = True

            if in_trade:
                continue

        if locked or not can_enter or in_trade:
            continue

        # ── Entry scan: check each POI in SETUP_FORMING state ─────────────────
        # Use the completed 5m bar for entry detection (no lookahead on 1m)
        # Entry is confirmed on the 1m close after the 5m setup is forming
        for ps in poi_states:
            if ps.state != POIZoneState.SETUP_FORMING:
                continue

            # Quick direction check
            direction = _entry_direction(ps.poi.bias, bar, ps.poi.price, cur_macd_sl)
            if direction is None:
                continue
            if direction == "long"  and not cfg.allow_longs:
                continue
            if direction == "short" and not cfg.allow_shorts:
                continue

            # Both indicators must now be aligned
            macd_ok = (not cfg.require_macd_slope_align or
                       _macd_aligned(ps.poi.bias, cur_macd_sl, cfg.macd_slope_threshold))
            ema_ok  = (not cfg.require_ema_slope_align or
                       _ema_aligned(ps.poi.bias, cur_ema_sl, cfg.ema_slope_threshold))

            if not (macd_ok and ema_ok):
                continue

            # Clean close: bar must not have wicked badly through the level
            if not _clean_close(direction, bar, ps.poi.price, cbuf):
                continue

            # Build entry
            ep  = bar.close + slip if direction == "long" else bar.close - slip

            if cfg.use_atr_stop and daily_atr > 0:
                atr_dist = cfg.stop_atr_pct * daily_atr
                stp = ep - atr_dist if direction == "long" else ep + atr_dist
            else:
                stp = (ps.poi.price - sbuf if direction == "long"
                       else ps.poi.price + sbuf)

            risk_pts = abs(ep - stp)
            nxt_poi  = _next_poi_target(direction, ep, poi_states)
            tgt      = (nxt_poi if nxt_poi is not None
                        else ep + risk_pts * cfg.target_r_multiple if direction == "long"
                        else ep - risk_pts * cfg.target_r_multiple)

            # Sanity: target must be beyond entry in the right direction
            if direction == "long"  and tgt <= ep:
                continue
            if direction == "short" and tgt >= ep:
                continue

            _set_poi_state(ps, POIZoneState.ENTRY_CONFIRMED,
                           bar.timestamp, f"entry_{direction}", bar.close)

            in_trade    = True
            t_dir       = direction
            t_entry     = ep
            t_stop      = stp
            t_target    = tgt
            t_ts        = bar.timestamp
            t_poi       = ps
            t_risk      = risk_pts
            t_bars      = 0
            t_zone_bars = ps.zone_bars
            t_macd_sl   = cur_macd_sl
            t_ema_sl    = cur_ema_sl

            break   # one trade at a time

    return trades, logs


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_poi_strategy_report(
    trades: list[POITrade],
    logs:   list[POITransitionLog],
    pois:   list[POI],
) -> None:
    if not trades:
        print("  No trades.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total  = sum(t.pnl_usd for t in trades)
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pf     = gw / gl if gl else float("inf")

    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    print("─" * 80)
    print(f"  Trades: {len(trades)}  ({len(wins)}W / {len(losses)}L)  "
          f"WR: {len(wins)/len(trades)*100:.1f}%  "
          f"PF: {pf:.2f}  Net: ${total:,.2f}  MaxDD: ${max_dd:,.2f}")
    print("─" * 80)

    hdr = (f"  {'Time (ET)':<10}  {'POI':>4}  {'Bias':<12}  {'Dir':<6}  "
           f"{'Entry':>8}  {'Stop':>8}  {'Target':>8}  {'Exit':>8}  "
           f"{'Reason':<12}  {'PnL':>9}  R")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for t in trades:
        et_ts = t.entry_time.astimezone(_ET)
        print(
            f"  {str(et_ts)[11:19]:<10}  #{t.poi_rank:>3}  {t.poi_bias:<12}  "
            f"{t.direction:<6}  {t.entry_price:>8.2f}  {t.stop_price:>8.2f}  "
            f"{t.target_price:>8.2f}  {t.exit_price:>8.2f}  "
            f"{t.exit_reason:<12}  ${t.pnl_usd:>8,.2f}  {t.pnl_r:>+.2f}R"
        )

    print()
    print("  POI USAGE:")
    for ps_rank in [1, 2, 3]:
        poi_trades = [t for t in trades if t.poi_rank == ps_rank]
        poi        = next((p for p in pois if p.rank == ps_rank), None)
        label      = poi.label if poi else "?"
        price      = poi.price if poi else 0.0
        if poi_trades:
            w = sum(1 for t in poi_trades if t.pnl_usd > 0)
            p = sum(t.pnl_usd for t in poi_trades)
            print(f"    POI #{ps_rank} @ {price:.2f}  [{label}]  "
                  f"→ {len(poi_trades)} trade(s), {w}W  ${p:,.2f}")
        else:
            status = next((lg.state_after for lg in reversed(logs)
                           if lg.poi_rank == ps_rank), "WATCHING")
            print(f"    POI #{ps_rank} @ {price:.2f}  [{label}]  → no trade ({status})")
